#!/usr/bin/env python3
"""The driver of the tests of `ca_file` (MFL051 B; J.7.4 of the plan): a subprocess, because the guard of `conftest.py`
forbids the sockets in the process of pytest. It makes a synthetic PKI (two authorities, a certificate of `localhost`
and one of `other.test`, both signed by the first; `cryptography`, already in uv.lock through google-auth), starts TLS
servers on 127.0.0.1 that answer like vLLM, and for each case runs the relay of the engine against one of them: its
readiness (the client of the probes) and one request of the child through its inbound socket (the client of the
requests). It prints one JSON object with the outcome of each case.

    python3 -I -B ca_file_driver.py WORKDIR
"""
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
from pathlib import Path
import socket
import ssl
import sys
import threading

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID  # noqa: E402

from markdown_for_llms.engine import relay, servers  # noqa: E402
from markdown_for_llms.engine.codes import EngineError  # noqa: E402

MODEL = "datalab-to/surya-ocr-2.5"
NOW = datetime.now(timezone.utc)


def authority(name):
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(NOW - timedelta(days=1))
            .not_valid_after(NOW + timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    return key, cert


def leaf(ca, dns):
    ca_key, ca_cert = ca
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, dns)]))
            .issuer_name(ca_cert.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(NOW - timedelta(days=1)).not_valid_after(NOW + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(dns)]), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    return key, cert


def pem(path, cert, key=None):
    data = cert.public_bytes(serialization.Encoding.PEM)
    if key is not None:
        data = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption()) + data
    path.write_bytes(data)
    return str(path)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def answer(self, status, doc, extra=None):
        data = json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.server.seen.append(("GET", self.path))
        if self.path == "/v1/models":
            return self.answer(200, {"data": [{"id": MODEL}]})
        if self.path == "/version":
            return self.answer(200, {"version": "0.25.1"})
        return self.answer(404, {"error": "no"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self.server.seen.append(("POST", self.path))
        if self.server.redirect:
            return self.answer(302, {"moved": True}, {"Location": "https://localhost:%d/elsewhere" % self.server.port})
        return self.answer(200, {"choices": [{"message": {"content": "ok"}}]})


def tls_server(cert_file, redirect=False):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_file)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    server.seen, server.redirect, server.port = [], redirect, server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def chat(path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(20)
    sock.connect(path)
    conn = http.client.HTTPConnection("relay", timeout=20)
    conn.sock = sock
    body = json.dumps({"model": MODEL, "messages": []}).encode()
    conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, None


def case(work, name, port, ca_file):
    raw = {"kind": "user", "url": "https://localhost:%d" % port}
    if ca_file is not None:
        raw["ca_file"] = ca_file
    ep = servers.endpoint(name, raw)
    out = {"case": name}
    try:
        r = relay.Relay(ep, {}, str(work / (name + ".sock")), ready_poll=0.1, probe_interval=60)
    except EngineError as exc:
        return dict(out, built=False, code=exc.code.value)
    r.start()
    try:
        try:
            r.ready(within=1.5)
            out["ready"] = True
        except EngineError as exc:
            out.update(ready=False, ready_code=exc.code.value)
        out["chat_status"], out["chat_body"] = chat(r.socket_path)
        out["stats"] = {k: r.stats[k] for k in ("transport_errors", "redirects", "by_status")}
    finally:
        r.close()
    return out


def main():
    work = Path(sys.argv[1])
    ca1, ca2 = authority("mdf test authority one"), authority("mdf test authority two")
    k, c = leaf(ca1, "localhost")
    good = tls_server(pem(work / "localhost.pem", c, k))
    k, c = leaf(ca1, "other.test")
    named = tls_server(pem(work / "other.pem", c, k))
    k, c = leaf(ca1, "localhost")
    moved = tls_server(pem(work / "moved.pem", c, k), redirect=True)
    ca1_file, ca2_file = pem(work / "ca1.pem", ca1[1]), pem(work / "ca2.pem", ca2[1])
    (work / "empty.pem").write_text("")
    results = [case(work, "right", good.port, ca1_file), case(work, "wrong", good.port, ca2_file),
               case(work, "name", named.port, ca1_file), case(work, "system", good.port, None),
               case(work, "redirect", moved.port, ca1_file),
               case(work, "missing", good.port, str(work / "absent.pem")),
               case(work, "empty", good.port, str(work / "empty.pem"))]
    seen = {"good": good.seen, "named": named.seen, "moved": moved.seen}
    print(json.dumps({"results": results, "seen": seen}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
