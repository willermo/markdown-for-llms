#!/usr/bin/env python3
"""A fake vLLM for the tests of the front of a GPU of the user (MFL051 B): an HTTP server on 127.0.0.1:PORT that
answers the paths the front forwards like vLLM 0.25.1 and records every request it receives (method, path) in
RECORD, one JSON line each, so a test sees what reached it. Any other path answers 200 "reached": a path that the
front must not forward shows up in the record. Standard library only.

    python3 fake_vllm.py PORT RECORD
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import signal
import sys
import threading

MODEL = "datalab-to/surya-ocr-2.5"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def record(self, body=b""):
        with self.server.lock, open(self.server.record, "a") as fh:
            fh.write(json.dumps({"method": self.command, "path": self.path, "bytes": len(body),
                                 "authorization": self.headers.get("Authorization")}) + "\n")

    def answer(self, status, doc, content_type="application/json"):
        data = doc if isinstance(doc, bytes) else json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_GET(self):
        self.record()
        path = self.path.split("?", 1)[0]
        if path == "/v1/models":
            return self.answer(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        if path == "/version":
            return self.answer(200, {"version": "0.25.1"})
        if path == "/metrics":
            return self.answer(200, b"vllm:num_requests_running 0\n", "text/plain")
        return self.answer(200, {"reached": path})

    do_HEAD = do_GET

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        self.record(body)
        if self.path == "/v1/chat/completions":
            return self.answer(200, {"id": "chat-1", "choices": [{"message": {"content": "ok %d" % len(body)}}]})
        return self.answer(200, {"reached": self.path})

    def do_DELETE(self):
        self.record()
        return self.answer(200, {"reached": self.path})

    do_PUT = do_DELETE


def main():
    port, record = int(sys.argv[1]), sys.argv[2]
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.record, server.lock = record, threading.Lock()
    signal.signal(signal.SIGTERM, lambda *a: threading.Thread(target=server.shutdown).start())
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
