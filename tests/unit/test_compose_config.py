"""The product in compose, judged on what compose itself reads (MFL051, plan r2.1 §3.1, §3.3, §5.1, App. F.2).

The object is the JSON of `docker compose config` for compose.yaml (and for compose.public.yaml over it), with the
defaults of an env file that is not there. `make check` and `make test` save it among their evidence
(scripts/check.sh: `compose-product.json`, `compose-public.json`), because the container of the suite has no
Docker; on a host with Docker the test asks compose itself.

What must hold, whatever else changes in the files:
- one published port, on 127.0.0.1; nginx is the only service on a network that publishes;
- no `env_file:` anywhere, and no name of a key in any `environment:`: the keys are read from the mounted file;
- the env file of the user is mounted only where keys are read (worker, cli, bench) and in `config`; never in
  the API, in nginx, in the launcher;
- the Docker socket in the launcher alone; the API without network;
- every service as the user of the host with a read-only root, no capability and `no-new-privileges`; UID 0 and
  the capability CHOWN in `init` alone, and CAP_FOWNER nowhere; the launcher with the group of the socket;
- a memory cap without swap on every service;
- `cli` and `bench` start no other service; the launcher has no `hostname:` (it finds itself by it).
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
UID, GID = str(os.getuid()), str(os.getgid())
KEY_NAME = re.compile(r"(_KEY|_SECRET|_TOKEN|_TOKEN_ID|_BEARER|PASSWORD)$")
ASK = ('. scripts/env-preflight.sh && compose_environment /nonexistent/.env && '
       'docker compose --env-file /dev/null %s --profile tools config --format json')


def load(name, files):
    proof = os.environ.get("FULL_CHECK_PROOF_DIR")
    if proof and (Path(proof) / name).is_file():
        return json.loads((Path(proof) / name).read_text())
    if shutil.which("docker") is None:
        raise unittest.SkipTest("no saved `docker compose config` (FULL_CHECK_PROOF_DIR) and no Docker here")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MDF_", "COMPOSE_"))}
    proc = subprocess.run(["sh", "-euc", ASK % files], cwd=ROOT, capture_output=True, text=True, env=env, timeout=120)
    if proc.returncode != 0:
        raise AssertionError("docker compose config: " + proc.stderr[-500:])
    return json.loads(proc.stdout)


def mounts(service, kind=None):
    return [v for v in service.get("volumes", []) if kind is None or v["type"] == kind]


class ProductComposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load("compose-product.json", "")
        cls.services = cls.config["services"]

    def test_the_services_of_the_product(self):
        self.assertEqual(sorted(self.services), ["api", "bench", "cli", "config", "init", "launcher", "nginx", "worker"])
        self.assertEqual({name for name, s in self.services.items() if s.get("profiles")}, {"cli", "bench"})
        for name in ("cli", "bench"):
            self.assertEqual(self.services[name]["profiles"], ["tools"])
        self.assertEqual(sorted(self.config["volumes"]), ["mdf-api", "mdf-cpu-server", "mdf-launcher", "mdf-models", "mdf-run",
                                                          "mdf-settings", "mdf-work"])

    def test_one_published_port_on_the_loopback(self):
        ports = [(name, port) for name, s in self.services.items() for port in s.get("ports", [])]
        self.assertEqual(len(ports), 1)
        name, port = ports[0]
        self.assertEqual((name, port["host_ip"], port["published"], port["target"]), ("nginx", "127.0.0.1", "23465", 8080))
        self.assertTrue(23460 <= int(port["published"]) <= 23479)                    # the block of the project

    def test_networks(self):
        nets = {name: (s.get("network_mode") or sorted(s.get("networks") or {})) for name, s in self.services.items()}
        self.assertEqual(nets, {"init": "none", "config": "none", "launcher": "none", "api": "none", "nginx": ["front"],
                                "worker": ["egress"], "cli": ["egress"], "bench": ["egress"]})
        for name, s in self.services.items():
            self.assertNotIn(s.get("network_mode"), ("host", "bridge"), name)

    def test_no_key_enters_the_environment_of_a_container(self):
        for name, s in self.services.items():
            with self.subTest(service=name):
                self.assertNotIn("env_file", s)
                for variable in s.get("environment") or {}:
                    self.assertIsNone(KEY_NAME.search(variable), variable)
        # the settings reach the services through settings.env: only the bench names of the one-off tools are in an environment
        mdf = {name: sorted(v for v in (s.get("environment") or {}) if v.startswith("MDF_")) for name, s in self.services.items()}
        self.assertEqual({name: names for name, names in mdf.items() if names},
                         {"cli": ["MDF_BENCH_CACHE_DIR", "MDF_BENCH_RUNS_DIR"], "bench": ["MDF_BENCH_CACHE_DIR", "MDF_BENCH_RUNS_DIR"],
                          "nginx": ["MDF_BODY_MIB"]})

    def test_the_env_file_is_mounted_read_only_and_only_where_it_is_read(self):
        seen = {}
        for name, s in self.services.items():
            for v in mounts(s, "bind"):
                if v["source"].endswith("/.env"):
                    seen[name] = (v["target"], v.get("read_only"), (v.get("bind") or {}).get("create_host_path"))
        self.assertEqual(seen, {name: ("/run/mdf/env", True, False) for name in ("config", "worker", "cli", "bench")})
        commands = {name: s.get("command") or [] for name, s in self.services.items()}
        self.assertIn("--keys-file=/run/mdf/env", commands["worker"])
        self.assertIn("--env-file=/run/mdf/settings/settings.env", commands["worker"])
        self.assertEqual(commands["api"], ["mdf", "--env-file=/run/mdf/settings/settings.env", "serve", "--uds=/run/mdf/api/api.sock"])
        self.assertFalse(any("keys-file" in argument for argument in commands["api"]))

    def test_the_docker_socket_is_in_the_launcher_alone(self):
        holders = [name for name, s in self.services.items() for v in mounts(s) if "docker.sock" in v["target"]
                   or "docker.sock" in str(v.get("source"))]
        self.assertEqual(holders, ["launcher"])
        launcher = self.services["launcher"]
        self.assertEqual((launcher["user"], launcher.get("network_mode")), ("%s:%s" % (UID, GID), "none"))
        self.assertEqual(len(launcher["group_add"]), 1)                              # the group of the socket, nothing else
        self.assertNotIn("hostname", launcher)
        self.assertNotIn("container_name", launcher)
        # it sees the volumes of the children read-only, and names them for the children by their exact names
        shared = {v["target"]: v.get("read_only", False) for v in mounts(launcher, "volume")}
        self.assertEqual((shared["/work"], shared["/run/mdf/run"]), (True, True))
        project = self.config["name"]
        for role, volume in (("work", "mdf-work"), ("run", "mdf-run"), ("models", "mdf-models"), ("cpuserver", "mdf-cpu-server")):
            self.assertIn("--volume=%s=%s_%s" % (role, project, volume), launcher["command"])
            self.assertEqual(self.config["volumes"][volume]["name"], "%s_%s" % (project, volume))

    def test_users_capabilities_and_roots(self):
        for name, s in self.services.items():
            with self.subTest(service=name):
                self.assertEqual(s["user"], "0:0" if name == "init" else "%s:%s" % (UID, GID))
                self.assertIs(s.get("read_only"), True)
                self.assertEqual(s.get("cap_drop"), ["ALL"])
                self.assertEqual(s.get("cap_add", []), ["CHOWN"] if name == "init" else [])
                self.assertEqual(s.get("security_opt"), ["no-new-privileges"])
                self.assertNotIn("privileged", s)
                self.assertNotIn("pid", s)
                self.assertNotIn("devices", s)
                if name != "launcher":
                    self.assertEqual(s.get("group_add", []), [])
        self.assertNotIn("FOWNER", json.dumps(self.config))
        init = self.services["init"]
        self.assertEqual(init["command"][:2], [UID, GID])
        self.assertEqual(sorted(init["command"][2:]), sorted(v["target"] for v in mounts(init, "volume")))
        self.assertEqual(len(init["command"]) - 2, len(self.config["volumes"]))      # the root of every named volume
        self.assertFalse([v for v in mounts(init) if v["target"] in ("/data", "/state", "/run/mdf/env")])

    def test_a_memory_cap_without_swap_on_every_service(self):
        for name, s in self.services.items():
            with self.subTest(service=name):
                self.assertTrue(int(s["mem_limit"]) > 0)
                self.assertEqual(s["memswap_limit"], s["mem_limit"])
                self.assertTrue(int(s["pids_limit"]) > 0)

    def test_the_one_off_tools_start_nothing_and_the_order_of_the_services(self):
        for name in ("cli", "bench"):
            self.assertNotIn("depends_on", self.services[name])
            self.assertEqual(self.services[name].get("restart"), "no")
        self.assertEqual(sorted(self.services["worker"]["depends_on"]), ["config", "launcher"])
        self.assertEqual(self.services["worker"]["depends_on"]["config"]["condition"], "service_completed_successfully")
        self.assertEqual(self.services["api"]["depends_on"]["config"]["condition"], "service_completed_successfully")
        self.assertEqual(self.services["config"]["depends_on"]["init"]["condition"], "service_completed_successfully")
        self.assertEqual(self.services["launcher"]["depends_on"]["init"]["condition"], "service_completed_successfully")
        self.assertEqual({name: s.get("restart") for name, s in self.services.items() if name not in ("cli", "bench")},
                         {"init": "no", "config": "no", "launcher": "unless-stopped", "api": "unless-stopped",
                          "worker": "unless-stopped", "nginx": "unless-stopped"})
        self.assertEqual((self.services["worker"]["stop_grace_period"], self.services["launcher"]["stop_grace_period"]),
                         ("11m0s", "30s"))

    def test_what_each_service_sees(self):
        def seen(name):
            return {v["target"]: ("ro" if v.get("read_only") else "rw") for v in mounts(self.services[name])}
        self.assertEqual(seen("api"), {"/data": "rw", "/run/mdf/api": "rw", "/run/mdf/settings": "ro"})
        self.assertEqual(seen("nginx"), {"/etc/nginx/templates/default.conf.template": "ro", "/run/mdf/api": "ro"})
        # `config` reads the archive before any service opens it (its identity against the settings), and only reads it
        self.assertEqual(seen("config"), {"/run/mdf/env": "ro", "/run/mdf/settings": "rw", "/data": "ro"})
        self.assertEqual(self.services["config"]["command"][-1], "--write")       # `--check` only when `make status` asks
        worker = {"/data": "rw", "/state": "rw", "/run/mdf/env": "ro", "/run/mdf/settings": "ro", "/run/mdf/launcher": "ro",
                  "/work": "rw", "/run/mdf/run": "rw", "/models": "ro", "/cpu-server": "ro"}
        self.assertEqual(seen("worker"), worker)
        # the one-off tools prepare the weights and the CPU server: those two volumes read-write, the rest as the worker
        self.assertEqual(seen("cli"), dict(worker, **{"/models": "rw", "/cpu-server": "rw"}))
        self.assertEqual(seen("bench"), seen("cli"))
        # the folders of the user are never made by Docker (it would make them as root)
        for name in ("api", "worker", "cli", "config"):
            for v in mounts(self.services[name], "bind"):
                self.assertIs(v["bind"]["create_host_path"], False, v["target"])
        # no anonymous volume: every volume has a name of the project, every bind a source
        for name, s in self.services.items():
            for v in mounts(s):
                self.assertTrue(v.get("source"), (name, v["target"]))

    def test_the_private_files_of_the_credentials_are_in_memory_and_of_the_user(self):
        for name in ("worker", "cli", "bench"):
            secrets = [t for t in self.services[name]["tmpfs"] if t.startswith("/run/mdf/secrets:")]
            self.assertEqual(len(secrets), 1)
            for option in ("mode=0700", "uid=" + UID, "gid=" + GID, "noexec"):
                self.assertIn(option, secrets[0])
        self.assertFalse([t for t in self.services["api"].get("tmpfs", []) if "secrets" in t])

    def test_nginx_by_digest(self):
        self.assertRegex(self.services["nginx"]["image"], r"^nginx:1\.30\.6-alpine@sha256:[0-9a-f]{64}$")


class PublicComposeTests(unittest.TestCase):
    """compose.public.yaml is never started: it has to stay a valid shape over compose.yaml, and keep its rules."""

    @classmethod
    def setUpClass(cls):
        cls.config = load("compose-public.json", "-f compose.yaml -f compose.public.yaml")
        cls.services = cls.config["services"]

    def test_the_archive_and_the_state_are_named_volumes(self):
        for name in ("api", "worker", "cli", "bench"):
            kinds = {v["target"]: (v["type"], v["source"]) for v in mounts(self.services[name]) if v["target"] in ("/data", "/state")}
            self.assertEqual(kinds["/data"], ("volume", "mdf-data"), name)
            if name != "api":
                self.assertEqual(kinds["/state"], ("volume", "mdf-state"), name)
        init = self.services["init"]
        self.assertEqual(sorted(init["command"][2:]), sorted(v["target"] for v in mounts(init, "volume")))
        self.assertEqual(len(init["command"]) - 2, len(self.config["volumes"]))

    def test_the_rules_of_the_local_file_still_hold(self):
        ports = [(name, port) for name, s in self.services.items() for port in s.get("ports", [])]
        self.assertEqual([(name, port["host_ip"]) for name, port in ports], [("nginx", "127.0.0.1")])
        for name, s in self.services.items():
            self.assertNotIn("env_file", s)
            self.assertEqual(s.get("cap_drop"), ["ALL"])
        nginx = {v["target"]: v["source"] for v in mounts(self.services["nginx"], "bind")}
        self.assertTrue(nginx["/etc/nginx/templates/default.conf.template"].endswith("/docker/nginx/public.conf.template"))

    def test_nothing_starts_it(self):
        for name in ("Makefile", "scripts/prepare-toolchain.sh", "scripts/env-preflight.sh"):
            text = (ROOT / name).read_text()
            self.assertNotIn("compose.public.yaml", text, name)
        self.assertEqual(len(re.findall(r"compose\.public\.yaml", (ROOT / "scripts/check.sh").read_text())), 1)   # `config` only


class TemplateTests(unittest.TestCase):
    def test_the_door_answers_to_the_loopback_names_and_closes_the_others(self):
        text = (ROOT / "docker/nginx/local.conf.template").read_text()
        self.assertIn("server_name 127.0.0.1 localhost;", text)
        self.assertRegex(text, r"listen 8080 default_server;\s+server_name _;\s+return 444;")
        self.assertIn("proxy_pass http://unix:/run/mdf/api/api.sock:;", text)
        self.assertIn("proxy_set_header Host $http_host;", text)              # the API checks Host, Origin and CSRF itself
        self.assertIn("client_max_body_size ${MDF_BODY_MIB}m;", text)
        for line in ("proxy_request_buffering off;", "proxy_buffering off;"):
            self.assertIn(line, text)
        self.assertNotRegex(text, r"listen\s+(?!8080)")
        # the only names the image substitutes are the ones of the service (NGINX_ENVSUBST_FILTER ^MDF_)
        self.assertEqual(set(re.findall(r"\$\{(\w+)\}", text)), {"MDF_BODY_MIB"})


if __name__ == "__main__":
    unittest.main()
