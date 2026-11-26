"""The commands of the product in the Makefile (MFL051, plan r2.1 §3.7, §5.5, App. F.2).

What must stay true of the recipes, read from the Makefile itself (and from `make -n` where `make` is installed:
the container of the suite has none):
- `make clean` never names the archive, the state or the weights: it removes what the services rebuild by
  themselves, and compose never gets `-v`;
- `make restart` creates the services again without building and without downloading;
- `make down` stops everything and removes no volume;
- `make clean-archive` deletes the archive only by its own identifier and only with the product stopped;
- compose never opens the env file of the user (`--env-file /dev/null`), and nothing starts compose.public.yaml.
"""
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
TEXT = (ROOT / "Makefile").read_text()
KEPT = ("mdf-models", "mdf-cpu-server", "MDF_COMPOSE_DATA", "MDF_COMPOSE_STATE", "/data", "/state")
# the one recursive removal of the Makefile: the archive, in `clean-archive`
REMOVAL = "rm " + "-rf"


def recipe(target):
    """The lines of the recipe of `target`, joined."""
    match = re.search(r"^%s:[^\n]*\n((?:\t[^\n]*\n)+)" % re.escape(target), TEXT, re.M)
    if match is None:
        raise AssertionError("no recipe for " + target)
    return match.group(1)


def variable(name):
    return re.search(r"^%s\s*[:?]?=\s*(.*)$" % re.escape(name), TEXT, re.M).group(1)


class MakefileTests(unittest.TestCase):
    def test_the_commands_of_the_product_exist_and_are_listed(self):
        for target in ("up", "down", "restart", "rebuild", "status", "logs", "prepare", "install-cli", "bench", "clean",
                       "clean-archive", "test", "check", "test-components", "images", "help"):
            with self.subTest(target=target):
                self.assertRegex(TEXT, r"(?m)^%s:.*## \S" % re.escape(target))
        phony = re.search(r"^\.PHONY:(.*)$", TEXT, re.M).group(1).split()
        for target in ("up", "down", "restart", "rebuild", "status", "logs", "prepare", "clean", "clean-archive"):
            self.assertIn(target, phony)

    def test_compose_never_opens_the_env_file_of_the_user(self):
        self.assertEqual(variable("COMPOSE"), "docker compose --env-file /dev/null")
        self.assertEqual(variable("TOOLS"), "$(COMPOSE) --profile tools")
        # every call of compose for the product goes through the two variables
        product = TEXT[TEXT.index("# --- the product"):TEXT.index("# --- the checks")]
        commands = [line for line in product.splitlines() if not line.startswith("#")]
        self.assertEqual([line for line in commands if "docker compose" in line], [])
        self.assertNotIn("compose.public", TEXT)

    def test_clean_never_names_the_archive_the_state_or_the_weights(self):
        body = recipe("clean")
        self.assertEqual(variable("EPHEMERAL").split(), ["mdf-work", "mdf-run", "mdf-launcher", "mdf-api", "mdf-settings"])
        for name in KEPT:
            self.assertNotIn(name, body)
            self.assertNotIn(name, variable("EPHEMERAL"))
        self.assertNotRegex(body, r"down[^\n]*\s(-v|--volumes)\b")                # compose would take every volume of the project
        self.assertNotIn(REMOVAL, body)
        self.assertNotIn("prune", TEXT)
        self.assertIn('docker volume rm "$${COMPOSE_PROJECT_NAME}_$$volume"', body)   # by exact name, of this project

    def test_down_removes_no_volume(self):
        body = recipe("down")
        self.assertIn("$(TOOLS) down --remove-orphans", body)
        self.assertNotRegex(body, r"\s(-v|--volumes)\b")
        self.assertNotIn("volume rm", body)
        # the children of the engine are not services of compose: the ones of this project only, by its label
        self.assertIn('--filter "label=mdf.project=$$COMPOSE_PROJECT_NAME"', body)

    def test_restart_creates_the_services_again_without_a_build_or_a_download(self):
        body = recipe("restart")
        self.assertIn("$(COMPOSE) up -d --force-recreate --no-build", body)
        for word in ("$(BUILD)", "buildx", "prepare", "$(MAKE)", "chmod"):
            self.assertNotIn(word, body)

    def test_up_is_the_only_recipe_that_gives_the_env_file_its_mode_and_only_in_the_state_mode(self):
        self.assertEqual(len(re.findall(r"chmod 600", TEXT)), 1)
        body = recipe("up")
        self.assertRegex(body, r'mode\) chmod 600 "\$\(ENV\)"')
        for state in ("missing", "link", "type", "owner"):
            self.assertRegex(body, r"%s\)[^;]*; exit 1 ;;" % state)
        self.assertLess(body.index("env_state"), body.index("compose_environment"))
        # the steps, in order: host, env file, images, access as the launcher, settings, prepare, services
        marks = ["docker_host_check", "env_state", "--target engine", "markdown_for_llms.launcher check", "$(GATE) 2> /dev/null) ||",
                 "prepare ENV=", "$(COMPOSE) up -d --no-build"]
        positions = [body.index(mark) for mark in marks]
        self.assertEqual(positions, sorted(positions))

    def test_an_env_file_the_archive_would_refuse_stops_the_command_before_the_services(self):
        """The `config` step is run as a gate: a service refused by its archive would be started again for ever."""
        self.assertEqual(variable("GATE"), "$(COMPOSE) run --rm -T config")
        for target, start in (("up", "$(COMPOSE) up -d --no-build"), ("restart", "$(COMPOSE) up -d --force-recreate")):
            body = recipe(target)
            # its answer is kept and shown only with a refusal
            gate = re.search(r"refused=\$\$\(\$\(GATE\) 2> /dev/null\) \|\| \{ echo \"\$\$refused\" >&2;[^}]*exit 1; \}", body)
            self.assertIsNotNone(gate, target)
            self.assertLess(gate.start(), body.index(start))
        status = recipe("status")
        self.assertIn("MDF_COMPOSE_CONFIG_MODE=--check $(COMPOSE) run --rm -T --no-deps config", status)   # it writes nothing
        self.assertNotIn("$(GATE)", status)

    def test_the_bench_runs_in_its_container_through_the_launcher_of_the_product(self):
        body = recipe("bench")
        self.assertEqual(variable("BENCH"), "run --level quick")
        self.assertIn("$(TOOLS) run --rm -T --no-deps bench mdf bench $(BENCH)", body)
        self.assertLess(body.index("ps --status running -q launcher"), body.index("run --rm"))    # no launcher, no bench
        self.assertNotIn(" up ", body)                                                            # and nothing is started

    def test_the_archive_goes_only_by_its_own_identifier_with_the_product_stopped(self):
        body = recipe("clean-archive")
        self.assertIn('if [ "$(CONFIRM)" != "$$id" ]', body)
        self.assertLess(body.index('"$(CONFIRM)" != "$$id"'), body.index(REMOVAL))
        self.assertLess(body.index("the product is running"), body.index(REMOVAL))
        self.assertEqual(re.findall(re.escape(REMOVAL) + r"[^\n;]*", TEXT), [REMOVAL + ' -- "$$MDF_COMPOSE_DATA"'])
        self.assertNotIn("MDF_COMPOSE_STATE", body)

    def test_the_cleanup_of_the_component_tests_removes_the_driver_before_it_lists_the_children(self):
        """Seen on a real daemon (MFL051 A2 fix): `docker compose run` ends at a signal and leaves its container
        running. A driver that still runs asks for a child after the children were listed: that child stayed, and
        the work volume with it. So the driver of this invocation goes first, then the launcher is stopped, then
        the children go by their label, then the project."""
        body = recipe("test-components")
        cleanup = body[body.index("cleanup() {"):body.index("trap cleanup EXIT")]
        steps = ('docker ps -a -q --filter "label=com.docker.compose.project=$$id" '
                 '--filter "label=com.docker.compose.service=suite"',
                 'docker container rm -f "$$driver"', "$$compose stop -t 30 launcher",
                 'docker ps -a --filter "label=mdf.project"', "$$compose down -v --remove-orphans")
        order = [cleanup.index(step) for step in steps]
        self.assertEqual(order, sorted(order))
        # nothing but what carries the identifier of the invocation is removed
        self.assertEqual(cleanup.count("rm -f"), 2)
        self.assertIn('case "$$project" in "$$id"|"$$id"-*) docker container rm -f "$$child"', cleanup)

    @unittest.skipIf(shutil.which("make") is None, "no make here (the container of the suite): the recipes are read above")
    def test_what_make_would_run(self):
        def dry(*args):
            proc = subprocess.run(["make", "-n", "--no-print-directory", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return proc.stdout
        clean = dry("clean")
        for name in KEPT[:2]:
            self.assertNotIn(name, clean)
        self.assertIn("for volume in mdf-work mdf-run mdf-launcher mdf-api mdf-settings; do", clean)
        restart = dry("restart", "ENV=some.env")
        self.assertIn('compose_environment "some.env"', restart)
        self.assertNotIn("buildx", restart)
        self.assertIn("-f", dry("logs", "FOLLOW=1", "SERVICE=worker").split("--tail=200")[1])


if __name__ == "__main__":
    unittest.main()
