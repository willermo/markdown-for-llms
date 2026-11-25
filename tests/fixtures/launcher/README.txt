Fixtures of the launcher
========================

(Plain text on purpose: `tests/product/test_product.py` keeps a closed list of the tracked Markdown files under `tests/fixtures/`.)

Used by `tests/unit/launcher/` (MFL051, tranche A1). Nothing here comes from a document or from a user: every file is produced by the code of this repository. In both kinds of file the strings `<supervisor>` and `<shell-probe>` stand for two scripts of the package, the supervisor of a child (`markdown_for_llms/engine/supervisor.py`) and the shell probe of the toolchain image (`containers.SHELL_PROBE`), which the tests put back before they compare.

`body-<kind>-<image>.json`: golden bodies
-----------------------------------------

The body of `POST /containers/create` that the launcher builds for one request of each kind and image (`launcher_support.golden_cases`), with a synthetic project (`mdf-unit`), user (`1000:1000`), image ID and instant (`mdf.t0` 1000.0). They are written from `launcher.body_of` and reviewed by reading them: a change of the body shows up as a change of these files in a diff.

`inspect-<kind>-<image>.json`: recorded inspects of real containers
-------------------------------------------------------------------

Each file holds the answer of a real Docker daemon to `GET /containers/{id}/json` for a container created, and never started, from the launcher's own body, together with the request and the context that produced it (`docker`, `context`, `request`, `inspect`). They are the positive fixtures of the verification after the creation: what a real daemon writes in the fields it generates or fills by itself (`Hostname`, `Image`, `Cmd`, `Entrypoint`, `Runtime`, `MaskedPaths`, `ConsoleSize` and the rest).

- Recorded by `tests/container/test_child_isolation.py` (`TestCreatedAsAsked`) during `make test-components`, which writes them to `.cache/test-components/<id>/fixtures/`; copied here by hand.
- Daemon: Docker Engine 29.10.0 (API 1.57, runc, cgroup v2, overlay2) on Linux x86_64, 2026-11-25.
- Made anonymous when recorded: the ID of the container (and so its hostname) is replaced by a fixed one, the name of the compose project by `mdf-fixture`, the user by `1000:1000`, the attempt by `0123456789abcdef`, the timestamps by a fixed instant and the layer directories of Docker by zeros. The image IDs are those of local builds. No path of a home directory, user name or host name is in them (`tests/unit/test_repository_hygiene.py` checks the paths).

Record them again when the body of a kind changes on purpose, or to add the inspect of another version of Docker.
