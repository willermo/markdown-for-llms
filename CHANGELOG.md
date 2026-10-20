# Changelog

All notable changes for users of this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); no version has been released yet.

## [Unreleased]

### Added

- Experimental local converter `mdf` (package `markdown_for_llms`): PDF to Markdown with assets through Marker 1.10.2 on CPU, Linux x86_64, single user on loopback. It ships a CLI, a FastAPI API and an inert text Web UI, a separate worker process and a SQLite and filesystem archive that keeps the source, the original extraction and every attempt. See the [first conversion tutorial](docs/tutorials/first-local-pdf.md).
- Result bundle states `ready`, `unchecked`, `incomplete` and `invalid`, kept apart from the job state; content quality is always reported as `unverified`. ZIP download, cancel, retry and worker recovery after an interruption.
- Typed `MDF_` configuration read from an explicit environment file; see `.env.example` and the [reference](docs/reference/local-product.md).
- Optional dependency groups `app`, `marker-cpu` and `marker-server`; `marker-cu126` is declared but GPU profiles are not verified.
- User and developer documentation in `docs/`, organized by tutorials, how-to guides, reference and explanation.
- Estimated conversion progress in the Web UI (a progress bar updated in place), in the API and CLI JSON (`progress` field, separate from `state`) and on stderr for `mdf convert --wait` on a terminal. The percentage is an estimate; the engine phase and its count are exact.
- `MDF_STALL_SECONDS` (default 900, minimum 60): the worker stops a conversion whose engine progress has not advanced for that long. See [ADR 0011](documentation/decisions/0011-stall-control-and-progress.md).
- `mdf downgrade-storage` returns an archive upgraded from schema 1 to the previous version.

### Changed

- Python 3.12 is required (`>=3.12,<3.13`); the minimum was `>=3.9`.
- Packaging moved to `pyproject.toml` with `uv.lock`; the legacy console scripts keep their names (`markdown-pipeline`, `markdown-config`, `markdown-clean`, `markdown-validate`, `markdown-chunk`). See the [uv environment guide](docs/how-to/uv-environment.md).
- `.env.example` now documents only the `MDF_` settings of the new converter. The legacy pipeline still reads its own environment variables (`CHUNK_SIZE`, `OVERLAP_SIZE`, `TARGET_LLM`, `MAX_WORKERS`, `LOG_LEVEL`, `CONVERSION_TIMEOUT`, `CONNECT_TIMEOUT`, `MARKER_*`, `MAX_POLLS`, `POLL_INTERVAL`), listed in the [legacy toolchain reference](docs/reference/toolchain-legacy.md).
- A missing local link destination whose last path segment has no file extension (for example a citation misparsed as a link, `k[[y, z, v]](y,z)`) now produces an `unchecked` warning instead of an `incomplete` bundle, so the ZIP stays downloadable. Missing images and missing link targets with a file extension still produce `incomplete`. An extension is a final dot followed by 1 to 10 ASCII letters or digits. See [what «bundle disponibile» means](docs/explanation/experimental-product.md).
- `MDF_TIMEOUT_SECONDS` is optional and defaults to 0, no total time limit; a positive value restores the deadline and the CPU time limit. Time limits are no longer part of the storage identity: changing them needs only a restart.
- Archives are upgraded automatically to identity and database schema 2 on the first opening, only when no other process has them open.
- The legacy Docker files build with uv; they cover the legacy pipeline only, not the new converter.
- README rewritten as the entry point to the experimental converter and to the legacy pipeline.

### Removed

- `setup.py` and `requirements.txt`, replaced by `pyproject.toml` and `uv.lock`.
- The optional dependency groups `dev` and `enhanced` of `setup.py`.
- The `--timeout-seconds` option of `mdf configure`, no longer needed.
