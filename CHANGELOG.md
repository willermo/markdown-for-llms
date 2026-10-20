# Changelog

All notable changes for users of this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); no version has been released yet.

## [Unreleased]

### Added

- Experimental local converter `mdf` (package `markdown_for_llms`): PDF to Markdown with assets through Marker 1.10.2 on CPU, Linux x86_64, single user on loopback. It ships a CLI, a FastAPI API and an inert text Web UI, a separate worker process and a SQLite and filesystem archive that keeps the source, the original extraction and every attempt. See the [first conversion tutorial](docs/tutorials/first-local-pdf.md).
- Result bundle states `ready`, `unchecked`, `incomplete` and `invalid`, kept apart from the job state; content quality is always reported as `unverified`. ZIP download, cancel, retry and worker recovery after an interruption.
- Typed `MDF_` configuration read from an explicit environment file; see `.env.example` and the [reference](docs/reference/local-product.md).
- Optional dependency groups `app`, `marker-cpu` and `marker-server`; `marker-cu126` is declared but GPU profiles are not verified.
- User and developer documentation in `docs/`, organized by tutorials, how-to guides, reference and explanation.

### Changed

- Python 3.12 is required (`>=3.12,<3.13`); the minimum was `>=3.9`.
- Packaging moved to `pyproject.toml` with `uv.lock`; the legacy console scripts keep their names (`markdown-pipeline`, `markdown-config`, `markdown-clean`, `markdown-validate`, `markdown-chunk`). See the [uv environment guide](docs/how-to/uv-environment.md).
- `.env.example` now documents only the `MDF_` settings of the new converter. The legacy pipeline still reads its own environment variables (`CHUNK_SIZE`, `OVERLAP_SIZE`, `TARGET_LLM`, `MAX_WORKERS`, `LOG_LEVEL`, `CONVERSION_TIMEOUT`, `CONNECT_TIMEOUT`, `MARKER_*`, `MAX_POLLS`, `POLL_INTERVAL`), listed in the [legacy toolchain reference](docs/reference/toolchain-legacy.md).
- The legacy Docker files build with uv; they cover the legacy pipeline only, not the new converter.
- README rewritten as the entry point to the experimental converter and to the legacy pipeline.

### Removed

- `setup.py` and `requirements.txt`, replaced by `pyproject.toml` and `uv.lock`.
- The optional dependency groups `dev` and `enhanced` of `setup.py`.
