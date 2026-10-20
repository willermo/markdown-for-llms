# Markdown for LLMs

Experimental local converter from PDF to Markdown with assets: the `mdf` CLI, an inert text Web UI, a separate worker and a SQLite and filesystem archive. It keeps the source, the original extraction and every attempt.

Status: the first increment I1 is integrated in `dev` and not released on `main`. The only profile is Linux x86_64 on CPU, single user, loopback access, with Marker 1.10.2. Quality is always `unverified`: check formulas and contents against the PDF.

- [First conversion in the browser or the CLI](docs/tutorials/first-local-pdf.md).
- [Prepare the package, runtime and local models](docs/how-to/local-product-setup.md).
- [Configuration, commands and API](docs/reference/local-product.md).
- [Bundle states and limits of the experimental product](docs/explanation/experimental-product.md).

## Legacy pipeline

The script pipeline stays separate from the new converter. Its `markdown-*` commands and the procedures below are legacy and have known losses in cleaning and in the copy of assets.

- [Prepare the uv environment and rebuild the distribution](docs/how-to/uv-environment.md).
- [Toolchain, commands, configuration and test layers](docs/reference/toolchain-legacy.md).
- [Prepare and verify Marker in Docker](docs/how-to/marker-legacy-docker.md).
- [JSON configuration of the pipeline](docs/reference/pipeline-config.md).
- [Backends, documents and conversion scenarios](docs/how-to/pipeline-backends.md).

The client needs only the Python runtime and an installed Pandoc; FastAPI and the Marker engine are separate extras. The shell or pyenv Python is not the project interpreter: use the virtual environment linked to the local managed Python, after rebuilding and verifying the ten installed modules against the sources and the canonical wheel.

From an external data workspace, with a verified environment, these are operational examples of the base paths (Pandoc for HTML; no Marker inference). `RUN_ENV` is the absolute path of the virtual environment; `PYTHONPATH` and `PYTHONHOME` are removed for the console scripts.

```bash
"$RUN_ENV/bin/python" -I -B -m config --create-default
"$RUN_ENV/bin/python" -I -B -m master_workflow --config-only
"$RUN_ENV/bin/python" -I -B -m master_workflow --step conversion --force
"$RUN_ENV/bin/python" -I -B -m master_workflow --step cleaning --force
"$RUN_ENV/bin/python" -I -B -m master_workflow --step validation --force
"$RUN_ENV/bin/python" -I -B -m master_workflow --step chunking --force --llm custom --chunk-size 1000 --overlap 100
```

Generate `pipeline_config.json` in the workspace and set the backend before converting. PDF uses the `local_marker` client by default; supported documents use Pandoc, unless `marker_document_formats` selects them explicitly. The cloud path is optional and needs a deliberate configuration and credentials. The migration does not verify local or remote Marker inference or GPU hardware.

The directories are `source_pdfs`, `source_documents`, `converted_markdown`, `cleaned_markdown`, `validated_markdown`, `chunked_markdown` and `pipeline_logs`. `source_ebooks` is a historical directory: new commands use the configured `source_documents`. `validation_report.json` is at the root of the workspace; `pipeline_state.json` is in the log directory.

Known limits of the legacy pipeline:

- The validator can discard content: exit code 0 does not certify that the document is complete.
- Legacy cleaning uses global transformations that can alter formulas, code, numbers, links and assets: compare them with the source.
- Chunks do not replace the complete Markdown; the legacy semantic overlap can be zero. There is no `--chunking-strategy` flag.
- The Marker wrapper returns empty images and reports options without applying them; `/health` measures liveness only, even after a model loading error. Build, offline contract and real inference are separate checks; the instructions for startup, weights, fonts and CPU/GPU stay outside the fast tests.
- The analysis, embedding, RAG and fine-tuning scripts cited by the old guide are not part of the repository; their earlier examples were external pseudocode, not commands of this project.

## Documentation and licence

[`docs/`](docs/README.md) collects the user and developer guides, [`documentation/`](documentation/README.md) the decisions, the [roadmap](documentation/roadmap.md), the limits and the technical debt, and the [changelog](CHANGELOG.md) the changes for users. Contributors and agents follow [`AGENTS.md`](AGENTS.md).

The application package `markdown_for_llms` and the flat legacy modules coexist. Licence [MIT](LICENSE); the conditions of the model weights are separate.
