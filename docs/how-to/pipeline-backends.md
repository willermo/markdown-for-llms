# Choose backends and scenarios of the pipeline

Prepare [environment and preflight](uv-environment.md), generate the [JSON configuration](../reference/pipeline-config.md) in the external workspace and keep the source and the original extraction. `RUN_ENV` is the absolute path of the canonical venv. Run `--config-only` first to check directories and routing.

## Backends and formats

PDF uses `pdf_converter`: `local_marker` requires a server deliberately prepared and started; `cloud_marker` uses the configured endpoint and the key named by `marker_cloud_api_key_env`. The cloud uploads documents to the service: select it explicitly only for admitted data. No ML acquisition or server activation is part of the client/Pandoc path.

Non-PDF uses Pandoc for epub/mobi/azw/azw3/html/htm/docx/rtf. Declared support does not guarantee the conversion of every variant. `marker_document_formats`, empty by default, enables an opt-in Marker routing also for formats other than PDF, including formats that the server must really support. The list does not install or test that engine. An unrecognized backend or a Marker error on a non-PDF can produce a **Pandoc** fallback, without implicit cloud upload; check converter_used and the messages. Operational sources: [unified_converter.py](../../unified_converter.py), [config.py](../../config.py).

The cloud sends the file, receives a check_url and polls. `max_polls=300` and `poll_interval=2` are finite defaults; `MAX_POLLS<=0` makes the polling unlimited. `CONVERSION_TIMEOUT=0` makes the wait for the response unlimited, distinct from `CONNECT_TIMEOUT`. Prefer finite and explicit values; increasing timeouts does not fix a wrong document or a server without memory. ML/OCR/imaging options do not prove that the local wrapper applies them. `/health` measures liveness.

## Documents and phases

Create `source_pdfs` and `source_documents` in the workspace, put admitted files there and use the full conversion or separate steps. The direct command of the converter requires source-dir/output-dir and produces its own results and reports:

```bash
"$RUN_ENV/bin/python" -I -B -m master_workflow --config-only
"$RUN_ENV/bin/python" -I -B -m master_workflow
"$RUN_ENV/bin/python" -I -B -m unified_converter --source-dir source_documents --output-dir converted_markdown
"$RUN_ENV/bin/python" -I -B -m master_workflow --step cleaning --force
"$RUN_ENV/bin/python" -I -B -m master_workflow --step validation --force
"$RUN_ENV/bin/python" -I -B -m master_workflow --step chunking --llm custom --chunk-size 1000 --overlap 100
"$RUN_ENV/bin/python" -I -B -m batch_monitor --status
"$RUN_ENV/bin/python" -I -B -m batch_monitor --check-queue
```

The pipeline can discard documents in validation; the monitor does not certify fidelity. Compare formulas, numbers, code, order, complete reports and assets. HTML/Pandoc can lose anchors; cleaning can remove content and assets are not always copied. An exit 0 does not cancel these losses.

For research and reading use the complete Markdown with assets. For retrieval or context limits produce chunks as a derivative, checking their sequence and overlap. No prepare_training_data/create_embeddings/RAG script is provided: external integrations require a separate project and data handling. Do not blindly copy only validated_markdown when the validation has excluded sources. The console forms and the isolated modules are described in the [CLI reference](../reference/toolchain-legacy.md).

For the local API, weights/fonts, CPU/GPU inference and deploy follow the [Docker limits and preparation](marker-legacy-docker.md): build and offline contract do not prove real Marker conversions. Do not run services for a fast test or use a dummy PDF as a content test.

[Index](README.md).
