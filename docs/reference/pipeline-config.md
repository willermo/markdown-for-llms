# JSON configuration of the legacy pipeline

`pipeline_config.json` belongs to the current data workspace, together with `.env`. Generate it with `"$RUN_ENV/bin/python" -I -B -m config --create-default`, after the preflight of the installed distribution. `--config-only` shows the configuration without starting conversions. The reference code is [config.py](../../config.py).

The keys actually read are `directories`, `llm_presets`, `validation_thresholds`, `cleaning_settings`, **`chunking_settings`**, `conversion_settings` and `pipeline_settings`. `chunking` is not a JSON alias. Valid example, to adapt only in the workspace:

```json
{
  "directories": {
    "source_pdfs": "source_pdfs", "source_documents": "source_documents",
    "converted": "converted_markdown", "cleaned": "cleaned_markdown",
    "validated": "validated_markdown", "chunked": "chunked_markdown",
    "logs": "pipeline_logs"
  },
  "llm_presets": {
    "custom": {"chunk_size": 4000, "overlap": 200, "description": "Configurazione locale"}
  },
  "validation_thresholds": {
    "min_content_length": 100, "max_content_length": 10000000, "min_words": 50,
    "max_artifact_ratio": 0.05, "min_readability_score": 20, "min_structure_score": 50
  },
  "cleaning_settings": {
    "aggressive_cleaning": true, "preserve_tables": true, "min_line_length": 3,
    "remove_html_tags": true, "normalize_whitespace": true
  },
  "chunking_settings": {
    "target_llm": "custom", "chunk_size": 4000, "overlap": 200,
    "min_chunk_size": 1000, "max_chunk_size": 8000, "strategy": "semantic"
  },
  "conversion_settings": {
    "pdf_converter": "local_marker", "document_converter": "pandoc",
    "marker_cloud_api_key_env": "MARKER_API_KEY",
    "marker_local_base_url": "http://localhost:8000",
    "marker_cloud_base_url": "https://www.datalab.to/api/v1",
    "marker_local_endpoint": "/convert", "marker_cloud_endpoint": "/marker",
    "pandoc_options": "--wrap=none --strip-comments --markdown-headings=atx",
    "supported_pdf_formats": ["pdf"],
    "supported_document_formats": ["epub", "mobi", "azw", "azw3", "html", "htm", "docx", "rtf"],
    "marker_document_formats": [], "conversion_timeout": 3600, "connect_timeout": 30,
    "max_retries": 3, "retry_delay": 10, "use_llm": false, "force_ocr": false,
    "paginate": false, "strip_existing_ocr": true, "disable_image_extraction": true,
    "max_polls": 300, "poll_interval": 2
  },
  "pipeline_settings": {"skip_existing": true, "max_workers": 3, "log_level": "INFO"}
}
```

| Section | Effect and limits of the current code |
| --- | --- |
| directories | Paths relative to the workspace; `source_ebooks` is a historical name, use `source_documents`. The validation report is in the root, the pipeline state in logs. |
| validation_thresholds | Validation thresholds and selection of the documents; they do not certify fidelity or the preservation of the assets. |
| cleaning_settings | Legacy options. The global transformations can lose content; keep the original extraction and compare the results. |
| llm_presets / chunking_settings | Size presets for tokenizer/chunk, not the selection of an inference provider. `semantic`, `fixed_size`, `sliding_window` are JSON values; the CLI has no `--chunking-strategy`. The semantic overlap can be zero. |
| conversion_settings | Routing, endpoint, timeout, retry and polling. The local wrapper can report options without applying them and return empty images. It does not imply the availability of an engine or multi-format fidelity. |
| pipeline_settings | Parallelism and logs. `skip_existing` can reuse previous outputs: keep the identities and do not infer success from the presence of a file. |

The legacy loader has no complete typed validation: unreadable JSON can produce a warning and use the defaults. Check the output of `--config-only`. The shell prevails over `.env`; `.env` prevails over the fields that have an explicit override. [Reference of the variables](toolchain-legacy.md); the [template](../../.env.example) now documents only the `MDF_` settings of the new converter, not these variables. Do not put keys in reports or snapshots.

Chunks and YAML metadata include chunk_id/source_file/index/total_chunks and content measures. `validation_report.json` describes summary and results per filename; the chunking reports depend on the mode. The historical examples of an A/B vote or of a universal report do not replace the schema actually produced.

[Backends and scenarios](../how-to/pipeline-backends.md), [index](README.md).
