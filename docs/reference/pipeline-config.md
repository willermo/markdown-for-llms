# Configurazione JSON della pipeline legacy

`pipeline_config.json` appartiene al workspace di dati corrente, insieme a `.env`.
Generarlo con `"$RUN_ENV/bin/python" -I -B -m config --create-default`, dopo il
preflight della distribuzione installata. `--config-only` mostra la configurazione
senza iniziare conversioni. Il codice di riferimento è [config.py](../../config.py).

Le chiavi realmente lette sono `directories`, `llm_presets`,
`validation_thresholds`, `cleaning_settings`, **`chunking_settings`**,
`conversion_settings` e `pipeline_settings`. `chunking` non è un alias JSON.
Esempio valido, da adattare soltanto nel workspace:

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

| Sezione | Effetto e limiti del codice attuale |
| --- | --- |
| directories | Percorsi relativi al workspace; `source_ebooks` è un nome storico, usare `source_documents`. Il report validation è nella radice, lo stato pipeline in logs. |
| validation_thresholds | Soglie di validazione e selezione dei documenti; non certificano fedeltà o conservazione degli asset. |
| cleaning_settings | Opzioni legacy. Le trasformazioni globali possono perdere contenuti; conservare estrazione originale e confrontare i risultati. |
| llm_presets / chunking_settings | Preset di dimensioni per tokenizer/chunk, non selezione di un provider di inferenza. `semantic`, `fixed_size`, `sliding_window` sono valori JSON; la CLI non ha `--chunking-strategy`. L'overlap semantico può essere nullo. |
| conversion_settings | Routing, endpoint, timeout, retry e polling. Il wrapper locale può riportare opzioni senza applicarle e restituire immagini vuote. Non implica disponibilità di un motore o fedeltà multi-formato. |
| pipeline_settings | Parallelismo e log. `skip_existing` può riusare output precedenti: conservare identità e non dedurre successo dalla presenza di un file. |

Il loader legacy non ha validazione tipizzata completa: JSON non leggibile può
produrre un warning e usare i default. Controllare l'output di `--config-only`.
La shell prevale su `.env`; `.env` prevale sui campi che hanno un override
esplicito. [Reference delle variabili](toolchain-legacy.md) e
[template](../../.env.example). Non mettere chiavi nei report o negli snapshot.

Chunk e metadati YAML includono chunk_id/source_file/indice/total_chunks e misure
di contenuto. `validation_report.json` descrive summary e risultati per filename;
i report di chunking dipendono dalla modalità. Gli esempi storici di voto A/B o
di un report universale non sostituiscono lo schema realmente prodotto.

[Backend e scenari](../how-to/pipeline-backends.md), [indice](README.md).
