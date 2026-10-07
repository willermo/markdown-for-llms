# Scegliere backend e scenari della pipeline

Preparare [ambiente e preflight](ambiente-uv.md), generare la
[configurazione JSON](../reference/pipeline-config.md) nel workspace esterno
e conservare sorgente/estrazione originale. `RUN_ENV` è il path assoluto della
venv canonica. Eseguire dapprima `--config-only` per controllare directory e routing.

## Backend e formati

PDF usa `pdf_converter`: `local_marker` richiede un server deliberatamente
preparato e avviato; `cloud_marker` usa l'endpoint configurato e la chiave
indicata da `marker_cloud_api_key_env`. Il cloud carica documenti al servizio:
selezionarlo esplicitamente soltanto per dati ammessi. Nessuna acquisizione ML
o attivazione server fa parte del percorso client/Pandoc.

Non-PDF usa Pandoc per epub/mobi/azw/azw3/html/htm/docx/rtf. Supporto dichiarato
non garantisce la conversione di ogni variante. `marker_document_formats`,
vuoto per default, abilita opt-in il routing Marker anche per formati diversi
da PDF, inclusi formati che il server deve supportare realmente. La lista non
installa né collauda quel motore. Un backend non riconosciuto o un errore Marker
su un non-PDF può produrre fallback **Pandoc**, senza caricamento cloud implicito;
controllare converter_used e messaggi. Fonti operative:
[unified_converter.py](../../unified_converter.py), [config.py](../../config.py).

Il cloud invia il file, riceve un check_url e fa polling. `max_polls=300` e
`poll_interval=2` sono default finiti; `MAX_POLLS<=0` rende il polling illimitato.
`CONVERSION_TIMEOUT=0` rende illimitata l'attesa di risposta, distinta da
`CONNECT_TIMEOUT`. Preferire valori finiti ed espliciti; aumentare timeout non
ripara un documento errato o un server senza memoria. Opzioni ML/OCR/imaging
non provano che il wrapper locale le applichi. `/health` misura liveness.

## Documenti e fasi

Creare `source_pdfs` e `source_documents` nel workspace, inserire file ammessi
e usare conversione completa o passi distinti. Il comando diretto del convertitore
richiede source-dir/output-dir e produce i propri risultati e report:

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

La pipeline può scartare documenti in validazione; il monitor non certifica
fedeltà. Confrontare formule, numeri, codice, ordine, report completi e asset.
HTML/Pandoc può perdere anchor; cleaning può rimuovere contenuto e asset non
vengono sempre copiati. Un exit0 non annulla queste perdite.

Per ricerca e lettura usare il Markdown completo con asset. Per retrieval o
limiti di contesto produrre chunk come derivato, controllandone sequenza e
overlap. Nessuno script prepare_training_data/create_embeddings/RAG è fornito:
integrazioni esterne richiedono un progetto e un trattamento dati separati.
Non copiare ciecamente soltanto validated_markdown quando la validazione ha
escluso sorgenti. Le forme console e i moduli isolati sono descritte nella
[reference CLI](../reference/toolchain-legacy.md).

Per API locale, pesi/font, inferenza CPU/GPU e deploy seguire i
[limiti e preparazione Docker](marker-legacy-docker.md): build e contratto
offline non provano conversioni reali Marker. Non eseguire servizi per un test
veloce né usare un dummy PDF come collaudo di contenuto.

[Indice](README.md).
