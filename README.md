# Markdown for LLMs

Convertitore locale sperimentale da PDF a Markdown con asset: CLI `mdf`, Web UI
testuale, worker separato e archivio SQLite/filesystem. Conserva sorgenti,
estrazione originale e tentativi distinti. Primo profilo Linux x86_64/CPU con
Marker1.10.2: qualità `unverified`, formule e contenuti da controllare sul PDF.

- [Prima conversione nel browser o nella CLI](docs/tutorials/primo-pdf-locale.md).
- [Preparare package, runtime e modelli locali](docs/how-to/prodotto-locale.md).
- [Configurazione, comandi e API](docs/reference/prodotto-locale.md).
- [Stati del bundle e limiti del prodotto sperimentale](docs/explanation/prodotto-sperimentale.md).

La pipeline a script rimane separata. I suoi comandi `markdown-*` e le procedure
seguenti sono legacy e hanno perdite note nel cleaning e nella copia degli asset.
Gli stati delle run e le integrazioni sono nei registri di sviluppo; il codice
implementato non equivale a un GO di review o a una versione integrata.

- [Preparare l'ambiente uv e ricostruire la distribuzione](docs/how-to/ambiente-uv.md).
- [Toolchain, comandi, configurazione e strati di test](docs/reference/toolchain-legacy.md).
- [Preparazione e verifica Docker Marker](docs/how-to/marker-legacy-docker.md).
- [Configurazione JSON della pipeline](docs/reference/pipeline-config.md).
- [Backend, documenti e scenari di conversione](docs/how-to/pipeline-backends.md).
- [Decisioni e roadmap](documentation/roadmap.md), [contributi](AGENTS.md).

Per il client bastano runtime Python e Pandoc già installato; FastAPI e il motore
Marker sono extra separati. Python della shell/pyenv non è l'interprete del
progetto. Usare la venv collegata al managed locale, dopo ricostruzione e verifica
dei dieci moduli installati contro i sorgenti e la wheel canonica.

Da un workspace di dati esterno, con ambiente già verificato, questi sono esempi
operativi dei percorsi base (Pandoc per HTML;
nessuna inferenza Marker). `RUN_ENV` è il path assoluto della venv;
`PYTHONPATH` e `PYTHONHOME` sono rimossi per le console script.

```bash
"$RUN_ENV/bin/python" -I -B -m config --create-default
"$RUN_ENV/bin/python" -I -B -m master_workflow --config-only
"$RUN_ENV/bin/python" -I -B -m master_workflow --step conversion --force
"$RUN_ENV/bin/python" -I -B -m master_workflow --step cleaning --force
"$RUN_ENV/bin/python" -I -B -m master_workflow --step validation --force
"$RUN_ENV/bin/python" -I -B -m master_workflow --step chunking --force --llm custom --chunk-size 1000 --overlap 100
```

Generare `pipeline_config.json` nel workspace e impostare il backend prima di
convertire. PDF usa il client `local_marker` per default; i documenti supportati
usano Pandoc, salvo una selezione esplicita di `marker_document_formats`. Il
percorso cloud è opzionale e richiede una configurazione deliberata e credenziali.
La migrazione non verifica inferenza Marker locale/remota o hardware GPU.

Le directory sono `source_pdfs`, `source_documents`, `converted_markdown`,
`cleaned_markdown`, `validated_markdown`, `chunked_markdown` e `pipeline_logs`.
`source_ebooks` è una directory storica: per i nuovi comandi usare la directory
configurata `source_documents`. `validation_report.json` è nella radice del
workspace; `pipeline_state.json` è nella directory dei log. Il validatore può
scartare contenuto: un exit 0 non certifica la completezza del documento.

Le formule, il codice, i numeri, i collegamenti e gli asset vanno confrontati con
il sorgente: il cleaning legacy usa trasformazioni globali che possono alterarli.
I chunk non sostituiscono il Markdown completo; l'overlap semantico legacy può
essere nullo. Non è disponibile un flag `--chunking-strategy`.

Il wrapper Marker restituisce immagini vuote e riporta opzioni senza applicarle;
`/health` misura liveness, anche dopo un errore di caricamento dei modelli. Build,
contratto offline e inferenza reale sono verifiche distinte. Le istruzioni per
startup/pesi/font e CPU/GPU rimangono fuori dalle prove veloci.

Gli script di analisi, embedding, RAG e fine tuning citati nella vecchia guida
non fanno parte del repository. Gli esempi precedenti di quegli script erano
pseudocodice esterno; non sono comandi di questo progetto.

Licenza [MIT](LICENSE); le condizioni dei pesi sono distinte. Package applicativo
`markdown_for_llms` e moduli flat legacy convivono;
[docs](docs/README.md) raccoglie le istruzioni operative e
[documentation](documentation/README.md) le decisioni, i limiti e gli avanzamenti.
