# Reference della toolchain e pipeline legacy

Pipeline legacy: installazione managed e verifiche S/B/I/E, suite separate e
contratto Docker offline. Linux x86_64 è il target dei diagnostici documentati,
non una restrizione del lock universale. Gli esiti delle run sono nei registri di sviluppo.
La nuova applicazione Web, DB e worker non sono disponibili.

| Selezione | Contenuto diretto | Limite |
| --- | --- | --- |
| base | requests, tiktoken, tqdm, python-dotenv | Nessun server, ML o dev |
| gruppo dev | pytest/cov/mock, black, flake8, mypy, httpx | Selezione esplicita |
| marker-server | FastAPI, uvicorn, python-multipart | Sufficiente per mock API, senza ML |
| marker-cpu | marker-pdf full 1.10.2, Surya 0.17.1, torch 2.7.1, bs4 | Indice torch CPU esplicito; candidato |
| marker-cu126 | stessi candidati, indice torch cu126 | Incompatibile con CPU nello stesso sync; GPU non verificata |

uv 0.10.10, managed CPython 3.12.13 locale e Setuptools 84.0.0 sono pin della run,
non una promessa di aggiornamenti automatici. `pyproject.toml` dichiara dipendenze,
`uv.lock` le risolve ed è presente nel worktree della run. I vincoli backend sono
separati dai pin runtime; Marker è solo wheel (`no-build-package`). Un sdist
transitivo con backend non identificato/vincolato blocca la preparazione.

Dieci moduli distribuiti: `config`, `logging_config`, `exceptions`,
`unified_converter`, `master_workflow`, `clean_markdown`, `validate_markdown`,
`chunk_markdown`, `batch_monitor`, `marker_api_server`. Il server richiede l'extra
API; nel base è soltanto letto/hashato. La wheel non contiene test, diagnostici,
fixture, dati, configurazioni operative o helper delle run.

| Console script | Target | Opzioni |
| --- | --- | --- |
| markdown-pipeline | master_workflow:main | --llm/--target-llm, --chunk-size, --overlap, --force, --step, --config-only, --dry-run |
| markdown-config | config:main | --create-default, --validate, --show |
| markdown-clean | clean_markdown:main | Nessun parser/help: usa il JSON nel CWD |
| markdown-validate | validate_markdown:main | Nessun parser/help: usa il JSON nel CWD |
| markdown-chunk | chunk_markdown:main | --input-dir, --output-dir, --target-llm, --chunk-size, --overlap |

Verificare gli help effettivi prima dell'uso. `--step` ammette conversion,
cleaning, validation, chunking. `--chunking-strategy` non esiste.
`--config-only` crea JSON ed esce prima dell'orchestratore e dei prerequisiti.
Clean/validate non rispondono a `--help`: l'invocazione avvia l'elaborazione.
`config --show` mostra il JSON e non applica gli override della pipeline.

La pipeline cattura il CWD come workspace al momento dell'istanza. JSON,
percorsi relativi, validation_report e log restano lì; i percorsi assoluti restano
assoluti. Per chiamarla da Python, passare un `ConfigManager` a
`UnifiedDocumentConverter`, non `config.conversion_settings`.

La pipeline e il converter standalone caricano solo `.env` del workspace,
senza risalire negli antenati o cercare nel site-packages, con `override=False`.
Nell'orchestratore: JSON → CLI → override applicativi; shell prevalente su `.env`.
Il converter standalone mantiene i propri override legacy. `.env.example`
elenca variabili effettive e valori innocui; non abilita da sola il cloud.
`MARKER_DOCUMENT_FORMATS` vuota non cancella una lista JSON già configurata.

| Variabili | Uso legacy |
| --- | --- |
| CHUNK_SIZE, OVERLAP_SIZE, TARGET_LLM | Override dei parametri di chunking |
| MAX_WORKERS, LOG_LEVEL | Pipeline |
| CONVERSION_TIMEOUT, CONNECT_TIMEOUT | Timeout conversione/connect; timeout 0 attende senza limite |
| MARKER_LOCAL_BASE_URL/ENDPOINT | Endpoint locale HTTP |
| MARKER_CLOUD_BASE_URL/ENDPOINT, MARKER_BASE_URL | Cloud; BASE_URL è alias legacy del cloud |
| MARKER_API_KEY | Segreto letto dal client cloud; nome selezionabile nel JSON |
| MARKER_DOCUMENT_FORMATS | Lista esplicita di estensioni non PDF per Marker |
| MARKER_USE_LLM, FORCE_OCR, PAGINATE, STRIP_EXISTING_OCR, DISABLE_IMAGE_EXTRACTION (prefisso MARKER_) | Opzioni client; booleani 1/true/yes/on |
| MAX_POLLS, POLL_INTERVAL | Polling cloud; 0 polls senza limite |
| TIKTOKEN_CACHE_DIR | Cache tokenizer pronta, identificata prima delle prove |

Le variabili UV_* e RUN_* sono toolchain/prove, non configurazione applicativa.
Non includere segreti negli snapshot o nei log delle sonde. Le opzioni del wrapper
Marker sono riportate nei metadata, senza applicazione al converter; immagini
vuote e health di liveness rimangono limiti. `local_marker` e server host non sono
verificati dopo la migrazione. La nuova politica locale/remota resta un progetto.

La convalida ufficiale richiede uno snapshot di esecuzione identificato,
R PASS nel runner e
catena S/B/I verificata prima della collection, con cache tokenizer e Pandoc
pronti. Non basta `uv --offline` per confinare requests/figli. `run_offline.py`
usa unshare primario, oppure Firejail noprofile/net none con blacklist D4;
stessa netns per padre/figli, socketpair locale positivo e daemon/IP negati.
La sonda preliminare della run non è un PASS ufficiale. Il wrapper non prepara
pacchetti, non crea freeze e non è una sandbox generale per codice ostile.

Ogni comando sotto è **da eseguire dopo preparazione e preflight**. `RUN_DEV_PY`
è la venv dev installata; `RUN_API_PY` contiene anche marker-server. Impostare
RUN_SOURCE_MANIFEST e RUN_INSTALLATION_RECEIPT ai file S/I correnti, verificati
nello stesso interprete. Per packaging: RUN_WHEEL_PY/RUN_WHEEL_RECEIPT,
RUN_BASELINE/RUN_BASELINE_WORKSPACES. Nessun prerequisito mancante diventa skip.

```bash
"$RUN_DEV_PY" -I -B -m pytest tests/ --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q
"$RUN_API_PY" -I -B -m pytest tests/api/ -q
"$RUN_DEV_PY" -I -B -m pytest tests/packaging/ -q
"$RUN_DEV_PY" -I -B -m unittest discover -s tests/governance -v
"$RUN_DEV_PY" -I -B -m pytest tests/ --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker --collect-only -q
"$RUN_DEV_PY" -I -B -m pytest --collect-only -q
"$RUN_DEV_PY" -I -B -m pytest . --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker --collect-only -q
"$RUN_API_PY" -I -B -m pytest tests/ --collect-only -q
"$RUN_API_PY" -I -B -m pytest . --collect-only -q
```

La collection globale usa l'ambiente API, importa gli harness senza ML e non
costruisce immagini. API/package/Docker restano strati obbligatori distinti;
la suite veloce selezionata non è un collaudo completo. I due smoke radice sono
manuali ed esclusi: non sono verifiche sicure di inferenza. I risultati storici
67 pass e V0 62 pass/5 fail non si trasferiscono al prodotto nuovo.

Per Docker, fuori dal runner che maschera il daemon, RUN_IMAGE è ID locale,
RUN_IMAGE_MANIFEST è image-source.json prodotto dal tool di contesto,
RUN_IMAGE_INPUTS è la receipt del tool supply, RUN_BACKEND_WHEEL è la wheel84
hash-pinned, RUN_IMAGE_CONTEXT_RECEIPT è la receipt del producer di contesto,
RUN_TEST_OUTPUT è una directory
nuova. C-docker richiede immagine CPU V7 già preparata: pull never/network none,
Python venv -I -B su stdin, senza mount o startup ordinario. Il test invoca
run_docker_contract.py versionato: cap-drop ALL/no-new-privileges, healthcheck
disabilitato, memoria/pids finiti, export filesystem del container fermo prima
di Python, full I e nove sottocasi. Preflight host corrente S/I, con modalità
official o standalone esplicita; nessun riferimento a snapshot storico fisso.
Per C-docker impostare `RUN_TEST_MODE=official|standalone` nell'ambiente:
il solo flag pytest `--test-mode` non seleziona il driver Docker; scelte
discordanti sono rifiutate.

```bash
"$RUN_DEV_PY" -I -B -m pytest tests/docker/test_marker_contract.py -q
```

Negativi D2/D3, probe di cache, R, F1–F6, S/B/I/E e receipt con hash/argv/CWD/env
sono requisiti della run. La suite non costruisce o installa per riparare un
mismatch. Una modifica invalida le prove dipendenti secondo la catena descritta nella guida uv;
il protocollo permanente è il [ciclo delle run](../../documentation/development/run-lifecycle.md).
Per un clone senza temp, i dettagli della catena sono nella
[guida uv](../how-to/ambiente-uv.md).

## Strumenti di sviluppo

Il gruppo `dev` del pyproject include pytest e gli strumenti Black, Flake8 e
Mypy. Preparare il profilo locked e reinstallare la canonica prima del preflight.
Eseguire un formatter o un controllo statico è una scelta esplicita: questi
strumenti non sostituiscono le suite selezionate né provano la fedeltà della
conversione. La suite non installa dipendenze per riparare la collection.
