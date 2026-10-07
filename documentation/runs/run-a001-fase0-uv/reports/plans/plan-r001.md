# Piano — run-a001-fase0-uv — r001

- Autore/provider/modello e chat: **Codex / OpenAI**; famiglia **GPT-6** indicata
  dalle istruzioni della sessione. Identificatore specifico del modello e ID della
  chat non esposti. Ruolo esclusivo: pianificatore in questa chat.
- Data: **2026-10-02, Europe/Rome**.
- Prompt di origine: [02-planning-r001.md](../prompts/02-planning-r001.md),
  riprodotto nella richiesta dell'utente.
- Branch verificato: `feature/run-a001-uv`.
- HEAD, `dev` locale e merge-base verificati:
  **`66ba82200e5def5a4db76f9bafccb0731b506091`**.
- Contesto: `planning-context-r001`; verifica iniziale **MATCH**, exit 0.
  Verifica finale registrata nelle [evidenze — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
  Questo snapshot identifica gli input, non approva il piano.
- Working tree iniziale: sei modifiche documentali di supervisione, conservate;
  nessuna implementazione uv. Bootstrap già integrato/pubblicato dall'utente.
- Brief: [brief.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), fonte dei sette criteri di accettazione.
- ADR: [0006](../../../../decisions/0006-python-toolchain-uv.md),
  [0007](../../../../decisions/0007-supervised-development-runs.md),
  [0001](../../../../decisions/0001-document-fidelity.md) per le prove.
- Giro: **iniziale**, nessun arbitrato di fix o GO precedente.
- Template seguito: [plan.md](../../../../development/templates/plan.md).

## Obiettivo e perimetro

Rendere la pipeline **legacy** installabile e riproducibile: interprete esplicito,
dipendenze dal lockfile, wheel completa, cinque comandi installati, fasi avviabili
in una directory di dati estranea ai sorgenti. Allineare installazione, sviluppo,
Docker/Compose e istruzioni alle stesse dipendenze. Preservare elaborazione e
formati degli output, salvo gli interventi necessari a packaging e avvio.

L'adozione di uv è già approvata. Questo documento propone come realizzarla e quali
prove consegnare; non richiede una nuova autorizzazione e non realizza la migrazione.
Non include nuova Web UI, FastAPI applicativo/worker, nuovi contratti del dominio,
scelta OCR definitiva, benchmark, revisione AI, riscrittura del cleaning, correzione
del successo parziale dei batch o modifica globale di pyenv/altri progetti.

Il server FastAPI esistente è **soltanto il wrapper Marker legacy**. Il Markdown
completo resta il risultato principale; il piano non cambia adesso le quattro fasi
o il comportamento legacy del chunking.

### Input ed evidenze

Letture effettuate nell'ordine del prompt: istruzioni e contesto comune; stato,
handover, brief e prompt della run; skill/protocollo; indice, changelog, ADR e fase
0.1; preparazione del supervisore e template; sorgenti e test pertinenti.

Fonti locali: [preparazione — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[protocollo](../../../../development/run-lifecycle.md),
[roadmap](../../../../roadmap.md),
[skill della run](../../../../../.agents/skills/manage-implementation-run/SKILL.md) e
[skill di fedeltà](../../../../../.agents/skills/verify-conversion-fidelity/SKILL.md).
L'inventario statico, i comandi eseguiti e i riferimenti primari datati sono nelle
[evidenze di pianificazione — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).

**Verificato:** uv 0.10.10; shell CPython 3.12.3 di pyenv con
`PYENV_VERSION=3.12.3`; Pandoc 3.1.3; CLI Docker/Compose presenti. Packaging flat
incompleto, `config.main` mancante, percorsi del codice legati al CWD e Docker da
Marker `master` sono confermati per lettura. Non esiste oggi un ambiente locale
del progetto o un lockfile; nella shell mancano requests/tiktoken. I test 67/6
citati nel bootstrap non sono stati rieseguiti né attribuiti alla migrazione.

**Proposto:** scelte P1–P7 sotto. **Da collaudare:** risoluzione, installazioni,
build e risultati delle prove V0–V11. Nessun esito futuro è precompilato come passato.

## Implementazione proposta

### P0 — Preparare baseline e prove, prima delle modifiche applicative

1. Il supervisore congela questo piano con evidenze essenziali, produce due prompt
   indipendenti ChatGPT/Claude, raccoglie i report reali e arbitra sullo stesso
   snapshot. L'implementatore inizia soltanto dopo il relativo GO.
2. L'implementatore verifica Git e snapshot approvato, conserva le sei modifiche
   documentali preesistenti e registra i propri file; nessun reset, commit o cambio
   di branch implicito. Non usare il vecchio snapshot del contesto per fingere che
   i sorgenti modificati debbano restare uguali durante l'implementazione.
3. Prima di modificare i moduli, prepara fixture sintetiche e un piccolo harness
   nella cartella della run; registra sorgenti, hash, provenienza e risultati
   baseline. Se il vecchio ambiente non è disponibile, ricrea un ambiente baseline
   **separato**, con le sole quattro dipendenze runtime e pytest, non passando per
   `setup.py` difettoso. I pacchetti risolti per la baseline sono congelati in una
   evidenza locale, con comandi/versioni; non sono una seconda fonte di produzione.
4. Esegue i vecchi script diretti dai sorgenti e la suite selezionata dove possibile.
   Per la baseline fuori dal repository può lanciare il percorso assoluto dello
   script originale: la ricerca dei moduli sorgente è naturale per quella prova.
   Non copia o corregge sorgenti per ottenere la baseline. Registra separatamente
   il fallimento atteso del vecchio orchestratore fuori dai sorgenti.
5. Prepara esplicitamente la piccola cache tiktoken richiesta dalla suite, oppure
   usa una cache già presente verificandone hash. Nessun modello OCR. Baseline
   impedita da un prerequisito viene registrata come tale, senza inventare output.
   Una baseline mancante per il contenuto modificabile va risolta prima del GO
   finale; l'output nuovo non può essere il proprio unico riferimento.

### P1 — Toolchain, metadati e singola fonte delle dipendenze

File: nuovi `pyproject.toml`, `uv.lock`, `.python-version`; `.gitignore`; rimozione
di `setup.py` e `requirements.txt` dopo il passaggio dei consumatori/documentazione.

| Scelta proposta | Motivazione, alternative e condizione di verifica |
| --- | --- |
| `setuptools.build_meta`, elenco esplicito `py-modules` | conserva i moduli flat senza anticipare fase 2; backend proposto `setuptools==84.0.0`, versione delle docs primarie consultate. Verificare disponibilità e Requires-Python prima della build. Hatchling o uv_build richiederebbero una configurazione/layout differente, senza vantaggio necessario in questa run |
| Nome/versione `markdown-for-llms`, `1.0.0` | conserva i metadati identificativi legacy; non è una nuova release pubblicata. Non mantenere URL example.com come riferimenti reali del progetto |
| `requires-python = >=3.12,<3.13` | limita il supporto dichiarato alla minor osservata e già usata da Docker; 3.9–3.11 e 3.13+ non sono collaudate in questa run. È una restrizione esplicita rispetto al vecchio metadata >=3.9, da evidenziare in README/report |
| `.python-version`: `3.12.13` | stessa minor della baseline, patch presente nel catalogo offline di uv 0.10.10. Consente gestione uv senza selezionare la vecchia pyenv 3.12.3. È un candidato da provare, **non l'ultima patch**: Python.org segnala già 3.12.15 |
| uv esatto `0.10.10`, `tool.uv.required-version = ==0.10.10` | versione presente e help verificato; documentare installazione del binario versionato e checksum, senza autoaggiornamento. Alternative: mantenere Python 3.12.3 per la sola baseline, oppure aggiornare insieme uv/pin alla nuova patch con prove ripetute e scostamento portato al supervisore |
| `tool.uv.default-groups = []` | sync ordinario leggero; sviluppo richiesto con `--group dev`, server/motori con extra espliciti |
| `.venv/`, `.venv-marker/`, `.venv-marker-gpu/`, `.venv-python/` ignorate | ambienti/interprete locali ricreabili, distinti da configurazioni/dati; uv.lock rimane versionabile |

Le scelte di versione sono proposte verificabili, non certificazioni di compatibilità.
Se la patch/immagine/backend non è reperibile o non passa le prove, non sostituire
silenziosamente con `latest` o Python 3.11: documentare il blocco e rientrare dal
supervisore per uno scostamento significativo. La scelta dei motori in fase 1 potrà
richiedere un altro interprete in ambiente separato entro ADR 0006.

La configurazione applicativa resta quella legacy JSON/`.env`. Gli strumenti della
toolchain non vanno trasformati in un nuovo schema applicativo o salvati insieme
alle credenziali. Non aggiungere una configurazione Poetry al progetto principale.

#### Selezione dell'interprete, anche con pyenv attivo

I comandi di verifica/installazione richiedono `--python 3.12.13 --managed-python`;
le esecuzioni successive usano l'eseguibile della `.venv` o `uv run` sul progetto
sincronizzato. In questo percorso `PYENV_VERSION=3.12.3` può restare attiva: non
seleziona l'interprete uv managed. `python3` fuori dalla `.venv`, invece, continua
a essere il Python della shell/pyenv. `.python-version` da sola non garantisce che
il comando della shell `python3` cambi interprete quando PYENV_VERSION è impostata.
La discovery e i flag sono documentati nelle [versioni Python uv](https://docs.astral.sh/uv/concepts/python-versions/).

Per l'implementazione/guida prevedere un download Python **esplicito**, separato
dai test: `uv python install 3.12.13 --no-bin --no-registry`; non usa `--default`,
`--global`, `--force`, shell update o comandi pyenv. Preferire installazione Python
locale al progetto, in `.venv-python/` ignorata, passando sempre
`UV_PYTHON_INSTALL_DIR` assoluta ai comandi managed; in alternativa una managed
già presente può essere riutilizzata dichiarandone il percorso. Registrare URL,
build e hash della distribuzione standalone: il solo numero Python non identifica
il binario. Nei test impostare `UV_PYTHON_DOWNLOADS=never` dopo la preparazione.
La cache uv può stare nella run per le evidenze o nella cache ordinaria autorizzata;
non pulire cache/interpreti condivisi della macchina.

#### Inventario e collocazione

| Collocazione | Dipendenze proposte | Uso effettivo / trattamento |
| --- | --- | --- |
| Runtime base | `requests>=2.31.0`, `tiktoken>=0.5.1`, `tqdm>=4.66.0`, `python-dotenv>=1.0.0` | client HTTP, chunking, cleaning/progresso e lettura .env; mantenere inizialmente i minimi legacy, fissare le versioni effettive nel lock e provarle |
| Gruppo `dev` | pytest, pytest-cov, pytest-mock, black, flake8, mypy con minimi legacy; httpx per TestClient | pytest è usato dai test; coverage/lint sono comandi documentati, pytest-mock non è richiesto dai test letti ma può restare come strumento dev. Nessuno nel runtime della wheel; httpx ha uso nei nuovi test del wrapper |
| Extra `marker-server` | fastapi, uvicorn, python-multipart | import top-level del server. Extra leggero, da richiedere esplicitamente anche per le prove API con fake converter |
| Extra `marker-cpu` | Marker `full` al commit proposto, `surya-ocr==0.17.1`, `torch==2.7.1` | motore opzionale pesante; torch dall'indice CPU esplicito; non installato da base/dev/marker-server |
| Extra `marker-cu126` | stesso Marker/Surya e `torch==2.7.1`, indice cu126 | candidato GPU Linux x86_64; incompatibile con marker-cpu nel sync; disponibilità del profilo non equivale a prova hardware |
| Transitive del motore | Pillow, transformers, pydantic, pdftext, ecc. | dal metadata del commit e dal lock, non ricopiate tutte come dipendenze dirette del progetto |
| Provider `full` | mammoth, openpyxl, python-pptx, ebooklib, weasyprint; bs4/filetype richiesti dalla registry | aggiungere beautifulsoup4 direttamente all'extra motore se il metadata risolto non la garantisce; filetype è già dichiarata upstream. Verificare import PPTX, che il vecchio Dockerfile non installava esplicitamente |
| Rimosse dal progetto | colorama, rich / extra enhanced | nessun uso negli import letti; logging usa direttamente sequenze ANSI. Non vendere una funzione di terminale aggiuntiva |
| Native / esterne | Pandoc; librerie del container; Docker/Compose e, per GPU, driver/Toolkit | non sono pacchetti Python. Il client Marker via HTTP non richiede motore/server nella propria `.venv` |

Torch 2.7.1 è proposto perché appartiene alla prima minor ammessa dai contratti
Marker/Surya esaminati ed esistono wheel ufficiali cp312 CPU e cu126. Non è una
versione dichiarata “compatibile” da una prova OCR già fatta. Non installare
torchvision/torchaudio se non richieste dagli import/transitive: se servisse
torchvision, la coppia ufficiale per torch 2.7.1 è 0.22.1.
[Versioni ufficiali PyTorch](https://pytorch.org/get-started/previous-versions/).

Nella risoluzione usare indici PyTorch nominati con `explicit=true`, sorgenti per
extra e conflitto CPU/cu126: [guida uv PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).
Non usare backend `auto`, fallback a PyPI per il profilo CPU o installazioni pip
successive che alterino la variante locked. Applicare marker Linux x86_64 ai
requisiti pesanti proposti e dichiarare quel solo target per il server in questa
run; non restringere per questo la wheel base agli stessi sistemi. Fuori dal
target un extra non installato non va presentato come server pronto.

`uv lock` risolve anche extra/gruppi non installati: la separazione impedisce
l'installazione ML implicita, non garantisce che il primo lock non legga metadata
Git/indici. Registrare il costo; consentire solo risoluzione/build metadata, non
startup, pesi o benchmark. Se il grafo è incompatibile, proporre al supervisore un
progetto uv separato del server con lock proprio, anziché escludere il motore da
un criterio già approvato. Non introdurre subito due fonti dello stesso ambiente.

La fonte dichiarativa sarà `pyproject.toml`; la risoluzione riproducibile sarà
`uv.lock`. **Rimuovere setup.py e requirements.txt**, aggiornando ogni comando che
li usa. Alternativa accettabile solo se emerge un consumatore necessario: setup
shim `setup()` senza dipendenze/metadati, oppure export requirements generato dal
lock con header e controllo di rigenerazione; mai un secondo elenco manuale.
Il percorso proposto non ne ha bisogno. Non importare automaticamente tutto il
vecchio requirements come runtime; i commenti TOML restano fuori dalle stringhe.

La build isolata del progetto deve usare il backend esatto, senza credere che il
lock runtime fissi da solo i backend. Per Marker Git, rilevare i build requirements
Poetry e fissarne le versioni effettive con vincoli di build uv; documentare anche
i backend di eventuali sdist. Per `uv build`, passare un file di build constraints
generato dalle versioni approvate se necessario. Verificare in log che nessun
backend unbounded sia risolto diversamente tra build pulite. Il piano non assegna
una versione Poetry-core senza avere verificato quel metadata.

### P2 — Distribuire i moduli flat e rendere config invocabile

Distribuire esattamente questi **dieci moduli** con `tool.setuptools.py-modules`:

```text
config
logging_config
exceptions
unified_converter
master_workflow
clean_markdown
validate_markdown
chunk_markdown
batch_monitor
marker_api_server
```

I primi nove sono importabili con il runtime base. `marker_api_server` è presente
nella wheel ma richiede `marker-server` per l'import; Marker resta importato
pigramente. Non distribuire `setup`, `test_pipeline`, `test_conversion`, fixture,
scripts della governance, dati, `.env`, temp o configurazioni operative. Ispezionare
anche la sdist per evitare file sensibili trascinati dalla discovery. README e
licenza possono essere inclusi come metadati del pacchetto.

Mantenere **i cinque entry point esistenti**, senza inventare una nuova CLI:

| Comando | Target |
| --- | --- |
| `markdown-pipeline` | `master_workflow:main` |
| `markdown-config` | `config:main` |
| `markdown-clean` | `clean_markdown:main` |
| `markdown-validate` | `validate_markdown:main` |
| `markdown-chunk` | `chunk_markdown:main` |

In `config.py` estrarre l'attuale blocco CLI in `main()` e lasciare il blocco
`__main__` come chiamata alla stessa funzione. Conservare `--create-default`,
`--validate`, `--show`, formato JSON, messaggi e comportamento del caricamento
legacy. Non trasformare `--validate` in una nuova validazione dello schema durante
questa migrazione. Un test confronta file JSON prodotti da script, `-m` e comando.

Cleaning e validation **non hanno oggi help/parser**: testare chiamate senza
argomenti su fixture, non usare `--help` come prova. Non aggiungere un parser solo
per far passare un controllo. Pipeline/config/chunk hanno invece help reale.
Mantenere invocazione diretta degli script dal clone, `python -m <modulo>` dalla
wheel e comandi installati. I ritorni `None`/exit legacy dei comandi di fase vanno
descritti; cambiare le politiche di errore dei batch richiederebbe un'altra run.

Alternativa al flat: package nominato con shim nei vecchi moduli. Ridurrebbe le
collisioni di nomi come config/exceptions, ma aumenta gli spostamenti/import e
anticipa la riorganizzazione del dominio. Per questa run preferire l'elenco esplicito,
registrando la limitazione e provando le origini reali degli import.
[Configurazione Setuptools](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html).

### P3 — Separare codice installato e directory dei dati

File: `master_workflow.py`, `unified_converter.py`; test mirati per l'avvio.

1. Usare una mappa delle fasi ai nomi di modulo (`clean_markdown`,
   `validate_markdown`, `chunk_markdown`) e subprocess con
   **`[sys.executable, '-m', modulo, ...]`**. Mantenere timeout, argomenti chunk,
   stdout/stderr, ordine e ritorni esistenti. Non usare `uv` nei subprocess:
   l'interprete che esegue l'orchestratore possiede già il package.
2. Sostituire il controllo dei file `CWD/*.py` con la risolubilità dei moduli
   installati (`importlib.util.find_spec` o controllo equivalente). Errore
   leggibile se manca un modulo. Non aggirare il controllo copiando file nel CWD.
3. Catturare la directory dei dati all'istanziazione dell'orchestratore, non
   all'import: CWD dell'avvio. Config JSON, `.env`, log e percorsi relativi dei
   dati restano riferiti a quella directory; percorsi assoluti configurati restano
   assoluti. Subprocess con `cwd` della stessa directory, report di validation
   letto da lì. Nessuna scrittura accanto al codice in site-packages.
4. Rendere esplicito il caricamento `.env` in `unified_converter.py` dal CWD
   dei dati, con la precedenza legacy delle variabili già presenti nell'ambiente.
   Il caricamento implicito dotenv può cercare dal modulo installato: confrontare
   un `.env` sintetico innocuo nel workspace esterno, con PYENV_VERSION ancora
   attiva e un override reale della shell. Nessuna credenziale vera nella prova.
5. Conservare il controllo legacy Pandoc/tiktoken anche per `--step`: Pandoc è
   presente nel contesto verificato e sarà prerequisito dichiarato delle prove
   dell'orchestratore. Una futura selezione dei prerequisiti per singola fase è
   alternativa utile, ma non necessaria alla correzione del packaging in r001.
   Aggiornare soltanto i suggerimenti di installazione per rimandare al flusso uv.

La directory del repository è richiesta per sincronizzare/sviluppare; la directory
dei dati è quella dalla quale si avvia la pipeline. Da un workspace esterno:
`uv run --project /percorso/clone --locked --no-default-groups markdown-pipeline ...`
usa l'ambiente del clone mantenendo il CWD dei dati. `--directory` cambia invece il
CWD e non va usato come equivalente. Per la prova di packaging rigorosa usare
direttamente l'interprete/comandi della wheel esterna, senza `uv run --project`.
Questa distinzione è confermata dall'[help locale run — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).

### P4 — Test proporzionati e confronto del contenuto

File proposti: `tests/packaging/test_installed_distribution.py`, helper/fixture
synthetic in `tests/fixtures/uv_migration/`, `tests/unit/test_config_cli.py`,
`tests/unit/test_marker_wrapper.py`; modifiche mirate a `tests/conftest.py` e
config pytest in pyproject. Un harness riutilizzabile della wheel può vivere in
`scripts/diagnostics/uv_migration.py`, se promosso con istruzioni e provenienza.
Nomi definitivi dei nuovi helper sono a scelta dell'implementatore e vanno nel report.

- Suite **selezionata**: `uv run --locked --group dev python -m pytest tests/ -q`.
  Specificare `testpaths=['tests']` per il comando senza percorsi. Escludere anche
  i due smoke alla radice con regole `--ignore` mirate nella configurazione del
  discovery ordinario, così che `pytest . --collect-only` non riapra collisione
  e HTTP implicito. Non rinominare/riscrivere gli smoke legacy come test nuovi.
  Una richiesta esplicita di quel file resta una prova manuale distinta.
  Configurare anche `norecursedirs` per temp/tmp, ambienti, build/dist e directory
  di dati generati, conservando le esclusioni standard di pytest: gli harness
  locali della run non devono entrare nel discovery dalla radice.
- Mantenere i sys.path legacy della suite attuale in questa run, dichiarandone il
  limite. I nuovi controlli interrogano processi della wheel estranei ai sorgenti,
  con ambiente ripulito da PYTHONPATH/PYTHONHOME e senza editable. Non sono validati
  dagli import del processo pytest che ha letto conftest.
- Correggere solo l'allestimento dei test di integrazione che deve provare davvero
  una fase: output inizialmente vuoti e `skip_existing=False`, senza modificare
  gli algoritmi. Preservare i casi di riuso ove necessari, identificandoli come tali.
- Preflight esplicito della cache tiktoken con hash/versione; nessuna acquisizione
  nella suite. Bloccare richieste di rete nei test in processo con fixture mirata;
  i subprocess di fase devono avere cache verificata prima dell'avvio e fixture
  instradate a Pandoc o fasi senza HTTP. Un `uv --offline` non blocca requests.
- Test del wrapper con startup/get_converter sostituiti prima del lifespan:
  converter fake, successo con oggetto `.markdown`, errore/import mancante,
  cleanup dei temporanei, verifica dei campi restituiti. Questi sono mock, anche
  quando eseguiti in un container. Non chiamare create_model_dict nei test veloci.
- Governance: eseguire anche `uv run --locked --group dev python -m unittest
  discover -s tests/governance -v`, con conteggio distinto. I commit di quelle
  fixture avvengono in repository temporanei, non sul progetto.

#### Fixture e confronto prima/dopo

Applicare la skill di fedeltà: conservare sorgente, baseline e output nuovo separati,
con hash e versioni; confrontare contenuti effettivi, non solo metadati. Piccolo corpus:

| Fixture | Scopo e osservabili |
| --- | --- |
| F1 Markdown stabile | almeno due sezioni in ordine, testo IT/Unicode, numeri distinti `123,45`, `-7`, `2026`, formula inline `$E=mc^2$`, riferimento testuale `[1]`, contenuto sufficiente per validazione; cleaning reale, output nuovo uguale ai byte baseline e invarianti pertinenti presenti |
| F2 Markdown sensibile | LaTeX con indici/graffe e formula multilinea, codice con graffe, link/bibliografia, immagine locale sintetica; caratterizzare i danni del cleaning legacy prima delle modifiche. Confronto byte prima/dopo, inventario di tutte le perdite rispetto al sorgente |
| F3 validation | F1/F2 rese valide e posizionate direttamente nella directory cleaned; la fase deve copiare il Markdown senza alterare formule, numeri, link o ordine. Confrontare report di validazione su campi deterministici e file byte per byte |
| F4 conversione Pandoc | HTML sintetico senza risorse remote: Unicode, numeri, due sezioni, formula testuale `E = mc²`, tabella/riferimento locale; conversione effettiva senza Marker/cloud. Verificare contenuto, ordine, summary con failed=0 e ritorno del processo |
| F5 asset | piccola immagine redistribuibile generata deterministicamente; hash del file, presenza e risoluzione dei riferimenti prima/dopo ogni fase pertinente. Registrare asset non copiati dal legacy, senza attribuire completezza al bundle |

La baseline serve a distinguere regressioni della migrazione da difetti già presenti.
Le regex del cleaning rimuovono già graffe/link e il wrapper/validator non garantisce
gli asset: **non correggerle né normalizzarle per far coincidere le prove**. F1 prova
la fase su contenuto preservabile; F2 documenta anche le perdite, non viene eliminata
dal corpus per ottenere un risultato positivo. Il requisito di questa run è nessuna
nuova alterazione rispetto alla baseline; non è un collaudo di fedeltà universale.
Ogni perdita nuova o baseline incerta va al supervisore. Non chiamare conversione
completa un caso con asset mancanti solo perché l'exit code è 0.

Per chunking confrontare contenuto dei chunk, sequenza/metadati deterministici,
copertura e overlap rispetto alla baseline. Escludere dal confronto solo campi
esattamente identificati come timestamp/percorso della fixture, mantenendo inalterati
Markdown e LaTeX. Non usare regex globali. Non promettere che ricomporre chunk legacy
ricostruisca sempre il documento senza perdita: documentare gli eventuali difetti.

### P5 — Marker immutabile e ambiente separato, entro ADR 0006

**Proposta motivata:** Marker v1.10.2 dal commit completo
`5a41cdbac6232a10aaf4fad29f8e6c1a0c2a0808`, anziché `master`.
Usare un riferimento Git completo nel requisito/source uv e verificare che lock,
metadata installato e evidenza puntino al medesimo commit. Il candidato è collegato
alla [release](https://github.com/datalab-to/marker/releases/tag/v1.10.2), non scelto
per il solo numero. La fonte diretta deve risultare anche nel metadata della wheel
se si intende installare quell'extra fuori da uv; le sources degli indici PyTorch
non vengono invece applicate automaticamente da pip sulla wheel.

Il confronto statico esaminato è nella [matrice Marker — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md):
import `PdfConverter`/`create_model_dict`, firma con artifact_dict/processor_list/
renderer, registry multi-formato, ritorno MarkdownOutput con `.markdown`, page_count.
Sostiene il candidato e l'esigenza dell'extra full; non dimostra inferenza riuscita.
La serie 2.0 corrente introduce una riscrittura e non è un aggiornamento implicito
di toolchain. Se 1.10.2 non passa lock/import/API, l'implementatore non sceglie un'altra
versione arbitraria: registra l'incompatibilità e torna al supervisore.

Preferire **due ambienti da un solo pyproject/lock**: `.venv` per client/pipeline e
`.venv-marker` con marker-server+marker-cpu. `UV_PROJECT_ENVIRONMENT` assoluta
seleziona il secondo solo nelle invocazioni esplicite; non esportarla globalmente.
Nessun fallback locale→cloud aggiunto. Per GPU usare `.venv-marker-gpu`, extra
marker-cu126 e TORCH_DEVICE=cuda; CPU e GPU non si mescolano nello stesso sync.

Il wrapper attuale ha limiti confermati: opzioni ricevute non applicate al converter,
immagini `{}`, page_count dipendente dal motore, startup che tollera errori e health
sempre positiva. Documentarli puntualmente. Non implementare in questa run gestione
immagini, nuova readiness o propagazione di tutti i parametri. Per il candidato
verificare che l'output venga estratto dal campo `.markdown`; se un tipo sconosciuto
produce solo `str(object)`, la prova deve fallire, non dichiarare conversione riuscita.
Un cambiamento ulteriore al wrapper necessario per altri contratti richiede valutazione
del supervisore; non ampliarlo sotto il nome di packaging.

Distinguere quattro livelli nelle evidenze: **dipendenze locked**, **immagine
costruita/importabile**, **pesi/font acquisiti e identificati**, **conversione reale**.
Non implicano automaticamente il livello successivo. Anche il constructor di
BaseConverter chiama download_font: import/firma o constructor con font/predictor
mock devono essere eseguiti offline e classificati correttamente.

### P6 — Docker/Compose coerenti con il lock

File: `Dockerfile`, `docker-compose.yml`, nuovo `.dockerignore`;
`docker-compose.gpu.yml` per override opzionale documentato, senza avvio implicito.

1. Verificare e fissare digest dell'immagine Python **3.12.13 slim-bookworm** e
   dell'immagine uv **0.10.10** per il target Linux x86_64. Non sono stati recuperati
   i manifest in pianificazione: i digest devono essere reali nel Dockerfile finale,
   non placeholder o tag minor mobili. Se il tag Python non esiste, fermare questo
   passo e portare al supervisore la scelta alternativa con prove.
2. Build multi-stage: uv copiato dalla propria immagine pinned, Python di container
   esplicito `/usr/local/bin/python`, download Python disabilitato. Il pin managed
   dell'host non deve provocare un altro download in Docker. Copiare pyproject/lock
   e i dieci moduli con lista esplicita, costruire e sincronizzare il package
   **non editable** con marker-server+marker-cpu e senza gruppi dev. Non usare pip
   install Marker/lib extra fuori dal lock. Se si ottimizza il layer iniziale con
   `--no-install-project`, eseguire comunque il sync finale completo.
3. Runtime: stesso digest Python, ambiente installato in percorso stabile,
   interpreter da quella venv, avvio `python -m marker_api_server`. Non affidarsi
   alla copia del solo `.py` server per mascherare il packaging. Separare /app del
   codice e /data dei temporanei/configurazioni; non montare dati sopra la venv.
4. Native: curl/CA per health e HTTPS; Pango/Harfbuzz/fontconfig/font per WeasyPrint;
   libcairo/glib/altre librerie solo secondo necessità verificata di provider/stack.
   Git/compiler possono servire nel builder per la dipendenza Git/sdist, non vanno
   automaticamente nel runtime. Pandoc è prerequisito della pipeline host, non del
   wrapper Marker salvo uso dimostrato. Poppler e librerie X/GL esistenti si conservano
   o rimuovono con evidenza degli import/provider, non per supposizione.
5. Congelare repository apt mediante timestamp effettivo Debian/debian-security
   snapshot e registrare versioni dpkg; partire dal 2026-10-02 e fissare gli import
   restituiti. [Debian snapshot](https://snapshot.debian.org/) documenta questa
   modalità. Nessun mirror mobile implicito se quello frozen fallisce. Conservare
   verifica firme; valid-until si gestisce solo per repository storici. Base/lock
   da soli non congelano i pacchetti apt o i pesi. Non promettere identità bit per bit
   di due immagini, ma medesimi input e versioni ricostruibili.
6. `.dockerignore`: escludere `.git`, `.venv*` (inclusa `.venv-python`), temp/tmp,
   `.env`/`.env.*` con eccezione esplicita solo `.env.example` se necessario,
   `local.env`, config JSON operativo, chiavi/credenziali/certificati privati,
   source_*, tutte le directory di conversione/log, cache modelli, build/dist,
   docker-data/.docker e dati locali. Preferire un contesto a lista ammessa ai
   soli file della build, con eccezioni provate. Gitignore non filtra il build context.
7. Compose CPU predefinito: build profile/arg `marker-cpu`, TORCH_DEVICE=cpu,
   volumi input/temp/cache separati. Controllare sul Surya installato il percorso
   cache effettivo; quello legacy `/root/.cache/datalab/models` non va assunto
   sufficiente senza evidenza. I pesi non entrano nell'immagine durante la build.
8. Override GPU: stesso Dockerfile con extra `marker-cu126`, TORCH_DEVICE=cuda e
   reservation NVIDIA con `capabilities: [gpu]`; verificare `docker compose ...
   config`. Il README attuale sostiene erroneamente che tutti i deploy block
   siano ignorati da Compose: correggere questa istruzione secondo le
   [docs ufficiali](https://docs.docker.com/compose/how-tos/gpu-support/).
   Il solo cambio di TORCH_DEVICE non cambia i binari torch CPU.
9. Eseguire build CPU e probe offline con entrypoint sostituito, **senza startup
   normale**. Healthcheck conserva il suo significato di liveness HTTP, non prova
   modelli/conversione. Avvio ordinario via Compose resta un passo operativo separato
   perché può scaricare pesi e font già allo startup.

Il profilo GPU può essere dichiarato/configurabile e locked senza essere collaudato
sull'hardware in questa run. Non attribuirgli supporto verificato. Incompatibilità
Python/torch fra motori può richiedere un progetto separato, ma solo con decisione
esplicita del supervisore e aggiornamento pertinente ad ADR 0006.

### P7 — Documentazione operativa e report

README: aggiornare prerequisiti/Python, quick start, struttura, installazione,
comandi di avvio e sviluppo/test, alternativa server locale e sezioni Docker/GPU.
Rimuovere il percorso Poetry inesistente e istruzioni requirements/setup/dev extra
obsolete. Usare comandi uv locked per il clone e spiegare comando/interprete
installato per la wheel e directory dei dati. Distinguere base/dev/server/CPU/GPU,
native, acquisizione tokenizer e startup con modelli. Verificare i comandi che
restano documentati; esempi di script downstream non presenti vanno indicati come
illustrativi, non come comandi disponibili del repository. Nessuna riscrittura
editoriale generale del README o dichiarazione che la nuova applicazione esista.

Guide italiane in `docs/`, secondo Diátaxis: `docs/how-to/ambiente-uv.md`
(ricreare ambiente/sviluppo/test), `docs/how-to/marker-legacy-docker.md`
(build/avvio opzionale con costi e prerequisiti), `docs/reference/toolchain-legacy.md`
(versioni, gruppi/extra, dieci moduli, cinque entry point, directory dati e limiti).
Aggiornare i soli indici docs pertinenti; per redigerle applicare la skill locale
write-diataxis-docs. `.env.example`: documentare variabili legacy effettive del
server/CPU/GPU solo se aggiunte o chiarite; non salvare UV_PROJECT_ENVIRONMENT
come variabile applicativa globale e non introdurre segreti.

L'implementatore aggiorna `documentation/CHANGELOG.md` con data, run, **lavoro
realizzato**, prove eseguite, limiti e prossimo passo; conserva le voci preesistenti.
Non registra merge/deploy o GO non avvenuti. Il supervisore cura passaggi di stato,
indici architetturali, arbitrati e stato di implementazione ADR. Se emerge una
nuova decisione significativa, l'implementatore la porta al supervisore e applica
la skill record-architecture-decision nel perimetro concordato.

## Criteri di accettazione e verifiche

### Matrice dei sette criteri del brief

| ID / criterio iniziale, invariato | Interventi | Prove e risultato atteso | Costi/prerequisiti e limiti |
| --- | --- | --- | --- |
| A1 — Ambiente pulito ricreabile usando il lockfile; scelta Python esplicita | P1 | V1: due sync in ambienti nuovi dal medesimo lock, interprete 3.12.13 managed, stesso inventario selezionato; lock invariato | rete per Python/pacchetti preparata esplicitamente; nessuna ML nel sync base; patch diversa dalla shell verificata per regressioni |
| A2 — Dipendenze di sviluppo separate dal runtime; package installato con moduli corretti | P1, P2, P5 | V2/V3: metadata wheel, dieci moduli, cinque target; runtime senza pytest/lint/torch/Marker; dev esplicito; server leggero distinto | extra motore può portare strumenti transitive upstream, da dichiarare; la wheel base non li installa |
| A3 — Import ed entry point verificati fuori dalla directory del codice | P2, P3 | V3/V4/V5: wheel non editable in temp esterna, origini site-packages, cinque comandi effettivi, almeno cleaning/validation orchestrate con output controllato e exit; -m e script diretti confrontati | Pandoc e cache tokenizer espliciti; help da solo insufficiente; nessun PYTHONPATH/copia dei sorgenti |
| A4 — Suite pertinente eseguita, distinguendo regressioni legacy, mock e prove reali | P0, P4 | V0/V5/V6/V8: suite selezionata, governance, confronti di contenuto, API mock e import reali separati; conteggi nuovi registrati; nessuna regressione introdotta | 67/6 storici; non chiamare suite selezionata collaudo completo; V10/V11 motori reali hanno prerequisiti diversi |
| A5 — README/guide aggiornati senza comandi inesistenti e senza dichiarare collaudi non fatti | P7 | V9: comandi di installazione/avvio/test provati nel perimetro indicato, link/diff validi, tabella verificato/non verificato coerente | esempi illustrativi identificati; istruzioni real-engine non eseguite restano dichiarate non collaudate |
| A6 — Build Docker coerente col perimetro del piano e limiti dei motori documentati | P5, P6 | V7/V8 obbligatorie: build CPU dal lock, digest/base/toolchain reali, native congelate, import/provider/API offline; Dockerignore e Compose validi | build pesante pianificata separatamente; pesi/inferenza CPU V10 e GPU V11 non impliciti; /health non dimostra conversione |
| A7 — Due review del piano, arbitrato GO, implementazione e due review con arbitrato finale | P0, consegne | V12: report reali ChatGPT/Claude sul medesimo snapshot in entrambi i giri, arbitrati del supervisore e rilievi tutti trattati | obbligatorio; questo piano e l'autore non producono review/GO; integrazione manuale dell'utente dopo GO finale |

### Catalogo delle prove

Costi indicativi, da misurare; un prerequisito mancante non trasforma una prova
obbligatoria in passata o opzionale. Le procedure seguenti sono **da eseguire
dall'implementatore**, non eseguite dal pianificatore.

| Prova | Procedura e osservabile | Classe/costo | Necessità proposta per l'arbitrato finale |
| --- | --- | --- | --- |
| V0 | baseline vecchi script/suite, fixture/hash/output/exit; vecchio avvio fuori dai sorgenti documentato | locale, secondi/minuti dopo preparazione runtime | obbligatoria per confronti pertinenti |
| V1 | lock check e due ambienti base/dev puliti; inventari Python/pacchetti/hash lock | locale con rete di installazione, minuti/decine MB; Python a parte | obbligatoria A1 |
| V2 | sdist/wheel build, ZIP/tar manifest, Requires-Dist/entry_points; build backend/versioni | locale/build, secondi/minuti senza ML | obbligatoria A2 |
| V3 | install wheel con runtime esportato dal lock in venv esterna; import origini, target metadata, pip check | locale installazione, poi nessuna rete | obbligatoria A2/A3 |
| V4 | config/pipeline/chunk help; config create/show; clean/validate senza argomenti; confronto -m/comandi/script | locale, secondi | obbligatoria A3/A5 |
| V5 | fasi orchestrate fuori sorgenti con F1–F5, output vuoti; confronto bytes/invarianti; processo negativo exit 1 | locale con Pandoc 3.1.3 e tokenizer preparato, secondi/minuti | obbligatoria A3/A4 |
| V6 | pytest tests selezionata, raccolta ordinaria/. senza smoke radice, unittest governance, controllo offline test | locale/no modelli, secondi/minuti | obbligatoria A4 |
| V7 | manifest digest, contesto Docker, compose config, build CPU senza modelli, lock e dpkg/pip inventari | Docker, rete/pacchetti ML, centinaia MB o GB e diversi minuti | obbligatoria A6; richiede fase operativa distinta dai test veloci |
| V8 | container senza rete, entrypoint diagnostico: import Marker/server, signature bind, provider selection, constructor/font/model mock, MarkdownOutput reale e wrapper fake HTTP | integrazione Docker senza inferenza, secondi/minuti dopo V7 | obbligatoria A6, etichette mock esplicite |
| V9 | comandi README/guide nel profilo indicato, link locali, diff check, coerenza documentazione/stati | documentale/locale, minuti | obbligatoria A5 |
| V10 | startup vero + PDF sintetico valido e almeno un documento non PDF; confronto sorgente/Markdown, formule/numeri/ordine/output/errori | motore CPU reale; pesi/font, RAM e tempo da misurare | separata; non necessaria a certificare la sola toolchain, necessaria prima di dichiarare Marker operativo/compatibilità reale |
| V11 | build variante GPU, disponibilità CUDA, stesso corpus V10, dispositivi/inventario/cache | hardware NVIDIA/driver/Toolkit, download e costi espliciti | separata; necessaria prima di dichiarare profilo GPU collaudato |
| V12 | controllo snapshot pre/post review e validità dei quattro report/due arbitrati | governance, senza modello OCR | obbligatoria A7; gestita dal supervisore/revisori |

Il brief A6 richiede una **build** coerente e limiti documentati: V7 non può essere
sostituita da `compose config` o da una lettura del Dockerfile. Se risorse/accesso
impediscono V7/V8, la consegna rimane con verifica essenziale mancante, da gestire
dal supervisore; non dichiarare la run completata. V10/V11 sono distinte perché
la run non sceglie/collauda definitivamente gli OCR. La loro assenza deve essere
esplicita nelle due review e nell'arbitrato, con limite “dipendenze/build e contratti
offline verificati; conversione Marker reale non verificata”. Se una modifica
necessaria cambia semantica del wrapper, il supervisore rivaluta la necessità di
V10; non si può usare il rinvio per nascondere un'incompatibilità osservata.

### Procedure concrete V1–V6

Le variabili indicate sono locali alla shell della prova, con percorsi assoluti.
In Linux usare una TemporaryDirectory esterna, per esempio creata con `mktemp -d`
in /tmp; conservarne risultati/evidenze prima della rimozione selettiva. Non
rinominare HOME/CODEX_HOME né toccare l'ambiente di altri progetti. Il report
specifica percorsi/hash effettivi, senza dump dell'intero ambiente.

Preparazione dell'implementazione, dopo disponibilità del pyproject:

```bash
uv --version
uv python install 3.12.13 --no-bin --no-registry
uv lock --python 3.12.13 --managed-python
uv lock --check
uv sync --locked --no-default-groups --python 3.12.13 --managed-python --no-python-downloads
uv run --locked --no-default-groups python -c 'import sys; print(sys.version); print(sys.executable)'
uv pip check --python .venv/bin/python
uv sync --locked --no-default-groups --group dev --python 3.12.13 --managed-python --no-python-downloads
```

Se si usa `.venv-python/`, anteporre a **tutti** i comandi managed
`UV_PYTHON_INSTALL_DIR` assoluta come descritto in P1; nessuna invocazione globale
è necessaria. Registrare una sola creazione iniziale del lock e poi usare locked.
`--frozen` omette il controllo di aggiornamento: utile solo se il layer Docker
incompleto lo richiede, seguito da locked completo; non è la prova di coerenza.

V1 usa `UV_PROJECT_ENVIRONMENT` con due nuovi percorsi assoluti (`base-a`, `base-b`)
e gli stessi flag base; poi due selezioni dev controllate. Confrontare inventari
nome/versione e `uv.lock` hash prima/dopo, Python/sys.base_prefix, assenza di ML/dev
dal base. Nessuna venv preesistente viene cancellata. Venv nuova prova installazione
pulita anche con cache wheel; una ricostruzione senza cache va indicata separatamente.

V2: build sdist e wheel in una cartella run univoca con `uv build --python 3.12.13
--managed-python --no-python-downloads --out-dir <destinazione>`; per default
verificare anche il percorso sdist→wheel. Ispezionare ZIP/tar senza eseguirli,
elenco moduli, assenza di dati/temp/segreti, metadata dei cinque script e dipendenze
standard. La riproducibilità qui riguarda input/risoluzione; non presumere wheel
byte-identiche senza controllo dei timestamp.

V3, con `RUN_REPO`, `RUN_VERIFY` e `RUN_WHEEL` impostate ai percorsi effettivi:

```bash
uv export --locked --no-default-groups --no-emit-project --format requirements.txt --output-file "$RUN_VERIFY/runtime.txt"
uv venv --python 3.12.13 --managed-python --no-python-downloads "$RUN_VERIFY/wheel-env"
uv pip sync --python "$RUN_VERIFY/wheel-env/bin/python" --require-hashes "$RUN_VERIFY/runtime.txt"
uv pip install --python "$RUN_VERIFY/wheel-env/bin/python" --no-deps "$RUN_WHEEL"
uv pip check --python "$RUN_VERIFY/wheel-env/bin/python"
```

Export base con hash include solo runtime/transitive; non rimuovere hash per
comodità. Installare la wheel con `--no-deps` dopo sync esatto evita una nuova
risoluzione. Richiedere un solo file wheel identificato da hash, non prendere
alla cieca il primo di una cartella dist con artefatti vecchi.

Il probe usa `subprocess.run` con `cwd=$RUN_VERIFY/data`, senza PYTHONPATH o
PYTHONHOME e con interprete assoluto wheel-env. `python -I -c` controlla import
dei nove moduli base, `module.__file__` sotto quel site-packages, assenza di path
del clone/editable nelle origini; `importlib.metadata.distribution` identifica
nome/versione, scripts e `direct_url.json` non editable. Nessuna copia dei moduli
applicativi nel workspace. Import validation/chunk può aprire log nel CWD: la
fixture deve essere scrivibile e l'effetto va controllato, non nascosto.

V4: `markdown-pipeline --help`, `markdown-config --help`, `markdown-chunk --help`
con binari assoluti wheel-env; `markdown-config --create-default` e `--show` con
confronto JSON. Eseguire davvero `markdown-clean` e `markdown-validate` con dati
preparati. Ripetere i casi pertinenti con `wheel-env/bin/python -m <modulo>` e,
per compatibilità sorgenti, interprete del progetto + script assoluto del clone.
`unified_converter` e `batch_monitor --help` possono essere controllati come
moduli aggiuntivi; non sono un sesto/settimo console entry point.

V5: in un nuovo CWD privo di `.py`, creare configurazione JSON sintetica,
output inizialmente vuoti e la sola fixture converted. Avviare effettivamente:

```text
<wheel-env>/bin/markdown-pipeline --step cleaning --force
<wheel-env>/bin/markdown-pipeline --step validation --force
<wheel-env>/bin/markdown-pipeline --step chunking --force --chunk-size 1000 --overlap 100
```

Confrontare ogni output pertinente con la baseline del singolo script, i dati
invarianti, validation report, index/chunk content e codice di uscita. Risultato
atteso positivo: 0 e output elaborato non vuoto con tutti i contenuti attesi nel
caso; niente file scritto nel clone/site-packages. Caso negativo: nuova directory
senza input converted, `--step cleaning` deve uscire 1 senza output di successo.
Registrare anche stdout/stderr del subprocess di fase per distinguere un modulo
mancante da un'assenza di input. Per conversione F4 usare `--step conversion`
con sorgente HTML unica e summary failed=0; non prova un batch misto o Marker.

V6, dopo preflight tokenizer esplicito e guardia rete:

```bash
uv run --locked --group dev python -m pytest tests/ -q
uv run --locked --group dev python -m pytest --collect-only -q
uv run --locked --group dev python -m pytest . --collect-only -q
uv run --locked --group dev python -m unittest discover -s tests/governance -v
```

Conservare elenco nodeid/count e distinzione governance/legacy/nuovi test. Nessuna
chiamata locale/cloud real-engine nella suite; mock e subprocess effettivi sono
indicati per caso. Fallimenti nuovi da dipendenze/Python vanno risolti o portati
al supervisore; non silenziare test/skips per replicare il numero storico.
Black/flake8/mypy possono verificare file nuovi o modificati con criterio; non
formattare tutto il repository né imporre che il debito legacy scompaia in r001.

### Procedure concrete V7–V11

V7: `docker compose config`, manifest inspection delle due immagini pinned,
`docker compose build marker-api` con plain log e progetto di prova identificato.
Non usare `up` per verificare la build. Il build scarica dipendenze ML ma non pesi:
richiede un passaggio operativo esplicito dell'implementazione con stima disco/rete
e limiti, distinto dai test veloci. Se non già incluso nel mandato operativo della
chat implementatrice, portare al supervisore quel bisogno prima del download
pesante, mantenendo il criterio A6 aperto.

Verificare contesto con sentinelle **non segrete** nei percorsi esclusi e un target
diagnostico del builder che ne dimostri l'assenza; rimuovere solo quelle sentinelle.
Il COPY finale a lista ammessa è un'ulteriore protezione, non una prova sufficiente
che temp/.env non siano stati inviati nel contesto. Salvare inventario del contesto,
digest immagine finale, lock hash, metadata pip e `dpkg-query`, log build e versione
Python/uv; verificare la rimozione di compiler/cache dall'immagine runtime prevista.

V8: `docker run --rm --network none` con entrypoint Python diagnostico e nessun
volume dei dati dell'utente. Controllare import di server, Marker, create_model_dict,
provider PDF/EPUB/DOCX/PPTX/XLSX/HTML/image, bs4/filetype e librerie full; confronto
signature bind con gli argomenti reali del wrapper; creare MarkdownOutput reale
e verificarne l'estrazione nel wrapper con converter fake. Per constructor patchare
download_font e predictor/processor che richiederebbero modelli, registrando quali
parti sono sostituite. Test di rendering/provider detection senza inferenza non
dimostra pipeline OCR. Nessuno startup normale non patchato, anche quando il
container non ha rete: i tentativi di download devono essere evitati.

V10, solo con perimetro download/hardware esplicito: volumi cache nuovi o verificati,
startup ordinario, identificazione di codice/pesi/font e log; PDF sintetico valido
con testo di riferimento e un HTML/DOCX sintetico del provider full. Nessun dummy
PDF dal vecchio test_conversion. Richiesta POST `/convert` e `/marker` (secondo
contratto effettivo), success/HTTP, Markdown non vuoto e corrispondente al documento,
numeri/formule/ordine e page_count; caso input corrotto con errore osservabile.
Conservare hash input/output/asset, cache e versioni effettive. Disabilitare LLM
remoti/credenziali; non usare health come criterio di inferenza riuscita. Se gli
asset non sono emessi dal wrapper, indicare conversione con quel limite, non bundle
completo. Acquisizione iniziale pesi separata, poi prova con egress disabilitato e
cache pronta dove tecnicamente possibile.

V11: prima di avviare modelli registrare driver, Toolkit e disponibilità GPU;
build con extra cu126, `torch.version.cuda` e dispositivo realmente utilizzato.
Poi corpus di V10 e misure minime tempo/memoria, senza benchmark comparativo degli
OCR. Se la GPU o il driver richiesto manca, registrare non eseguita; nessun fallback
CPU o cloud implicito presentato come risultato GPU. Stop/cleanup solo dei container
di prova con ID verificati; nessun `docker system prune`, `down -v` o cancellazione
di cache/volumi dell'utente.

### Registrazione degli esiti e dei fallimenti

Ogni prova ha ID, comando/argomenti, CWD, prerequisiti, profilo, branch/HEAD e hash
degli artefatti esaminati, versioni, ora/durata, stdout/stderr, exit code e risultato
atteso/osservato. Usare subprocess senza `check` che cancelli il contesto del
fallimento, oppure catturare esplicitamente l'eccezione e l'output. Conservare file
prodotti e diff deterministici; niente soli screenshot di “passed”. Stati del
report: PASS, FAIL, NON_ESEGUITA, IMPEDITA, con motivazione. Mock separato da reale.

Un codice 0 con contenuto mancante è FAIL della prova di contenuto; una health
positiva con startup Marker fallita non prova il motore. I difetti legacy sono
identificati per fixture, sorgente/base e output baseline, con effetti e rinvio
motivato. Non attribuire al solo packaging la correzione o preservazione di ciò
che non è stato misurato. Non salvare API key, `.env` reale o dump environment.

## Rischi, migrazione e recupero

| Rischio / incertezza | Trattamento e criterio di recupero |
| --- | --- |
| Restrizione Python rispetto a metadata legacy e upgrade patch | documentare supporto 3.12 e pin; baseline e confronto separano effetti della patch da packaging/dipendenze; altre minor restano non verificate |
| uv 0.10.10 / Python 3.12.13 non ultime versioni | scelta riproducibile per la run, nessuna dichiarazione “ultimo/sicuro per deploy”; aggiornamento esplicito con ripetizione delle prove, mai pin mobile |
| Backend/digest/base non ancora recuperati o risoluzione Marker fallita | preflight obbligatorio; evidenza del blocco, scelta alternativa tramite supervisore; nessun placeholder nel risultato finale |
| Grafo universale CPU/CUDA incompatibile | conflitti/marker espliciti; se insufficienti, piano rivisto per progetto server separato e aggiornamento ADR, senza appesantire il base |
| Dipendenza Git/build backend non congelato dal lock runtime | SHA pieno, metadata verificato, vincoli backend, inventari di build e hash delle wheel; non dichiarare pesi bloccati da uv.lock |
| Moduli flat con nomi generici / editable maschera omissioni | origine di ogni import e wheel esterna non editable; eventuale namespace rimandato a fase 2 |
| Differenza CWD/.env/report fra script e wheel | singola directory dei dati, subprocess -m con stesso interprete/CWD, override .env sintetico e output controllati |
| Cleaning/copia asset/exit di batch già difettosi | baseline distinta, nessun nuovo danno; report fattuale delle perdite; niente fix algoritmico mascherato da migrazione |
| Tiktoken, font e pesi scaricati in test/startup | preparazione tokenizer esplicita; fake startup e constructor offline; build e inferenza in passaggi separati |
| Health server positiva anche senza Marker pronto | dichiarare liveness; verifica API/contenuto e logs, non affermare conversione dal solo endpoint |
| Cache/volumi legacy non corrispondono a Surya pinned | verificare percorso sul pacchetto installato, usare volumi di prova distinti, conservare quelli operativi |
| Docker apt o variante hardware mobile | digest/base, snapshot apt, indici espliciti, target documentato; no supporto multipiattaforma/GPU dichiarato senza prova |

Migrazione dei dati: nessuna modifica automatica dei documenti dell'utente; JSON
legacy resta compatibile. Provare solo workspace sintetici. La migrazione del clone
crea ambienti nuovi, non disinstalla pyenv o altri ambienti. La rimozione dei due
vecchi file di packaging avviene dopo che README/Docker/guide usano la nuova fonte.

Recupero: conservare base Git, patch delle sole modifiche della run, fixture ed
evidenze prima/dopo. In caso di fallimento, fermare solo processi/ambiente di prova
identificati e consegnare checkpoint, senza reset/cancellazione del working tree o
di temp. Gli ambienti baseline e precedenti rimangono riutilizzabili; una venv nuova
può essere eliminata selettivamente dopo aver archiviato i risultati. Tornare ai
vecchi sorgenti non ricrea magicamente il Marker installato da master: se esiste
un'immagine precedente registrarne digest/versioni prima di sostituirla, senza
avviarla o scaricare modelli come parte del recupero. In assenza di quell'immagine,
il precedente motore non è una baseline riproducibile nota.

Nessun commit/merge/push o deploy da parte degli agenti. Dopo l'arbitrato finale il
supervisore prepara i comandi contestualizzati per l'utente; una promozione/deploy
richiede il proprio perimetro operativo e le prove pertinenti.

## Consegne e handover

Consegna del pianificatore:

- Questo piano r001, senza implementazione o esito di review.
- [Evidenze datate — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), inventario AST e help uv;
  [verifiche finali — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
- [Checkpoint del ruolo — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), con prove fatte/non fatte,
  incertezze e nessun processo lasciato implicito.

Consegna attesa dall'implementatore dopo il GO sul piano: codice/lock e documenti
previsti, report per P0–P7 e matrice A1–A7, evidenze V0–V9 e V10/V11 se eseguite,
conteggi effettivi, manifest degli artefatti/build e checkpoint. Qualsiasi scostamento
sostanziale va al supervisore prima di sostituire il perimetro approvato.

**Prossimo destinatario: supervisore.** Deve leggere piano/checkpoint/evidenze,
verificare il contesto, creare lo snapshot comune includendo evidenze essenziali,
aggiornare i propri registri e preparare i due prompt indipendenti ChatGPT/Claude.
Le review non leggono il report corrente dell'altra chat. L'implementazione attende
entrambe le review e l'arbitrato GO riferito a questo piano identificato.
Il pianificatore non modifica STATE, handover comuni, indici, eventi o arbitrati.
