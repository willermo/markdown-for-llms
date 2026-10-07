# Review — implementazione — run-a001-fase0-uv — r001 — Claude

- Autore/provider/modello e chat: **Anthropic, Claude Opus 5.5**
  (`claude-opus-5-5[1m]`), Claude Code nell'estensione VSCode. Nuova chat
  dedicata esclusivamente a questa review; ID chat non esposto. Nessun subagente,
  workflow o delega.
- Prompt di origine: [prompt54](../prompts/54-review-implementation-r001-claude.md)
  e [criteri comuni](../prompts/review-implementation-r001-common.md), ricevuti dall'utente.
- Oggetto: delta completo della run a001 rispetto a
  `66ba82200e5def5a4db76f9bafccb0731b506091` (worktree modificato e file nuovi
  non ignorati), report-r017 e prove selezionate.
- Snapshot: `impl-r001-stage-final-s001`, manifest SHA-256
  `77a3098c78531475fd94e18cfff2b9e945a88c51e86eb273b9abf1fe91a35374`, uguale a
  [freeze-identity.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
  `run_context.py verify` con il managed 3.12.13 locale: **MATCH prima**
  (09:38:35) e **MATCH dopo** (vedi evidenze `identity-before.txt` e `identity-after.txt`).
  Branch `feature/run-a001-uv`, HEAD/dev `66ba822…`, indice vuoto.
- Piano r003 SHA `462dc1f4…a2066f38`, arbitrato r003 SHA `f14ebe2f…4540df0d`:
  ricalcolati e conformi a quelli indicati.
- Indipendenza: **non ho letto la review ChatGPT dell'implementazione**. Un solo
  `ls` di `reviews/` all'inizio mostrava soltanto le sei review storiche del piano.
  Nessuna altra review del codice letta.
- Giro: **iniziale**, prima review dell'intero delta. Nessun rilievo aperto in ingresso.
- Esito: **NO_GO**.

## Ambito e prove

### Letture

- AGENTS, skill `manage-implementation-run`, template review; `temp/HANDOVER.md`,
  STATE/HANDOVER della run, brief.
- Piano r003 integrale; arbitrato r003 (D1–D5); addenda R013–R018 integrali.
- Report-r017; report-r016, sezione Docker. Matrice, equivalenza, final-checks e
  final-static r017. Workload della fedeltà installata r002, compose receipt e
  config risolte, driver V7/V8 `build_cpu_s053.py` e `probe_cpu_r004.py`
  (solo parti pertinenti).
- **Codice del delta:** diff integrale di `config.py`, `master_workflow.py`,
  `unified_converter.py`, `tests/conftest.py`, `tests/integration/test_pipeline.py`;
  `pyproject.toml`, `MANIFEST.in`, `build-constraints.txt`, `.python-version`,
  `.dockerignore`, `.gitignore`, `.env.example`, `Dockerfile`, i tre Compose.
  Test nuovi `tests/unit/test_phase_launch.py`, `test_config_cli.py`,
  `tests/api/test_marker_wrapper.py`, `tests/packaging/test_installed_distribution.py`,
  `tests/docker/test_marker_contract.py`.
- **Diagnostici:** `make_source_manifest.py`, `check_preflight.py`,
  `check_source_inputs.py`, `verify_distribution.py` (receipt/installazione),
  `image_prepare.py`, `run_offline.py` (flusso di controllo).
- **Documentazione:** README, guide `ambiente-uv.md`, `marker-legacy-docker.md`,
  reference `toolchain-legacy.md`, diff AGENTS; ADR0006 solo per lo stato.

### Prove eseguite dal revisore

Tutte producono output solo in
`evidence/review-implementation-r001-claude/`.

1. **Identità prima/dopo.** SHA dello snapshot e `verify` MATCH, Git in sola lettura.
2. **`equivalence_recheck.py`**, sola lettura, con il managed 3.12.13 `-I -B`.
   - I dieci moduli correnti coincidono con S-s055 e S-s021 (base della fedeltà
     riusata); build input e diagnostici coincidono con S55.
   - SHA di sdist/wheel B46 confermati.
   - La wheel contiene esattamente i dieci moduli, con byte uguali al worktree.
   - Nella sdist non c'è nulla di operativo.
   - I 25 input filtrati dell'immagine r005 coincidono con i file correnti.
3. **`official_gate_probe.py`**: S sintetico in memoria, nessuna scrittura.
   - `validate_current(official=True)` rifiuta uno S standalone.
   - `snapshot_check` accetta solo `run_id == "run-a001-fase0-uv"`.
   - `tests/conftest.py` non passa mai `--standalone`.
4. **`netns-probe`**: due build `--output type=cacheonly`, `--pull=false`,
   `--no-cache` sul builder `default`, `FROM run-a001-marker-cpu:r005` locale.
   - Rete `default` (quella usata da Compose senza `build.network`):
     interfacce `eth0 lo`, netns diverso dall'host.
   - Rete `none`: solo `lo`.
   - Nessun pull, download, immagine prodotta o modifica di immagini esistenti.
     Restano due piccoli record nella build cache del builder `default`, non
     rimossi (niente cleanup). Il costo non è stato addebitato alle quote
     dell'implementatore: è trascurabile e lo dichiaro qui.
5. **Coerenza statica `uv.lock`/pyproject** con `tomllib`, senza risoluzione uv:
   - SHA del lock uguale al root promosso in R014;
   - requires-dist base, gruppo dev ed extra coincidono;
   - conflitto CPU/cu126 presente;
   - varianti torch CPU/cu126 dagli indici espliciti.

### Verificato senza rilievi

- **Codice prodotto P2/P3:**
  - `load_workspace_env` prima di `apply_env_overrides`, `override=False`, solo
    nel workspace.
  - `load_dotenv()` all'import rimosso.
  - Fasi lanciate con `[sys.executable, -I, -B, -m, modulo]`, `cwd=workspace`,
    PYTHONPATH/PYTHONHOME rimossi.
  - Preflight in figlio `-I` che verifica l'origine sotto purelib/platlib.
  - Workspace catturato all'istanza; percorsi assoluti conservati.
  - Epilogo help corretto; ordine legacy JSON → CLI → env conservato.
  - Nessuna modifica algoritmica a cleaning/validation/chunking.
- **Fedeltà:** le asserzioni della suite packaging e del workload r002 sono
  byte-exact rispetto alla baseline, senza normalizzazioni:
  - cleaned, validated e report per campo;
  - chunk e metadata, chunks_index, sliding window;
  - F4 Pandoc;
  - casi .env 1200/80 e shell 1600/120;
  - 14 sentinelle;
  - perdita legacy dell'asset preservata;
  - negativo di cleaning vuoto.
  Il riuso sullo S55 è legittimo: moduli, entry point e header METADATA sono
  identici e cambia solo il corpo del README (punto 2).
- **Packaging:** dieci `py-modules`, cinque script, `packages=[]`, extra separati,
  `default-groups=[]`, `reinstall-package`, `no-build-package` Marker,
  `python-downloads=manual`, `required-version`.
- **API mock:** sette casi coerenti con P4 e CLA-S6 r002.
- **Runner:** il comando parte solo dopo PASS del runner e dei preflight, senza
  fallback non confinato.
- **Costi:** coerenti con STATE e R018.
  - Core: H850288640, residuo 22020096 byte, 3940,5 s.
  - Docker: 568,7 s, quote invariate.

### Non eseguito, con motivo

- Nessuna riesecuzione di V1, installer, build, sync, `uv lock --check`, suite o
  campagne: costo e mutazione di cache, nessun dubbio concreto sui PASS ricevuti
  per gli input verificati al punto 2.
- Letti solo in parte: `check_runner.py`, `run_baseline.py`, `make_fixtures.py`,
  `probe_image_contract.py` e i test dei diagnostici (`test_uv_diagnostics.py`,
  `test_package_diagnostics.py`, `test_core_diagnostics.py`).
- Non rivisti nel merito, in quanto dominio del supervisore: CHANGELOG, ADR,
  protocollo e template di governance.
- Nessuna `docker compose build` reale: avrebbe richiesto la supply e un
  `image-source.json` che non esistono nel clone. La prova 4 dimostra il punto
  bloccante senza download.

## Rilievi

| ID | Severità e blocco | Posizione | Raggiungibilità/input ammessi | Attribuzione e requisito | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **CLA-I001** | **Alta — bloccante sì** | `tests/conftest.py:248-262`; `scripts/diagnostics/run-a001-fase0-uv/make_source_manifest.py:98,145`; `check_preflight.py:22`; `AGENTS.md:99`; `docs/reference/toolchain-legacy.md:85,120` | Qualunque `pytest`, compreso il C-fast prescritto da AGENTS, da: utente normale con S standalone come da guida uv; clone senza `temp/` (ignorata da Git); una run successiva con altro `run_id` | Run, codice nuovo. Arbitrato r003 **D2**: «variante normale senza temp/run … nessuno snapshot di run richiesto all'utente normale». **CLA-S2 r003**: modalità locale ridotta ammessa. A4 e A5 | `pytest_configure` esige RUN_SOURCE_MANIFEST/RUN_INSTALLATION_RECEIPT e chiama il preflight senza `--standalone`, quindi `official=True`. S standalone rifiutato; S run-stage accettato solo da uno snapshot `temp/run-a001-fase0-uv/snapshots/…` con `run_id` cablato. Esito: `UsageError` prima della collection. Fuori da questa run, e anche qui dopo la pulizia di `temp/`, **nessuna suite del repository è eseguibile**. La reference (riga 120) rinvia per un «clone senza temp» alla guida, ma il percorso documentato non arriva ai test | `official_gate_probe.out.json`; lettura del codice | Il harness permanente accetta lo S standalone documentato (selezione esplicita, non fallback silenzioso). Le prove ufficiali conservano la modalità run-stage. Il gate permanente non è legato al solo `run-a001-fase0-uv`. Prova reale: C-fast con S standalone, senza snapshot di run, più negativo S stantio → FAIL prima della collection. Guida/reference/AGENTS allineate |
| **CLA-I002** | **Alta — bloccante sì** | `docker-compose.yml:4-14`, `docker-compose-build.yml:4-14`; `Dockerfile:5,12,22,32-41`; `image_prepare.py:48,77`; `docs/how-to/marker-legacy-docker.md`; strumenti solo in `temp/…/docker-cpu-r001/` | `docker compose -f docker-compose.yml build` o `up --build`, e lo stesso con `docker-compose-build.yml`, dal clone; ricostruzione per aggiornamento (P6.10) | Run. Piano **P6** («Conservare il Compose build evita di rompere un percorso esistente», punto 8); V7 prescriveva `docker compose … build marker-api`; P2:469-470 (diagnostici nel repository). A6 | (a) Nessun Compose dichiara `build.network: none`. Con la rete di default il RUN vede `eth0 lo` e `image_prepare.py:48` interrompe con «Namespace build non isolato». (b) Il contesto Compose è il clone, privo di `image-source.json` (la guida lo vuole solo nel contesto usa e getta), quindi la riga 77 fallisce comunque. (c) `verified-inputs` (APT locale firmato, ~94 wheel, sdist, backend) e `image-source.json` non hanno produttore nel repository: acquisizione, resolver APT, cattura e driver di build stanno solo nella `temp/` ignorata. V7 è passata con `buildx --network none` su una copia; i Compose hanno solo `config` PASS. Lo scostamento dalla prova di piano non è registrato. Entrambi i percorsi Compose di build sono rotti e la build non è riproducibile dal repository | `netns-probe/result.txt`; `compose-cpu.json` (build senza `network`); `build_cpu_s053.py:48`; report-r016:130-157 | Opzione 1: rendere funzionante la build Compose (`network: none`, generazione contesto/`image-source.json` e supply con strumenti versionati e indipendenti dalla run) e dimostrarla una volta, o motivarne l'equivalenza. Opzione 2: scostamento approvato dal supervisore, con Compose solo runtime (`image:`) e build `buildx` documentata, più strumenti versionati per supply e `image-source.json`. In entrambi i casi la guida riporta una procedura eseguibile dal repository |
| **CLA-I003** | Media — bloccante no | `tests/docker/test_marker_contract.py:17-32`; guida Docker:84-95; `temp/…/probe_cpu_r004.py` | C-docker documentato nella reference; ripetizioni V8 future (P6.10) | Run. P4: il driver V8 è `tests/docker/test_marker_contract.py`; P2:469-470 | Il test versionato **non è mai stato eseguito**: V8 PASS proviene da un driver solo in `temp/`. Il test omette l'irrobustimento che la guida prescrive e che il driver eseguito applica: cap-drop ALL, no-new-privileges, limiti memoria/pids, healthcheck disabilitato, verifica dei file di startup a container fermo. Lo strato C-docker del repository resta non provato e più debole di quello dimostrato | Assenza di receipt pytest Docker; report-r016:156-157; confronto argv | Allineare il test versionato al driver eseguito, oppure versionare quel driver come C-docker ufficiale. Eseguirlo una volta contro r005 con receipt (nessuna rebuild necessaria), oppure documentare il limite |
| **CLA-I004** | Media — bloccante no | `README.md` (1545 → 67 righe); `final-static-r017.json` `old_74_blocks`/`old_9_inline` | Lettori del README e della documentazione utente | Run. Piano P7:891-894 («assegnare a ogni blocco/inline precedente mantenuto/modificato/rimosso/spostato … delta motivati»); CLA-P004 r002; A5 | Tutti i 74 blocchi hanno una sola disposizione generica (`replaced_or_removed…`, `current: null`); i 9 inline un'altra. Nessuna distinzione spostato/rimosso né destinazione o motivo. Sono sparite senza destinazione sezioni valide non legate alla toolchain: struttura del `pipeline_config.json` (directories, validation_thresholds, cleaning_settings, chunking, conversion_settings), scelta del metodo di conversione, Marker locale multi-formato, polling cloud, scenari d'uso. La reference copre solo CLI e variabili | Conteggi in `final-static-r017.json`; titoli del README legacy | Disposizione per elemento con destinazione o motivo. Ripristinare o spostare in `docs/reference` la documentazione ancora valida della configurazione JSON e dei backend, oppure registrare una decisione esplicita |
| **CLA-I005** | Bassa — bloccante no | `README.md:9-12`; guide uv/Docker/reference (righe di stato iniziali); `AGENTS.md:119` | Lettori dopo GO/merge; il README è anche il long_description della wheel (input B46) | Run. A5 («senza dichiarare…»), coerenza degli stati | Il README incorpora lo stato transitorio («non integrata e senza GO finale»; «Docker CPU V7/V8 è in convalida», incoerente con le guide che la danno eseguita). Diventerà falso alla chiusura e cambiarlo invalida B. `AGENTS.md:119` dice ancora che il README «contiene ancora la guida legacy», ora falso | Lettura dei file | Alla chiusura, aggiornamento documentale con la regola di invalidazione README → B, oppure stato della run spostato fuori dal README. Correggere `AGENTS.md:119` |

### Suggerimenti, non requisiti

- **S1.** Un `uv sync` o `uv run` senza `--no-editable` produce un'installazione
  editable che il preflight rifiuta come «Origine non installata». Il comportamento
  è voluto; il messaggio potrebbe indicare esplicitamente `--no-editable`.
- **S2.** `image_prepare.py:49` include il path `/run/user/1000/…`. Dentro
  BuildKit è innocuo, ma conviene generalizzarlo se lo strumento diventa permanente.
- **S3.** Il runtime contiene `/opt/build-proof` e `/opt/prepare-proof` con copie
  dei diagnostici. Sono accettabili come provenienza; valutare etichette o
  esportazione fuori dall'immagine per un deploy.

## Motivazione dell'esito e limiti

**NO_GO per CLA-I001 e CLA-I002.**

La migrazione del codice prodotto è corretta e ben dimostrata. Non trovo difetti
nei tre moduli modificati, nel packaging né nelle prove di fedeltà, i cui legami
con gli input correnti ho ricalcolato in modo indipendente. A1–A3 sono
sostenuti dalle prove ricevute e dai miei ricalcoli.

**A4 (CLA-I001).** La suite funziona solo dentro questa run. L'arbitrato D2
richiedeva esplicitamente un percorso normale senza run e la reference lo
promette. Dopo il merge e la pulizia di `temp/`, i test del repository non sono
eseguibili da nessuno, compresa la run successiva. Non è una rarità: è il primo
comando che un contributore esegue.

**A6 (CLA-I002).** L'immagine r005 e i nove sottocasi V8 sono reali e legati agli
input correnti. Però i due Compose di build, che il piano chiedeva di conservare,
falliscono in modo deterministico. La build non è riproducibile dal repository,
perché i produttori degli input stanno solo in `temp/`. Il PASS V7/V8 resta
valido come prova dell'immagine; non dimostra il percorso consegnato.

CLA-I003–I005 non bloccano da soli; conviene chiuderli nello stesso giro di fix.

### Limiti

- Non ho rieseguito le campagne.
- La lettura dei diagnostici di runner e baseline e dei loro test è parziale.
- Non ho tentato una `docker compose build` completa.
- La review vale per lo snapshot `impl-r001-stage-final-s001` verificato MATCH
  prima e dopo.

Nessuna modifica a prodotto, snapshot, registri condivisi o arbitrati. Nessuna
operazione Git, commit, merge, push o deploy.

Consegna al **supervisore** per l'arbitrato con il report ChatGPT.
