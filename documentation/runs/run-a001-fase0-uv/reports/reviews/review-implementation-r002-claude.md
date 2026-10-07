# Review — implementazione — run-a001-fase0-uv — r002 — Claude

- Autore/provider/modello e chat: **Anthropic, Claude Opus 5.5**
  (`claude-opus-5-5[1m]`), Claude Code nell'estensione VSCode. Nuova chat
  dedicata solo a questa review; ID chat non esposto. Nessun subagente, workflow
  o delega.
- Prompt di origine: [prompt58](../prompts/58-review-implementation-r002-claude.md)
  e [criteri comuni r002](../prompts/review-implementation-r002-common.md).
- Oggetto: fix di CLA-I001–I005 e GPT-I001 e regressioni delle parti toccate.
  Delta: 30 path tra final-s001 e la ricezione, secondo
  [delta-s001-to-received — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
  Inclusi report-r018/r019 e le prove `fix-review-r001`, `fix-review-r002` e
  `docker-cpu-r001/fix-review-r001`.
- Snapshot: `impl-r001-stage-final-s002`, manifest SHA-256
  `e0cb33d16a5766f653923c2d25cf365f910d7e738d542893db2a1ac589272c1d`
  (1152887 byte), worktree `361da837…ee866142`, 133 file e 4050 artefatti,
  uguali a [freeze-identity — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
  `run_context.py verify` con il managed 3.12.13: **MATCH prima** (12:22:38) e
  **MATCH dopo** (vedi `verify-before.txt` e `verify-after.txt`).
  Branch `feature/run-a001-uv`, HEAD/dev `66ba822…`, indice vuoto.
- Indipendenza: **non ho letto la review r002 di ChatGPT** né il suo checkpoint.
  Ho letto come storia comune le review r001 e l'arbitrato r001.
- Giro: **fix**. Sei rilievi arbitrati aperti in ingresso, tutti verificati.
- Esito: **GO**, con CLA-I006 da correggere nei registri (non bloccante).

## Ambito e prove

### Letture

- **Governance:** AGENTS, template review, arbitrato implementazione r001, review
  r001 Claude, R019, R020, report-r018/r019, reception, freeze-identity,
  final-checks e documentation-delta del supervisore.
- **Codice del delta, integrale:**
  - `tests/conftest.py`, `tests/docker/test_marker_contract.py`;
  - `make_source_manifest.py`, `check_preflight.py`, `check_source_inputs.py`,
    `run_tests.py`, `make_input_inventory.py`, `prepare_docker_inputs.py`,
    `image_prepare.py`, `run_docker_contract.py`;
  - `run_offline.py`, parte `outside()`, e `verify_distribution.py`, parti
    `validate_receipt` e `main`;
  - `Dockerfile`, `.dockerignore`, i tre Compose;
  - i test LauncherCleanup e la fixture Git di `test_package_diagnostics.py`.
- **Documenti:** README, AGENTS, `ambiente-uv.md`, `marker-legacy-docker.md`,
  `pipeline-backends.md`, `pipeline-config.md`, `toolchain-legacy.md`
  (sezione test/Docker). Le affermazioni sono state confrontate con `config.py`
  (dataclass e loader), `unified_converter.py` (argparse, polling, timeout,
  fallback) e `batch_monitor.py`.
- **Prove ricevute:**
  - `standalone_and_runid.py`, le 18 call e il workload/result r003;
  - E-standalone-fast ed E-pytest fast;
  - V1 r003 (fasi e 51 call);
  - receipt s060 del wrapper R;
  - receipt di supply e contesto r001–r004;
  - Compose r004/r007: invocation, stdout, config 0/1/2, own-image, history;
  - C-docker r002: invocation, result e contract/result;
  - final-docker-ledger, equivalence-final, matrix-final, delivery,
    final-checks, entry, approval-wait, resource-authorization-r002 e gli script
    `deliver_r002.py`, `finish_host_r003.py`, `run_core_proof_r002.py` (costs).

### Sonde eseguite

Tutte stdlib, in sola lettura, con il managed `-I -B` e timeout finiti. Gli
output sono in `evidence/review-implementation-r002-claude/`.

| Sonda | Esito | Secondi interni |
| --- | --- | --- |
| `readme_reconciliation_check.py` | 74/74 fence del README originale (SHA `2f154ee4…`) con riga e hash reali uguali; 9/9 inline trovati; nessuna destinazione mancante o motivo vuoto; 5 hash di pianificazione r003 distinti e conservati separatamente | 0,002 |
| `b56_readme_modules_check` | sdist/wheel B56 (`a6a85fe1…`/`6fff8af1…`): README, dieci moduli e pyproject uguali al worktree; descrizione METADATA = README corrente | 0,011 |
| `time_accounting_check` | ricostruzione del cumulativo dalle formule e dai mtime: vedi CLA-I006 | 0,001 |
| `make_input_inventory.py` reale + `verify_input_inventory` | PASS, 3767 record; il tool documentato funziona. Output da 827 KB rimosso dopo il controllo | 0,84 + 0,49 |
| `doc_links_check` + `git diff --check` | 54 link locali validi, nessun errore di whitespace | 0,001 |

Costo: circa 1,4 s interni, meno di 4 s con l'avvio degli interpreti (quota R020
di 30 s). Output finale sotto 20 KB, più report e checkpoint. Nessuna rete,
Docker, build, installer, suite, ML o scrittura fuori dalla cartella evidenze.

### Non eseguito, con motivo

- Nessuna suite, R, V1, build, Docker o Compose: quota implementativa esaurita;
  le prove ricevute sono leggibili e legate agli input.
- I rifiuti di `--test-mode`/`RUN_TEST_MODE` invalidi o discordanti nel conftest
  sono verificati solo dalla lettura del codice. Nessun test li esercita: nessun
  dubbio concreto, il codice è lineare.
- Il ramo `--acquire` di `prepare_docker_inputs.py`, quello per un clone nuovo,
  non è mai stato eseguito: richiede rete non autorizzata. La supply r004 usa
  `--reuse-in-place` e riverifica ogni payload contro lock, firme e Packages.
  Il codice letto è coerente: setuptools 84 è nel lock e quindi entra nei
  payload scaricati.

## Chiusura dei rilievi originari

| ID originario | Esito | Motivo e prova |
| --- | --- | --- |
| CLA-I001 (bloccante) | **Risolto** | Vedi nota CLA-I001 |
| CLA-I002 (bloccante) | **Risolto** | Vedi nota CLA-I002 |
| CLA-I003 | **Risolto** | Vedi nota CLA-I003 |
| CLA-I004 | **Risolto** | Vedi nota CLA-I004 |
| CLA-I005 | **Risolto** | Vedi nota CLA-I005 |
| GPT-I001 | **Risolto** | Vedi nota GPT-I001 |

### CLA-I001 — modalità standalone e gate generalizzato

**Codice:**

- Il conftest seleziona la modalità solo esplicitamente (`--test-mode` o
  `RUN_TEST_MODE`, default official). Valori invalidi e scelte discordanti
  sollevano `UsageError`; con standalone passa `--standalone` al preflight
  `-I -B`.
- `run_tests.py` replica la stessa selezione. Fa preflight prima di
  `import pytest` e il postcheck nella stessa modalità.
- `validate_current(official=True)` rifiuta lo S standalone.
- `snapshot_check` non è più cablato ad a001:
  - run_id e label sono validati con regex;
  - il path deve essere `host/temp/<run_id>/snapshots/<label>.json`;
  - schema int, `run_context verify` del clone stesso.
- Git standalone tollera HEAD/dev assenti.

**Prova reale (`proof-standalone-runid-r003`):**

- Copia senza `temp/` né snapshot: S, B e I standalone, poi C-fast con il
  `run_tests.py` permanente della copia.
- Esito E-standalone-fast: scope standalone, 137 test raccolti, exit 0,
  0 skip, postcheck `--standalone` PASS.
- Negativi senza sync, tutti con exit 1 prima di collection e import:
  - S stantio: `Input corrente diverso da S: exceptions.py`;
  - I con `source.id` alterato: `Receipt vecchia`;
  - I con file installato alterato: `Installazione diversa`;
  - official che riceve S standalone: rifiutato, così come il preflight official.
- Run_id diverso `run-b002-fase0-uv`: snapshot e verify reali, S official,
  I e preflight PASS.
- Rifiutati `../escape`, schema bool, label/path difforme e SHA errato.

Le prove esercitano codice permanente, non surrogati. Il mock Git compare solo
nei test unitari.

### CLA-I002 — percorso Compose di build

- **Compose:** entrambi dichiarano `network: none`, contesto preparato e
  `additional_contexts` verified-wheels/verified-apt.
- **Tool permanente** `prepare_docker_inputs.py` (SHA `2e8df0da…`): è lo stesso
  file registrato nelle receipt supply r004 e context r001–r004.
  - Supply: export uv locked, wheel ABI, SHA/size del lock, ZIP/RECORD,
    backend84; gpgv InRelease, Packages SHA, resolver apt nativo in container
    `--network none`/cap-drop/no-new-privileges.
  - Context: copia solo i file di S più Dockerfile e diagnostici, e scrive
    `image-source.json` solo nella copia.
- **Build reale Compose r007** (`docker compose … build marker-api`):
  - `image_prepare` è stato eseguito davvero (step 7/9, 7,0 s):
    - netns isolato con sola `lo`;
    - allowlist del contesto;
    - EbookLib/backend84;
    - hook nativo `build_wheel`.
  - Eseguiti anche `image_package` e l'I del builder.
  - Lo step APT è CACHED da una build con LLB e rete identiche.
  - History Completed 27/27 (08:54:01Z), immagine creata alle 08:53:57Z:
    `sha256:b3db6dc2…c3a8`.
  - Il campo `native_completion:false` è una euristica storica sul log. History
    e inspect dimostrano il completamento.
- **Secondo Compose:** `build` risolto identico al primo (config-0 = config-1).
  GPU solo config, con `MARKER_EXTRA=marker-cu126`.
- **Equivalenza:** i 24 oggetti del contesto consumato coincidono con il
  worktree corrente (ricalcolato).
- **Riserva sul clone nuovo:** il percorso documentato vale anche per un clone
  nuovo, ma `--acquire` è dimostrato solo dal codice (limite dichiarato).

### CLA-I003 — C-docker versionato

- `tests/docker/test_marker_contract.py::test_marker_contract` eseguito una
  volta: un nodeid, 1 passed, nove sottocasi, nessun tentativo di rete,
  container raccolto.
- Il test invoca `run_docker_contract.py` versionato, che applica:
  - container per ID, pull never, rete none, nessun mount;
  - cap-drop ALL, no-new-privileges, healthcheck disabilitato,
    pids128/mem2g/cpu2;
  - export del filesystem fermo (pth/customize/pyvenv/backend84) prima di
    Python;
  - stdin fidato con AST validato e full I prima degli import;
  - receipt anche sul FAIL e collect/rm.
- Lo SHA di test, driver, probe e producer è uguale in s058 (esecuzione),
  s060 e worktree. Ricalcolato: il probe in stdin-identity coincide.

### CLA-I004 — riconciliazione README

- Disposizioni individuali dei 74 blocchi: 40 modificati, 10 spostati,
  24 rimossi. Dei 9 inline: 4 modificati e 5 rimossi.
- Ogni elemento ha identità originale, motivo e destinazione esistente.
  Gli hash reali delle fence sono distinti dai 5 hash normalizzati r003
  (ricalcolo indipendente).
- `pipeline-config.md` e `pipeline-backends.md` coincidono con il codice:
  - chiavi `*_settings`;
  - default;
  - polling `max_polls<=0` e `conversion_timeout<=0`;
  - fallback Pandoc per i non-PDF;
  - flag reali di `unified_converter` e `batch_monitor`.
- Pseudocodice RAG/embedding rimosso con motivo.

### CLA-I005 — documentazione durevole

- README senza stato di run o GO; rimanda ai registri. `AGENTS.md:119` corretto.
- README di 4070 byte (`33ce6c4b…`) uguale a quello in sdist e METADATA B56:
  è precedente a B56.
- Le modifiche a Dockerfile e `.dockerignore` hanno invalidato r005 e portato
  alla nuova build r007.
- L'applicazione nuova resta descritta come pianificata.

### GPT-I001 — cleanup del socket

- Il `finally` rimuove il path reale `socket_dir/'s'` e la directory propria.
- L'errore di cleanup va in `cleanup_error`, distinto da `error`. Non sostituisce
  la diagnosi primaria e non produce mai PASS.
- La receipt viene scritta sempre.
- Tre test mirati (EACCES, timeout, cleanup EACCES), inclusi e PASS nel fast
  ufficiale.
- R reale s060 con wrapper finale (SHA `2946aa16…` uguale al file):
  `temporary_socket_cleaned:true`, padre e figlio con sola `lo`, nessun
  indebolimento.

### Regressioni delle parti toccate

Nessuna regressione bloccante. Verificati:

- preflight `-I -B` prima della collection;
- guardia rete installata prima del preflight;
- official invariato per lo stage s060;
- I/E correnti (sei profili), fast 137+300 subtest, packaging 11, API 7,
  discovery 137/156/156 con zero skip e postcheck;
- FAIL discovery-tests-r001 conservato;
- V1 con quattro fasi e 51 call PASS;
- `image_prepare` S2 generalizzato (`/run/user/*`).

Le osservazioni minori sono CLA-I007 e CLA-I008.

## Rilievi nuovi

### CLA-I006 — sforamento temporale dichiarato sottostimato

| Campo | Contenuto |
| --- | --- |
| Severità e blocco | **Media — blocco: no** |
| Posizione | `evidence/implementation-r001/fix-review-r002/final-checks.json` (`core_cumulative_seconds`, `delivery_accounting_overrun_seconds`); `delivery.json` (`time_compliance`); `implementation/report-r019.md` (paragrafo «Contabilità di chiusura»); R020 e `reception-r001/final-checks.json` che li recepiscono |
| Raggiungibilità e input | Ledger temporale della consegna, letto da supervisore e arbitrato |
| Attribuzione e requisito | Run, consegna dell'implementatore. AGENTS: registrare la chiamata reale ed esiti fedeli. R020 chiede di verificare la misura e le attività oltre il cap |

**Problema.** Il valore dichiarato di 8285,287 s è calcolato da
`deliver_r002.py` quando scrive `matrix-final` e `checkpoint-before`
(mtime 1791367862,04–,20, formula 8285,16–8285,31: coerente).

`final-checks.json`, `delivery.json`, `manifest.json`, report-r019 e il
checkpoint hanno però mtime 1791367898,28–,30. Con la stessa formula
`costs()` di `run_core_proof_r002.py` corrispondono a un cumulativo di
**8321,42 s**.

La formula è verificata:

- l'offset monotonic/wall è identico in `entry.json` e in `approval-wait.json`;
- l'attesa esclusa di 27,524 s coincide con l'intervallo tra i due file;
- V1 risulta 8251,36 dal mtime contro 8251,006 dichiarato.

I campi `time_compliance`, `delivery_accounting_overrun_seconds`,
`last_workload_cumulative_seconds` e il paragrafo di chiusura non compaiono in
`deliver_r002.py`. Sono stati scritti da un passo successivo non conservato,
36 s dopo la misura.

**Impatto.** Lo sforamento reale è **≥41,42 s**, non 5,287 s. È un limite
inferiore: l'attività dell'autore dopo l'ultima scrittura non è misurabile.
L'esito dei workload non cambia: V1 si chiude a 8251,006 s, sotto 8280, e dopo
non c'è nessun workload prodotto. La quota resta non PASS in ogni caso. Il
registro però sottostima il costo e la consegna contiene contenuti prodotti
fuori dai comandi registrati.

**Evidenza.** `time_accounting_check.out.json`; `ls --time-style=+%s` della
cartella; grep su `deliver_r002.py`.

**Criterio di risoluzione.** Il supervisore registra in arbitrato e ledger uno
sforamento ≥41,42 s con questa derivazione. Indica che i campi e il paragrafo di
chiusura vengono da un passo non conservato successivo alla misura. Nessuna
nuova prova, sanatoria o riesecuzione. Per le run future, il gate temporale va
applicato anche agli script di consegna e ogni riscrittura va registrata.

### CLA-I007 — comando inesistente e variabile mancante nelle guide

| Campo | Contenuto |
| --- | --- |
| Severità e blocco | **Bassa — blocco: no** |
| Posizione | `docs/how-to/ambiente-uv.md:132`; `docs/reference/toolchain-legacy.md:106-110` |
| Raggiungibilità e input | Utente che segue la sezione «Test locali espliciti senza run» o la reference C-docker |
| Attribuzione e requisito | Run, fix CLA-I004/I001: «niente comandi inesistenti» |

**Problema.**

- La guida prescrive `verify_distribution.py --standalone`, ma `main()` non
  definisce quel flag. Argparse esce con «unrecognized arguments». Lo stesso
  file, alle righe 76-80, mostra il comando corretto senza flag.
- La reference elenca le variabili C-docker ma omette
  `RUN_IMAGE_CONTEXT_RECEIPT`, che il test richiede (`KeyError`). La guida
  Docker invece la riporta.

L'errore è fail-closed e non produce falsi PASS.

**Evidenza.** Lettura di `verify_distribution.py:376-382` e
`test_marker_contract.py:8`.

**Criterio di risoluzione.** Rimuovere il flag inesistente, oppure dire che la
modalità segue lo scope di S. Aggiungere la variabile alla reference. Verifica:
link e whitespace.

### CLA-I008 — `--test-mode` ignorato dal test C-docker

| Campo | Contenuto |
| --- | --- |
| Severità e blocco | **Bassa — blocco: no** |
| Posizione | `tests/docker/test_marker_contract.py:15` |
| Raggiungibilità e input | `pytest tests/docker/test_marker_contract.py --test-mode standalone` senza `RUN_TEST_MODE`, selezione dichiarata equivalente in `ambiente-uv.md:136` |
| Attribuzione e requisito | Run, fix CLA-I001/I003: modalità esplicita coerente |

**Problema.** Il conftest esegue il preflight standalone e lo supera. Il test
però passa al driver `--mode` letto solo dall'ambiente, con default
`official`. Il driver rifiuta allora lo S standalone e il test fallisce.

Non c'è falso PASS. La guida Docker indica `RUN_TEST_MODE` e quel percorso
funziona.

**Evidenza.** Lettura del conftest (righe 265-272) e del test.

**Criterio di risoluzione.** Il test usa la modalità risolta dal conftest, per
esempio con `request.config.getoption` più l'ambiente. In alternativa, la
documentazione limita C-docker a `RUN_TEST_MODE`. Serve un test mirato senza
Docker oppure la nota nella guida.

### Suggerimenti, non requisiti

- I Compose interpolano `${…:?}` per tutte le variabili di build. Un
  `docker compose up` della sola immagine richiede quindi anche i path della
  supply. Lo startup del servizio è fuori perimetro: valutarlo quando si
  autorizza il runtime.
- L'inventario prodotto da `make_input_inventory.py` include
  `.venv-python/.lock`. Se uv lo modifica, S diventa stantio: va solo tenuto
  presente.

## Motivazione dell'esito e limiti

**GO** per l'oggetto congelato `impl-r001-stage-final-s002`.

**I due bloccanti sono chiusi con prove reali sul codice permanente.**

- CLA-I001: suite eseguibile senza `temp/` in modalità standalone esplicita,
  gate generalizzato a qualsiasi run_id valido, negativi prima della collection.
- CLA-I002: Compose CPU costruito davvero con rete none da input prodotti da
  tool versionati. Il secondo Compose ha una build identica.

I/E, fast, packaging, API, discovery e V1 ricevuti sono legati agli input
correnti. Ho ricalcolato B56, README, contesto Docker e driver/test/probe.
CLA-I003–I005 e GPT-I001 sono chiusi.

**CLA-I006 non blocca il prodotto.** Nessun workload supera il cap e la quota
era già dichiarata non PASS. Il dato registrato è comunque inesatto: il
supervisore deve correggerlo prima della chiusura e non riportare 5,287 s come
costo finale. CLA-I007 e CLA-I008 sono difetti documentali e di coerenza minori,
fail-closed. Si possono chiudere con un fix mirato o metterli in backlog, a
scelta dell'arbitrato.

### Limiti

- Nessuna riesecuzione di campagne.
- `--acquire` e la context `--mode standalone` sono verificati solo dal codice.
- Sui path conftest invalidi o discordanti non esiste un test eseguito.
- La review vale per final-s002, verificato MATCH prima e dopo.

Nessuna modifica a prodotto, prove, snapshot o registri comuni. Nessuna
operazione Git, commit, merge, push o deploy. Consegna al **supervisore** per
l'arbitrato con il report ChatGPT.
