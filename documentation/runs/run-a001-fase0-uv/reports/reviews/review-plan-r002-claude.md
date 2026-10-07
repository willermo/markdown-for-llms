# Review — piano — run-a001-fase0-uv — r002 — Claude

- Autore/provider/modello e chat: **Claude Opus 5.5** (ID modello `claude-opus-5-5[1m]`),
  provider **Anthropic**, Claude Code nell'estensione VS Code. Nessun ID chat è esposto.
  La directory di scratchpad contiene `a17a68b7-a77f-47a3-8902-4ee513a42db7`, non
  confermato come identificativo della chat. Nessuna sostituzione di provider.
  Il revisore Claude r001 dichiarava lo stesso modello. Questa è però una **nuova chat**,
  senza memoria di quella sessione. Ruolo esclusivo: revisore Claude r002. Non ho svolto
  pianificazione, supervisione, arbitrato né la review r001, e non ho delegato ad altri agenti.
- Prompt di origine: `temp/run-a001-fase0-uv/prompts/review-plan-r002-claude.md`, incollato
  dall'utente. SHA-256 `84b1e1775e41ee443210658484ef13da610dbefd628a4de81b239da44c87f565`,
  uguale alla voce del manifest.
- Oggetto e revisione esatta: [plan-r002.md](../plans/plan-r002.md), revisione **r002**,
  SHA-256 `df23588de1247820a797e033ba7e93f60ed7d6b5f1f64e1948f012646039ca71`, ricalcolato e
  uguale alla voce del manifest.
- Snapshot, HEAD e impronta verificati prima/dopo: snapshot [plan-r002.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md);
  branch `feature/run-a001-uv`; HEAD, `dev` e merge-base `66ba82200e5def5a4db76f9bafccb0731b506091`;
  worktree_sha256 `46d68f51823734faf1dfc1a9f004d052ed76a395c04851428149905038f3ae0a`; 55 artefatti.
  `verify` **MATCH prima** (exit 0, 19:12:49+02:00) e **MATCH dopo**
  ([identity-pre.txt — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
  [identity-post.txt — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)). Il ricalcolo indipendente
  delle 134 voci del manifest (79 file e 55 artefatti) dà 0 differenze. Sono presenti soltanto le sei
  modifiche documentali di supervisione previste.
- Indipendenza: **non** ho letto il report, il checkpoint, le evidenze o il prompt del revisore
  ChatGPT r002, né l'arbitrato r002, che non esistono o non sono stati aperti. Ho letto le review r001
  e l'arbitrato r001, consentiti come antecedenti. Nessun coordinamento con altre chat.
- Esito: **NO_GO** sul piano r002, per il rilievo bloccante **CLA-P001 r002**. Gli altri rilievi r002
  non sono bloccanti. I tre blocchi dell'arbitrato r001 sono affrontati in modo adeguato rispetto ai
  rispettivi criteri; il nuovo blocco riguarda un difetto distinto.

## Ambito e prove

**File letti, nell'ordine del prompt.**

1. Prompt, `AGENTS.md`, skill `manage-implementation-run`, `run-lifecycle.md`, template review.
2. `STATE.md`, `brief.md`, `prompts/02-planning-r002.md`, `arbitrations/arbitration-plan-r001.md`.
3. `documentation/README.md`, indice ADR, ADR 0001/0006/0007, roadmap (quadro e fase 0.1),
   changelog (voci recenti).
4. **Piano r002 per intero** (940 righe), checkpoint `handovers/planning-r002.md`, `findings.md`,
   `checks.json` con il sidecar `checks.sha256` (coincidente), `static-inventory.json`, `local-help.json`.
5. `supervisor-plan-review-r002/receipt.json`; le due review r001; `pyenv-shim-probe.json`
   dell'arbitrato e `pyenv-probe.txt` della review Claude r001; help uv congelati r001.
6. Sorgenti: `setup.py`, `requirements.txt`, `.gitignore`, `.env.example`, `Dockerfile`, entrambi i
   Compose, `config.py`, `master_workflow.py` per intero e `marker_api_server.py`. Di
   `unified_converter`, `chunk_markdown`, `clean_markdown`, `validate_markdown`, `logging_config`,
   `batch_monitor` ed `exceptions` le parti pertinenti (import, CLI, dotenv, metadati, codici d'uscita).
   Inoltre `tests/conftest.py`, il test d'integrazione per intero, le intestazioni di unit e
   governance, i due smoke alla radice e il README per l'inventario.
   Ho applicato la skill `verify-conversion-fidelity` a F1–F6 e al confronto dei chunk.

**Controlli eseguiti** (dettagli in [static-checks.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)):

- identità Git/snapshot pre/post e ricalcolo degli hash;
- confronto di ogni flag uv dei comandi futuri con gli help 0.10.10 congelati: tutti presenti;
- riconteggio indipendente dei blocchi README;
- lettura di sysctl e profili AppArmor rilevanti per il runner `unshare`, senza creare namespace;
- verifica sul codice degli osservabili V5;
- versioni di docker, buildx e pandoc.

**Fonti primarie** consultate il 2026-10-02 (solo testo e JSON, nessun pacchetto):

- sorgenti e documentazione di uv al tag 0.10.10 (cache, `RunArgs`, settings, download Python);
- metadata PyPI di Marker 1.10.2, Surya 0.17.1, Setuptools 84.0.0, FastAPI e Starlette;
- l'API di integrità PyPI;
- i sorgenti di FastAPI 0.142.2 e Starlette 1.7.0.

URL, HTTP e SHA-256 sono in [sources.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), con fatti e
inferenze separati. Estratti: [uv-cache-excerpts.txt — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
e [uv-cli-settings-excerpts.txt — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).

**Affermazioni del piano confermate.**

- 3.12.13 è nel catalogo uv 0.10.10.
- Setuptools 84.0.0 esiste (Requires-Python `>=3.10`).
- La wheel Marker 1.10.2 ha SHA-256 `f631737d…09fb`. Non ha attestazione PEP 740, quindi il confronto
  metadata/file con il tag proposto in P5 è necessario.
- Marker vincola `transformers<5` e `weasyprint>=63.1,<64`.
- Surya 0.17.1 ha una wheel: nessuna build Surya da sdist.
- `-I` implica `-P` ed `-E`.
- Le chiavi `[tool.uv]` proposte esistono in 0.10.10.
- `--no-registry` riguarda Windows.
- Ambiente prima della CLI e figlio chunk senza override: l'osservabile V5 è discriminante.
- FastAPI 0.142.2 mantiene `on_event` come deprecato.

**Non eseguito, per vincolo del prompt.** Nessun lock, sync, run o build uv; nessuna installazione,
suite o collection; nessuna conversione; nessun Docker config/manifest/build/run; nessun namespace,
startup Marker o download di pacchetti, modelli o font. Le procedure V0–V12 sono valutate solo per
fattibilità e sufficienza. Nessun esito futuro è attribuito al piano.

### Riscontro per criterio A1–A7

| Criterio | Valutazione motivata | Rilievi |
| --- | --- | --- |
| **A1** Ambiente ricreabile dal lock, Python esplicito | Il pin 3.12.13 è motivato e `required-version` è fissato. Il percorso managed è unico, `$RUN_REPO/.venv-python`, tramite `UV_PYTHON_INSTALL_DIR` assoluta e `--no-bin`, senza `--default`/`--global`/pyenv. La matrice shim/bin/`uv run`/venv con e senza `PYENV_VERSION` è coerente con i due probe. V1 osserva `sys.executable`, realpath, prefix, base_prefix, `sys.path` e origine dello stdlib; prevede due sync puliti e `uv lock --check`. Help statico e selezione reale restano distinti, e i comandi sono fattibili in 0.10.10. Resta aperto un fallback implicito: senza le variabili della subshell, uv scarica 3.12.13 nella directory predefinita | CLA-P003 |
| **A2** Dev separato, moduli corretti | Il gruppo `dev` è fuori da Requires-Dist; gli extra `marker-server`/`marker-cpu`/`marker-cu126` sono in conflitto e usano indici `explicit`. Dieci `py-modules` espliciti, cinque script con `config:main`, ispezione di sdist/wheel/metadata, backend Setuptools fissato con build constraints a doppia rappresentazione. Marker è installato con `no-build-package`; gli altri sdist sono bloccati se il backend non è vincolato. Impianto adeguato | — (suggerimento S1) |
| **A3** Import/entry point fuori dai sorgenti | Wheel non editable in un ambiente esterno. Figli lanciati come `sys.executable -I -m`, con PYTHONPATH/PYTHONHOME rimossi, preflight in un sottoprocesso `-I`, origini applicative e transitive, sentinelle e PYTHONPATH avverso. Console e script del clone sono documentati con ambiente ripulito. Il disegno dell'isolamento è adeguato. Manca però il legame tra il codice installato negli ambienti e i sorgenti in esame, anche per la compatibilità con lo script del clone. Le sentinelle non coprono il modulo padre | **CLA-P001**, CLA-P005 |
| **A4** Suite, mock e prove reali distinti | La matrice copre ogni test nuovo o mutato. C-fast, C-api (dev + `marker-server`), C-package, C-docker, governance e discovery hanno comandi esatti; gli ignore precedono l'import, la raccolta è inerte, conteggi e partizione sono riconciliati, nessuno skip essenziale. La preparazione è separata e la cache tiktoken è un prerequisito. Baseline F1–F6 con `validated` e argomenti dell'orchestratore, almeno due chunk, complemento sliding, confronti incrociati. Però gli ambienti non editable possono eseguire nei figli una build del progetto **non aggiornata** senza che il piano lo rilevi, e il runner offline scelto è probabilmente negato dalla policy dell'host | **CLA-P001**, CLA-P002 |
| **A5** Istruzioni senza comandi inesistenti | Epilogo `run_full_pipeline.py`, `--chunking-strategy`, `source_ebooks`, percorso del report, costruttore, Poetry, pulizie globali e dump delle chiavi sono trattati. Gli script downstream restano pseudocodice illustrativo e la nuova app non è presentata come disponibile. AGENTS è assegnato all'implementatore, i registri condivisi al supervisore. L'inventario dichiarato completo omette però cinque blocchi bash indentati e un comando inline | CLA-P004 |
| **A6** Docker coerente e limiti dei motori | V7/V8 sono obbligatorie con sottocasi concreti: build CPU dal lock, digest nativi, dpkg, contesto con sentinelle, entrambi i Compose e l'override GPU solo in config, contratti e rendering WeasyPrint offline con constructor patchato prima dell'uso. La fonte è la wheel PyPI con hash; non presume equivalenza al tag. Bookworm, la patch storica e la manutenzione sono motivati e dichiarati come limite. V10/V11 sono rinviate, `local_marker` è dichiarato non verificato e c'è un criterio di ritorno a essenziale | — (note S3, S6, S7) |
| **A7** Due review, arbitrati, Git manuale | V12 prevede quattro report reali e due arbitrati su snapshot, chat nuove e scostamenti riportati al supervisore; commit, merge, push e promozione restano manuali. Questa review non autorizza l'implementazione | — |

### Coerenza P0–P7 / V0–V12, costi, uscite e recupero

L'ordine è coerente: baseline (P0/V0) prima delle modifiche, poi P1–P3, preparazione esplicita
degli ambienti e V1–V6. P5/P6 e V7/V8 stanno in un passo operativo distinto, con stima di rete e
disco presentata al supervisore. Seguono P7/V9 e il report. Gli stati PASS/FAIL/IMPEDITA/NON_ESEGUITA
sono definiti. «Codice 0 senza contenuto atteso è FAIL» e «health non prova la conversione» sono
corretti. Il recupero esclude reset, stash implicito, `git clean`, prune e cleanup di volumi
operativi. I costi sono dichiarati: download di Python e pacchetti nella preparazione, GB di ML in
V7, V10/V11 non autorizzate.

Due lacune di coerenza:

- **(a)** V0 è collocata nel runner `unshare`. Se il runner è negato, si blocca il primo passo
  essenziale (CLA-P002).
- **(b)** Ogni prova registra «branch/HEAD/hash», ma il piano non lega ambienti, wheel e immagine
  al contenuto dei sorgenti dello snapshot in esame (CLA-P001). Con HEAD invariato senza commit, la
  sola identità Git non distingue codice modificato.

### Disposizione degli undici rilievi accolti nell'arbitrato r001

| ID r001 | Disposizione nel piano r002 | Sezioni ed evidenza |
| --- | --- | --- |
| **CLA-P001 r001** (blocco A1) | **Affrontato adeguatamente** rispetto al criterio: pin motivato, percorso unico, matrice e osservabili V1, nessuna modifica globale. Il fallback implicito in assenza della subshell è un nuovo rilievo non bloccante, CLA-P003 r002 | P1 righe 105–165; V1 righe 684–692; probe confermati |
| **CLA-P002 r001** (blocco A4) | **Affrontato adeguatamente** nei punti del criterio r001: matrice per strato, api-env con l'extra, preparazione separata, ignore e conteggi, progetto installato nello stesso interprete. Restano due difetti distinti: freschezza del progetto installato (CLA-P001 r002, **bloccante**) e fattibilità del runner (CLA-P002 r002) | P4 righe 324–377; V6 righe 741–800 |
| **CLA-P003 r001** | **Adeguato.** Wheel PyPI scelta e motivata, alternativa Git con requisiti di build, hash confermato. La verifica metadata/RECORD/file rispetto al tag è necessaria, perché non esiste attestazione PEP 740. Torch, Surya e le transitive restano candidati; altri sdist vincolati | P1 righe 201–214; P5 righe 431–466 |
| **CLA-P004 r001** | **Adeguato.** Helper in `config` prima di `apply_env_overrides`, `override=False`, solo il workspace. L'osservabile 1200/80 → 1600/120 su `chunking_parameters` e `pipeline_state` è discriminante secondo il codice | P3 righe 263–281; V5 righe 720–726 |
| **CLA-P005 r001** (blocco A3) | **Adeguato** per l'isolamento: `-I -m`, ambiente ripulito, preflight `-I`, origini transitive, sentinelle e PYTHONPATH avverso. Miglioria minore in CLA-P005 r002 | P3 righe 283–316; V5 righe 728–737 |
| **CLA-P006 r001** | **Adeguato.** Compose build conservato e allineato, `config -f` per entrambi i file e per l'override GPU, progetto di prova distinto, nessuno startup | P6 righe 487–491, 537–550; V7 righe 806–819 |
| **CLA-P007 r001** | **Adeguato.** Bookworm motivato rispetto a trixie e Alpine, date LTS, patch storica non ricostruita, digest nativi obbligatori, snapshot apt, politica di aggiornamento con ripetizione V7/V8 | P6 righe 493–531, 551–556 |
| **CLA-P008 r001** | **Parziale.** Responsabilità su AGENTS, epilogo, opzioni e script inesistenti risolti; inventario incompleto (CLA-P004 r002) | P7 righe 567–616; findings |
| **CLA-P009 r001** | **Adeguato.** Stessi input da `validated` e stessi argomenti dell'orchestratore, versioni e tokenizer registrati, confronto incrociato circoscritto, nessuna normalizzazione | P4 righe 401–427 |
| **CLA-P010 r001** | **Adeguato.** V10/V11 non autorizzate, limite dichiarato, punto in cui il supervisore propone il perimetro, ritorno a essenziale in caso di incompatibilità, nessun PASS per rinvio | P5 righe 474–483; V10/V11 righe 646–647, 858–866 |
| **GPT-P001 r001** | **Adeguato.** F6 con almeno due chunk in baseline e wheel, fixture congelata prima dell'esito, sequenza, contenuto, metadati e overlap; complemento con `chunk_by_sliding_window` esistente senza nuove opzioni | P4 righe 392–418 |

## Rilievi

Nuovi rilievi del giro r002. Gli ID ricominciano: CLA-P001 r002 non è CLA-P001 r001.

| ID | Severità e blocco sì/no | Posizione | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- |
| CLA-P001 | **Media-alta — bloccante: sì** (A4; A3 per la compatibilità con lo script del clone) | plan-r002.md:244–249 (P2, «occorre risincronizzare»), 297–298 (preflight spec/origine), 326–333 (matrice), 664–667 e 744–755 (sync `--no-editable`), 868–871 (registrazione) | Il piano usa ambienti host **non editable** (dev-env, api-env, base-*, `.venv`). I figli `-I -m` del test d'integrazione e della compatibilità con lo script del clone eseguono quindi la **copia installata**, mentre il padre usa i sorgenti del clone. In uv 0.10.10 una directory locale viene ricostruita **solo** se cambiano `pyproject.toml`, `setup.py` o `setup.cfg`. La build in cache viene riusata anche in un ambiente nuovo, quindi il rimedio «risincronizzare» non aggiorna i moduli dopo una correzione. Il preflight controlla nome e origine, non il contenuto. Il piano non lega ambienti, wheel (`RUN_WHEEL` identificata solo dal proprio hash) e immagine al contenuto dei sorgenti dello snapshot. **Impatto:** C-fast e V4 possono dare PASS su codice diverso da quello revisionato, dopo una modifica successiva al primo sync. Contrasta con il principio di AGENTS sull'identificazione degli artefatti senza fermarsi al nome. Il difetto verrebbe anche reso permanente dalla nuova istruzione C-fast in AGENTS | uv 0.10.10 `docs/concepts/cache.md` (sezioni «Dependency caching» e «Dynamic metadata»); `crates/uv-distribution/src/source/mod.rs` `source_tree` (shard `WheelCache::Path`) e `source_tree_revision` (riuso se `CacheInfo` coincide); `cache_info.rs` righe 64–93 (chiavi predefinite). Estratti in [uv-cache-excerpts.txt — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), fonti e inferenze in [sources.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md) | Il piano r003: (1) prescrive un meccanismo di freschezza per ogni sync host di verifica e d'uso. Sono ammessi `tool.uv.reinstall-package = ["markdown-for-llms"]`, oppure `tool.uv.cache-keys` con `pyproject.toml` e i dieci moduli, oppure `--reinstall-package markdown-for-llms` nei comandi, con motivazione. (2) Prima di C-fast, C-api, C-package, V4 e V5 confronta gli SHA-256 dei dieci moduli installati con i file del clone nello snapshot in prova; per la wheel usa il suo RECORD, per l'immagine `/opt/venv`. Una differenza produce FAIL o IMPEDITA, non PASS. (3) Ogni esito V registra l'impronta `run_context` del worktree; una modifica successiva dei sorgenti invalida le prove dipendenti e ne richiede la ripetizione. (4) Corregge il testo P2 e il futuro testo AGENTS/guida su «risincronizzare» |
| CLA-P002 | Media — bloccante: no | plan-r002.md:372–375, 757–771; V0/V5/V6 righe 636, 641–642 | Il runner scelto per V0/V4/V5/V6 è `unshare --user --map-root-user --net`. Sull'host di riferimento (Ubuntu 24.04.5) `kernel.apparmor_restrict_unprivileged_userns=1`. Il profilo `unprivileged_userns` nega tutte le capability e non esiste un profilo per `unshare` o `bwrap`. Questi indizi statici rendono **probabile** che `--net` venga negato. Il piano prevede correttamente IMPEDITA senza allentare il vincolo, ma non indica un'alternativa: il primo passo essenziale (V0) e A3/A4 rischiano di fermarsi all'avvio, in attesa di una ricerca del supervisore durante l'implementazione. È un'**inferenza**: non ho creato namespace, come prescritto | [static-checks.md §3 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md): sysctl, `/etc/apparmor.d/unprivileged_userns` (`audit deny capability`), assenza di profili dedicati; `firejail` setuid presente; utente nel gruppo docker | Prima di P0 il supervisore o il piano: (1) autorizza e registra una sonda innocua di fattibilità del runner, per esempio namespace creato e sole interfacce `lo`; (2) predefinisce un'alternativa equivalente approvata, con criterio di equivalenza (nessuna interfaccia esterna e figli che la ereditano). Esempi: `firejail --noprofile --net=none` già installato e setuid; un network namespace creato con privilegi esplicitamente autorizzati dall'utente; un altro host Linux. (3) Stabilisce che V0 non inizia finché il runner non è confermato. `uv --offline`, proxy o sola guardia in-process non sono equivalenti |
| CLA-P003 | Media-bassa — bloccante: no | plan-r002.md:110, 142–158, 657–659, 793–797; matrice righe 127–134 | Il percorso managed unico esiste solo se la subshell esporta `UV_PYTHON_INSTALL_DIR` e `UV_PYTHON_DOWNLOADS`. In 0.10.10 `python-downloads` vale `automatic` per default e la directory d'installazione non ha una chiave di progetto. Un qualsiasi comando uv nel clone senza quelle variabili scarica 3.12.13 nella **directory predefinita** e la usa. Esempi: `uv run pytest`, un IDE, un agente che segue AGENTS. Questa è un'alternativa implicita, con accesso di rete, a quanto chiesto da CLA-P001 r001. Sull'host quella directory è già in uso (`cpython-3.14.3`). V1 non prova il caso «shell non preparata» | [sources.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), fatto 2; [uv-cli-settings-excerpts.txt — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md); [static-checks.md §5 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md) | Il piano aggiunge una guardia di progetto. Per esempio `[tool.uv] python-downloads = "manual"`: in quel caso solo `uv python install` acquisisce interpreti, e il parsing va verificato in V1. In alternativa motiva un percorso unico diverso, senza variabili obbligatorie. V1 include il caso «comando uv dal clone senza variabili»: errore esplicito, nessuna nuova voce in `~/.local/share/uv/python`, nessun download. README, guida e AGENTS lo dichiarano |
| CLA-P004 | Bassa — bloccante: no | plan-r002.md:56–60, 567–578; findings, sezione inventario; `static-inventory.json` | L'inventario dichiara «ogni blocco shell/Python/YAML operativo del README». Il README ha 74 blocchi recintati: ne mancano cinque **bash indentati**, operativi (L1367–1395: `export CONVERSION_TIMEOUT/CONNECT_TIMEOUT/MAX_WORKERS`, `docker stats`, `docker compose logs --follow`), e il comando inline `git checkout -b feature-name` (L1538). Undici blocchi json, markdown e albero sono esclusi senza una regola dichiarata; tra questi la configurazione completa L199–271, che V4 potrebbe confrontare con `--create-default`. La rigenerazione post-implementazione prevista da P7 erediterebbe la stessa lacuna di estrazione | [static-checks.md §2 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), riconteggio con fence indentati | L'estrattore riconosce i fence indentati. I cinque blocchi e il comando inline sono classificati. È dichiarata una regola per json, markdown e alberi: verificabili contro output reali, per esempio L199–271 con V4, oppure illustrativi. V9 riconcilia i 74 blocchi e i comandi inline con l'estrattore corretto |
| CLA-P005 | Bassa — bloccante: no | plan-r002.md:728–737 | Le sentinelle V5 coprono config, exceptions, logging_config, i tre moduli di fase e quattro transitive. Mancano `master_workflow.py` e `unified_converter.py`, cioè il modulo lanciato con `-I -m master_workflow` sotto PYTHONPATH avverso e il suo import diretto. Mancano anche `batch_monitor.py` e `marker_api_server.py`. L'isolamento `-I` le renderebbe innocue per costruzione, ma la prova non lo dimostra per il processo padre | Piano V5; `master_workflow.py:25–28` (import del padre) | Le sentinelle V5 coprono i dieci nomi di modulo, o almeno `master_workflow` e `unified_converter`, oltre alle quattro transitive. Nessun marcatore risulta in tutte le modalità di avvio |

**Suggerimenti opzionali, non rilievi.**

- **S1.** Valutare `tool.uv.environments` per limitare il lock universale alle piattaforme target, con
  motivazione. Riduce le risoluzioni e le build di metadata da sdist per piattaforme non usate
  durante il lock.
- **S2.** Includere `temp`, `tmp` e `documentation/runs` in `norecursedirs`, se la copia baseline di
  P0 o test archiviati vi risiedono. Evita collisioni in C-discovery radice.
- **S3.** Se il pyproject usa `readme` o `license-files`, la lista ammessa di `.dockerignore` deve
  includere quei file per la build del progetto nell'immagine.
- **S4.** La guardia di rete di `conftest` deve consentire `socketpair` AF_UNIX: il loop asyncio di
  TestClient/anyio lo usa, altrimenti C-api fallisce per cause estranee al wrapper.
- **S5.** `uv run --locked --no-sync` è accettato dal parser 0.10.10, ma `--no-sync` implica
  `--frozen`. L'asserzione sul lock resti affidata a `uv lock --check`, già in V1.
- **S6.** FastAPI e Starlette non hanno limiti. Oggi verrebbero risolte FastAPI 0.142.2 e
  Starlette 1.7.0. `on_event` funziona tramite la copia di `_DefaultLifespan` in FastAPI, con
  DeprecationWarning: non trasformare i warning in errori e registrare le versioni.
- **S7.** Il percorso host `.venv-marker` (server fuori Docker) non ha una V dedicata. In V9 va
  etichettato come non verificato, come `local_marker`.

## Motivazione dell'esito e limiti

Il piano r002 è completo, ordinato e verificabile nella grande maggioranza delle parti. Conserva A1–A7
e tratta tutti gli undici rilievi con interventi e prove future concrete. Ha chiuso in modo adeguato i
criteri dei tre blocchi r001:

- percorso managed e matrice pyenv;
- strati e prerequisiti dei test;
- isolamento `-I` dei figli dal workspace dati.

I comandi uv proposti esistono in 0.10.10. Gli osservabili V5 sono discriminanti secondo il codice
legacy. V7/V8 restano obbligatorie; V10/V11 sono rinviate senza alcun PASS implicito.

L'esito è **NO_GO** per **CLA-P001 r002**: un difetto di correttezza nella catena di prova di A4, e di
A3 per lo script del clone. Il piano fonda la soluzione di CLA-P002 r001 sul «progetto installato
nello stesso interprete dei figli». In uv 0.10.10 un'installazione non editable non viene ricostruita
dopo modifiche ai soli moduli, e il rimedio indicato dal piano non basta. Mancando un legame tra
artefatti installati e sorgenti dello snapshot, le suite possono certificare codice diverso da quello
in review, senza segnale. La correzione è circoscritta (criterio sopra) e non richiede di rivedere
l'impianto. Un piano r003 può essere una revisione mirata di P2, P4, V1–V6 e P7 per CLA-P001, con
CLA-P002–P005 r002 integrabili nello stesso giro o rinviabili dall'arbitro con motivazione.

**Limiti della review.**

- Il comportamento della cache uv è documentato e letto nel sorgente 0.10.10, ma non riprodotto. Non
  ho verificato nel sorgente il percorso di `uv build`; il criterio copre comunque la wheel.
- La negazione del runner `unshare` è un'inferenza da sysctl e profili, non una prova.
- Nessun lock reale: compatibilità del grafo CPU/cu126, sdist e native non verificate.
- Le pagine e i metadata web possono cambiare dopo il 2026-10-02.
- Questa review vale solo per lo snapshot `plan-r002` con l'identità sopra indicata. Una modifica a
  piano o input richiede una nuova revisione.
- Questo NO_GO non autorizza né blocca altro che il passaggio di stato del piano r002, che spetta
  all'arbitrato.
