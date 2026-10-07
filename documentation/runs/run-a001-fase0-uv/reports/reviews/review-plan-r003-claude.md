# Review — piano — run-a001-fase0-uv — r003 — Claude

- Autore/provider/modello e chat: **Claude Opus 5.5** (ID modello `claude-opus-5-5[1m]`),
  provider **Anthropic**, interfaccia **Claude Code nell'estensione VS Code**. Nessun ID
  chat è esposto. La directory di scratchpad contiene `162bb358-c201-4ccb-81b7-23dd8f6be5e6`,
  che non assumo come ID della chat. Nessuna sostituzione di provider o interfaccia.
  I revisori Claude r001 e r002 dichiaravano lo stesso modello. Questa è però una **nuova
  chat** senza memoria di quelle sessioni. Ruolo esclusivo: revisore Claude r003. Non ho
  pianificato, supervisionato, arbitrato né svolto altre review, e non ho delegato ad agenti.
- Data: 2026-10-03, Europe/Rome.
- Prompt di origine: [review-plan-r003-claude.md](../prompts/review-plan-r003-claude.md),
  incollato dall'utente. SHA-256 del file
  `e84bb6af6b6c63b10f263784def75340d96ac2a213d65962a5b04d621c176b5a`, uguale alla voce del
  manifest; inizio e fine coincidono con il testo incollato.
- Oggetto e revisione esatta: [plan-r003.md](../plans/plan-r003.md), revisione **r003**,
  SHA-256 `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`, ricalcolato.
  Il piano dichiara il ruolo di pianificatore; non è un'implementazione.
- Snapshot, HEAD e impronta verificati prima/dopo: [plan-r003.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
  SHA-256 `99f33ba4c9ab452fd0015a654ce59251b68d1467fa97763b2c5c3de921fe4046`; branch
  `feature/run-a001-uv`; HEAD, `dev` e merge-base `66ba82200e5def5a4db76f9bafccb0731b506091`;
  worktree_sha256 `abbab72ee487228cd2a4733592d336770c0914dfdcd9242a510c6172b1c971a4`;
  79 file e 90 artefatti. `verify` **MATCH prima** della lettura tecnica e **MATCH dopo**
  la scrittura degli output ([identity-pre — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
  [identity-post — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)). Sono presenti
  soltanto le sei modifiche documentali di supervisione previste.
- Indipendenza: **non** ho letto report, checkpoint, evidenze o prompt del revisore ChatGPT
  r003, né un arbitrato r003. Un `ls` di `evidence/` ha mostrato l'esistenza della directory
  `review-plan-r003-chatgpt/`: non l'ho aperta. Ho letto i report e gli arbitrati r001/r002,
  consentiti come antecedenti comuni. Nessun coordinamento con altre chat.
- Esito: **GO sul piano r003** identificato sopra, con **cinque rilievi non bloccanti**
  (CLA-P001…P005 r003) e quattro suggerimenti opzionali. Il blocco CLA-P001 r002 e i tre
  blocchi r001 sono affrontati in modo adeguato **nel disegno**; nessuno è chiuso
  operativamente. Questo GO non avvia l'implementazione: decide l'arbitrato.

## Ambito e prove

**File letti, nell'ordine del prompt.**

1. Prompt; `AGENTS.md`; skill `manage-implementation-run`; `run-lifecycle.md`; template review.
2. `STATE.md`, `brief.md` (A1–A7), `prompts/02-planning-r003.md`, arbitrati r002 e r001.
3. `documentation/README.md`, indice ADR, ADR 0001/0006/0007, roadmap (quadro e fase 0.1),
   voci recenti del changelog.
4. **Piano r003 per intero** (1400 righe); checkpoint `handovers/planning-r003.md`;
   `evidence/planning-r003/findings.md`; struttura e contenuto pertinente di `checks.json`.
5. `supervisor-plan-review-r003/receipt.json` (identità, ricevuta, transizione); review
   r002 Claude e ChatGPT; estratti cache e settings uv della review r002 Claude.
6. Sorgenti: `setup.py`, `requirements.txt`, `.gitignore`, `Dockerfile`, `docker-compose.yml`,
   **`docker-compose-build.yml`**; le parti pertinenti di `master_workflow.py`, `config.py`,
   `chunk_markdown.py` e `marker_api_server.py`; `tests/conftest.py`, il test di integrazione,
   l'intestazione dei test di governance; README per l'inventario. Ho applicato la skill
   `verify-conversion-fidelity` per giudicare F1–F6 e il confronto dei chunk, senza conversioni.

**Controlli eseguiti** (dettagli in [static-checks.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)):

- Identità Git e snapshot, prima e dopo.
- Ricalcolo dei 90 artefatti del manifest: 0 differenze. `checks.json` ha SHA-256
  `8f965823…8eba`, uguale al manifest e alla ricevuta.
- Ricalcolo dei 183 input di `checks.json`. Le 12 differenze sono tutte file di
  supervisione aggiornati dopo la consegna: i sei documenti della transizione e sei
  registri comuni in `temp/`. Nessun sorgente o antecedente è cambiato.
- Confronto dei flag di tutti i comandi futuri con gli help uv 0.10.10 congelati:
  tutti presenti.
- Riconteggio indipendente del README: 74 fence, 63 operative, cinque indentate,
  9 inline. Le coordinate coincidono con l'inventario r003.
- Verifica nel codice degli osservabili V5, del metodo sliding, del parser del chunk,
  dell'epilogo e del wrapper.
- Metadati di `firejail`, `unshare` e del socket Docker, senza eseguirli.

**Fonti primarie** ([sources.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)):

- `settings.rs` al tag 0.10.10, letto: `reinstall-package` «implies `refresh-package`».
  `python-downloads` è un'opzione globale ammessa in `pyproject.toml`. Non esiste una
  chiave di progetto per la directory managed.
- `docs/reference/settings.md` al tag: HTTP 404, perché il file è generato.
- Pagina locale `network_namespaces(7)`.

**Non eseguito, per vincolo del prompt:** nessun lock, sync, run, build, export o pip
uv; nessuna installazione o download; nessuna suite o collection; nessuna conversione;
niente Docker, namespace, Firejail, startup Marker o probe runtime. Le procedure
§R/V0–V12 sono valutate per fattibilità e sufficienza: nessun esito futuro è attribuito al piano.

### Riscontro per criterio A1–A7

| Criterio | Valutazione motivata del disegno | Rilievi |
| --- | --- | --- |
| **A1** Ambiente ricreabile dal lock, Python esplicito | Pin 3.12.13 motivato e `required-version`. Managed solo in `$RUN_REPO/.venv-python` con `--no-bin`, senza `--default`/`--global`/pyenv. Matrice shim/bin/`uv run`/venv con e senza PYENV_VERSION (P1 L204–217) e osservabili `sys.executable`/realpath/prefix/base_prefix/stdlib. Due sync puliti, `uv lock --check` separato. Guardia `python-downloads = "manual"`: è ammessa in pyproject secondo settings.rs 0.10.10, va provata in V1 e non è presentata come guardia del path (L247–253). Il caso senza variabili si svolge dentro R, senza `--offline`/`--no-python-downloads` (L268–281), con rifiuto dell'origine prima degli import. Stop su origine diversa | — (suggerimento S4) |
| **A2** Dev separato, moduli corretti | Runtime, `dev`, `marker-server`, `marker-cpu`/`marker-cu126` in conflitto, con indici `explicit`. Dieci `py-modules` e cinque script, `config:main` da estrarre. Setuptools 84.0.0 in build-system, in `build-constraints.txt` e in `build-constraint-dependencies`, con controllo di corrispondenza. Marker `no-build-package`; stop sugli sdist con backend ignoto. Ispezione degli archivi, non di `find_packages` | — |
| **A3** Import/entry point fuori dai sorgenti | Wheel esterna non editable. Figli `sys.executable -I -m`, PYTHONPATH/PYTHONHOME rimossi, preflight in subprocess `-I`. Catena S/B/I prima di V3/V4/V5. Console, `-I -m` e script del clone sono separati. Quattordici sentinelle con PYTHONPATH avverso; server verificato con spec/hash senza importare FastAPI | CLA-P002 |
| **A4** Suite, mock e prove reali distinti | §R PASS prima di V0. Matrice di ogni test nuovo o mutato; C-fast/API/package/docker/governance/discovery con comandi esatti; ignore prima dell'import; zero skip essenziale. Ambienti non editable con `reinstall-package`, wheel canonica come ultimo passo, preflight dei dieci moduli in ogni interprete pertinente. Matrice d'invalidazione e probe discriminante con negativo e ripristino. F1–F6, ≥2 chunk, overlap e confronti incrociati | CLA-P001, CLA-P003, CLA-P004 |
| **A5** Istruzioni senza comandi inesistenti | Inventario verificato in modo indipendente: 74 fence (60 bash, 7 json, 3 senza linguaggio, 2 yaml, 1 markdown, 1 python), cinque indentate, 9 inline incluso L1538. Regola di estrazione a stati, quattro classi, 11 non operativi illustrativi, riconciliazione pre/post senza totale fisso. Epilogo, `--chunking-strategy`, script assenti come pseudocodice, AGENTS all'implementatore, registri al supervisore. Nessuna nuova app presentata come disponibile | — |
| **A6** Docker coerente e limiti dei motori | Wheel PyPI Marker 1.10.2 con hash, metadata/RECORD e confronto file/contratto col commit, senza presumere equivalenza. Torch/Surya/native candidati. Bookworm, patch storica, LTS, digest nativi, apt congelato, `.dockerignore` a lista ammessa con README/licenze. Entrambi i Compose e override GPU solo in config. Catena S/B/I nel builder e in `/opt/venv`. **V7/V8 obbligatorie**, `--pull=never --network none`, constructor patchato prima, rendering WeasyPrint sintetico; health e config non bastano | CLA-P005 |
| **A7** Due review, arbitrati, Git manuale | V12 con quattro report reali e due arbitrati su snapshot correnti. Scostamenti al supervisore; commit/merge/push manuali dell'utente. CLA-S1 rinviato senza restrizioni implicite delle piattaforme | CLA-P001 (handoff) |

### Coerenza §R/P0–P7/V0–V12, comandi, costi, uscite e recupero

**Ordine.** L'ordine è coerente e implementabile:

1. §R preflight dopo il GO.
2. P0/V0 baseline con copia identificata dei dieci originali.
3. P1, P2, P3.
4. V2: S, sdist e wheel dalla sdist, ispezione.
5. Sync con rebuild, wheel canonica e preflight in ogni ambiente.
6. Probe discriminante con ripristino, prima dei test finali (L519–526).
7. V3–V6.
8. Passo Docker distinto (V7/V8).
9. P7/V9.

V2 precede esplicitamente l'installazione canonica (L983–999).

**Comandi.** Coerenti con P1 e con gli help: `uv lock --check` separato (CLA-S5);
`uv pip sync` con `--build-constraints`; le installazioni della wheel con
`--no-deps --no-build --reinstall-package`. La riga `uv export` (L1040) omette i flag
Python espliciti delle altre righe, ma la subshell preimposta `UV_PYTHON_DOWNLOADS=never`
e la directory locale (L979–981): nessun effetto pratico.

**Costi e uscite.**

- I costi sono dichiarati: download Python e pacchetti nella preparazione, cache
  tokenizer, GB di ML e native in V7, presentati prima al supervisore.
- V10/V11 non sono autorizzate.
- Gli stati PASS/FAIL/IMPEDITA/NON_ESEGUITA sono definiti.
- «Exit 0 senza contenuto è FAIL» copre anche i ritorni silenziosi del chunk legacy
  (`main()` → `None` senza input).

**Recupero.** Niente reset, stash, clean, prune o `down -v`; si fermano solo oggetti
di prova identificati. Un mismatch S/B/I ferma la prova e non diventa mai PASS.

### Controlli comuni richiesti, in sintesi

- **CLA-P001 r002, catena S/B/I/E.**
  - `reinstall-package` persistente più flag: semantica confermata al tag 0.10.10
    (doc cache e settings.rs).
  - Ricostruzione a metadata invariati: verificata dal probe.
  - Wheel canonica come ultimo passo di `.venv`, base, dev, dev-env, api-env, server
    host e wheel-env (L1020–1028, L1043): presente.
  - Dieci moduli byte per byte in sdist, wheel, RECORD decodificato, installato e
    `/opt/venv`: presente.
  - Base senza FastAPI: presente. Ispezione sicura degli archivi: presente.
  - Probe in copia isolata con cache popolata: negativo prima del sync, verifica
    prima della wheel canonica, ripristino, nessuna pulizia globale.
  - Lacune non bloccanti: handoff degli snapshot di stage (CLA-P001), produttore di
    S e rifiuto obbligatorio delle receipt vecchie (CLA-P002), attribuzione della
    sola configurazione persistente (CLA-P003).
- **A1/A2.** Adeguato (vedi tabella). GPT-P001 r002 è applicato; `--no-sync` non
  certifica il lock.
- **Runner (CLA-P002 r002).** Unshare primario e Firejail alternativa, con argv
  equivalenti. Il diagnostico verifica namespace diverso, sole interfacce lo,
  assenza di rotte, connect fallito verso 192.0.2.1/2001:db8::1, figli e
  socketpair AF_UNIX positivo. AppArmor resta un indizio. Se entrambi falliscono,
  le prove sono IMPEDITE. Daemon Docker separato. Manca la verifica dei socket UNIX
  su path (CLA-P004).
- **Isolamento e .env.** Il codice conferma gli osservabili V5: `CHUNK_SIZE` e
  `OVERLAP_SIZE`, CLI applicata prima degli override, `pipeline_state.settings`,
  `chunking_parameters`. Con .env 1200/80 e CLI 1000/100 l'esito atteso 1200/80
  discrimina un caricamento tardivo; la shell 1600/120 discrimina `override=False`.
- **Fedeltà.** Stessi input `validated` e argomenti dell'orchestratore; F6 congelata
  prima dell'esito con ≥2 chunk; confronto di sequenza, frontmatter, index, conteggi,
  `heading_context` e overlap. `chunk_by_sliding_window` esiste (`chunk_markdown.py:455`).
  Niente normalizzazione di Markdown/LaTeX; perdite legacy inventariate. Confronti
  incrociati circoscritti alla fixture divergente.

### Disposizione delle tre matrici

La valutazione riguarda il **disegno**. Nessuna voce è chiusa operativamente: le prove
indicate sono future e da registrare con S/B/I/E.

**Matrice 1 — undici rilievi r001**

| ID | Disposizione | Sezioni / evidenza | Prove future richieste |
| --- | --- | --- | --- |
| **CLA-P001 r001** (blocco A1) | **Adeguato** | P1 L197–243, guardia L245–299; V1 L1059–1081 | Receipt V1 con origini prima/dopo il pin, con/senza PYENV_VERSION, `python3`/`uv run`/venv |
| **CLA-P002 r001** (blocco A4) | **Adeguato** | P4 L599–644; V6 L1143–1199 | C-* con liste nodeid, conteggi e partizione; preflight S/B/I per interprete; zero skip essenziale |
| **CLA-P003 r001** | **Adeguato** (precisato da GPT-P001 r002) | P1 L335–348; P5 L720–757; V3 L1042, L1049–1057 | Inventario backend/sdist effettivi; metadata, RECORD e file di contratto Marker rispetto al commit; V7/V8 |
| **CLA-P004 r001** | **Adeguato** | P3 L538–556; V5 L1109–1115; codice `config.py:266–270`, `master_workflow.py:51, 558–560` | V5: 1200/80 da .env, 1600/120 da shell, in processi separati |
| **CLA-P005 r001** (blocco A3) | **Adeguato** | P3 L558–591; V5 L1117–1136 | 14 sentinelle mai eseguite, origini padre e figli, PYTHONPATH avverso, contenuti uguali |
| **CLA-P006 r001** | **Adeguato** | P6 L778–782, L836–849; V7 L1224–1226 | Config `-f` di entrambi i Compose e dell'override GPU; inventario risorse/mount/cache |
| **CLA-P007 r001** | **Adeguato** | P6 L784–800, L826–830, L850–855 | Digest nativi index/amd64, snapshot apt, dpkg; V7/V8 ripetute agli aggiornamenti |
| **CLA-P008 r001** | **Adeguato** (completato da CLA-P004 r002) | P7 L864–938; findings §inventario | V9: riconciliazione pre/post, help reali, AGENTS aggiornato dall'implementatore |
| **CLA-P009 r001** | **Adeguato** | P0 L155–170; P4 L692–718 | V0/V5: stessi input `validated`, versioni, tokenizer e cache registrati; incroci solo se diverge |
| **CLA-P010 r001** | **Adeguato** | P5 L759–774; V10/V11 L1272–1288 | Nessuna in questo mandato; proposta del supervisore dopo V7/V8; V10 essenziale se emerge incompatibilità |
| **GPT-P001 r001** (multi-chunk) | **Adeguato** | P4 L683–709 | F6 ≥2 chunk in baseline e wheel; complemento sliding con intersezione >0 |

**Matrice 2 — sei rilievi r002**

| ID | Disposizione | Sezioni / evidenza | Prove future richieste |
| --- | --- | --- | --- |
| **CLA-P001 r002** (blocco A3/A4) | **Adeguato nel disegno**; precisazioni non bloccanti CLA-P001/P002/P003 r003 | P2 L389–526; V1–V6 L983–1057, L1138–1141, L1179–1186; P6.4; V7/V8 L1243–1248; P7 L900–903, L929–934 | Probe negativo/positivo e ripristino; receipt V2 S/B; preflight I in ogni ambiente; receipt E di ogni C, V3–V5 e immagine; invalidazioni registrate |
| **CLA-P002 r002** | **Adeguato**; precisazione CLA-P004 r003 | §R L84–147; V6 L1165–1177 | Receipt R con namespace host/padre/figlio, interfacce/rotte, connect fallito, AF_UNIX positivo, prerequisiti; IMPEDITA se negato |
| **CLA-P003 r002** | **Adeguato** | P1 L245–299; V1 L1071–1081 | Parsing di `manual` con uv 0.10.10; caso senza variabili con e senza venv; inventari managed prima/dopo |
| **CLA-P004 r002** | **Adeguato**, verificato in modo indipendente | P7 L866–896; findings; checks | V9: estrazione post con lo stesso scanner, delta motivati |
| **CLA-P005 r002** | **Adeguato** | V5 L1117–1136 | Assenza dei 14 marcatori nelle tre modalità; server con spec/hash nel base, runtime solo api-env mock |
| **GPT-P001 r002** (pip/constraints) | **Adeguato** | V3 L1042; L1049–1057; P1 L335–348 | Manifest backend effettivi, input e output di ogni build; stop su sdist o backend ignoti (suggerimento S3 per la baseline) |

**Matrice 3 — disposizioni CLA-S1…S7 r002**

| Voce | Disposizione | Sezioni / prova o limite |
| --- | --- | --- |
| **CLA-S1** | **Adeguato** (rinvio rispettato) | Matrice 3 L1337; lock universale senza `environments` impliciti |
| **CLA-S2** | **Adeguato** | P4 L628–644: `norecursedirs` nominati, esclusioni standard conservate, conteggi |
| **CLA-S3** | **Adeguato** | P2 L428–433; P6 L812–814: README e licenze consumati nel manifest e nel contesto |
| **CLA-S4** | **Adeguato** | §R L125–126; P4 L658–661: socketpair positivo, blocco AF_INET/AF_INET6 |
| **CLA-S5** | **Adeguato** | V1 L991; Matrice 3 L1341 |
| **CLA-S6** | **Adeguato** | P4 L648–651: versioni locked e warning osservati, niente refactoring di `on_event` |
| **CLA-S7** | **Adeguato** | P2 L411; P5 L759–763; V9 L1277–1278: server host non verificato senza prova dedicata |

## Rilievi

Nuovi rilievi del giro r003. Gli ID ricominciano: CLA-P001 r003 non è CLA-P001 r001 né r002.

| ID | Severità e blocco sì/no | Posizione | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- |
| **CLA-P001** | **Media — bloccante: no** (A4/A7, operatività della catena E) | plan-r003.md:424–427, 474–481, 983, 1290–1294 | Gli snapshot di stage sono affidati al supervisore, ma il passaggio non è concreto. Mancano: schema delle label, comando e lista `--artifact`, chi lo esegue, cosa fa l'implementatore in attesa. L'**ordine** tra snapshot e S è ambiguo: S cita snapshot e SHA del manifest (quindi lo segue), mentre L475 dice che l'implementatore consegna «manifest e receipt per snapshot». Servono almeno cinque stage più uno per ogni invalidazione. Il supervisore è una chat distinta, non un servizio continuo. **Impatto:** blocchi o improvvisazioni. Per esempio uno snapshot creato dall'implementatore, uno vecchio riusato, oppure receipt senza l'impronta richiesta da E. Non produce PASS falsi: S/B/I è basato sul contenuto e il riciclo è vietato | `run-lifecycle.md` (solo il supervisore modifica lo stato condiviso; chat non continua); `scripts/run_context.py` offre solo `snapshot`, che scrive in `snapshots/`, e `verify`, che richiede uno snapshot esistente: nessun calcolo dell'impronta in sola lettura | L'arbitrato o il prompt d'implementazione fissano: (a) chi esegue `snapshot` per gli stage, uno schema di label progressive (per esempio `impl-r001-stage-<nome>-sNN`) e gli artefatti inclusi; (b) l'ordine snapshot → S (con SHA del manifest) → B/I/E; (c) arresto dell'implementatore con checkpoint e richiesta esatta, **oppure** uno scostamento registrato che lo autorizzi a creare snapshot append-only con prefisso riservato, poi verificati dal supervisore; (d) quando serve un nuovo stage, legato alla matrice d'invalidazione |
| **CLA-P002** | **Bassa-media — bloccante: no** (A3/A4) | plan-r003.md:407, 416–434, 466–467, 514–517, 995–999, 1027, 1045 | `verify_distribution.py` consuma `--source-manifest`, ma nessuno strumento produce S, e schema e ID non sono definiti. Il legame S ↔ clone corrente ↔ voci dello snapshot di stage è affidato alla frase «verificare… input hash», senza un meccanismo nel diagnostico. Il rifiuto di una receipt vecchia è facoltativo («può essere usata»). Il percorso d'uso normale `.venv` richiede S/B, ma un utente non ha snapshot della run. **Impatto:** l'anello sorgente → S, il più delicato della catena, resterebbe un passo manuale o non revisionato. Un S prodotto in un altro momento supererebbe I == S | Piano P2: nessuna CLI di generazione; i tre diagnostici elencati in L609 verificano soltanto | Nominare un produttore stdlib visibile nel repository, per esempio un modo `--write-source-manifest` o un quarto diagnostico, con schema S deterministico e ID. Il preflight deve ricalcolare i dieci file del clone e confrontarli con S **e** con le voci `files` dello snapshot di stage, registrandone lo SHA. L'harness deve rifiutare in modo obbligatorio le receipt con ID/hash di S diverso dal corrente, come caso negativo del probe. P7 deve documentare una variante utente senza snapshot della run (S da clone e stato Git) |
| **CLA-P003** | **Bassa — bloccante: no** (A4) | plan-r003.md:391–399, 509–512 | Il probe esegue il secondo sync con configurazione persistente **e** flag insieme, quindi non può attribuire il rebuild alla sola configurazione. È però quella a proteggere i sync impliciti: `uv run` senza `--no-sync`, IDE, `uv sync` senza flag. Il piano la dichiara parte del meccanismo scelto e per `python-downloads` richiede il parsing reale in V1 (L249), mentre per `reinstall-package` no. **Impatto:** una chiave non letta resterebbe mascherata dal flag. Il preflight continua a bloccare le prove, quindi il rischio riguarda l'uso documentato, non i PASS | Doc cache 0.10.10 e settings.rs 0.10.10 («Implies `refresh-package`»), [sources.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md); nessuna prova runtime | Aggiungere al probe, nella stessa copia e cache, un caso con la **sola** configurazione: preflight FAIL, poi `uv sync` senza `--reinstall-package`, log del rebuild e hash installato uguale a S-probe. In alternativa dichiarare la configurazione come difesa non verificata e imporre il flag e il preflight in tutti i comandi documentati (README, AGENTS, IDE) |
| **CLA-P004** | **Bassa — bloccante: no** (A4, CLA-P002 r002) | plan-r003.md:118–126, 144–147, 656–661, 1174–1177 | Il diagnostico R prova l'assenza di egress solo per AF_INET/AF_INET6. I network namespace isolano il solo namespace **astratto** dei socket UNIX. `/run/docker.sock` è `root:docker` e l'utente è nel gruppo `docker`, quindi dentro `unshare --net` o `firejail --net=none` il daemon resta verosimilmente raggiungibile. Lo stesso vale per figli, uv e strumenti senza la guardia pytest, e il daemon può avviare container con rete. Il piano afferma «Non esporre il socket Docker nei figli…» senza meccanismo né verifica. Il codice legacy non usa Docker: è una lacuna della prova di isolamento, non un PASS falso probabile | [static-checks.md §5 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md): permessi del socket, `id -Gn`, `network_namespaces(7)`; inferenza, nessun namespace creato | Il diagnostico R elenca i socket dei daemon noti (`docker.sock`, containerd) e ne prova la sola `connect`, senza inviare richieste, registrandone la raggiungibilità. Per gli strati fast/API/packaging va poi scelta una delle due vie: (a) mascherarli senza modifiche all'host, per esempio con `firejail --blacklist=` o un mount namespace privato; (b) dichiarare il canale residuo e far negare alla guardia pytest la `connect` AF_UNIX su path diversi da socketpair. Il criterio d'equivalenza tra i due runner include questo esito |
| **CLA-P005** | **Bassa — bloccante: no** (A6; dati non fidati e privati) | plan-r003.md:831–835, 1236–1238 | V7 richiede sentinelle non segrete «in ogni categoria esclusa» (`.env`/`.env.*`, `local.env`, config JSON operativo, `source_*`, output, cache, `.git`, `.venv*`, temp), ma non dice dove crearle né con quali nomi. Nel clone reale, una sentinella `.env`, `pipeline_config.json` o in `source_*` può sovrascrivere o alterare dati dell'utente: oggi `.env` è assente qui, ma la procedura è generica. È in contrasto con AGENTS (dati privati, verificare prima di sovrascrivere) | P6.7 e V7; `.gitignore` (queste categorie sono ignorate e quindi non visibili all'impronta) | Creare le sentinelle in una copia usa e getta del contesto, **oppure** solo con nomi univoci inesistenti (per esempio `a001-sentinel-<uuid>`) dopo un controllo d'assenza. Mai sovrascrivere né toccare `.env`, config o dati reali. Registrare gli hash, rimuovere le sentinelle dopo la prova e verificare stato Git e snapshot di stage invariati |

**Suggerimenti opzionali, non rilievi.**

- **S1.** Forma non interattiva del runner per un implementatore agente: un'unica
  invocazione `unshare … -- bash --noprofile --norc -c '<diagnostico R> && <comando C>'`,
  oppure un wrapper versionato. Così diagnostico e test condividono lo stesso namespace;
  registrare `/proc/self/ns/net` anche dal processo di test. Con `unshare --net` lo `lo`
  è DOWN, con Firejail è UP: il criterio d'equivalenza non deve dipenderne, ma lo stato va registrato.
- **S2.** Controllo di freschezza dentro la suite. Le fixture d'integrazione e di
  avvio confrontano i byte dei moduli del clone con le origini risolte da un subprocess
  `-I` di `sys.executable`. Rende C-fast autoprotetto anche se lanciato da IDE o AGENTS
  senza S/B; non sostituisce S/B/I.
- **S3.** Ambiente baseline P0.3: specificare lo strumento, per esempio `uv pip install
  --python <pyenv 3.12.3 assoluto>` in una venv fuori dal clone, con `--no-build` o
  build constraints e freeze registrato. Applicare la regola GPT-P001 r002 anche alla
  baseline e ai confronti incrociati.
- **S4.** Nel caso V1 senza variabili, uv legge la configurazione utente e di sistema:
  registrare presenza e hash di `~/.config/uv/uv.toml`, `/etc/uv/uv.toml` e i file di
  configurazione riportati nel log verboso.

## Motivazione dell'esito e limiti

**GO sul piano r003** per lo snapshot e l'hash indicati. Il piano è autosufficiente,
conserva A1–A7, P0–P7/V0–V12 e la specifica degli undici rilievi r001. Risponde al blocco
CLA-P001 r002 con un meccanismo motivato e confermato nelle fonti del tag 0.10.10:
`reinstall-package` persistente con flag, che implica refresh. Lo rafforza con un
controllo **basato sul contenuto**, indipendente dalla semantica della cache: wheel
canonica dalla sdist come ultimo passo di ogni ambiente e confronto dei dieci moduli fra
S, sdist, wheel, RECORD, installazione e `/opt/venv` prima di ogni prova pertinente.
Prevede inoltre una matrice d'invalidazione e un probe discriminante con controllo
negativo e ripristino. Questo chiude nel disegno il difetto «PASS su codice diverso da
quello revisionato». Un mismatch ferma la prova.

Gli altri cinque rilievi r002 e S1–S7 sono trattati con prove osservabili. I tre blocchi
r001 restano affrontati adeguatamente. Comandi, flag, inventario README e osservabili V5
sono stati verificati in modo indipendente su help congelati, scansione e codice.

I cinque nuovi rilievi sono precisioni non bloccanti:

- CLA-P001 e CLA-P002 riguardano l'operatività della catena E/S;
- CLA-P003 l'attribuzione del meccanismo;
- CLA-P004 la completezza della prova d'isolamento;
- CLA-P005 la sicurezza delle sentinelle Docker.

Nessuno può produrre un PASS falso con i preflight previsti. I loro criteri sono
verificabili staticamente nel prompt o nel report d'implementazione e nella review
dell'implementazione. L'arbitro decide se richiederli come precisazioni del GO o in
una nuova revisione.

**Limiti della review.**

- La semantica uv è letta in documentazione e sorgenti al tag (settings.rs tramite
  strumento di estrazione), non riprodotta. Grafo CPU/cu126, sdist, native e digest
  restano da risolvere nelle prove future.
- L'accessibilità del socket Docker nel runner e la negazione di `unshare` sono
  inferenze statiche. Nessun namespace o Firejail è stato avviato.
- Le date Debian sono riprese dalle fonti antecedenti, coerenti tra loro, e non
  ricontrollate online. Pagine e metadata web possono cambiare dopo il 2026-10-03.
- La review vale solo per lo snapshot `plan-r003` con l'identità indicata. Una modifica
  a piano o input richiede una nuova revisione. Questo GO non autorizza da solo
  l'implementazione, che attende l'arbitrato su entrambi i report.
