# Arbitrato — piano — run-a001-fase0-uv — r003

- Data: **2026-10-03, Europe/Rome**. Supervisore Codex/OpenAI, famiglia GPT-6;
  identificatore specifico del modello e ID chat non esposti. Nuova chat di
  supervisione, nessuna terza review, implementazione o delega.
- Oggetto: [plan-r003.md](../plans/plan-r003.md), SHA-256
  `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`.
- Manifest comune delle review: [plan-r003.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
  SHA-256 `99f33ba4c9ab452fd0015a654ce59251b68d1467fa97763b2c5c3de921fe4046`;
  79 file +90 artefatti, worktree
  `abbab72ee487228cd2a4733592d336770c0914dfdcd9242a510c6172b1c971a4`.
- Branch `feature/run-a001-uv`; HEAD/dev/merge-base
  `66ba82200e5def5a4db76f9bafccb0731b506091`; indice Git vuoto.
- Ingresso autonomamente verificato **MATCH** su
  [arbitration-context-r003.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
  SHA-256 `f8ba45c0aafa1500f077bdb9785282fd8fc6d78ee286bb1b1db26dec6dd1553c`,
  worktree `0906017032537c60e398a8b6ca99585250b65d563cf9a79ab4be3726ba130188`.
- Report [ChatGPT r003](../reviews/review-plan-r003-chatgpt.md): **GO**, nessun
  nuovo rilievo. Autore effettivo dichiarato Codex/OpenAI, famiglia GPT-6,
  Codex IDE/API, ruolo ChatGPT assegnato dall'utente; differenza da ChatGPT Web
  dichiarata. ID specifici non esposti.
- Report [Claude r003](../reviews/review-plan-r003-claude.md): **GO**, cinque
  rilievi non bloccanti e quattro suggerimenti. Autore dichiarato Anthropic,
  Opus5.5 `claude-opus-5-5[1m]`, Claude Code VSCode; ID chat non esposto,
  scratchpad non assunto come ID.
- **Esito del supervisore: GO sul piano r003 con le precisazioni operative
  definite qui.** Stato successivo **IMPLEMENTATION**, in attesa di una nuova
  chat implementatrice. Migrazione non iniziata; nessun GO sul codice.

## Validità, evidenze e ambito del GO

Letti integralmente piano e due report, checkpoint e findings; esaminati in modo
strutturato inventari/checks e manifest. Entrambi dichiarano nuova chat esclusiva,
indipendenza, nessuna delega e nessuna lettura dell'altro report corrente o
arbitrato r003. Claude dichiara soltanto ls della cartella ChatGPT. MATCH pre/post
sul medesimo oggetto nei rispettivi file. Gli hash identificano i file; autore,
modello e indipendenza restano dichiarazioni, non proprietà certificate dagli hash.

La [ricevuta propria — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
riconferma 90 artefatti comuni, 15 output reali, otto hash output ChatGPT e undici
voci hash post Claude. Plan-r003 è storico soltanto in files/worktree_sha256:
confrontati i before/after dei sei documenti con la ricevuta del precedente
supervisore, senza ulteriori divergenze. Piano/review/antecedenti non modificati.
Il nuovo contesto implementation-context-r001 conserverà questa corrispondenza;
la transizione e i controlli finali sono nelle evidenze proprie, separate dalla
ricevuta di handover congelata.

Il GO autorizza il disegno della migrazione, non certifica runtime, runner, lock,
backend, installazioni, fedeltà, Docker o Marker. Le prove obbligatorie future
non sono blocchi di pianificazione ancora senza soluzione: hanno azione, criterio
e uscita definite. Un fallimento o scostamento sostanziale durante l'implementazione
ritorna al supervisore; non diventa PASS per effetto di questo GO.

## Decisione sulle nove voci r003

Gli ID sono r003, distinti dagli omonimi r001/r002. Le severità sono confermate
come rilievi di precisazione del disegno; nessuna voce è chiusa operativamente.

| ID | Disposizione | Motivazione ed evidenza | Azione e criterio futuro |
| --- | --- | --- | --- |
| **CLA-P001 r003** | **Accolto, non bloccante** | P2:474–481 affida snapshot al supervisore senza un passaggio concreto; S a P2:424 cita un manifest che deve già esistere. La lacuna è operativa e non cambia la catena prevista. | D1 conserva il proprietario, fissa label/comando/artefatti e checkpoint in attesa. Ordine input → snapshot → S → B/I/E; nessuna auto-delega o dipendenza circolare. |
| **CLA-P002 r003** | **Accolto, non bloccante** | P2:416–434 definisce contenuti di S, E e V6 vietano mismatch, ma manca produttore visibile e confronto esplicito corrente/S/files. ChatGPT ha ragione sul divieto già vincolante; Claude ha ragione sull'assenza di un negativo obbligatorio e sul percorso utente. | D2 concreta produttore stdlib, schema/ID, legame clone/snapshot, negativo receipt e variante standalone. Stale S/receipt deve fermare l'harness prima della collection; non basta I == vecchio S. |
| **CLA-P003 r003** | **Accolto, non bloccante** | P2:509–512 usa configurazione e flag insieme: non attribuisce il rebuild alla sola configurazione. La fonte uv taggata documenta la chiave, senza provarne il parsing locale. | D3 aggiunge caso config-only, dopo negativo, prima dell'installazione canonica; anche il caso flag conserva la sua prova separata. Metadata/lock invariati, ripristino verificato. |
| **CLA-P004 r003** | **Accolto, non bloccante** | §R e V6 distinguono daemon/namespace ma non verificano il path socket. network_namespaces(7)/unix(7) confermano la distinzione; raggiungibilità locale resta inferenza, non risultato osservato. | D4 sceglie confinamento filesystem dei daemon noti con Firejail blacklist quando necessario, più guardia pytest. R verifica sola connect/close, padre/figlio, nessuna richiesta daemon; unshare semplice non è equivalente se lascia il canale aperto. |
| **CLA-P005 r003** | **Accolto, non bloccante** | V7:1236–1238 omette luogo/nomi delle sentinelle. P6/AGENTS già escludono dati privati e modifiche operative. Precisare la copia evita sovrascritture senza cambiare la prova. | D5 usa copia usa e getta del solo contesto ammesso e sentinelle sintetiche nuove, con hash, confronto e cleanup limitato; mai sentinelle in file reali del clone. |
| **CLA-S1 r003** | **Accolto, non bloccante** | Forma non interattiva rende concreta R/V6 per un agente e mantiene diagnostico/test nello stesso namespace. Stato lo UP/DOWN non è il criterio di egress. | Wrapper fidato versionato, argv senza shell interpolata; ripete R e preflight, poi comando C solo su PASS. Receipt del processo test e figli con namespace effettivi; lo registrato, nessuna attivazione richiesta. |
| **CLA-S2 r003** | **Accolto, non bloccante** | Fixture di integrazione/avvio già devono verificare ambiente nello stesso sys.executable; il controllo IDE rafforza il requisito esistente. | Confronto byte clone/installato in subprocess -I prima dei test interessati; mismatch FAIL. Modalità IDE senza S ammessa solo come controllo locale ridotto, mai prova ufficiale A3/A4 senza S/B/I/E/R. |
| **CLA-S3 r003** | **Accolto, non bloccante** | P0.3 indica venv baseline ma non strumento. P1/GPT-P001 r002 già impongono constraints o sole wheel anche agli altri pip. | Venv esterna via interprete pyenv 3.12.3 assoluto e -I -m venv; uv pip install con --python assoluto e --no-build, versioni/hash/freeze. Se serve sdist, backend constraints identificati prima; stessa regola negli incroci. Nessuna installazione nel pyenv originale. |
| **CLA-S4 r003** | **Accolto, non bloccante** | Rimuovere UV_* non elimina le configurazioni persistenti utente/sistema. Fonte uv e antecedenti sostengono la necessità di registrare gli input effettivi. | V1 registra presenza/path/hash di config uv progetto/utente/sistema e dei file effettivamente scoperti, compresi path XDG; nessun contenuto o segreto nei log. Differenze prima/dopo invalidano l'attribuzione. |

## D1 — Protocollo concreto degli stage e proprietario

**Solo il supervisore esegue snapshot e aggiorna i registri comuni.** Non autorizzo
l'implementatore a scrivere snapshots/, STATE, HANDOVER comune, eventi o arbitrati.
Il supervisore non lavora fra i turni: ogni freeze richiede una consegna esplicita
alla chat di supervisione. Questa attesa è un passaggio di ruolo, non una nuova
richiesta di approvazione di uv o del singolo file.

Label riservate: `impl-r001-stage-baseline-s001`, `impl-r001-stage-package-s001`,
`impl-r001-stage-tests-s001`, `impl-r001-stage-image-s001`,
`impl-r001-stage-final-s001`. A ogni invalidazione incrementare s002, s003 ecc.
**per quello stage**, mai sovrascrivere né riusare una label. Lo stage finale
congela report/evidenze per le review del codice; non genera un S che includa
se stesso o un hash del proprio manifest dentro un artefatto congelato.

1. L'implementatore prepara input stabili, diagnostici/fixture nel repository,
   ambienti/cache in preparazione distinta e inventari innocui. Non esegue ancora
   la prova dello stage. Scrive `implementation/stages/<label>/request.json`:
   schema 1, run/revisione/label/stage, branch/HEAD/dev/merge-base, elenco dei
   file rilevanti con byte/hash o assenza esplicita, artefatti da congelare,
   comandi/profili/costi autorizzati, prove previste e invalidazioni. I path degli
   artefatti sono relativi alla radice e confinati alla run; niente segreti.
2. Aggiorna `handovers/implementation-r001.md` con
   `WAITING_FOR_STAGE_SNAPSHOT`, richiesta precisa, processi propri e lavoro
   compiuto. Consegna al supervisore usando il prompt stage predisposto. Nessuna
   prova dipendente del nuovo stage in attesa; niente poll o freeze autonomo.
3. Il supervisore confronta richiesta e file, conserva GO/piano, controlla Git e
   perimetro; completa prima del freeze ogni aggiornamento tracciato di stato
   (changelog/metadata), poi forma argv `python3 scripts/run_context.py snapshot
   run-a001-fase0-uv --label <label>` con un `--artifact <path>` per ogni input
   esplicito. Include piano/arbitrato/prompt implementativo, request.json e
   inventari/fixture/evidenze di ingresso dello stage; per baseline include
   copia dei dieci legacy e manifest dei suoi input. I file codice/test/diagnostici
   non ignorati sono già in files. Per final include anche report e precedenti
   S/B/I/E/runner/output/confronti selezionati. Non include output futuri.
4. Verifica MATCH, salva risposta propria con label/path/SHA del manifest e
   worktree, elenco congelato, ruolo successivo e prompt di ripresa. Solo allora
   l'implementatore produce S riferito al manifest già esistente e prosegue B/I/E.
   La risposta/checkpoint successivi restano fuori dagli artefatti dello stesso
   freeze; un contesto di ripresa distinto può includerli senza cicli.
5. Prima/dopo ogni prova verify dello stage e confronto degli input pertinenti.
   Una modifica attesa o inattesa invalida gli esiti dipendenti: nuova richiesta,
   nuova label, rebuild/install/prove secondo P2:484–493. Un vecchio PASS conserva
   la sua impronta; equivalenza al finale solo con confronto motivato.

Baseline: preparare diagnostico R e input prima del freeze, poi R PASS prima
**dell'esecuzione** P0/V0. Tuning esplorativo di F6 è identificato e non PASS finale;
se cambia fixture, nuovo stage baseline prima del confronto definitivo.
Package/tests/image: S precede build/install/prove; gli input consumati da build
(anche README/licenza) devono essere stabili. Final: prima ripetere le prove
invalidate, poi congelare i report già prodotti; nessuna autoreferenzialità S/E.

## D2 — Produttore di S, schema e controlli obbligatori

Realizzare dopo GO `scripts/diagnostics/run-a001-fase0-uv/make_source_manifest.py`,
stdlib visibile alla review, distinto dal prodotto/wheel. CLI minima:
`--repo PATH --snapshot PATH --output PATH` oppure
`--repo PATH --standalone --output PATH`; rifiutare output già presente,
symlink/percorsi evasivi, schema ignoto, file mancanti/duplicati e input cambiati
mentre si acquisiscono. Il verificatore P2 conserva la CLI prevista, con
`--repo PATH` esplicito per confrontare il clone fidato corrente.

Schema JSON S v1: envelope `schema`, `id`, `payload`, `captured_at_utc`.
Payload deterministico: scope `run-stage` o `standalone`, repo reale, identità
Git (branch/HEAD/dev/merge-base, stato dei file), riferimento snapshot
label/path/SHA/worktree per run-stage, `modules` ordinati (esattamente dieci
path, kind, bytes, sha256), `build_inputs` ordinati con stato file/assenza,
toolchain/interprete/stdlib/piattaforma e inventari identificati di dipendenze
**di ingresso** e backend/config effettiva senza segreti. Include tutti gli input
P2:424–434 e hash del produttore/diagnostici. Non include B/I/E futuri né
contenuti della distribuzione di progetto installata come input di se stessa.

`id = "sha256:" + SHA256(JSON(payload, sort_keys=True, ensure_ascii=True,
separators=(",", ":")))`; timestamp escluso dall'ID. Hash esterno del file S
registrato separatamente. L'ID distingue baseline, stage nuovi, copia probe e
uso standalone. Non usare soltanto nome/versione della distribuzione.

In run-stage il produttore verifica snapshot immutabile/schema/identità e
MATCH prima/dopo, ricalcola i dieci sorgenti e ogni input build corrente e
confronta path/kind/hash con `files` del manifest (o `artifacts` per copie
baseline/probe della run). Un file rilevante assente dal manifest è FAIL; non basta
confrontare S con una tabella del piano. Il verificatore ripete il confronto
corrente ↔ S ↔ snapshot prima di B/I/E e legge byte degli archivi/installazione
come in P2. Per le copie baseline/probe, i path locali dei dieci moduli devono risolvere
agli esatti artefatti della richiesta congelata: mappa relativa alla radice
host dichiarata in S, repo della copia distinto e provenienza dal clone
registrata, non confronto con i moduli immutati della radice. Sono dati di
prova identificati, mai surrogati del prodotto finale.
Per Docker il driver confronta anche copia-contesto e S fidato;
per la baseline distingue esplicitamente sorgenti originali e nuovo prodotto.

Receipt I/E v1 identifica proprio ID, stage/snapshot, S.id e SHA file S, B/archivi,
interpreter/profile, diagnostici/test/fixture/config/cache, argv/CWD e esiti.
L'harness ricalcola gli input: una receipt PASS registrata non è prova di
freschezza se S/snapshot/input cambiano. **Negativi obbligatori:** S vecchio
contro clone-probe modificato; receipt vecchia contro S nuovo; ID corretto ma
SHA/file o uno degli input diverso. Exit nonzero **prima di collection/import
applicativi**, senza build/installazione/riparazione. Poi receipt corrente
positiva e ripristino dei dieci hash/metadata/lock. Mantiene il negativo già
prescritto del modulo installato stantio prima del sync. Questi controlli
operativi chiudono il dettaglio soltanto dopo prova reale.

Variante normale senza temp/run: stessa catena clone → S-standalone → sdist →
wheel → installazione, directory di lavoro nuova locale ignorata `.cache/uv-fidelity/`,
ID e stato Git, confronto sorgenti correnti prima dell'uso. Nessuno snapshot di
run richiesto all'utente normale; le receipt dichiarano scope standalone e non
sono ammesse come evidenza ufficiale della run. La guida descrive generazione,
rebuild dopo modifiche, installazione canonica/preflight e rifiuto sul mismatch.

## D3 — Configurazione sola e flag

Nel probe isolato/cache propria già popolata aggiungere **due casi indipendenti**
a metadata/lock invariati. Dopo il primo sync, modifica innocua di exceptions.py,
nuovo S-probe e negativo installazione stantia. Primo caso: sync con sola
`tool.uv.reinstall-package`, senza flag reinstall/refresh e senza altre override
che ne mascherino la lettura; log/config/hash installato prima della wheel canonica.
Secondo caso, dopo riallineamento e nuova modifica innocua: comando previsto
con flag esplicito e stesso controllo prima della wheel canonica. Non rimuovere
la configurazione del clone per simulare i casi. Nessun fetch implicito nel probe.
Ripristinare la sola copia e dimostrarne identità; prove finali nel clone reale.

Le istruzioni README/AGENTS/IDE mantengono flag e preflight espliciti. Config-only
PASS permette dichiarare la guardia provata; FAIL torna al supervisore, senza
rimuovere la chiave o fingere che il caso con flag la abbia verificata.

## D4 — Runner, UNIX su path e invocazione unica

Scelta: **confinamento dei daemon noti ereditato dai figli**, oltre alla guardia
pytest. R inventaria path canonici/alias Docker (`/run/docker.sock`,
`/var/run/docker.sock`, rootless/XDG e DOCKER_HOST locale), containerd e altri
socket di daemon segnalati dalla configurazione target. Niente scansione dei
contenuti di directory private o endpoint remoti. Il diagnostico tenta soltanto
connect con timeout breve e close sui path noti, senza send/recv o richieste
operative; prima/dopo e padre/figlio. Un path assente è registrato come tale,
non come prova di una blacklist funzionante.

Unshare semplice resta primario per verificare il namespace IP; può essere
usato per le prove solo se R dimostra che tutti quei path sono inaccessibili.
Se un path è raggiungibile, non ammettere quel runner: usare l'alternativa
Firejail esistente `--noprofile --net=none`, aggiungendo `--blacklist=PATH` per
path/alias/target canonici identificati. Nessun profilo persistente, chmod,
modifica socket/daemon/rete host o nuovo setuid. Mascheramento fallito o policy
negata = IMPEDITA; non basta la monkeypatch pytest per uv/Pandoc/figli.

Controllo discriminante del mascheramento con socket sintetico in directory di
prova nuova (helper fidato, nessuna richiesta daemon), visibile fuori e negato
dentro padre/figlio; socketpair AF_UNIX senza nome positivo. Test pytest nega
connect/connect_ex/sendto ai path AF_UNIX e a Internet, conservando socketpair
per API. Nessun fd daemon ereditato nei test; argv/environment ripuliti dalle
configurazioni daemon non necessarie. R, preflight S/B/I e C condividono una
singola invocazione non interattiva del runner tramite wrapper stdlib versionato
`run_offline.py`, che usa argv e avvia C solo dopo PASS; namespace dei processi
osservati e hash del wrapper nella receipt. Lo UP/DOWN registrato, non requisito.

Equivalenza fra candidati: IP senza route/egress, stessi prerequisiti, figli,
socketpair positivo, daemon noti negati e nessuna guardia disattivata. Limite:
non è una sandbox generale contro codice ostile o ogni IPC host sconosciuto;
è isolamento delle prove fidate su fixture sintetiche e dei canali inventariati.
Un nuovo canale individuato richiede aggiornamento del target/preflight.
C-docker/V7/V8 restano fuori dal runner degli altri strati, daemon locale e
perimetro separato, immagine locale/pull never/network none per V8.

## D5 — Sentinelle Docker e dati reali

Copia usa e getta in directory nuova: solo i file ammessi realmente consumati
dalla build, Dockerfile/.dockerignore identificati, niente `.git` reale, .env,
configurazioni, documenti, cache o venv dell'utente. Creare lì con apertura
esclusiva sentinelle non segrete per ogni categoria esclusa, anche `.env`,
`.env.*`, local.env, config JSON, .git, .venv*, temp/tmp, source_*, output/cache.
Registrare path/hash e assenza iniziale. Manifest del contesto filtrato prodotto
usando quelle stesse regole deve escluderle; la copia ammessa deve coincidere
con i file S, non contenere sostituti dei moduli per riparare packaging.

La prova diagnostica del filtro e la build finale identificano i rispettivi
contesti; non trasferire PASS fra contesti diversi senza confronto di input e
regole. Cleanup soltanto della directory di prova identificata dopo conservazione
evidenze; verificare clone/snapshot/Git invariati e sentinelle reali mai create.
Non copiare ricorsivamente il clone con file ignorati o dati privati.

## Vincoli conservati e motivazione complessiva

Verificati A1–A7 e le tre matrici del piano (11 r001, 6 r002, 7 disposizioni r002).

| Specifica antecedente | Valutazione nel disegno r003 e prova futura |
| --- | --- |
| CLA-P001 r001, A1 | Pin/managed/pyenv/origini espliciti in P1/V1; matrice reale richiesta |
| CLA-P002 r001, A4 | Strati/API/wheel/governance/discovery in P4/V6; conteggi e zero skip essenziale |
| CLA-P003 r001 | Fonte Marker/contratto e backend P1/P5; pip constraints GPT-P001 r002 conservato |
| CLA-P004 r001 | Dotenv prima override P3/V5, processi e osservabili 1200/80 e 1600/120 |
| CLA-P005 r001, A3 | -I/ambiente/PYTHONPATH e origini/sentinelle P3/V5, wheel esterna richiesta |
| CLA-P006 r001 | Entrambi i Compose P6/V7, config esplicite e limiti |
| CLA-P007 r001 | Candidati/base/digest/native/manutenzione P6; V7/V8, stop se non verificabili |
| CLA-P008 r001 | P7/V9 inventario completo, epilogo e AGENTS; CLA-P004 r002 conserva 74/9 pre |
| CLA-P009 r001 | P0/P4/V0/V5 stessi validated/input/argomenti e incroci se diverge |
| CLA-P010 r001 | V10 rinviata nel mandato, essenziale se incompatibilità; limite local_marker |
| GPT-P001 r001 | F6 almeno due chunk/overlap, complemento sliding esistente, difetti legacy |
| CLA-P001 r002, A3/A4 | P2 S/B/I/E e rebuild ora adeguati nel disegno con D1–D3; nessuna chiusura runtime |
| CLA-P002 r002 | R prima baseline, alternativa concreta e IMPEDITA; D4/S1 rendono esecutivo il dettaglio |
| CLA-P003 r002 | Manual/origine/caso senza variabili P1/V1; S4 registra config reali |
| CLA-P004 r002 | P7/checks inventario completo, delta pre/post; non correggere evidenze storiche |
| CLA-P005 r002 | Dieci moduli + quattro transitive, padre/figli/base senza FastAPI V5 |
| GPT-P001 r002 | Vincoli backend o no-build per ogni pip; S3 rende esplicita anche la baseline |
| CLA-S1 r002 | Rinvio conservato, nessuna piattaforma ristretta implicitamente |
| CLA-S2 r002 | Discovery/esclusioni nominate/conteggi, nessun ignore per nascondere test |
| CLA-S3 r002 | README/licenze/input backend inclusi in S e Docker se consumati |
| CLA-S4 r002 | Socketpair positivo conservato, IP/daemon negati senza disabilitare API |
| CLA-S5 r002 | Lock --check indipendente, hash prima/dopo; no-sync non prova freschezza |
| CLA-S6 r002 | Versioni API locked/warning osservati, niente refactoring on_event |
| CLA-S7 r002 | Server host non verificato senza prova propria; nessun PASS ereditato da Docker |

R, P0–P7 e V0–V12 mantengono ordine e obiettivi. A2 conserva separazione runtime/
dev/API/motore, dieci moduli flat/cinque script; A5 inventario completo e guide;
A6 richiede build/contratti reali futuri, A7 due nuove review del codice e
arbitrato finale. F1–F6 devono verificare bytes/formule/numeri/ordine/asset e
perdite legacy; niente normalizzazione o correzione algoritmi in questa run.

Il blocco CLA-P001 r002 è superato **nel disegno** da un meccanismo documentato
più un confronto del contenuto indipendente dalla cache. Le precisazioni D1–D5
realizzano obblighi già presenti: proprietario, strumenti diagnostici, input
correnti, isolamento e dati sintetici. Non cambiano candidati/versioni, layout,
requisiti, output o perimetro di prodotto. Non serve r004/doppia review del piano
per queste specificazioni; verranno esaminate nelle due review dell'implementazione.
Non lascio un blocco valido irrisolto con un GO condizionale. I due GO dei revisori
non sono la premessa probatoria della decisione, che usa testo e riscontri.

Fonti e limiti: [sources.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
**V7/V8 obbligatorie future**, preparazione GB/native distinta dai test veloci e
perimetro presentato al supervisore prima di acquisizioni pesanti non già coperte.
**V10/V11, pesi/font/inferenza non autorizzati**. Nessun invio remoto implicito.
local_marker e server host non verificati dopo migrazione senza prove pertinenti.

## Passaggio successivo

Prompt completo per **nuova chat implementatrice**:
[04-implementation-r001.md](../prompts/04-implementation-r001.md).
Contesto d'ingresso: `snapshots/implementation-context-r001.json`, creato dopo
aggiornamenti di stato e verificato con checks finali propri. Piano e review
restano invariati. Prompt per consegne di stage al supervisore:
[05-supervisor-stage-r001.md](../prompts/05-supervisor-stage-r001.md).

NO_GO r001/r002 conservati. Nessuna implementazione in questa chat: soltanto
Git/hash/letture/fonti e verifiche documentali. Nessun namespace, Docker,
lock/sync/build/installazione, suite/collection, conversione o download.
Nessun commit/merge/push/promozione/deploy: operazioni Git manuali dell'utente
solo dopo GO finale sul codice. Temp ignorata, Git non la trasferisce.
