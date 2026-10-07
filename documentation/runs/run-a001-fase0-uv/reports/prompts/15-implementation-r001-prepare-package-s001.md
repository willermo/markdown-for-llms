# Preparazione implementativa r001 — package s001 — run-a001-fase0-uv

Agisci come **implementatore** in una nuova chat nel repository
`/home/davide/workarea/markdown-for-llms`, riprendendo
`handovers/implementation-r001.md`. Il supervisore ha ricevuto e verificato
la baseline s005: **BASELINE_ACCEPTED / WAITING_FOR_STAGE_PREPARATION package-s001**.
GO piano r003 invariato; nessun GO codice. Quattro R s005 complete PASS,
S canonico corrente dello stage e V0 **PASS di caratterizzazione**. Suite legacy
**FAIL/exit1**:67nodeid/201eventi/62pass/5fail osservati, cause ENOENT/AssertionError
individuali. V0 s003/s004 FAIL e impedimenti s001/s002 rimangono storia intatta.

Mandato: **sola preparazione statica P1/P2/P3**, diagnostici e richiesta D1
`impl-r001-stage-package-s001` WAITING_FOR_STAGE_SNAPSHOT. Non eseguire acquisizioni,
uv python install/lock/sync/build/pip, R/S/B/I/E, collection/probe/applicazione.
Il supervisore riceverà la request, congelerà input reali e consegnerà un prompt
operativo successivo. Non produrre un lock sintetico né input/output futuri.
Questa consegna non è ancora lo stage package o il GO delle prove V1–V3.

Chat distinta, nessun subagente, review simulata, auto-snapshot o servizio di
polling. Non modificare STATE/HANDOVER/eventi/indici/arbitrati/ADR condivisi.
Prompt04 origine; prompt14 completato. La roadmap non avvia fasi aggiuntive.

## Letture in ordine e identità

1. AGENTS, CHANGELOG, PROJECT-CONTEXT/HANDOVER globali e STATE/HANDOVER della run;
   manage-implementation-run, protocollo e template request/role/checkpoint.
2. Brief A1–A7, piano r003 e arbitrato r003 integralmente per questa chat fresca;
   D1–D5 e24disposizioni antecedenti vincolanti; NO_GO r001/r002 conservati.
3. Prompt04 e14, report/checkpoint implementatore, completion-r001 s005,
   delivery/manual-content-audit e controlli in resume-baseline-s005.
4. In evidence/supervisor-implementation-r001/baseline-completion-r001/:
   decision.md, reception.json, content-observations.json, suite-observations.json,
   sources.json, transition.json, metadata-transition.json, response.json,
   snapshot-verify.json e checks-final.json; checkpoint
   handovers/supervisor-baseline-completion-r001.md. Non rieseguire helper supervisore.
5. Contesto snapshots/package-preparation-context-r001.json; identità esatta
   SHA/worktree/file/artefatti nella response del supervisore. Leggi manifest
   baseline s005 e S per risultati originali, non come preflight del prodotto nuovo.
6. Poi soltanto moduli/config/launcher/metadata/test/README/consumatori pertinenti,
   help uv0.10.10 e fonti primarie versionate pertinenti. Verifica flags/API dubbie
   prima di proporre argv; conserva URL/data/esiti e limiti, niente esiti inventati.

Percorsi senza prefisso relativi alla run, codice/AGENTS/documentation alla radice.
Non caricare tutta temp; recupera intervalli troncati. Inventari grandi strutturati,
ma non omettere record. Riepilogo chat non sostituisce receipt o codice.

| Oggetto | SHA-256 |
| --- | --- |
| plans/plan-r003.md | `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b` |
| arbitrations/arbitration-plan-r003.md | `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d` |
| prompts/04-implementation-r001.md | `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035` |
| prompts/14-implementation-r001-baseline-s005.md | `f813d53f1424c1a8c4b39eb023122b88a59ae3da4a221107cd2b34731c54ab4c` |
| implementation/stages/impl-r001-stage-baseline-s005/request.json | `2d82e003935a2936fd1c200027f3d860ea6bdf5e21887b3584864c489929a324` |
| implementation/stages/impl-r001-stage-baseline-s005/completion-r001.json | `d58fb122b545caf6cc6a9ec4af9e34d06b241cda11410606fe8b5fd1c21a0531` |
| snapshots/impl-r001-stage-baseline-s005.json | `aba4fc260a9e3de766bd5a5022bac60f9703881f128a443c61671015fc627ed0` |
| implementation/stages/impl-r001-stage-baseline-s005/sources.json | `874b8528d20d87372354df46f4da558115c90733c1ee66093e70689825deb68d` |
| evidence/implementation-r001/resume-baseline-s005/manual-content-audit.md | `7ff5c35d2f1d05f827a17b947c8b78dbbf777befd085174b4d5513177197c054` |
| evidence/implementation-r001/resume-baseline-s005/runtime-output-inventory.json | `857b3cd2b762189352cf2d9acbec9429ca2515d8f35e911092c79d084c3bd936` |
| scripts/diagnostics/run-a001-fase0-uv/run_offline.py | `a937e0067f81f9d09c2fec49f4b47de00eb9a8b254b64d75caed1029b3c71018` |
| scripts/diagnostics/run-a001-fase0-uv/check_runner.py | `086f7dbb12798ca990a04dec575ae72e5066cc32721ee522aa26e41644adf1ed` |
| scripts/diagnostics/run-a001-fase0-uv/make_source_manifest.py | `7dc068734a5fd00b63a652995402727264b9d8e909ffa1b3c93841e5948ce44d` |
| scripts/diagnostics/run-a001-fase0-uv/run_baseline.py | `750e120edca98f00c543e497218d8f56cb7cfac56d90859292f83753f3ad2e44` |

Prima di intervenire verifica Git reale e il nuovo contesto:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
git diff --cached --name-only
python3 scripts/run_context.py verify run-a001-fase0-uv --label package-preparation-context-r001
```

Atteso MATCH all'ingresso. Branch feature/run-a001-uv,HEAD/dev/merge-base
66ba82200e5def5a4db76f9bafccb0731b506091, indice vuoto; sei metadata documentali e
17file preparatori nuovi. Nessun prodotto/lock migrato a questa consegna.
Ricalcola la response e confronta i suoi descriptor e quello del prompt15 con
il manifest: il contesto contiene questo prompt, quindi il prompt non incorpora
lo SHA del proprio contesto (ciclo vietato). Response/checkpoint/checks successivi
sono fuori dal freeze; identità completa letta da response e snapshot-verify.
Se input/base/mandato differiscono, registra scostamento e consegna al supervisore.

Lo stage baseline s005 è storico alla ricezione soltanto per CHANGELOG, poi sei
metadata del supervisore prima del nuovo contesto; tutti903artefatti invariati.
180output/120workspace/84Markdown-report s005 sono ricevuti e preservati, non
riscrivere S/snapshot/receipt/rapporti storici. Durante la preparazione i delta
tecnici autorizzati rendono storico il nuovo contesto: registra before/after,
non chiamarlo MATCH per input nuovi. Il prossimo freeze package sarà distinto.

## P1 — Dichiarazione, pin e piano di preparazione operativa

Realizza pyproject.toml, .python-version esatta, build-constraints.txt e ignore
secondo P1/P2. Candidati **vincolanti**: uv0.10.10 required-version==0.10.10,
CPython3.12.13, requires-python>=3.12,<3.13, setuptools.build_meta e
setuptools==84.0.0, markdown-for-llms1.0.0. Nessun upgrade tacito alla patch attuale.
Managed solo RUN_REPO/.venv-python,UV_PYTHON_INSTALL_DIR assoluta nelle invocazioni/
subshell; python-downloads="manual" + origine/preflight, download esplicito iniziale
separato,poi UV_PYTHON_DOWNLOADS=never e flag managed/no-downloads. Niente pyenv
install/local/global,PATH/global config/altro progetto o directory predefinita.

Un solo pyproject/lock. Runtime quattro dipendenze P1; dev separato con httpx,
default-groups=[]; marker-server e marker-cpu/cu126 espliciti,in conflitto CPU/GPU,
indici torch explicit,versioni Marker1.10.2/Surya0.17.1/Torch2.7.1 candidate.
Colorama/rich/enhanced e nostri tool dev non passano nel runtime. Non installare
ML/native o server qui. Universal lock dovrà risolvere metadata anche extra non
installati: costo e eventuali sdist/backend non congelati da dichiarare, niente
risoluzione/build implicita o rimozione silenziosa di un extra/A6.

Tre pin backend uguali:build-system,build-constraints.txt e
tool.uv.build-constraint-dependencies; configurazione reinstall-package persistente.
README/licenze realmente consumati dal backend devono essere identificati e
stabili prima della futura build. Metadata statici; rimuovi example.com ingannevole
senza inventare contatti. Non rimuovere setup.py/requirements.txt finché tutti i
consumatori non sono allineati; inventaria quelli ancora legacy,nessuno shim nuovo
come fonte runtime alternativa. P6/P7 fuori da questa ripresa: registro chiaro
dei consumatori rinviati e rimozione successiva,non dichiararla completata.

**uv.lock resta da generare operativamente**, non fabbricare resolved pins.
Preparazione attuale produce una richiesta concreta che distingue:

- Acquisizione esplicita futura del managed, metadata del lock, backend/deps leggere
  di ingresso e cache proprie. URL/build/hashes/provenienza/versioni,comandi exact
  verificati su help congelato,fonti,stima rete/disco/tempo/target e limiti.
- Input realmente risultanti (lock/interprete/deps/backend/config) da inventariare
  prima di S. Se l'acquisizione/lock cambia gli input congelati,nuova request/label
  package s002 o successiva **prima** di S/build/install/prove dipendenti.
- S finale stage→sdist→wheel dalla sdist→I/preflight→V1–V3,solo nella ripresa
  operativa successiva prevista dal supervisore. Mai receipt per output futuri.

Il primo package-s001 può identificare ingresso e piano d'acquisizione; non vale
come snapshot immutabile del futuro lock. Nessuna build/install progetto prima
S/B pertinenti. Eventuale preparazione dipendenze senza progetto deve essere
esplicita e distinta da I,senza circolarità nell'inventario di ingresso S.
Identifica fin da ora directory nuove ignorate per cache/tmp/output package nel
filesystem scelto,senza usare/cancellare la baseline; inventaria spazio reale.
A queste acquisizioni serve una successiva disposizione dopo verifica della
request,non una riapprovazione generica dell'adozione uv.

## P2 — Packaging e diagnostici stdlib

Dieci py-modules espliciti,esattamente:
config,logging_config,exceptions,unified_converter,master_workflow,
clean_markdown,validate_markdown,chunk_markdown,batch_monitor,marker_api_server.
Cinque console script invariati a target P2; config.main va estratta dalla CLI
corrente,richiamata anche da __main__,preservando JSON/help/messaggi/limiti.
Nessun namespace/layout fase2,nuovo launcher generato o extra console script.
Wheel/sdist esclusione test/fixture/governance/setup/dati/.env/temp/segreti/cache;
non assumere correttezza da find_packages o dal nome/versione.

Completa nel repository scripts/diagnostics/run-a001-fase0-uv/:
check_python_origin.py,verify_distribution.py e make_source_manifest.py D2.
S schema1 ID canonico payload sort_keys/ensure_ascii/separators,timestamp escluso;
run-stage e variante standalone reali con repo esplicito,snapshot/correnti/path/
clone-copy mappati. Per standalone nessuna temp/run obbligatoria; receipt scope
standalone non ammesse come prova ufficiale della run. Nessun output già presente,
path evasivo/symlink/dup/input ignoto accettato. I/B/E futuri esclusi da S.

Verificatore CLI esatta P2 --source-manifest/--sdist/--wheel/--profile
base|api|cpu/--expected-managed/--receipt e --archives-only. Dieci hash/byte in
sorgenti/sdist/wheel/RECORD/installazione,Name/Version/entrypoints,direct_url
non editable;tar/zip path sicuri e digest RECORD decodificati. Legge il server
in base senza import FastAPI/Marker. stdlib non installa,costruisce,scarica o
ripara. Origine managed/sys.base_prefix/stdlib esatta prima import applicativo.

Definisci I/E schema1 e legame clone corrente↔S↔files/artifacts snapshot.
Ricevuta vecchia o mismatch deve fermare prima collection/import. Negativi reali
D2/D3 futuri obbligatori:modulo installato stantio;S vecchio vsclone-probe nuovo;
receipt vecchia vsS nuovo;ID corretto ma SHA/file/input diverso;positivo corrente
e ripristino dieci hash/metadata/lock. Config-only e flag sono due casi indipendenti,
cache-probe popolata e solo exceptions.py,copie identificate diverse dal clone.
Non eseguirli nella preparazione né considerarli chiusi da mock/test puri.

Prima di modificare un diagnostico conserva la versione baseline s005 con hash
come dato storico identificato,fuori dalla wheel; il nuovo codice resta visibile
nel repository. Mantieni guardie D4/TMPDIR/blacklist e run_baseline.py; non toccare
fixture/algoritmi per ottenere coincidenza. Un diverso scope origin/preflight
richiede modifica motivata del solo diagnostico pertinente e nuovi target/R.
Nessun PASS R/S/V0 baseline ammette nuovo prodotto/interprete/deps/diagnostici.

## P3 — Avvio workspace, dotenv e figli isolati

Moduli autorizzati:config.py,master_workflow.py,unified_converter.py per il delta
minimo del piano. workspace=Path.cwd().resolve() catturato prima override;
percorsi relativi/JSON/report/log nel workspace,assoluti conservati,
get_directory_path legacy. Helper condiviso config.py carica solo
workspace/.env con override=False prima apply_env_overrides/costruzione converter;
converter standalone chiama lo stesso helper prima della propria lettura config.
Rimuovi load_dotenv implicito all'import,niente ricerca parent/site-packages.
CLI inizializzata poi override ambiente legacy,shell precede .env.

Figli fasi [sys.executable,'-I','-m',modulo,...],cwd=workspace,stdout/stderr/timeout/
exit e argomenti legacy. Env rimuove PYTHONPATH/PYTHONHOME senza perdere applicativo.
Preflight find_spec/origine in subprocess -I nello stesso interprete,non nel
solo padre pytest. Nessun CWD/*.py o fallback al clone/site-packages copiati.
Da wheel console con env ripulito oppure python -I -m,clone script assoluto fidato
con package sincronizzato nello stesso sys.executable. Prerequisiti Pandoc/
tiktoken anche per --step preservati,suggerimento pip aggiornato a uv pertinente.
Nessuna nuova feature/FastAPI/algoritmo normalizzazione regex o promessa config
--show/health/readiness non prevista dal legacy.

Le prove reali CLI/fasi/origini/.env/precedenza/sentinelle/V3–V5 sono future dopo
S/B/I e ambienti pronti,non avviare l'app nella preparazione. P4 test integrazione
per i cinque fail non va anticipato con skip/xfail; se necessario prepara soli
casi di regressione del nuovo avvio/diagnostico chiaramente nominati,senza
rieseguire suite/conftest. Tutti i quattro moduli legacy test immutati qui.

## Verifiche pure e inventari prima della request

Ammessi AST,tomllib/JSON,lettura sorgenti/diff/link/help informativi e test stdlib
puri dei nuovi diagnostici con dati sintetici/memoria. Test visibili alla review,
comando bootstrap assoluto -I -B,classi nominate; nessuna pytest/conftest/app/
Pandoc/tokenizer/rete/runner,nessuna falsa esecuzione S/B/I/E. Test significativi
per mismatch,path/archivi/RECORD/ID e mancata presenza,non doppioni della sola
implementazione. Non rieseguire i23 test baseline invariati per abitudine.

Prima del delta conserva report/checkpoint e sorgenti/diagnostici/metadata pertinenti
ricevuti in evidence/implementation-r001/preparation-package-s001/ con hash.
Tieni originali baseline e nuovi sorgenti/fixture/output separati. Baseline
pyenv3.12.3/17dipendenze+pip e /tmp/a001-uv-baseline-pi8cvs6x preservati interamente.
La sua S rimane storica e non verrà ricalcolata contro il clone prodotto mutato.
Confronti futuri distinguono fonte→legacy,legacy→nuovo prodotto e origini/interpreti/
constraints/tokenizer/Pandoc; differenza nuova/incerta FAIL al supervisore.

Leggi solo stat/hash pertinenti di interpreti/binari/venv/cache/policy/daemon,
config uv utente/sistema/XDG/progetto senza contenuti/segreti. Host read tramite
exec_command require_escalated con justification specifica/auto-review per comando,
nessun socket,namespace o modifica host. /proc limitata ai processi propri,
spazio/disco reali e timestamp,binari/UID/gate D4 inventariati. Non leggere .env
reale o dati privati; nessuna scansione ricorsiva indiscriminata. Root/tmp/devices
vanno distinti nelle stime:spazio baseline /tmp finale274366464byte non garantisce
costo managed/cache/build. Nessuna pulizia implicita o modifica rete/AppArmor/sysctl/
setuid/privilegi/profili. Rifiuto tool:registra azione/motivo,IMPEDITA,nessun canale
alternativo; runner disponibile non prova nuovi input. Futuri comandi R non
interattivi e per singolo passo,stesso host UID1000/Firejail,tutti gate padre/figlio.

Conserva costi futuri leggeri e pesanti separati. Universal lock metadata e backend
possono richiedere rete:scope concreto e sdist requirements/pin prima di chiedere
il freeze; se non determinabili,richiesta esplicita d'inventario operativo limitato,
nessuna acquisizione tacita o backend mobile. V7/V8 ML/native/GB e Docker fuori
mandato attuale; V10/V11,pesi/font/inferenza/cloud/deploy/remoto esclusi.

## Richiesta D1 e consegna

Produci implementation/stages/impl-r001-stage-package-s001/request.json schema1:
run/revisione/stage/label/status/autorità; ingresso/context/plan/arbitration/prompts;
Git reale; input esistenti con byte/hash o assenza esplicita; scope/comandi/argv/CWD/
profili/tool/timeouts/costi; outputs futuri separati; invalidazioni e fasi ammesse.
Solo artefatti esistenti relativi alla radice e confinati alla run,niente symlink/
.. o path assoluti in artifacts. Inventari host possono descrivere path assoluti
ma il loro file è confinato. Lista exact senza duplicati,request inclusa senza
self-hash; piano/arbitrato/prompt04/15/copie/inventari/fixture/prove ricevute.

Report/checkpoint propri e sei metadata supervisore escludi dagli input stabili;
conserva copie,non congelare registri mutabili. Non includere S/B/I/E/lock/venv/
receipt future o SHA del futuro package snapshot. Un uv.lock ancora assente è
assenza dichiarata,non errore mascherato. Etichetta stage subphase PREPARATION_INPUTS:
le acquisizioni operative future modificano input e richiedono label successiva
prima della catena finale. Non chiamare tale stato package installato.

Aggiorna solo report/checkpoint propri: file modificati,motivazioni P1/P2/P3,
AST/test puri e limiti,risultati veri,hash before/after,input/inventari/costi/
futureargv,processi e prossima request. Alla consegna CHANGELOG del lavoro reale
rendere storico il contesto per i delta autorizzati identificati; nessun metadata
condiviso dell'implementatore. Checkpoint **WAITING_FOR_STAGE_SNAPSHOT** con path
preciso. Consegna al supervisore con [prompt05](05-supervisor-stage-r001.md).

Se non puoi completare un input/flag/costo,documenta impedimento concreto nel
checkpoint; non inventare la request valida o un lock e non avviare altro stage.
Nessun auto-freeze/retry/runtime. Successiva chat operativa solo dopo freeze/
nuovo prompt. V1–V9,S/B/I/E/negativi reali e review codice/arbitrato finale ancora
obbligatori; V7/V8 futuro costo distinto,V10/V11 non autorizzate. Commit/merge/
push/promozione manuali utente dopo GO; nessun deploy o invio remoto.
Temp,venv/cache e workspace reali non viaggiano con Git.
