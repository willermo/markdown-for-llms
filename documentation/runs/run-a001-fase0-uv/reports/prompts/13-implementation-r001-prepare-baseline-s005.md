# Preparazione implementativa r001 — baseline s005 — run-a001-fase0-uv

Agisci come **implementatore** in una nuova chat nel repository
`/home/davide/workarea/markdown-for-llms`. Riprendi il checkpoint reale
`handovers/implementation-r001.md`, attualmente FAIL_V0 /
WAITING_FOR_SUPERVISOR_DISPOSITION s004. Il supervisore ha accolto IMPL-V0-004
e assegna **sola preparazione s005**. Stato comune IMPLEMENTATION /
BASELINE_V0_FAIL / WAITING_FOR_STAGE_PREPARATION. Nessun GO codice.

R/S s004 sono positivi storici; **V0 s004 resta FAIL/exit2, suite FAIL/exit1**
(67 nodeid/201 eventi osservati, 62 pass/5 fail). Non trasformare la receipt
fallita o l'audit in PASS V0. Request/stage/S/R/V0 s005 ancora assenti.
Nessun retry nello stage vecchio, packaging P1–P3 o fase dalla roadmap.

Chat distinta dal supervisore, nessun subagente o review simulata. Non sei il
supervisore: scrivi diagnostici ammessi, report/checkpoint propri e request
reale, non snapshot/arbitrati/STATE/HANDOVER/eventi/indici/ADR comuni. Il
supervisore non è un servizio continuo; consegna con prompt05 alla fine.

## Letture e identità

1. AGENTS, CHANGELOG, PROJECT-CONTEXT/HANDOVER globali e STATE/HANDOVER della run;
   manage-implementation-run, protocollo e verify-conversion-fidelity.
2. Brief A1–A7, piano r003 integrale se chat fresca, arbitrato r003 D1–D5 e
   prompt04. GO r003 invariato, NO_GO r001/r002 e 24 disposizioni vincolanti.
3. Prompt11 preparazione e prompt12 runtime completati; checkpoint/report
   propri, impediment-r004 s004, audit.md e final-handoff-checks in
   evidence/implementation-r001/resume-baseline-s004/. Non confondere issue
   r004 con una nuova revisione del piano, che non è stata disposta.
4. [Disposizione IMPL-V0-004](../../evidence/supervisor-implementation-r001/baseline-recovery-r002/decision.md),
   reception.json, suite-observations.json, content-observations.json, sources.json,
   metadata-transition.json, transition.json, proposed-commands-baseline-s005.json
   e snapshot-verify.json nella stessa cartella;
   [risposta supervisore](../../evidence/supervisor-implementation-r001/baseline-recovery-r002/response.json)
   e [checkpoint](../handovers/supervisor-baseline-recovery-r002.md).
5. Manifest baseline-recovery-context-r002, snapshot/request/S s004, dati raw
   legacy-collection-observed/legacy-run-observed e stdout/stderr in
   evidence/implementation-r001/baseline-results-s004/. Driver/test diagnostici
   integrali e sorgente pytest installata pertinente, con hash dalle fonti.
   Inventari grandi strutturalmente, intervalli troncati da recuperare; non tutta temp.

Percorsi senza prefisso relativi a `temp/run-a001-fase0-uv/`.

| Oggetto | SHA-256 |
| --- | --- |
| plans/plan-r003.md | `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b` |
| arbitrations/arbitration-plan-r003.md | `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d` |
| prompts/04-implementation-r001.md | `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035` |
| prompts/12-implementation-r001-baseline-s004.md | `9f46db8fa9a8377506549d0ad6f05c5440be520f4fb5e51a8eff3d229eb6d073` |
| snapshots/baseline-recovery-context-r002.json | `81d6f349860bf797c0dcad3a828cf370398cd370025ffeefead2e9187f7ccff3` |
| snapshots/impl-r001-stage-baseline-s004.json | `0540c6a2a9e1bd2acf94710cda39a7f388a0e55d6200cc5bbf35526aa781e7ae` |
| implementation/stages/impl-r001-stage-baseline-s004/request.json | `3ce55c0532c925443e68cdb429a7c22f9cef59a702102cc7213af17170d37e19` |
| implementation/stages/impl-r001-stage-baseline-s004/impediment-r004.json | `ad88de8bec0ee08790061d379c5f1423aaf6058afdcd78313291296870cbd615` |
| implementation/stages/impl-r001-stage-baseline-s004/sources.json | `85a3b5b9b621c4817ba2921bd2bba48ab8bb0e732e4dcc0ea586edae07cc06a5` |
| evidence/supervisor-implementation-r001/baseline-recovery-r002/decision.md | `4b79ff17adf7d333ac252ed35127e93eb1a769e11c3c67a0c30de76341745ad6` |
| evidence/supervisor-implementation-r001/baseline-recovery-r002/sources.json | `0a2ed545e903e8f7c3bd02a5661d006f9473de63f29408a28e2cb591acd0cbb3` |
| evidence/supervisor-implementation-r001/baseline-recovery-r002/proposed-commands-baseline-s005.json | `b6dacd938bd5839404111c6d918f6c259f3de29648bbb5ea3ad35d78bbab8a17` |
| scripts/diagnostics/run-a001-fase0-uv/run_baseline.py | `bf7fd085fade82061da92d24c86f69831a7c9d13b624c89fd1443adeffb2b022` |
| tests/unit/test_uv_diagnostics.py | `1a4853dde8da76197c6403ed8377f1e910969286224659744704ac95bb29ad81` |

Contesto **baseline-recovery-context-r002 MATCH: 96 file + 718 artefatti**,
worktree `5f3b51d5d532c43275c0f4a81de911f16422d84a87ca3086ebfcebed240428aa`; è l'ingresso della preparazione,
**non lo snapshot stage s005**. Prompt13/risposta/checkpoint/checks successivi
esclusi da quel contesto, identità finale nella risposta. Branch
`feature/run-a001-uv`, HEAD/dev/merge-base
`66ba82200e5def5a4db76f9bafccb0731b506091`, indice vuoto. Sei metadata documentali
modificati e 17 file preparatori nuovi; prodotto/pyproject/lock non migrati.

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
git diff --cached --name-only
python3 scripts/run_context.py verify run-a001-fase0-uv --label baseline-recovery-context-r002
```

Conferma identità/MATCH prima dei delta. S004 è storico alla ricezione per
sola modifica CHANGELOG e ora per sei metadata supervisore; nessuna modifica
tecnica. I suoi 455 artefatti e antecedenti sono immutati. Dopo i delta ammessi
questo contesto diventerà storico per driver/test e CHANGELOG alla consegna,
con transizione precisa e 718 artefatti invariati. Non rigenerare snapshot
vecchi o chiamare MATCH un oggetto nuovo con la loro label. Preserva prima
le versioni proprie correnti; copie report/checkpoint/driver/changelog ricevuti
sono nella cartella del supervisore. I riferimenti all'output CHANGELOG s004
si verificano sulla copia ricevuta dopo l'aggiornamento metadata supervisore.

## Compito: confronto configurazione, senza attenuare i gate

Il difetto è confermato nei raw s004 e nella sorgente installata pytest9.1.1:
inicfg include gli override -o. Collection contiene il proprio cache_dir;
run contiene il proprio cache_dir e verbosity_test_cases="1". Questi valori
erano prescritti, ma l'equality integrale li rifiuta. Gli altri campi comuni
confrontati coincidono. I test suite precedenti fornivano solo plugins e
pytest_version, senza inicfg: questo caso reale non era coperto.

Modifica soltanto `scripts/diagnostics/run-a001-fase0-uv/run_baseline.py` e
`tests/unit/test_uv_diagnostics.py`, limitatamente al confronto config e ai
test necessari. Helper stdlib puri ammessi nel driver. Mantieni runner,
preflight/S, algoritmo, fixture, producer, deps/cache, test legacy, campi
output raw, F4/sliding e relativi gate/test invariati. Nessuna correzione
del prodotto o nuova funzionalità. Se occorre estendere il perimetro,
consegna evidenza al supervisore prima di procedere su quella parte.

Separa confronto comune e validazione esatta per fase; conserva gli observed
raw senza modificarli o fabbricarli. Non basta ignorare inicfg, eliminare
qualsiasi chiave diversa o usare get(key)==get(key) quando mancano campi.
Campi obbligatori mancanti/tipo errato devono produrre FAIL; null esplicito
per configfile/record attualmente assenti è distinto dal campo non registrato.

Restano comuni rootdir/configfile/configfile_record/hash/addopts/effective_args/
pytest_version/plugin_distributions e tutte le chiavi comuni di inicfg.
Solo cache_dir e verbosity_test_cases possono essere separati **dopo** avere
verificato esattamente chiave, valore, fase, path, argv e config effettiva.
Per la baseline attuale il comune inicfg è vuoto, configfile/record null,
addopts [], effective_args i quattro file. Non introdurre config del clone
o override ambientali; le nuove osservazioni dovranno confermare questi dati.

| Controllo per fase | Collection | Run |
| --- | --- | --- |
| cache_dir in inicfg/config/cache_before/cache_after | cwd/pytest-cache-collection assoluto | cwd/pytest-cache-run assoluto |
| -o verbosity_test_cases | assente in argv e inicfg attuali | esattamente una volta, valore "1" |
| verbosity globale/test_cases | -1/-1 | 1/1 |
| display flags | --collect-only -q --verbosity=-1 | -v -rA --verbosity=1 |
| pytest_main argv completo | quattro file + display + -o cache_dir | quattro file + display + -o verbosity_test_cases=1 + -o cache_dir |

Deriva repo/CWD/cache attesi dai parametri della propria invocation, non da
un path raw che si auto-dichiara valido. Le cache sono sotto legacy-suite del
workspace della nuova label. Rifiuta cache uguali/scambiate/esterne/stale,
override extra/duplicati/confliggenti, valore/fase errati o mancanti, differenze
comuni inattese, invocation/effective config discordanti. Mantieni selezione
esatta dei quattro file e nessun addopts che la alteri. I nomi plugin numerici
automatici sono ID di processi distinti: conservarli raw, senza pretendere
uguaglianza di tali ID; controllo distribuzioni/versioni già richiesto invariato.

La correzione deve mantenere tutti i gate esistenti: 67 nodeid unici raccolti
ed eseguiti; 201 report setup/call/teardown; setup/teardown PASS; 62 call PASS
e cinque call FAIL con cause ENOENT/AssertionError individuali; zero extra,
duplicati, mancanti, deselected, skip/xfail/errori. Cache fresh e summary/exit
coerenti. Non cambiare questi requisiti o accettare il solo complemento dei fail.
Il risultato della caratterizzazione conserva **suite FAIL/exit1**; un test
puro o replay corretto non è PASS V0, né PASS della suite o della migrazione.

## Verifica puramente preparatoria

Test stdlib con config completa di forma realistica, costruita in memoria
nei test, senza dipendenza obbligatoria da temp per la suite futura. Positivo
per gli override esatti sopra e config comune coerente. Negativi per campi
mancanti/null impropri/tipi, override inattesi/duplicati/mancanti/valore sbagliato,
cache/path/fase/argv discordanti, config comune/versioni/hash/selection mutati.
Mantieni i negativi precedenti per nodeid, eventi, cause, skip/errori/cache e
i test F4/sliding. Un test con config comune diversa non deve passare soltanto
perché le due cache sono diverse. Non fissare un numero di test per simulare esiti.

Audit puro separato dei due observed JSON reali s004 conservati: leggi i dati
immutati, invoca solo la funzione stdlib di confronto/riconciliazione, registra
hash input, parametri, confronto prima/dopo e risultato. Non chiamare run(),
producer S, preflight runtime, collection o applicazione. Usa il CWD/repo s004
realmente dichiarati nel replay, non trasforma i dati in s005. Il risultato
deve restare FAIL/exit1 della suite con caratterizzazione completa, senza
riscrivere receipt/log/raw e senza dichiarare V0 s004 ora PASS. Errori del solo
lettore/pure test preservati con codice/log, non nascosti o dati corretti ad hoc.

Esegui solo classi di unità pure con Python bootstrap `-I -B`, direttamente
via unittest; syntax AST e argv unitari espliciti. Nessun import pytest,
conftest, pipeline, tokenizer, Pandoc o modello; niente collection/suite
legacy o R. Leggere file/cache pubblica in stdlib è audit dei dati, non
tokenizzazione nuova. Conserva diff e hash del driver/test prima/dopo.

## Inventari, target e richiesta reale s005

1. Preserva report/checkpoint ricevuti e versioni dei due diagnostici prima
   del delta, in nuova cartella
   `evidence/implementation-r001/preparation-baseline-s005/`. Non correggere
   cartelle r001–r004 o le receipt storiche; owned output nuovi esclusivi.
2. Conserva `/tmp/a001-uv-baseline-pi8cvs6x`: venv/tmp/workspace/uv-cache/tiktoken-cache.
   Bootstrap `/home/davide/.pyenv/versions/3.12.3/bin/python3`, baseline
   `/tmp/a001-uv-baseline-pi8cvs6x/venv/bin/python`, Python3.12.3 e deps/cache
   già presenti. Host solo read/stat/hash tramite profilo require_escalated
   con justification specifica e review automatica, se serve la sua osservazione;
   nessun socket/namespace/app/probe/install/fetch/build/sync adesso.
3. Nuovi runner-target-bootstrap/baseline, baseline-inputs, input-inventory,
   transizione rispetto a s004 e inventario host pertinente. Ricalcola tutti
   hash attuali/copie/HEAD, fixture/cache/absenze; non ricopiare un PASS storico.
   Target aggiornano hash driver/test e gli eventuali nuovi input puri dichiarati,
   preservano guardie/binari/daemon/TMPDIR/interpreti/deps/assenze/clean_env.
   Report/checkpoint mutabili e sei metadata supervisore esclusi dagli input
   stabili; conservane copie. Hash/byte mancanti storici non retro-riempiti.
4. Quattro argv esterni futuri in
   [proposed-commands-baseline-s005.json](../../evidence/supervisor-implementation-r001/baseline-recovery-r002/proposed-commands-baseline-s005.json):
   solo sostituzione s004→s005 degli argv/cwd/output/timeouts; Firejail/TMPDIR/
   blacklist e step_tool_profiles invariati. Confrontali esattamente, mantieni
   may_execute_now=false. Due argv suite interni stesso schema/fasi/timeout300
   con CWD/cache s005; conserva argc/flag exact, non cambia i file selezionati.
5. Inventaria output futuri assenti: snapshot/S s005, quattro runner directory,
   baseline-results-s005, workspace v0-s005, cache collection/run e observed/
   log futuri. Nessuna directory output R o placeholder receipt prima del freeze.
   S s004 resta storia; S s005 nascerà soltanto dopo snapshot e R completi.
6. Prepara `implementation/stages/impl-r001-stage-baseline-s005/request.json`
   schema1 D1, **WAITING_FOR_STAGE_SNAPSHOT**. Run/revision/label/stage/author,
   Git, authority e contesto attuali, elenco input/assenze con hash/byte,
   artefatti confinati alla run, target/argv/profili, costi/limiti, future assenze
   e invalidazioni. Include piano/arbitrato/prompt13, richiesta stessa senza
   self-hash, inventari/fixture/copie originali, evidenze s004 e pure preparatorie.
   Niente path evasivi/assoluti/symlink nell'elenco artifacts, output futuri,
   snapshot proprio o digest futuro, report/checkpoint mutabili o self-hash.
   Non dichiarare la request già congelata: il proprietario del freeze è il supervisore.

Stesso host UID1000 e Firejail esistente 0.9.72. Negativo unshare s002 equivalente
solo per scelta fallback con binario/ramo/host/path invariati; R PASS s004
non ammette driver s005. Niente sudo/nuovo privilegio/setuid/profilo persistente,
modifica sysctl/AppArmor/rete/permessi/socket/daemon host. Per future R, sole
connect/close D4 senza API e sintetico proprio esclusivo/socketpair/IP/gate
completi, padre/figlio, stessa invocation programma. Tutto è ancora futuro.
Rifiuto automatico o R incompleta → IMPEDITA e supervisore, nessun terminale/
host/canale alternativo o attenuazione dei gate. S/input mismatch → FAIL prima
di import/collection, niente vecchie receipt ammesse. No prefix_rule ampia.

## Costi, limiti e consegna

Spazio autore finale s004 288.968.704 byte alle15:57:24UTC, non misura futura:
ricontrolla spazio in preparazione, nessuna pulizia implicita di cache/venv/
workspace/evidenze. Rete/acquisizioni previste zero; output stimati 50 MB,
R/S due minuti, V0/audit 35 minuti (audit manuale fuori timeout wrapper1.200s).
Distinguere stima/misura. Nessun runtime pesante implicito nella preparazione.

Contenuti s004: F4 contestuale, sliding16/15 con100ID/326–349caratteri,
51chunk semantic e84Markdown/report identici s003. Sono dati identificati,
non PASS del nuovo driver. Perdite F2/formule/codice/link, F5/asset/PNG downstream,
anchor/separatori e bundle legacy incompleto restano; ASCII sliding soltanto.
Warning overlay/proc campionato/daemon assenti limitano la prova R ai canali
inventariati. S/B/I/E e negativi D2 prodotto ancora futuri.

Aggiorna report/checkpoint propri con delta, pure checks reali, hash/costi/
limiti/owned processi e request s005; consegna via
[prompt05](05-supervisor-stage-r001.md). Changelog solo del lavoro reale alla
consegna, dopo controlli: registra transizione del contesto per i due diagnostici
e CHANGELOG e 718 artefatti intatti; nessuna identità ereditata senza confronto.
Supervisore riceve request reale → metadata prima del nuovo freeze s005 →
nuovo prompt per R-bootstrap/R-baseline/S/V0/audit. **Non eseguire R/S/V0 o
collection/suite adesso, né fare retry della s004.** Input/retry dopo il futuro
freeze richiederanno s006 o successivo. Nessun auto-snapshot/polling.

P1–P3 attendono baseline adeguata e mandato successivo; nessuna r004 del piano
o review simulata. V7/V8 future obbligatorie con costo/perimetro pesante distinto;
V10/V11/pesi/font/inferenza non autorizzati, nessun deploy/invio remoto.
Review codice e arbitrato finale futuri. Commit/merge/push/promozioni spettano
all'utente dopo GO, mai eseguiti da te. Temp e venv/cache esterne non trasferiti
da Git: preserva il lavoro, non soltanto i manifest.
