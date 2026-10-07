# Ripresa implementativa — diagnostici V0 e preparazione baseline s004

Agisci come **nuova chat implementatrice** della run `run-a001-fase0-uv` in
`/home/davide/workarea/markdown-for-llms`, continuando il mandato04. Il supervisore
ha ricevuto R/S positivi e V0 FAIL s003 e dispone questa preparazione concreta.
**Adesso solo correzioni diagnostiche, verifiche pure e request s004.**
Nessuna R/S/V0, suite legacy/collection o conversione prima del freeze D1 e
del nuovo prompt del supervisore. Non assumere il suo ruolo, creare snapshot,
subagenti o review simulate. P1–P3 restano in attesa.

## Letture e ingresso

1. AGENTS, CHANGELOG, temp/PROJECT-CONTEXT e temp/HANDOVER; STATE/HANDOVER run.
   Skill manage-implementation-run, protocollo run-lifecycle e template ruolo;
   verify-conversion-fidelity per audit dei dati conservati, senza autorizzare runtime.
2. Brief A1–A7, piano r003 **integrale** se chat nuova, arbitrato r003 D1–D5,
   prompt04 e precedenti NO_GO r001/r002. Recupera ogni intervallo troncato.
3. Report/checkpoint propri correnti, impediment-r003 dello stage s003,
   sources.json S-s003, receipt/inside/runner dei quattro passi, receipt V0,
   audit-v0/content-audit e source-checks/runner-analysis/handoff-checks in
   evidence/implementation-r001/resume-baseline-s003. Inventari grandi per
   struttura/campi, senza omissioni pertinenti. Prompt10 è sequenza eseguita.
4. In evidence/supervisor-implementation-r001/baseline-recovery-r001 leggi
   **decision.md**, reception.json, content-verification.json, sources.json,
   proposed-commands-baseline-s004.json e response.json; confronta le copie
   report/handover/driver ricevuti. Leggi handovers/supervisor-baseline-recovery-r001.md.
   Poi soli sorgenti/fixture/help primari pertinenti.

Percorsi dei punti2–4 relativi alla run salvo prefisso. Fonti web non sono
esiti del supervisore. Identità/autore degli output è dichiarata, non provata
dagli hash. Questa chat implementa solo il delta assegnato, senza terza review.

| Oggetto immutabile | SHA-256 |
| --- | --- |
| plans/plan-r003.md | 462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b |
| arbitrations/arbitration-plan-r003.md | f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d |
| prompts/04-implementation-r001.md | fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035 |
| snapshots/impl-r001-stage-baseline-s003.json | f518b327c4aebe8b0a3c29ec9c5aa153e95c78d28fb63970b0040be0f5b1d32a |
| implementation/stages/impl-r001-stage-baseline-s003/impediment-r003.json | d3925c5111175a2089ef53870e06429128c732509c2ccfa9f3d0d599ae718015 |
| implementation/stages/impl-r001-stage-baseline-s003/sources.json | b9109b36ec4760b110eb6972010aa868b1a288591557f0c5c71862f0451d6551 |
| scripts/diagnostics/run-a001-fase0-uv/run_baseline.py ricevuto | 6707138d05294ffd5a2de186aad7c25db25d10abf2c0f9edbf540c8c8c01073d |
| scripts/diagnostics/run-a001-fase0-uv/run_offline.py | a937e0067f81f9d09c2fec49f4b47de00eb9a8b254b64d75caed1029b3c71018 |

Contesto ingresso **snapshots/baseline-recovery-context-r001.json**.
SHA manifest/worktree/lista esatta in response.json prodotta dopo snapshot
per evitare autoreferenzialità. Confrontali e richiedi MATCH **prima** delle modifiche:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
git diff --cached --name-only
python3 scripts/run_context.py verify run-a001-fase0-uv --label baseline-recovery-context-r001
```

Branch feature/run-a001-uv, HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto; sei metadata documentali e17nuovi file preparatori, nessun prodotto
migrato. S003 storico: alla ricezione solo CHANGELOG, poi sei metadata del
supervisore identificati in transition.json,181artefatti invariati. Non
pretendere MATCH storico né rigenerare s003. Il contesto di recupero non è
uno stage baseline e diverrà storico dopo il delta autorizzato; registra la
transizione. Request/snapshot s004 devono ancora essere prodotti nei ruoli corretti.

## Risultati acquisiti e correzioni assegnate

R s003 completa PASS quattro volte, S schema1 generato/verificato, V0 exit2/FAIL.
S ID sha256:9b3b15edf3a698426f2b003aa01964a34bd1ca5c2539ebf230b2e10687a5f889.
R non è IMPEDITA s003 e non passa automaticamente a s004; sette test preparatori
precedenti distinti. Suite4file distinta exit1:62pass/5fail,67eseguiti/0skip/error;
cinque integrazioni lookup script CWD. Nessun V0 PASS retroattivo.

Preserva request/manifest/receipt/log/output s001–s003 e gli originali.
Prima di aggiornare report/checkpoint salvali in nuovi file propri, ad esempio
evidence/implementation-r001/preparation-baseline-s004/report-before-preparation.md
e handover-before-preparation.md, confrontandoli con le copie ricevute.
Conserva driver prima-delta con SHA sopra. Nessun file storico corretto.

È autorizzato modificare **run_baseline.py** e **tests/unit/test_uv_diagnostics.py**
soltanto per i tre punti sotto. Helpers stdlib nello stesso driver; nessuna
nuova dipendenza, modulo di prodotto o nuova architettura. Se serve un ulteriore
delta oltre questi file, descrivilo e riconsegna prima di eseguirlo.

1. **IMPL-V0-001 F4.** Due marker escaped conservati nei contesti richiamo e
   bibliografia. Sostituisci l'invariante letterale con verifica scoped di
   quantità/identità/posizioni/contesti/ordine dei riferimenti sorgente-output.
   Ammesso riconoscere `[1]` oppure `\[1\]` nel testo F4 pertinente, con
   scansione esatta/dati del fixture; nessun replace/unescape globale del Markdown,
   codice/LaTeX/link o modifica del documento salvato. Non accettare due marker
   spostati entrambi nella bibliografia o un riferimento diverso.
   Mantieni summary failed0, numeri, Unicode, formula, tabella e due sezioni;
   rafforza il controllo ordine con esistenza di entrambe prima del confronto
   degli indici. L'anchor HTML omesso rimane perdita legacy esplicita.
2. **IMPL-V0-002 sliding.** Durante il futuro sliding-child dentro R usa una
   sola tokenizzazione del sorgente identificato cl100k_base e i parametri1000/100
   del chunker originale. Deriva finestre dal passo e lunghezza, controlla numero,
   ordine/corpo integrale vs decode della slice con la sola `.strip()` già
   applicata dall'originale, e registra i margini esclusi. Calcola ogni intersezione
   sugli indici/ID originali, verifica gli ID reali e **anche il testo effettivo**
   condiviso suffix/prefix e la sua posizione nel sorgente. Identifica byte/char
   offset, copertura continua e fine testo; gap o testo/ID alterato deve fallire.
   Per questa F6 ASCII i confini sono univoci; non generalizzare a UTF-8 arbitrario
   o mascherare U+FFFD. Le API bytes/offsets del tiktoken installato sono leggibili
   ora, eseguibili solo dopo freeze/R. Non dichiarare100token da metadata senza
   verificare stringhe/finestre. Richiedi >=2chunk e tutti confini osservabili>0.
   Mantieni i vecchi overlap dopo ritokenizzazione come misura separata che può
   essere0, senza gate di equivalenza non previsto dal piano. Zero **intersezione
   effettiva**, ordine errato, perdita/duplicazione o difformità resta FAIL.
   Semantic overlap0 legacy e fixture/algoritmo rimangono invariati.
3. **SUP-V0-003 provenance suite.** Prepara nel driver, per la futura V0 e
   dopo R/S preflight, collection esplicita degli stessi quattro file seguita
   dalla loro esecuzione con log per-nodeid observed. Non includere test nuovi
   diagnostici nel conteggio legacy né cambiare CWD/test/conftest/fixture.
   Conserva config/plugin/versioni/argomenti/exit/rootdir e due cache nuove
   **separate** collection/run sotto il nuovo workspace esclusivo, con
   `-o cache_dir=<path-assoluto>` e assenza/inventario prima/dopo. Non cancellare
   la cache reale del repo né riusare quella s003. Collection `--collect-only -q`,
   test `-v -rA` o logging equivalente con mapping completo observed; configura
   la verbosità effettiva senza cambiare selezione/addopts del progetto.
   Riconcilia lista collected, esiti per-nodeid, summary/cache fresh e exit;
   identità passed non inferite dal solo complemento cache. Attesi gli stessi
   67nodeid e cinque failed identificati nella decisione/evidenza; enumera i
   62PASS reali nuovi e conserva tutti cinque FAIL/cause lookup e negativo
   orchestratore exit1. Nessun xfail/skip/deselect/riparazione dei test/sorgenti,
   cambio CWD o copia clean_markdown.py nel workspace per farli passare.
   Esito/mapping inatteso, skip/error o prove incomplete devono bloccare la
   caratterizzazione V0 e tornare al supervisore. Se tutto documenta fedelmente
   la baseline, V0 può caratterizzarla mantenendo **legacy_suite FAIL/exit1**;
   nessun PASS trasferito alla suite o alla futura wheel.

## Verifiche preparatorie ammesse

Sintassi/diff/hash e test **stdlib puri** dei nuovi helper su dati conservati
o sintetici: riferimenti escaped positivi e marker mancante/diverso/fuori
contesto negativi; finestre/tokenID/testo corretti e ID/corpo/intersezione/
copertura/ordine difformi negativi; parser/nodeid di log con esiti mancanti,
duplicati, cache stale o conteggio incoerente rifiutati. Sono controlli del
diagnostico, non V0/R o nuovi contenuti convertiti. Se usi l'entrypoint unittest
del file test, invocalo diretto con Python -I -B per evitare bytecode, collection pytest e
conftest. Non importare chunk_markdown/tiktoken, avviare Firejail/unshare,
socket/netlink, Pandoc o suite legacy. Non invocare sliding-child/driver run.
Conserva dati/hash/errori/exit/stdout/stderr; nessun golden adattato all'output
nuovo. Verifica anche il caso positivo reale s003 leggendo gli output intatti.

Originali10/buildmetadata4/fixture F1–F6/provenienza, wrapper/check_runner,
producer S, deps/interprete/uv/Pandoc/cache/policy host rimangono invariati.
Tutti i PASS R/S s003 sono storia riferita a s003. Nessun runtime nuovo ora,
nessun allentamento dei gate D2/D4 o confronto contenuti rimosso.

## Preparazione, request e handoff s004

Preserva /tmp/a001-uv-baseline-pi8cvs6x: venv/tmp/workspace/cache. Zero acquisizioni
previste, nessuna reinstallazione o pulizia dei workspace storici. Prepara nuovi
target-bootstrap/target-baseline/baseline-inputs e inventari in
evidence/implementation-r001/preparation-baseline-s004/, riferiti agli hash
nuovi driver/test e ai prerequisiti immutati; verifica clone/copia/HEAD,
S-s003 storico e le assenze pertinenti secondo schema reale (non inventare
conteggi storici). Scope/hash/assenze esatti nella request, byte ignoti vecchi
restano ignoti. Registra tutti i delta e l'equivalenza limitata scelta Firejail
dopo negativo unshare s002, non PASS ereditati.

Puoi riconfermare host/tool/policy/spazio mediante **sola lettura** nel contesto
exec_command require_escalated UID/GID1000, justification specifica e review
automatica per ogni comando, senza prefix_rule ampia. Stat/hash/versioni/route
già esposte e path daemon noti/alias/XDG, nessun connect/socket/netns applicativo,
segreto o modifica host. R s004 non è provata da tale inventario. Rifiuto review
→ registra azione/motivo e consegna al supervisore, senza altro terminale/host
per aggirarlo. Nessun sudo/nuovo privilegio/setuid/profilo/policy/rete/permessi.

Usa i quattro argv **solo proposti** in proposed-commands-baseline-s004.json:
stesso Firejail/wrapper/TMPDIR/gate, nuovi target/stage/output S/receipt/workspace.
Verifica pura delle differenze s003→s004 e dipendenze; costi/timeouts/stime
aggiornati, nessuno di questi argv eseguibile ora. Predisponi collection/log
interni nuovi nell'harness, dichiarandoli nella request. Non creare S o altri
output futuri in anticipo e non riusare S-s003 per ammettere driver nuovo.

Crea **request reale dell'implementatore**, schema1 D1,
implementation/stages/impl-r001-stage-baseline-s004/request.json,
WAITING_FOR_STAGE_SNAPSHOT, con autore/run/revision/stage/label/Git:

- files/hash/byte/assenze, input esterni e target, diff/matrice test nuovi,
  prove pure/costi, argv/cwd/timeout/output/profile/invalidazioni esatti;
- artifacts_to_freeze confined relativi: piano/arbitrato/mandato04/prompt11,
  request nuova senza self-hash, originali/inventari/target/diagnostici visibili,
  decisione/fonti/argv del supervisore, contesto di recupero e s003/request/
  impedimento/S/receipt/log/audit/copie esistenti con hash verificati;
- escludi report/checkpoint mutabili, response/checks futuri e output futuri,
  nessun digest del medesimo snapshot in S; input → snapshot → S → B/I/E;
- tutti i comandi futuri may_execute_now=false; R/S/V0 s004 NON_ESEGUITI,
  nuova R completa obbligatoria, S nuovo dopo snapshot, nessun PASS trasferito.

Aggiorna solo report/checkpoint propri e CHANGELOG per il lavoro effettivo
prima della consegna. Non STATE/HANDOVER comuni/indici/eventi/ADR. Checkpoint
WAITING_FOR_STAGE_SNAPSHOT, path request s004/hash/delta/unità pure/input/
processi e limiti. Consegna al supervisore tramite prompt05; non è servizio
continuo e non creare snapshot da solo. Se manca input o serve un delta diverso,
documenta e riconsegna senza prove dipendenti.

Dopo freeze e nuovo prompt soltanto: R-bootstrap completo → R-baseline
completo → nuovo S → V0/audit/suite identificata. D2 mismatch prima import/
collection FAIL, D4 gate incompleto IMPEDITA, receipt V0 FAIL resta FAIL.
Ogni input/tuning/retry successivo richiede s005 o label successiva.
P1–P3 solo dopo baseline adeguata, nel successivo mandato; niente pin/lock/
pyproject/install/build/download ora. GO piano r003/NO_GO antecedenti conservati,
nessuna r004 e nessun GO codice. V7/V8 future obbligatorie/costo separato;
V10/V11/pesi/font/inferenza esclusi. Review codice/arbitrato finale ancora
necessari; Git manuale utente, nessun deploy/invio remoto. Temp/ambienti/lavoro
non committato non viaggiano con Git; preservare separatamente.
