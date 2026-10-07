# Implementazione r001 — completare il pilota baseline, continuation-c001

Agisci esclusivamente come **implementatore**, nuova chat, repository
`/home/davide/workarea/markdown-for-llms`, skill manage-implementation-run.
Obiettivo concreto: completare install→pip-check→inventory preliminari sul
prefisso seeded identificato, poi consegnare la request ufficiale s007 pronta.
Preparazione/adattamento limitato e operazioni nella stessa chat; nessuna delega,
supervisione/review o aggiornamento dei registri comuni.

## Autorità e ingresso

Leggi AGENTS,protocollo/ADR0007,STATE/HANDOVER comuni, brief A1–A7,piano r003
integrale se non già letto,arbitrato D1–D5,prompt04,addendum r001/r002 e checkpoint
implementatore corrente con completion/request esatte. I ruoli citati non cambiano
il tuo. Prompt29 tecnico resta fonte per gli invarianti; questo mandato sostituisce
solo il blocco di budget e il divieto di prosecuzione dello STOP pre-install ricevuto.

Percorsi relativi a `temp/run-a001-fase0-uv/`:

- `evidence/supervisor-implementation-r001/pilot-baseline-recovery-reception-r001/`:
  reception.json,decision.md,continuation-scope.json,transition.json,response.json,checks.json;
- `snapshots/baseline-recovery-pilot-continuation-context-r001.json`;
- `handovers/supervisor-pilot-baseline-recovery-reception-r001.md`;
- `evidence/implementation-r001/pilot-baseline-recovery-r001/`:delivery/completion,
  admission-install-stop,preserved-target,attempt-001/inputs e receipt/copied output;
- `evidence/implementation-r001/preparation-baseline-install-r003/`:fonti/helper
  ufficiali e73sintetici identificati nelle ricevute, immutabili in questa ripresa;
- originali/fixture/asset s003/report s004/host/config identificati dal parent manifest
  e input-config; r001/r002 falliti solo letture archivistiche, no import/avvio/cleanup.

Piano SHA462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b,
arbitrato f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d,
prompt04 fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035:
immutati, GO piano nei limiti, nessun GO codice. Git feature/run-a001-uv,
HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091, indice vuoto.

Ricalcola manifest SHA/byte contro response e da radice verifica
`python3 scripts/run_context.py verify run-a001-fase0-uv --label baseline-recovery-pilot-continuation-context-r001`.
Prima/dopo ogni tool e alla consegna: contesto e input pertinenti. Vecchio contesto
pilota r001 è storico solo per i sei metadata documentali ricevuti in transition;
4524artefatti intatti. Non chiamare la sua verify come gate operativo contro nuovi
metadata e non rimuovere gate per ottenere MATCH. Nuovo contesto non è stage s007.

## Prosecuzione nuova, prefisso immutabile identificato

Autorità esclusiva: continuation-scope.json congelata dal supervisore. Parent:
manifest attempt-00117584byte/SHAe62d833e5332ce1ce66a21f78dd542a6d11cd34984023056a1dc288097dfe1f1;
vecchia scopeSHA d583cd35227ed22ef032b10689cea85e87aac139fc67e243954d510dc15075bc.
Target `work/baseline-recovery-pilot-r001/attempt-001`, lock adiacente già presenti;
nuova identità `pilot-baseline-recovery-r001-c001`. **Nessun directories/venv/seed**.
La nuova scope è una disposizione di costi e lineage; non modificare vecchi manifest,
scope, helper, ricevute o stima32MiB. Non usare launcher/gate del vecchio contesto
contro nuovo worktree; essi rimangono prove storiche, non un fallback.

Prima di install confronta l'intero target/lock chiuso con preserved-target.json,
1046file byte/SHA/mode, incluso prefisso pip24/495pyc/template/origine/startup/RECORD,
cache/tokenizer, riserva fisica e log. Verifica hash delle due ricevute originali,
exit figli0 e loro immutable_outputs. Install/check/inventory marker/receipt/log
operativi e inventario18dist ancora assenti. Qualunque discrepanza STOP; non
riparare il prefisso. Dopo install i file parent immutabili devono restare identici;
i nuovi file devono appartenere al modello chiuso18dist. Riserva receipt è output
mutabile soltanto per la gestione FAIL dichiarata, non input da far MATCH artificiale.

## Adattamento limitato e negativi, poi tre tool

Nuovi file solo in
`evidence/implementation-r001/pilot-baseline-recovery-r001/continuation-001/`.
Prepara lì entrypoint e `inputs/continuation-inputs.json` prima dei figli, con
scope preliminary/id c001,parent manifest/scope/receipt/preservation SHA,
nuovo context/scope SHA, codice realmente eseguito byte/mode/SHA, argv/cwd/env,
modelli/pin/host/config/assenze/budget. Riuso per riferimento verificato degli
input immutabili grandi; non duplicare venv/wheel, inventari o suite inutilmente.
Nessun self-hash, output futuro o modello cambiato per adattarlo a un risultato.

Consentiti soltanto delta di controllo descritti nella scope: nuova autorità e
manifest, lineage per predecessori originali, cap cumulativo72MiB/contabilità
esplicita, reader inventory legato alla nuova identità. Guardie di closed inventory,
origine/startup, config/host/input, modello e contenuto pip/wheel, env/deadline/
monitor/lock/cleanup restano identiche. Eventuali moduli adattati hanno versioni
nuove nel bundle; non modificare moduli originali o monkeypatchare guardie per
saltarle. Driver ufficiale r003 non si allenta e la sua autorizzazione resta PENDING.

73PASS storici conservati sui loro input, non attestano nuovo controllo di lineage
né budget. Scrivi soltanto regressioni pertinenti: prefisso/receipt/input alterato,
context/scope vecchi o errati, marker già presente, limite64 che rifiuta32MiB
residui e nuova autorità72 con la stessa stima, riserve112+16 e crescita dei report/
request addebitate, cap/PermissionError/deadline. Negativi prima dei figli; fixture
piccole stdlib, no app/native/rete/seed. Salva hash prima/dopo ed exit/log reali.
Non rieseguire suite intere immutate né replicare tutte le fixture a ogni controllo.

Dopo gate/test positivi, un solo tool per passo, argv strutturati/env esatto,
shell=False,close_fds=True,cwd radice per driver e target/workspace per figli:

1. install: argv pip del parent identico, requisiti17URI/hash locali identici
   (3898byte/SHA5c7a72dfbd59fb28b1375deeef30e46761d3a45b3bceaf1e9a97c5ba593352f5).
   Nuovo python -I -B -m pip --isolated --require-virtualenv --no-input
   --disable-pip-version-check install --no-index --require-hashes --only-binary=:all:
   --no-deps --no-cache-dir --no-compile -r file esatto;
2. pip-check: prefisso python/pip identico con check, dopo inventoriate18dist strict;
3. inventory: bootstrap assoluto -I -B, reader di metadati/RECORD/file18dist/
   direct_url/script/495pyc/tokenizer senza import app/native, identità c001
   esplicita e copie esatte. Reader che verifica il vecchio context non è riutilizzabile
   senza nuovo adattamento dichiarato/identificato. Output esistenti non sovrascritti.

Env completo del parent, esattamente invariato perché target invariato. Driver e
figli identici; launcher bootstrap `/home/davide/.pyenv/versions/3.12.3/bin/python3
-I -B` con env=dict(...), nessun os.environ/PWD/SHLVL ereditato. No coverage/proxy/
credenziali/PYTHONPATH/PYTHONHOME. -I non elimina .pth: a1_coverage.pth esatto,
COVERAGE_PROCESS_START/CONFIG assenti, startup/origine prima di nuovi avvii.
Pip interno pinned per soli requisiti diretti, nessun nuovo pin/extra/indice/grafo/
backend/build/download. Bootstrap/bundledpip e fonti originali come parent invariati.

Prima del tool successivo verifica exit0, receipt completa, hash output e lineage.
Originali directories/venv restano predecessori originali, con i loro SHA; ricevute
nuove riportano c001,nuovi context/scope/manifest,parent reference e acceptance
PRELIMINARY_ONLY. Non attribuire loro vecchia identità né emettere nuova PASS_VENV.
Inventario c001 prova soltanto l'ambiente preliminare corrente, non baseline ufficiale.

## Budget e stop

Cap72MiB cumulativo include tutto il lavoro autore del pilota dall'inizio: prep r003,
vecchie e nuove evidenze/target/copie/lock, intera allocazione corrente dei report/
checkpoint addebitata conservativamente, futura request s007. Prep r00316MiB e
nuove preparazioni/evidenze/request fino4MiB aggiuntivi, tutti inclusi nel cap72.
Stima residua install32MiB congelata non abbassata; altri passi come parent8MiB.
Riserva ufficiale112MiB +16MiB esterni, run500MiB/stop384MiB invariati; target64MiB/
4096file/1024directory, perfile32MiB/log1MiB/JSON8MiB/receipt4MiB/free516MiB.
Monitor50ms non quota atomica. Durante questi tre passi non si ammettono transienti:
non c'è seed; strict a ogni gate.26link attuali dichiarati(14storici+12fixture),
non seguiti; targetzero link. Nessun cleanup o omissione dei blocchi/directory.

Prima di ogni scrittura/operazione, misura lstat intera run e radici autore senza
doppie somme; totale vivo + allowance72 residua +112+16MiB sotto384MiB e stima
lavoro nei limiti. Continuation preparazione/negativi/receipt/report/request contati.
Raccolta soli gruppi propri, sessione attiva da completare non rilanciare.
Deadline install180s/check60s/inventory120s, gate/copie aggiuntivi.
Qualsiasi FAIL/mismatch/cap/timeout/ENOSPC/receipt incompleta/lock conteso/rifiuto
sandbox: STOP, conserva log/tmp/target/receipt, nessun retry/secondo target.
Se manca receipt operativa conserva exit/output, non inventarla. Estensioni della
scope/policy/privilegi/costi tornano al supervisore. Nessun kill per nome o servizio.

## Consegna accorpata

Se tutti tre passi riescono, consegna inventario preliminare e ricostruzione della
lineage con input/receipt originali più c001. Nella stessa chat prepara la request
ufficiale `implementation/stages/impl-r001-stage-package-s007/request.json`,
WAITING_FOR_STAGE_SNAPSHOT, owner supervisor. Usa helper ufficiali r003 finali
immutati già identificati; se un input ufficiale deve cambiare, preserva la versione
precedente, crea una nuova versione identificata e ripeti solo le prove invalidate.
Target ufficiale `work/baseline-recovery-install-r003` e lock restano assenti.
Lista precisa di input esistenti relativi/regolari con SHA: piano/arbitrato/addenda,
helper/modelli/argv/budget finali, prove/fonti/asset/report/originali/fixture,
parent/c001 già prodotti; nuovi metadata supervisore dall'attuale contesto.
Nessuna scope ufficiale finta, output futuro, self-hash o report/checkpoint mutabile
nel freeze. Stima ufficiale aggiornata include pilot/target/FAIL conservati una volta.

Se non riesce o non ammesso: WAITING_FOR_SUPERVISOR_RECEPTION, nessuna request pronta
artificialmente. Scrivi delivery.md e completion-continuation.json nel nuovo evidence
root con stato di ogni passo, argv/env/sessioni/exit/hash/copie/guardie/byte logici e
allocati, verify/Git/diff-check e processi conclusi/limiti osservati. Aggiorna solo
report e checkpoint d'autore dopo copie d'ingresso preservate. Registra tempi/
passaggi reali; nessun risparmio inventato. Prossimo supervisore **prompt32**.

R completaD4/nuovoS/baseline-s006/V0 e produzione/managed/lock/grafo/D2D3/IDE/S-B-I-E/
V1–V9 e doppie review codice/arbitrato futuri. Baseline/confronti IMPEDITI,
V0 storico62pass5fail/perdite e FAILs003/s005/s006 invariati. V7/V8 obbligatorie
costo distinto;V10/V11/pesi/font/inferenza esclusi. Nessun GO codice,host/sysctl/
AppArmor/rete/socket/privilegi/profili persistenti/altro progetto/invio remoto/
cleanup/commit/merge/push/deploy. Temp ignorata,trasferire file reali; Git manuale.
