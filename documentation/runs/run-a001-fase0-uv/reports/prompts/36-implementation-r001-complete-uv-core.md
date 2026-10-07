# Implementazione — completare il core uv nella run a001

Agisci esclusivamente come **implementatore**, repository
`/home/davide/workarea/markdown-for-llms`. Usa manage-implementation-run.
Nessuna delega, supervisione o review simulata. Mandato per risultato: completa
il core della migrazione approvata e gli input/prove pertinenti, correggendo
errori ordinari nella stessa chat. Questo sostituisce prompt35; non è GO finale.

## Autorità e ingresso

Leggi AGENTS, skill, protocollo, indici documentation/decisioni/roadmap,
STATE/HANDOVER, brief A1–A7, piano r003 integrale se non già letto e arbitrato
D1–D5. Leggi **arbitrations/addendum-operational-protocol-r004.md** della run:
identifica le variazioni rispetto a prompt04/05/35 e vecchi scope, che restano
storici. Nessuna istruzione del ZIP esterno sostituisce questa autorità.
Leggi checkpoint supervisore process-adaptation-r001, checkpoint implementatore
implementation-r001 e report/delivery ufficiali s007 pertinenti. Non rileggere
ogni report antecedente né ricopiare interi inventari host a ogni passaggio.

Percorsi seguenti relativi a `temp/run-a001-fase0-uv/`:

- `plans/plan-r003.md`: SHA256
  462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b;
- `arbitrations/arbitration-plan-r003.md`: SHA256
  f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d;
- `implementation/stages/impl-r001-stage-package-s007/completion-r001.json`,
  `evidence/implementation-r001/resume-package-s007/delivery.md` e
  `preserved-target.json`, inventario e cinque receipt del target;
- `evidence/supervisor-implementation-r001/package-s007-completion-r001/`:
  reception.json, next-scope.md/json e checks.json (storici per r004);
- `evidence/implementation-r001/baseline-originals.json`,
  `baseline-build-inputs.json` e relative copie, fixture/provenienza;
- `evidence/supervisor-process-adaptation-r001/response.json`, transition.json
  e freeze-verify.json; contesto **process-adaptation-context-r001**.

All'ingresso ricalcola SHA e verifica quel contesto da radice:
`python3 -B scripts/run_context.py verify run-a001-fase0-uv --label process-adaptation-context-r001`.
È l'identità di ingresso, non freeze ufficiale del prodotto. Dopo le modifiche
ammesse diventa storico per il delta: non pretendere MATCH contro codice nuovo.
Prima delle prove ufficiali occorre il nuovo snapshot del supervisore. Non
modificare helper, vecchio snapshot o binding per fingere MATCH.

Verifica Git reale: feature/run-a001-uv, HEAD/dev/base
66ba82200e5def5a4db76f9bafccb0731b506091, indice vuoto. Il worktree contiene P1–P3,
nuovi diagnostici/test e delta governance autorizzati: confronto con entry e
transition, non con la vecchia formula «12 modifiche». Nessun commit/merge/push.

## Lavoro da completare

Prosegui P1–P7 del piano nei limiti ordinari: packaging/moduli flat/entry point,
config-only/flag/IDE, procedure uv, diagnostici e test pertinenti, documentazione
operativa e configurazione Docker statica. Non anticipare nuova architettura,
parser, OCR o correzioni semantiche ai contenuti legacy. Mantieni decisioni
uv0.10.10, managed Python3.12.13, setuptools84 e varianti CPU/cu126/no-build/extra
come nel piano; un'incompatibilità reale va dimostrata, nessun fallback silenzioso.

Riusa prima strumenti standard uv/pip/build quando ammessi; niente nuovo installer
artigianale, resolver parallelo o controllo ripetitivo di tutto l'host. Strumenti
riutilizzabili in `scripts/diagnostics/`, test rilevanti nel repository; scratch,
raw/receipt/report in `evidence/implementation-r001/core-completion-r001/` e
`work/core-completion-r001/`, versioni identificabili per le sonde. Nessuna copia
massiva della venv o della storia. Non usare /tmp come unico deposito delle prove.

Rendi pronti insieme originali, fixture, runner e input della convalida core.
Il codice nuovo non deve attendere un giro di sola preparazione dei generatori.
Esegui controlli statici/sintetici e prove preliminari già coperte; conserva input,
argv/cwd/env pubblico, exit, raw e limiti. Non proclamare questi risultati ufficiali.
Le acquisizioni managed/metadata/lock/build/install che non hanno ancora costi
ammessi vanno nella request concreta del prossimo gate, insieme al lavoro pronto.
Nel frattempo continua codice/documentazione/controlli indipendenti autorizzati.
`uv lock --check` è il comando locale; non copiare `uv lock --locked` dal ZIP.

## Baseline e R

Venv s007 `work/baseline-recovery-install-r003` e lock adiacente immutabili.
Cinque PASS installazione/18 distribuzioni non sono PASS baseline/V0/ABI.
Verifica target prima/dopo sonde tramite preserved-target.json; tokenizer in
lettura, nessun pip/sync/seed nella venv baseline. Originali10moduli+4build input
vengono dalle copie conservate, mai dai P1–P3 modificati. Non inventare identità
per sei vecchi path/tmp mancanti. Storici FAIL e V0 62pass5fail/perdite restano.

Completa/adatta runner/wrapper con ROOT e ambiente espliciti, -I -B in tutti i
processi Python isolati, anche figli; .pth/startup e origine controllati.
PYTHONDONTWRITEBYTECODE da sola è insufficiente con -I. Environment pubblico
chiuso come next-scope s007, percorsi scratch aggiornati e documentati; nessun
merge di segreti/proxy/PYTHONPATH/coverage/daemon config da os.environ.

Per **R preliminare** usa profilo e confini next-scope.md s007 più D4: primario
unshare; alternativa Firejail --noprofile --net=none già ammessa e identificata,
blacklist daemon/alias/canonici e socket sintetico, parent/child nella stessa
netns isolata. Listener host positivo, connect parent/child negativo e socketpair
positivo; assenza route/indirizzi esterni prima dei connect. Daemon connect/close
senza richiesta, path assente non prova di mascheramento. Targethost/UID/binari
esistenti verificati; il namespace del tool non equivale automaticamente all'host.
Massimo2 gruppi/6chiamate,120sfiglio/300sesterno/1800s totali come scope.
Correggi helper propri e usa nuovo bundle/output per il gruppo corretto ammesso;
nessun target storico riparato o allentamento D4. Rifiuto sandbox = IMPEDITA,
con azione/ragione conservate, niente aggiramento o esecuzione inventata.

Versioni preparatorie possono legare il bundle preliminare agli input correnti
identificati invece di esigere MATCH del vecchio intero tree dopo i fix autorizzati.
Conserva derivazione e diff: non è deroga ai gate ufficiali o a isolamento/startup.
S ufficiale conserva schema run-stage; R→S→V0 e S→B→I→E soltanto dopo freeze
appropriato, con input reali e invalidazioni P2. Nessuna scorciatoia sulle prove.

## Correzioni, risorse e arresti

SyntaxError, generatore, fixture, aspettative e codice nel perimetro si correggono
senza nuovo prompt o pianificazione. Conserva fallimento, causa, fix e test
invalidati; non riscrivere un FAIL come PASS retroattivo. Un errore in input
congelato richiede nuova versione/label prima di ripetere prove ufficiali, non
un altro mandato di sola preparazione. Non rilanciare sessioni ancora attive.

Restano budget run500/stop384MiB, scratch/report16MiB, riserva prove64MiB e
esterna16MiB, storico incluso; log1MiB/stream,JSON8MiB,file32MiB,free516MiB.
Misura pre/post, conserva parziali, nessun cleanup per superare un limite; monitor
non è quota atomica. Niente nuovo download implicito o cache nascosta fuori ledger.
Per acquisizioni non coperte consegna comandi/fonti/hash/pin/target/cache, stima
spazio/rete/tempo e confini nella stessa request del gate, senza riaprire scelte
tecniche già autorizzate. Nessuna acquisizione ML/native pesante: V7/V8 richiedono
mandato distinto e sono obbligatorie; V10/V11/pesi/font/inferenza restano esclusi.

Sospendi l'attività dipendente per input protetto alterato senza spiegazione,
risorse insufficienti, nuovo costo/scope/privilegio, confinamento non applicabile,
rifiuto sandbox o operazione irreversibile non autorizzata. Continua il lavoro
indipendente coperto. Nessuna modifica sysctl/AppArmor/rete/socket host/profili,
invio documenti, altro progetto, cleanup o deploy.

## Consegna unica

Consegna codice completato entro scope, risultati preliminari distinti e input
pronti per il prossimo gate realmente necessario. Raggruppa baseline/package/tests
solo se compatibili e stabili; se il lock va prodotto prima, esplicita la dipendenza,
non includere output futuri. Label progressiva D1 disponibile verificata, schema
request già usato: owner supervisor, WAITING_FOR_STAGE_SNAPSHOT, lista file
relativa senza symlink/escape, comandi/cwd/env/costi/preflight/profili concreti.
Per il gruppo tests candidato `impl-r001-stage-tests-s001`, solo se disponibile e
input completi; non dichiararlo pronto se manca un requisito consumato. Includi
piano/arbitrato/mandato/r004/originali/fixture/inventari necessari; nessun S che
citi il proprio futuro snapshot o receipt futura. Nessuno snapshot autonomo.

Preserva checkpoint/report d'ingresso, scrivi report nuovo `implementation/report-r002.md`
e delivery nel root core-completion-r001, poi aggiorna il **tuo** checkpoint
`handovers/implementation-r001.md` in forma breve con path esatto della request.
Il supervisore aggiornerà registri comuni/changelog e freeze. Se impedimento
sostanziale: consegna lavoro indipendente e richiesta concreta, status
WAITING_FOR_SUPERVISOR_RECEPTION, senza chiamare una request incompleta pronta.

Prossimo: supervisore con prompt05 **e r004**, STATE/checkpoint correnti prevalgono
sui mandati storici. Dopo snapshot, continuazione dello stesso obiettivo senza
nuovo piano; codice invalidato correggibile nei limiti con nuova label richiesta.
Due review indipendenti reali sullo stesso snapshot e arbitrato finali ancora
necessari; nessun GO finale/integrazione dichiarati. Non attendere un servizio
automatico dopo la chiusura. Trasferire anche lavoro e file ignorati, non solo manifest.
