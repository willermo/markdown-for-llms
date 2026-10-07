# Implementazione r001 — eseguire il backend EbookLib già auditato

Agisci esclusivamente come **implementatore**, repository
`/home/davide/workarea/markdown-for-llms`. Puoi proseguire nella stessa chat
implementatrice; questo prompt è autosufficiente anche per una chat nuova.
Obiettivo: due build native offline della sola EbookLib0.18, validare wheel/
backend effettivi e concretizzare il seguito del lock. Esegui i risultati ammessi,
correggi launcher e attese ordinarie nella stessa chat; nessun giro di sola
preparazione, delega, supervisione o review simulata.

## Autorità e identità

Leggi AGENTS, skill manage-implementation-run, protocollo, STATE/HANDOVER, brief
A1–A7, indici architettura/decisioni/roadmap, piano r003 integrale se non già letto,
arbitrato D1–D5 e checkpoint implementatore. Report corrente **report-r004.md**,
checkpoint supervisore **handovers/supervisor-ebooklib-backend-reception-r001.md**.
I ruoli dei documenti non cambiano il tuo ruolo implementatore.
I percorsi di seguito sono relativi a `temp/run-a001-fase0-uv/`:

- Piano `plans/plan-r003.md`, SHA256
  **462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b**;
  arbitrato `arbitrations/arbitration-plan-r003.md`, SHA256
  **f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d**.
- Addenda r004–r006 conservati; **r007** ammette questa tranche di backend,
  comportamento uv nativo e risorse. Nessun GO finale né nuovo piano.
- Request nuova `implementation/stages/impl-r001-stage-package-s009/request.json`:
  **103249byte**, SHA256
  **dbb4fe8d9aa077d439a0670da0e6f1eb42e98b07f52a3771d653b4813cd86365**.
- `evidence/supervisor-implementation-r001/impl-r001-stage-package-s009/`:
  scope `authorized-scope.json`, **27842byte**, SHA256
  **c5923c454a398b91bf05d719b2486816d7292582ac0a443e20bc20da33525037**;
  target-a.json/target-b.json, reception/transition/checks/sources/freeze-verify/
  response.json. Scope e target sono input immutabili.
- `snapshots/impl-r001-stage-package-s009.json`: **83329byte**, SHA256
  **85649c577a0715100f19304a48b660a0054efd65ad7fe66f5ef4889167ed2009**,
  worktree **518749e5472ae9a69384b3a2255f659c218a441c84a7773fdad8e2b39428b00e**,
  **123file/211artefatti**, MATCH alla consegna supervisore.

Byte/hash di manifest/request/scope e piano/arbitrato prima delle attività;
da radice prima delle operazioni e alla consegna:

```bash
python3 -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-package-s009
```

Mismatch resta FAIL, nessun autofreeze/allentamento helper. Baseline-s007 era
MATCH alla ricezione e ora è storica per soli cinque metadata in transition.json;
**baseline originale completa resta acquisita**, non rifarla per questa tranche.
S.id,67nodeid/201eventi/62pass5fail e perdite legacy sono osservazioni reali ricevute,
non PASS del prodotto migrato. Richiesta d’autore audit/next-package-gate-r001
immutata/not-authorized alla consegna; nuova scope del supervisore è distinta.

Git feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto, core36 e metadata identificati. Verifica Git reale. Non cambiare
tracciati/root uv.lock/.venv in questa tranche; preserva report/checkpoint d’ingresso.

## Esecuzione nativa pronta, due chiamate separate

La nuova request **proposed_operations** e scope **operations** contengono
gli argv completi del wrapper per a e b e gli argv/env/cwd nativi. Sono identici;
verificali e usali strutturati, shell=False, close_fds=True, env=dict(...), senza
merge con os.environ. Bootstrap launcher assoluto pyenv3.12.3 `-I -B`.

Prima crea **esclusivamente** `work/ebooklib-native-r001/` e la base evidenze
`evidence/implementation-r001/resume-package-s009/`, già assenti alla ricezione.
Per ogni operazione crea il suo cwd e cache/tmp/config/dist della lista dichiarata;
non creare il suo future_wrapper_output: lo crea il wrapper. Directory a/b e
output non si riusano. Distinguere log/receipt del launcher esterno da quelli del
wrapper `build-a-r001`/`build-b-r001`; non sovrascrivere un output già prodotto.

Ogni argv avvia il wrapper congelato: R padre/figlio con target nuovo, D4 e
Firejail noprofile/net=none, quindi launcher inline stdlib `os.execve` e uv nativo
**nello stesso namespace**, nessuna shell. Il launcher inline è già nella request:
non riscriverlo sopra helper né cambiare il wrapper per estenderne l’ambiente.
Env R e native_environment coincidono sulle16 variabili pertinenti; il nativo
aggiunge soltanto le tre variabili uv esplicite della scope. TMPDIR/cache/config/
HOME/PATH e guardie privacy sono concordanti. Le receipt devono mostrare comando
uv e discendenti osservati nella netns isolata; non basta solo una R precedente.

Il nativo è `uv --verbose build --wheel` sull’archivio115484byte/SHA
**38562643a7bc94d9bf56e9930b4927e4e93b5d1d0917f697a6454db5a1c1a533**,
offline/no-index/find-links locale, Python managed3.12.13 assoluto, no-downloads,
constraints congelati e out-dir specifico. Tutte le opzioni sono negli argv,
non ricostruire un comando abbreviato. Nessun sync/install della dipendenza nel
prodotto, import EbookLib o dipendenze runtime lxml/six.

Prima di ciascuna chiamata verifica hash uv, managed binary/BUILD/origine,
archivio, wheel backend e constraints; assenze/config della scope. La directory
find-links deve offrire come wheel top-level **soltanto setuptools84**:
818216byte/SHA51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670.
Zero index/download/fallback di backend. Il resolver interno nativo per quei soli
requisiti di build è ammesso; il resolver runtime/prodotto resta escluso.
Controlla integrità target/lock s007 e startup bootstrap prima/dopo, usando
check_preserved.verify soltanto per integrità. Non reinstallare managed/backend.

Build isolation rimane uv nativa: venv temporanea e installazione della sola
wheel backend sono ora ammesse. **Gli hook nativi usano Python -c, senza -I/-B**,
come nel frontend pinned; ambiente chiuso PYTHONDONTWRITEBYTECODE=1 e
PYTHONNOUSERSITE=1 resta effettivo. Non fingere flag né creare shim/patch dei
figli. Tutti i lanci Python sotto controllo dell’agente/runner mantengono -I -B.
Startup temporaneo _virtualenv/setuptools pinned accettato entro r007; distingue
quello inferito dal sorgente da ciò che hai realmente osservato.

Child120s/outer300s per chiamata; una sessione viva si raccoglie, non si rilancia.
Per il runner usa il profilo host già impiegato con autorizzazione concreta se
serve require_escalated. Rifiuto automatico sandbox resta IMPEDITA, ragione/azione
preservate; nessun aggiramento o timeout inventato. Nessun cambio host/rete/daemon/
privilegi/sysctl/AppArmor/profili persistenti. Solo processi/gruppi propri raccolti.

Dopo a controlla exit0, R/wrapper PASS, namespace, identità e wheel prima di b.
Un FAIL sostanziale del backend/cap/integrità blocca la seconda build dipendente;
nessuna acquisizione per soddisfare hook/requirements mancanti. Errori ordinari
di reader/launcher si correggono nella stessa chat, conservando l’esito precedente.
Non fare retry ciechi su build già iniziata; un input congelato modificato richiede
nuova label. Non consegnare solo sintassi mentre lavoro indipendente ammesso resta.

## Budget e validazione degli output

Scope `budget`: run intera più `.venv-python`, storico/cache/tmp/log/dir/link
metadata compresi senza seguirli. Max1GiB/stop896MiB, libero minimo1GiB.
Tranche nuova **32MiB incrementali** dalla ricezione:24 build/cache/tmp e8 registri/
strumenti/snapshot; riserva esterna16MiB, quota residua contata una volta secondo
formula della scope. Quota74MiB precedente conclusa, non un permesso aggiuntivo.
Entrambe le chiamate/parziali contano; cap d’attività include **tutto il ledger**,
non soltanto out-dir. Monitor0,5s/gap target≤1s, periodico non quota atomica;
log1MiB/stream,JSON8MiB,RLIMIT_FSIZE32MiB/file. Zero rete/download/costi remoti.

Non usare main storico500/384 o rami baseline del vecchio launcher per ammissione.
Adatta la tua guardia al budget nuovo prima dell’avvio; verifica sintassi/formula
localmente. Conserva versioni rilevanti, senza ricopiare tutta la storia.
**Nessun cleanup dell’agente per passare i gate**. La normale rimozione di tmp
da uv stesso è lifecycle nativo ammesso: inventaria ciò che resta e dichiara
ciò che non è osservabile. Non attribuire un picco istantaneo a pochi campioni.

Leggi le wheel senza import: percorsi/tipi sicuri, Name/Version distribuzione
EbookLib0.18, WHEEL/tag, METADATA lxml/six, entrypoint eventuali, RECORD e payload
ebooklib rispetto ai membri sorgente. Nessuna riscrittura0.18→0.18.1: costante
interna diversa è parte del source da conservare. Confronta byte/hash dei due
archivi e payload; conserva differenze timestamp/RECORD/metadata. Se outer hash
diverge, non dichiarare riproducibilità byte o normalizzare per PASS.

Usa trace native per backend e requirements effettivi; non fabbricare receipt
degli hook. Se il ritorno raw get_requires viene eliminato da uv e non osservato,
scrivilo esplicitamente; log, wheel e metadata autentici restano la prova.
Non imporre un backend alternativo per catturare quel raw né presentare la
selezione nel sorgente uv come esecuzione osservata. FAIL reali rimangono FAIL.
PASS vale solo per backend/wheel della dipendenza: niente B/I/E/ABI/applicazione.

## Consegna e seguito del lock

Evidenze nuove resume-package-s009: chiamate effettive/argv/cwd/env pubblico,
timestamp/sessioni/exit/receipt/log/input-output/hash/byte, inventari e guardie
pre/post. Completion `implementation/stages/impl-r001-stage-package-s009/completion-r001.json`
legata a manifest/request/scope, ogni operazione ed esito distinto, NOT_EXECUTED
se non avviata. Report nuovo **implementation/report-r005.md**, delivery nuovo
e checkpoint **handovers/implementation-r001.md**; report-r004/precedenti immutati.
Nessun registro comune/changelog/arbitrato/snapshot dell’implementatore.

Nella stessa chat esamina in sola lettura help/sorgenti uv pinned e cache reali
per una richiesta concreta del passo successivo: come ottenere metadata nativi
del registry originale mantenendo no-build/guardie o con un’eccezione effettivamente
circoscritta da ricevere. Non avviare resolver, altri backend o nuove acquisizioni.
Due build locali non provano che la cache del registry sia riusabile: niente
spostamento/forgiatura di cache, metadata/lock manuale, fonte find-links promossa
a prodotto o rimozione generale di --no-build. Non cambiare pin/Marker/indici/
grafo universale. Riporta il limite concreto se manca un metodo supportato.

Lock FAIL storico, lock-check NOT_EXECUTED, tests-s001 non pronto senza lock/S/B/I.
S/B/I/E del prodotto, V7/V8 con mandato pesante distinto, due review indipendenti
reali e arbitrato finale ancora da ottenere. V10/V11/pesi/font/inferenza esclusi,
local_marker non convalidato. Nessun GO finale, product build/install/sync/test/
Docker, invio documenti, cleanup, altro progetto, commit/merge/push/promozione/deploy.
Git manuale dell’utente. Stato conclusivo **WAITING_FOR_SUPERVISOR_RECEPTION**,
prossimo supervisore prompt05+r004–r007 con checkpoint corrente. Nessun servizio
da attendere; trasferire file reali ignorati/managed/cache/work e lavoro non
committato, non soltanto manifest.
