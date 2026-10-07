# Implementazione49 — ripresa S/B/I/E con approvazione degli strumenti

Agisci esclusivamente come **implementatore** in
`/home/davide/workarea/markdown-for-llms`, nella stessa chat48 se disponibile.
Completa l'intera sequenza22operazioni S/B/I/E già ricevuta. Il supervisore ha
ricevuto report-r013 e corretto la clausola di arresto: EPERM del processo
non equivale a una richiesta di escalation respinta. Prepara il comando
concreto e richiedi l'approvazione tramite gli strumenti, senza un'altra
consegna al supervisore per chiedere il permesso di farlo.

## Leggi e verifica l'ingresso

AGENTS, skill manage-implementation-run e protocollo aggiornati, STATE/HANDOVER,
checkpoint autore/report-r013. Path successivi relativi a
`temp/run-a001-fase0-uv/`:

- `arbitrations/addendum-operational-protocol-r015.md`: disposizione corrente,
  prevale sulla clausola generica di STOP sandbox di48 e antecedenti.
- `evidence/supervisor-implementation-r001/sandbox-recovery-reception-r001/`:
  reception.json, freeze-disposition.json, response.json e final-checks.json.
- `evidence/implementation-r001/product-sbi-s016-r001/delivery.json` e
  sandbox-diagnosis.json; non rifare la diagnosi bind già disponibile.
- `prompts/48-implementation-r001-execute-frozen-product-sbi.md`, piano r003,
  arbitrato r003 D1–D5 e r013/r014 per sequenza, requisiti e limiti conservati.
  In una chat fresca leggere il piano completo, senza tutta la storia della run.
- Request storica `implementation/stages/impl-r001-stage-product-s016/request.json`,
  inventory `work/product-sbi-r001/package-inputs.json` e template/launcher in
  `evidence/implementation-r001/product-inputs-r001/`. La label effettiva è s017
  come disposto da R015; non trattare i riferimenti s016 storici come vincoli
  che impediscono l'adattamento operativo.

**Snapshot effettivo** `snapshots/impl-r001-stage-product-s017.json`,293347byte,
SHA0949ad6715795fe8ee91c387a6e2d1714f2958ac8dd7530a3f81226a4e5216bc;
worktree72951e483c8e9509fd09462591937650dd25f133005d26e0cd4f15f9448c05ed,
127file/938artefatti,MATCH verificato dal supervisore. Ripeti verify prima/dopo
le prove. S016 resta immutato/storico; non ristabilire MATCH del vecchio stage
contro il checkpoint aggiornato o il protocollo nuovo. Il freeze s017 usa
copie immutabili del checkpoint d'ingresso: il file vivo non è più artifact.

Branch feature/run-a001-uv, HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto. Ricezione propria:1027record s016 ricalcolati, unico delta prima
della governance =checkpoint autore; inventory805file invariati. Prodotto,
lock/backend/diagnostici ufficiali/launcher invariati. Nessun S/B/I/E avviato,
rootvenv/S/dist/base-a/base-b/esterno assenti. FAIL S tentativo1 conservato.

Piano SHA462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b;
arbitrato SHAf14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d.
Scope ricevuto `evidence/supervisor-implementation-r001/product-input-reception-r001/authorized-scope.json`,
63176byte/SHA3fd09c86e4cd4d124c3df035cf84ee102b97fee7c372a5e27689d8463134e4c7.
GO solo piano/NO_GO storici; nessun GO finale.

## Prepara la ripresa operativa nella stessa chat

Crea un adapter/driver proprio in
`evidence/implementation-r001/product-sbi-s017-r001/resume_product_s017.py`.
Il path è un output futuro da produrre, non un input già attestato. Sono
delegate le correzioni del dispatcher operativo, non quelle dei requisiti.
Mantieni intatti launcher/template congelati e report/receipt precedenti.

Il launcher originale ha la label hardcoded s016: il nuovo adapter deve
selezionare s017 per SNAPSHOT/check iniziale/finale/wrapper e per l'argomento
--snapshot di `S-frozen-product`. Può caricare il dispatcher identificato,
validare i bytes del template storico e applicare override in memoria dei
soli riferimenti operativi. Conserva la configurazione effettiva, argv/env/cwd,
hash del driver e degli input realmente consumati. Non modificare la semantica
di snapshot_check, guardie source/inventory, R/D4 o audit B/I/E.
Se il proprio adapter richiede un fix ordinario, correggilo e continua qui.
Non tornare al supervisore per la sola label/argv/output/contabilità del driver.

Contabilità autorevole di ingresso: **111.10300820460543s già addebitati**
su7200s, residuo **7088.896991795395s**. Il dispatcher congelato conosce solo
prefix/result interni e sovrastima il residuo: il nuovo driver deve limitare
realmente ogni chiamata con il cumulativo completo delivery-r013 e i successivi
intervalli non sovrapposti. Il vecchio driver include già3.1342911049723625s
del launcher; non sommarli di nuovo. Le stime interne restano ricevute
storiche, non il gate autorevole. Sottrai guardie/preparazione/tentativi nuovi
una sola volta. Non basta correggere il numero nel report dopo l'esecuzione.

## Richiedi l'approvazione del contesto operativo

Il report-r013 ha già diagnosticato bind AF_UNIX/EPERM nel sandbox. Non è
stato consultato l'auto-review. Dopo aver preparato il driver reviewabile e
verificato staticamente argv/env/cwd/gate, usa il meccanismo ufficiale dello
strumento disponibile. Con exec_command, richiedi
`sandbox_permissions="require_escalated"` per il comando concreto della
ripresa. Non chiedere in chat un altro permesso al supervisore prima di inviarlo.

Prima chiamata concreta, dopo creazione del driver che espone questi argomenti:

```text
/home/davide/workarea/markdown-for-llms/.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -I -B /home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/evidence/implementation-r001/product-sbi-s017-r001/resume_product_s017.py --operation S-frozen-product --attempt 2
```

CWD radice; il driver avvia i figli con argv strutturati,
shell=False/close_fds=True e environment pubblico puro dello scope,
senza merge os.environ. Prima degli interpreti controlla origine/config/site.

Giustificazione della richiesta: il launcher deve creare un socket UNIX
**sintetico nella run** e avviare Firejail/R; il sandbox corrente nega bind.
Il workload S è già autorizzato, rimane dentro Firejail net=none e parte
soltanto dopo nuovi R/D4 PASS. R conserva probe connect/close dei daemon
inventariati senza send/recv operativo, mascheramento padre/figlio e AF_UNIX
locale positivo. Specifica directory, dati sintetici, budget e nessun accesso
operativo daemon, rete/download o modifica host. Il sistema di approvazione
decide; non usare prefix_rule generici che approvino qualunque script Python.

Se approvata, esegui nel contesto concesso e raccogli receipt/processi, poi
prosegui gli altri21ID usando approvazioni richieste dagli strumenti secondo
la policy effettiva. È ammessa una chiamata a driver completo delimitato e
reviewabile se la policy la approva; non presumerla autorizzata dal solo prompt.
R/D4 resta nella medesima invocazione dei workload, stesso interprete/namespace
dei figli. L'approvazione esterna non è prova di isolamento o PASS prodotto.

Se l'approvazione è respinta, conserva la decisione e il motivo; non aggirarla
con altri strumenti/comandi equivalenti e non chiamare il FAIL di syscall
«rifiuto auto-review». Se l'escalation è indisponibile per istruzione della
sessione, documenta quel limite concreto. Continua lavoro indipendente coperto
e consegna un blocco reale. Nessun sudo, cambio sysctl/AppArmor/rete host,
setuid/profili/socket operativi o workload fuori da R.

## Completa S/B/I/E e consegna

Stessi22ID e dipendenze della request/template: S ufficiale s017 → build uv
nativa offline sdist/wheel-dalla-sdist con backend84/no-build-isolation,
startup e import/get_requires prima, rawtrace veri hook → audit B/RECORD
prima install → root/base-a/base-b sync locked runtime no-install-project/
no-build, stessa canonica no-deps/no-build, I/diecihash/origini ed E → esterno
export hash-locked, managedvenv, runtime/canonica/pip-check/I/E. Nomi/digest
archivi derivati da B reale. Nessun PASS trasferito dallo scratch o dalle copie.
S usa attempt>=2/output nuovi; altri ID non avevano output, sempre directory
esclusive e retry soltanto dopo diagnosi/raccolta processi/costi. Non dichiarare
riuscita la sequenza se un prerequisito manca o fallisce.

Budget **112MiB cumulativi/Hentry553541632** invariato:
H=max(logical,allocated) lstat nofollow su run/.venv-python/.venv/
/tmp/run-a001-product-wheel-env-r001; Delta=max(0,H-Hentry)<117440512 e
H+max(0,117440512-Delta)+16777216<939524096. Pool1GiB/stop896MiB,
repository libero>=1GiB e /tmp>=128MiB. Storage residuo di consegna in
response/final-checks, da rimisurare prima e durante le chiamate. Monitor0,5s/
gap target1s non atomico; file32MiB/JSON8MiB/stream1MiB/log normali.
Workload<=900s per chiamata entro residuo, outer+180; nessun reset/residuo32
sommato/cleanup/cache o tmp fuori ledger, paid0. Nessuna nuova rete/download
Python/backend/dev/API/ML/pesi/font, import app/CLI/suite/V0 impliciti.

Fix reversibili r013 restano delegati. Cambi a input della prova si correggono
qui e si raggruppano per il freeze ufficiale necessario, senza nuovo piano.
Arresti: decisione negativa effettiva dell'approvazione, costo/requisito/fonte
logica nuovi, impossibilità dei confini, baseline alterata o irreversibilità.

Consegna **report-r014** reale, evidenze22esiti/exit/comandi/hash/S/B/I/E/costi
e checkpoint autore aggiornato. Il checkpoint vivo è escluso da s017: non
serve renderlo storico per quell'aggiornamento. Stato comune e snapshot al
supervisore. V1 config-only/rebuild/stale/ripristino, V3–V9/import/fiveCLI/
fasi/dev/API/suite/comparativi/negativi restano aperti; V7/V8 obbligatorie con
costi distinti, V10/V11 escluse. Due review reali in nuove chat indipendenti
ChatGPT/Claude sullo stesso snapshot finale/arbitrato/GO ancora futuri.
FAIL storici/baseline62pass5fail/perdite preservati, ABI managedLinuxx86_64.
Nessun agente, Git write/commit/merge/push/promozione/deploy/cleanup globale
o servizio automatico da attendere. Il prossimo risultato è l'esecuzione
completa o il rifiuto effettivo documentato, non un'altra sola preparazione.
