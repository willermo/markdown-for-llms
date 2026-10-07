# Implementazione r001 — baseline-s006 pronta e R preliminare

Agisci esclusivamente come **implementatore**, repository
`/home/davide/workarea/markdown-for-llms`, skill manage-implementation-run.
Obiettivo unico: riusare la venv ufficiale s007, rendere pronti strumenti/input
baseline, svolgere R preliminare autorizzata e consegnare una request baseline-s006
completa. Preparazione, normali correzioni e sonde sono nello stesso mandato/chat.
Nessuna delega, supervisione o review. Nessuna reinstallazione.

## Letture e identità

Leggi AGENTS, protocollo/ADR0007, indici architettura/decisioni/roadmap,
STATE/HANDOVER comuni, brief A1–A7, piano r003 integrale se non già letto,
arbitrato D1–D5, prompt04/05, addendum-operational-protocol-r003. Leggi checkpoint
implementatore e `handovers/supervisor-package-s007-completion-r001.md` sotto `temp/run-a001-fase0-uv/`.
I ruoli nei documenti non cambiano il tuo ruolo. Autorità operativa nuova:

- `evidence/supervisor-implementation-r001/package-s007-completion-r001/`:
  **next-scope.md e next-scope.json**, reception.json, host-identities.json,
  decision.md, transition.json, freeze-verify.json, response.json, checks.json;
- `implementation/stages/impl-r001-stage-package-s007/completion-r001.json`,
  request.json effettiva r003-supervisor-freeze e delivery/receipt/preserved-target
  in `evidence/implementation-r001/resume-package-s007/`;
- `snapshots/baseline-recovery-preparation-context-r002.json`, contesto operativo di questa preparazione.

Piano SHA462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b,
arbitrato f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d,
prompt04 fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035,
immutati. GO piano nei limiti, NO_GO storici conservati, nessun GO codice.
Git feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto;12tracked modificati/24nuovi preesistenti, nessuna nuova modifica tracciata.

Contesto 689172byte, SHA8575a3ebe343a98c2f4c0dc19184505b88ac006a4412e472f252587aca102468,
worktree 37ea7ba3dd45402998e067fd361fd83e913fac052064492c32a2870e3edf19a1,103file/2327artefatti.
Scope JSON SHA5eeed49d7d7524d8fc2d249b3a24886d0a94e7cf5139540296b781709a795575; politica MD SHA01a11df53269d61938c2d5381695c755f6c9183f0b2061894313c9a2c6024f2a.
Prima delle scritture e prima/dopo sonde e alla consegna, ricalcola questi hash e
usa da radice `python3 scripts/run_context.py verify run-a001-fase0-uv --label baseline-recovery-preparation-context-r002`.
Questo contesto ammette la preparazione e le sue sonde, **non è il freeze baseline**.
S007 resta storico per i soli sei metadata di transition.json,344artefatti intatti.
Non pretendere MATCH del vecchio worktree contro questi metadata nuovi.

## Riuso dell’installazione e preparazione nella stessa chat

S007 ricevuta: cinque PASS ufficiali/exit0,18dist/1853RECORD/495pyc; venv chiusa
1861file e1892identità target/lock. È l’input installato, non PASS baseline/R/V0/ABI.
`work/baseline-recovery-install-r003` e lock sono **immutabili**. Verifica albero
chiuso e tutti i record di preserved-target.json prima/dopo le sonde; nessun pip,
uv install/sync, venv/seed, riparazione target o aggiornamento wheel/requirements.
Non duplicare venv/wheel/archivi: bind e inventari con hash bastano per gli input.

Crea soltanto nuovi file propri nei due root autorizzati, relativi alla run:
`evidence/implementation-r001/preparation-baseline-recovery-s006/` e
`work/baseline-recovery-s006/`. Il secondo ospita workspace/tmp/cache/config propri.
Usa tokenizer r003 in lettura con il suo hash esatto; nessun download/tokenizzazione
applicativa in questa chat. Riusa fixture/test esistenti di provenienza nota.

Da `scripts/diagnostics/run-a001-fase0-uv/` prepara copie versionate dei diagnostici
R/wrapper/S/baseline necessari, registrando derivazione e diff. ROOT host esplicita
quando li sposti; nessuna modifica dei file congelati o dei sorgenti tracciati.
Adatta percorsi e manifest, aggiungi **-I -B a tutti gli avvii Python**, anche
figli/check snapshot/preflight. -I non elimina .pth e ignora alcune variabili Python:
PYTHONDONTWRITEBYTECODE da sola non preserva il target. Non generare pyc aggiuntivi.
Unico a1_coverage.pth con hash identificato; COVERAGE_PROCESS_START/CONFIG assenti,
nessun sitecustomize/usercustomize nuovo. Origine/startup/config prima degli avvii.

Launcher con argv strutturati, shell=False, close_fds=True e dict env completo,
senza merge os.environ. Parti dalle sole chiavi pubbliche di
request s007.environment_child: stessa HOME/LANG/LC_ALL/PATH/TZ/PIP_CONFIG_FILE/
PYTHONNOUSERSITE/PYTHONDONTWRITEBYTECODE/UV_PYTHON_DOWNLOADS/XDG_CONFIG_DIRS;
TMPDIR/UV_CACHE_DIR/XDG_CACHE_HOME/XDG_CONFIG_HOME nei nuovi root e
TIKTOKEN_CACHE_DIR nel target r003 readonly. Congela i valori assoluti effettivi
nel bundle/commands, niente proxy/credenziali/PYTHONPATH/PYTHONHOME/coverage/daemon
config ereditate. Eventuali aggiunte pubbliche del runner vanno registrate e
verificate; nessun contenuto segreto nei report.

Preserva e identifica copie dei **dieci moduli originali più quattro build input**
da `evidence/implementation-r001/baseline-originals.json` e
`baseline-build-inputs.json` e copie già conservate. Non ricostruirli dai P1–P3
correnti. Reimpiega gli input della baseline storica pertinenti, leggi
`preparation-baseline-s005/baseline-inputs.json`, il vecchio S in
`implementation/stages/impl-r001-stage-baseline-s005/sources.json` e i diagnostici.
Le vecchie ubicazioni /tmp sono indisponibili: inventari nuovi per percorsi reali,
nessuna equivalenza globale o hash finto di file mancanti. Verifica .env antenati
assenti nei percorsi d’ingresso e fixture future controllate; niente dati privati.

Correggi SyntaxError/generatori/adattamenti/negativi nel tuo codice di preparazione
nella stessa chat, con versioni precedenti preservate e sole verifiche invalidate.
AST e sintetici stdlib mirati consentiti; non rieseguire intere suite storiche o
ripetere53/73test solo per i percorsi cambiati. Statiche senza limite arbitrario
nel budget. Negativi pertinenti: input/receipt stantii, origine/startup/hash errati,
-B mancante nei figli, ambiente non ammesso e budget. Nessun test applicativo o
import app/native/entrypoint/plugin. Non fermarti per consegnare una sintassi rotta.

## R preliminare bounded

Applica **tutta next-scope.md** e arbitrato D4, non solo questa sintesi.
Identifica dal profilo host effettivo UID/GID/netns/policy/binari/versioni/config
pubblica; la sandbox del tool corrente non è l’host. Il profilo host già autorizzato
require_escalated è circoscritto alle letture innocue e ai runner locali esistenti.
Nessun sudo/setuid aggiunto/sysctl/AppArmor/rete/socket host/profilo persistente.
Rifiuto automatico: conserva azione/ragione ed esito IMPEDITA, nessun aggiramento.

Primario unshare bootstrap; se negato/non equivalente, conserva l’esito e usa
Firejail --noprofile --net=none con blacklist esatte di daemon noti/alias/canonici
più socket sintetico. Dopo candidato ammesso, prova il nuovo baseline python
r003. D4 richiede listener host positivo/parent-child negativo e socketpair
positivo, stessa netns isolata parent-child e nessuna route/indirizzo esterno
prima dei connect Internet di documentazione. Daemon solo connect/close, nessuna
richiesta; path assenti non prova di mascheramento. Wrapper unico e prerequisiti
Python/Pandoc/Git/tokenizer/tmp/workspace completi. Niente S/V0 o conversioni.

Ogni gruppo identifica prima degli avvii bundle con helper/target/argv/cwd/env/hash
esatti e contesto; conserva receipt/output/raw/log/parziali. Scope **PRELIMINARY**,
non ufficiale. Massimo2gruppi/sei chiamate, figlio120s/esterno300s, totale1800s.
Un FAIL operativo sospende il gruppo; alternativa candidata è espressamente
ammessa. Secondo gruppo solo dopo correzione ordinaria motivata ammessa, nuovo
bundle/tmp/output, nessun retry cieco o target riparato. Garanzie non allentabili.
Raccogli le sessioni avviate prima di proseguire; non rilanciarle se attive.

## Budget e consegna unica

Ledger nuova fase: prep16MiB incluse tutte le nuove evidenze/work/request/report;
riserva future R/S/V0 ufficiali64MiB più16MiB esterni. Run500/stop384MiB invariati;
consumo storico, vecchi FAIL e pilota inclusi. Ammissione install112MiB storica
conclusa, non sommarla di nuovo. Response/checks supervisore danno il consumo
iniziale di questo mandato; misura logico e allocato prima/dopo attività. Gate:
max(run live) più quote nuove ancora non consumate sotto384MiB e free516MiB.
Nessun cleanup per passare. Log1MiB/stream,JSON8MiB,RLIMIT_FSIZE32MiB/file.
Limiti effettivi nei launcher, misure complete pre/post, link storici non seguiti;
nessuna quota atomica dichiarata. Solo socket/sentinel propri chiusi e rimossi
come da R, mantenendo evidenze. Zero download/costi remoti/pesi/font.

Se strumenti/input pronti e R preliminare valida, consegna request schema1
`implementation/stages/impl-r001-stage-baseline-s006/request.json`, status
WAITING_FOR_STAGE_SNAPSHOT, owner supervisor. Comandi/profili/env/costi completi
per **R→S→V0 ufficiali dopo il freeze**, nessun placeholder operativo irrisolto
salvo SHA del futuro stage/S esplicitamente tipizzati. Includi originali/fixture/
inventari/scope/bundle esistenti; lista artifact solo relativa root e dentro run,
file root già in snapshot.files separati. Non includere output futuri o S che
citi il proprio futuro snapshot. Nessun snapshot autonomo, nuova R/S/V0 ufficiale
richiede ripetizione dopo freeze. F1–F6/contenuto e suite sono future prove, non
attestazioni di questa chat. Riserva64MiB deve coprire i comandi proposti.

Preserva report/checkpoint d’ingresso, aggiorna solo i tuoi report/checkpoint,
request e nuove evidenze; consegna delivery.md con lineage e risultati separati.
Se impedimento sostanziale, consegna parziali/causa e WAITING_FOR_SUPERVISOR_RECEPTION,
nessuna request detta pronta. Prossimo supervisore prompt05 **con scope e contesto
r002 correnti**. Nessun servizio automatico da attendere o consegna intermedia
per normale correzione statica.

Baseline s005/V0storico62pass5fail/perdite/FAIL e path mancanti preservati. Produzione
managed/lock/grafo/backend/S-B-I-E/V1–V9/V7V8 e due review indipendenti reali
ChatGPT/Claude/arbitrato finale futuri,V10V11/pesi/font/inferenza esclusi.
Nessun GO codice/commit/merge/push/promozione/deploy o invio remoto; Git manuale.
