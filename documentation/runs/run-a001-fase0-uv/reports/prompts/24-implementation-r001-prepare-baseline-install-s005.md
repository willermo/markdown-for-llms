# Implementazione r001 — preparare installazione baseline package-s005

Agisci esclusivamente come **implementatore**,nuova chat,repository
`/home/davide/workarea/markdown-for-llms`. Nessuna delega,supervisione o review.
Mandato **STATIC_PREPARATION_ONLY**: preparare nuova request e strumenti di
installazione offline della baseline. Nessun venv/install/pip/uv/download o
test applicativo/R in questa chat. Gli archivi sono già ricevuti e verificati.

## Letture e identità

Leggi AGENTS,skill manage-implementation-run,protocollo,indici architetturali/
decisioni/roadmap,STATE/HANDOVER comuni,brief A1–A7,piano r003 integrale se non
già letto,arbitrato D1–D5,prompt04/05. Leggi checkpoint autore corrente
`temp/run-a001-fase0-uv/handovers/implementation-r001.md` e request/completion
esattamente indicati. Non trasformare ruolo supervisore nel tuo ruolo.

Piano SHA256 `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`,
arbitrato `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`,
prompt04 `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035`.
GO piano r003 nei limiti,NO_GO r001/r002 storici,nessun GO codice.

Percorsi seguenti relativi a `temp/run-a001-fase0-uv/`:

- `evidence/supervisor-implementation-r001/package-s004-reception-r001/`:
  decision.md,reception.json,startup-member-readings.json,install-input-review.json,
  next-scope.json,transition.json,response.json,checks.json;
- `handovers/supervisor-package-s004-reception-r001.md`;
- `snapshots/baseline-install-preparation-context-r001.json`;
- `implementation/stages/impl-r001-stage-package-s004/` request.json e completion-r001.json;
  `evidence/implementation-r001/resume-package-s004/` delivery/tool/output audit;
- `work/baseline-recovery-inspection-r002/archive-report.json`,receipt/copia;
- `evidence/implementation-r001/preparation-recovery-inspection-r002/`
  expectations.json,driver/helper,budget e checks pertinenti;
- `evidence/implementation-r001/preparation-baseline-recovery-r001/`:
  allowlist,selected-requirements,driver/venv_guard/inventory_recovered,commands,
  pinned-sources/pinned-source-readings,originals-reconciliation e budget;
- asset18 e copie s003 in `work/baseline-recovery-r001/`,in sola lettura.

Il nuovo contesto di preparazione ha identità in **response.json supervisore**,
esterna al manifest per evitare self-hash. Prima di scritture verifica SHA/byte
manifest e run_context verify della label **baseline-install-preparation-context-r001**.
Se assente/mismatch,STOP,non inventare hash. Il vecchio s004 è storico solo per
sei metadata prima del contesto,delta in transition.json,2150artefatti invariati.
Non richiedere MATCH del worktree s004 contro i nuovi metadata.

S004 manifest SHA `a2032cd82d7791b4d2c55fda6a9fb49615a0430234a7db6c6302304c4bc4ccb2`,
request SHA `319ce185f3ffd04f85d2449617d25447adf8d883c8f09fec87eee2a87ba684e3`.
Archive-report196956byte,SHA **b515660170eabc5618ee04d92e467acb2e882cedcd756966ffd7051283b1ffcd**,
expectations SHA **bdbfff72fde92537e5e515edfddd578ed29d7388bf149564393699fa4eb002a4**.
17wheel/784voci/73directory/16691324byte espansi,PASS_ARCHIVES_AND_CLOSURE_ONLY e
PASS_CLOSURE ricevuti;18asset s0036555654byte e copie conservati. ABI nativa
NOT_TESTED;nessuna equivalenza venv o baseline recuperata.

Git feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto,11tracked modificati/24nuovi. Verifica Git reale;P1–P3 predisposti
non sono divergenza ignota. Copia report/checkpoint ricevuti prima di aggiornarli.

## Review startup ricevuta: condizioni vincolanti

Unico `.pth`: **a1_coverage.pth**,byte/SHA esatti nel report e nella review.
La riga esegue import stdlib ed exec di una stringa fissa;il ramo coverage è
condizionato da **COVERAGE_PROCESS_START** o **COVERAGE_PROCESS_CONFIG**.
**-I non disattiva i .pth**. Prepara ambiente figli ricostruito che escluda
entrambe le variabili,e gate prima di ogni avvio dell'interprete della nuova venv.
Prima del primo avvio dopo installazione verifica byte/SHA del .pth installato
con lettura dal bootstrap,oltre assenza di startup estranei/non inventariati.
Non dichiarare il .pth ineseguito in un futuro avvio:il percorso stdlib verrà
eseguito,il ramo coverage deve restare inattivo. Non mutare il .pth per evitarlo.

Nessun sitecustomize/usercustomize o .data negli archivi ricevuti;entry_points
console/pytest11 recensiti ma non invocabili in questa tranche. Non lanciare
pytest o caricare plugin,nemmeno per fare pip-check. CLI dotenv richiede extra
non selezionato,non includerlo tacitamente. Test certifi e completion shell tqdm
non devono essere eseguiti/registrati;cinque ELF non caricati per verificarne
versioni. Nessuna collisione file tra17wheel o col pip24bundled rilevata.

install-input-review accetta i contenuti **per la preparazione**,non autorizza
installer o nuovi argv. Non falsificare campi per soddisfare il driver s003:
quel driver resta immutabile,legato a stage/report/target vecchi. Il budget
installer dovrà essere valutato dal supervisore sulla request reale s005.

## Pacchetto statico richiesto

Nuova directory evidenze `evidence/implementation-r001/preparation-baseline-install-r001/`.
Nuovo target futuro `work/baseline-recovery-install-r001`,lock adiacente dedicato,
assenti ora. Non crearli in questa chat,non riutilizzare target/marker/lock s003/s004.
Origini wheel/tokenizer rimangono nei path s003,report s004 immutabile. Nessun
GET o copia massiva della storia. Se target/request s005 già esistono,STOP e
chiarisci lo stato reale,non sovrascrivere o scegliere nomi ad hoc.

Prepara driver e argv fissi per passi distinti **directories,venv,install,
pip-check,inventory**. Separare i tool futuri e i gate di precedenza. Tutto deve
essere concreto e verificabile prima della nuova request;nessun passo si esegue
ora e nessuna auto-freeze. Include solo output attuali come input dello snapshot.

Bootstrap assoluto `/home/davide/.pyenv/versions/3.12.3/bin/python3`,alias verso
python3.12,SHA `b7e2da38f91d2b8dd6d08afb6262c9c413d8bd222a25ef831da9e27ef4e0a807`.
Pip bundled **24.0**,2110226byte,SHA
`ba0d021a166865d2265246961bec0152ff124de910c5cc39f1156ce3fa7c69dc`.
Leggi fonti locali pinned già identificate (venv/ensurepip/site/pip) e riferisci
flag/config al codice effettivo. Nessuna invocazione pip/help/venv in preparazione.

- Directories:target esclusivo con TMPDIR/cache/config/log/receipt/workspace e
  riserva;predisporre copia **locale** del tokenizer con chiave cache e hash,
  in passo esplicito con costo/provenienza,senza eseguirlo. Non copiare wheel
  salvo necessità concreta motivata;requirements file URI verso archivi congelati.
- Venv:bootstrap `-I -B -m venv --copies`,pip24bundled soltanto,no upgrade-deps.
  Risolvere concretamente lib64 senza symlink secondo sorgente venv pinned;
  nessuna modifica pyenv/global/site o rimozione delle guardie contro link.
  Verifica origine dei binari copiati,pyvenv.cfg,stdlib e pip24 prima di eseguire
  pip. Se manca o differisce qualcosa,STOP,nessun aggiornamento pip automatico.
- Install:nuovo python venv `-I -B -m pip --isolated --require-virtualenv`,
  no-input/disable-version-check,install no-index,require-hashes,only-binary=:all:,
  no-deps,no-cache-dir,no-compile e requirements congelati17URI/hash esatti.
  Nessun resolver/backend/sdist/build o indice remoto. Chiusura s004 è input
  ricevuto;non cambiare pin/extra per far passare installazione.
- Pip-check:passo separato dopo guardie startup/origine,nessun import app/pytest.
- Inventory:17distribuzioni più pip24 con importlib.metadata o letture equivalenti,
  RECORD installati,digest confronto wheel/installato,entrypoint/direct_url,
  assenze startup estranei,origine interpreter/stdlib e tokenizer/copie/cache.
  Identifica eccezioni reali RECORD/pyc/shebang generati dal pip pinned senza
  ignorare file arbitrari;verifica confinamento e symlink. Non importare pacchetti
  nativi/app per leggerne versioni. Baseline-originals10moduli+4build input e
  fixture rimangono ingressi separati e identificati,non copie daP1–P3.

Nessun nuovo R o confronto P0/V0 concatenato ai passi installazione. Un inventario
riuscito non ricrea l'identità completa dell'ambiente /tmp scomparso.

## Preflight, costi, failure e prove statiche

Request/artifacts/host/config e assenze persistenti verificati prima di scritture;
SHA stage futuro unico segnaposto,sei metadata supervisore separati e verificati
contro il futuro snapshot s005. Copie report/checkpoint ricevuti congelate;
correnti esclusi. Ricalcola e collega review/report s004/expectations e18origini/
copie s003;nessuna lettura errata del campo bytes nei record snapshot.

Ambiente figlio ricostruito,HOME invariata,PATH bootstrap:/usr/bin:/bin,locale
C.UTF-8/TZ UTC,TMPDIR/cache/XDG nuovo target,UV_PYTHON_DOWNLOADS=never,
PIP_CONFIG_FILE=/dev/null,PYTHONDONTWRITEBYTECODE=1,PYTHONNOUSERSITE=1.
Identifica anche eventuale pip.conf della nuova venv e startup host pertinenti;
nessun valore segreto nei log. Escludi variabili coverage citate,proxy/credenziali,
PYTHONPATH/PYTHONHOME e override pip/uv/pytest impliciti. Padre tool distinto.
Verifica da sorgenti pinned come isolated e PIP_CONFIG_FILE interagiscono;
ensurepip interno pulisce PIP_* e può compilare pyc nonostante -B del padre.

Mandato500MiB non ampliato. Stima **aggiornata** comprensiva di storia corrente,
16,7MB wheel espansi,pipbundled/pyc/venv,copie/tmp/receipt/log,evidenze e margine;
contabilità logica/allocata senza doppie somme o symlink storici seguiti.
Nuovo target/budget incrementale espliciti,stop384MiB e riserve;monitor50ms è
periodico,non quota atomica. Valuta burst dell'installer,limiti reali dei file
e del lavoro,nessuna garanzia fisica inventata. Nessuna modifica host per quota.
Timeout futuri venv120s/install180s/pip-check60s/inventory120s o modifica motivata
da presentare al supervisore;tempo preflight/copie padre aggiuntivo. Nessuna
stima o gate autorizza implicitamente software operativo in questa chat.

Argv strutturati,shell=False,close_fds,lock/marker/receipt/copie esclusivi,
deadline effettiva,log bounded,raccolta del solo processo proprio. Preserva
partial su FAIL/timeout,nessun retry/reset;preflight senza receipt ha evidenza
tool. Riserva receipt e assenza/incompletezza non PASS. Errori di copia/inventario
o dipendenze devono fermare la catena. Nessuna pulizia dei raw/ambienti falliti.

Ammessi controlli AST e test **sintetici stdlib** nel nuovo spazio evidenze:
positivi/negativi dei gate identità,config/startup,COVERAGE_PROCESS_*,budget,
inventari/RECORD/confinamento/receipt. Nessun subprocess venv/pip/uv,network/
socket,vera installazione/estrazione dei wheel,app/pytest/R. Non scrivere test
che si limitino a rispecchiare l'implementazione;prova le guardie significative.
Conserva log e versioni realmente testate. Se modifichi helper dopo i test,
ripeti prove pertinenti o dichiara hash/delta e limite;non chiamare il driver
finale collaudato da prove fatte su altra versione.

## Request e consegna

Produci `implementation/stages/impl-r001-stage-package-s005/request.json`,schema1,
stage package,label **impl-r001-stage-package-s005**,subphase
**BASELINE_RECOVERY_INSTALL_INPUTS**,status **WAITING_FOR_STAGE_SNAPSHOT**,
snapshot_owner supervisor. Lista esatta input esistenti,hash/byte,comandi/costi/
gate,review e report archivi ricevuti,originali/fixture e request senza self-hash.
Path artefatti relativi canonici,non symlink;identità host separate. Nessun
manifest/venv/receipt/output futuro o S/B/I/E futuro nella lista.

preparation.md,commands,budget,requirements/manifest/helper,test/log/check e
proposta changelog nelle nuove evidenze. Aggiorna solo implementation/report-r001.md
e handovers/implementation-r001.md con request esatta,esiti/limiti/processi propri.
Non scrivere registri comuni,metadata tracciati,prodotto/test/fixture/diagnostici
o evidenze supervisore. Verifica contesto d'ingresso,hash,preservazione e Git/
diff-check,poi consegna **supervisore prompt05**. Nessun servizio in attesa;
nessuna esecuzione prima di ricezione request/freeze/nuovo mandato.

Baseline6path/tmp assenti/2662record indisponibili:confronti IMPEDITI,pulizia
probabile riferita non provata. V0s005 caratterizzazione PASS,suiteFAIL/exit1/
67nodeid/201eventi/62pass5fail/perdite F2/F5/anchor/separatori/bundle/ASCII conservati,
S003 FAIL conservato,nessuna equivalenza venv o PASS trasferito. Dopo recupero
ricevuto baseline-s006/R completa D4,parent/child/TMPDIR/namespace/daemon/
blacklist/socketpair e S. Regex2026baseline incompatibile Marker<2025,nessun pin
prodotto trasferito. Otto software produzione differiti;uv0.10.10/managed3.12.13/
setuptools84/CPU-cu126/no-build/extra invariati,selectorGNU/build20260310/redirect/
cap/grafo universale aperti. D2D3/config-only/flag/IDE/S-B-I-E/V1–V9,due review
ChatGPT/Claude/arbitrato futuri. V7/V8 obbligatorie/costo distinto,V10/V11/pesi/
font/inferenza esclusi. Nessun GO codice,cleanup/modifica host/privilegi/altro
progetto,commit/merge/push/promozione/deploy/invio documenti. Git manuale utente
dopo arbitrato finale;temp ignorata:trasferire file reali.
