# Implementazione r001 — ispezione offline package-s004

Agisci esclusivamente come **implementatore** in nuova chat,repository
`/home/davide/workarea/markdown-for-llms`. Nessuna delega,supervisione o review.
Mandato: **directories → inspect**,due chiamate distinte,poi consegna al supervisore.
Nessuna rete,nuova acquisizione,estrazione/import dei pacchetti,venv/installazione,
pip/uv,R o prova applicativa. La presenza degli archivi non è recupero baseline.

## Letture e autorità

Leggi AGENTS,skill manage-implementation-run,protocollo,indici architetturali e
roadmap,STATE/HANDOVER comuni,brief A1–A7,piano r003 integrale se non già letto,
arbitrato D1–D5,prompt04/05. Leggi **handovers/implementation-r001.md corrente**
e la request esatta indicata lì;il checkpoint supervisore non cambia il tuo ruolo.
Tutti i percorsi seguenti sono relativi a `temp/run-a001-fase0-uv/`:

- `implementation/stages/impl-r001-stage-package-s004/request.json`;
- `evidence/supervisor-implementation-r001/impl-r001-stage-package-s004/`:
  decision.md,reception.json,authorized-scope.json,transition.json,response.json,
  test-scope-equivalence.json,driver-accounting-diff.txt,audit-stop-r001.json;
- `handovers/supervisor-stage-package-s004.md`;
- `evidence/implementation-r001/preparation-recovery-inspection-r002/`:
  preparation.md,expectations.json,commands.json,budget.json,host-config-inputs.json,
  driver.py,recovery_common.py,inspect_inputs.py,archive_verify.py,requirements_gate.py,
  delivery-audit-r001.json,static-checks-r001.json,synthetic-test-runs-r003.json;
- `evidence/supervisor-implementation-r001/package-s003-reception-r001/`:
  decision.md,requires-dist-comparison.json e reception.json;
- completion/request s003 e `evidence/implementation-r001/resume-package-s003/`
  delivery.md,impediment-r001.json;asset/copie/receipt nel vecchio target.

Prompt22 descrive preparazione già consegnata; questo prompt dispone l'esecuzione.
Non caricare tutta temp. Piano SHA256
`462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`,arbitrato
`f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`,prompt04
`fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035`.
GO piano r003 nei limiti,NO_GO r001/r002 storici,nessun GO codice.

## Identità congelata

Stage **impl-r001-stage-package-s004**:

- manifest `snapshots/impl-r001-stage-package-s004.json`,**606807byte**;
- SHA256 **a2032cd82d7791b4d2c55fda6a9fb49615a0430234a7db6c6302304c4bc4ccb2**;
- worktree **52947956e5b408236e865fcd7baa9611596d26c11b496250046a4e7b3dc87a88**;
- **103file/2150artefatti**,MATCH alla consegna;2131input richiesti+19evidenze proprie;
- request **891752byte**,SHA256
  **319ce185f3ffd04f85d2449617d25447adf8d883c8f09fec87eee2a87ba684e3**;
- 201input stabili/2130record/9host/27config ricalcolati,sei metadata separati.

Branch feature/run-a001-uv,HEAD/dev/base
`66ba82200e5def5a4db76f9bafccb0731b506091`,indice vuoto,11tracked modificati e
24file nuovi. Verifica Git reale e hash prima di qualsiasi scrittura operativa;
le modifiche P1–P3 previste non sono divergenze ignote. Ricalcola manifest/request/
piano/arbitrato e verifica stage s004 con run_context dalla radice.
Mismatch => STOP e consegna,nessuna alterazione di helper/request per superarlo.

Contesto precedente baseline-recovery-inspection-preparation-context-r001 storico
solo per sei metadata aggiornati prima del freeze,delta in transition.json;
2023artefatti invariati,1889artefatti s003 invariati. Entry_worktree della request
identifica l'ingresso;il driver verifica i metadata contro **s004 corrente**.
Non pretendere MATCH dal vecchio snapshot o trasferire PASS al worktree nuovo.
Response/questo prompt/checkpoint successivi sono fuori dal manifest,senza cicli.

Bootstrap `/home/davide/.pyenv/versions/3.12.3/bin/python3`,alias host verso
python3.12,SHA `b7e2da38f91d2b8dd6d08afb6262c9c413d8bd222a25ef831da9e27ef4e0a807`.
Nessun interprete alternativo o software aggiunto. Le identità host/config nella
request restano autorità;non esporre segreti o valori di configurazioni sensibili.

## Disposizione sui test e sulle equivalenze

S003 acquisizione18asset/6555654byte e18copie accolti;inspect FAIL/exit1 conservato,
archive-report/copia assenti. Nessun reset dei marker/lock/receipt s003.
Quindici coppie finite Requires-Dist di idna6/pygments1/pytest-cov3/urllib3 5
identificate con SHA e fonti sono equivalenti,altre13wheel confronto esatto.
Il gate nuovo proietta solo quelle stringhe,conservando molteplicità;nessuna
normalizzazione generale o accettazione automatica di nuovo mismatch. L'inferenza
del primo FAIL su idna non proviene dal traceback storico e resta dichiarata tale.

55test sintetici autore PASS documentati. Supervisore ha rilevato driver cambiato
dopo le prove:versione testata SHA5df93526f95cc7c8d195ef9cfff8e1eb168c7fd8938058ca0b360b1a78689855,
finale SHAa19b8b24c899c6e436684d3fa614405d69a028d09135044c598c8702e9764c18.
Sola inizializzazione budget ora include l'addebito storico;require_report testata
identica e resto AST fuori budget invariato. Equivalenza esplicita solo per
quella funzione;altri helper/test con hash identici. Non dichiarare driver finale,
deadline/processi/lock/budget collaudati dai55test. Il primo rilievo è preservato.
Non ripetere i sintetici per sostituire la prova operativa ora autorizzata.

## Due operazioni esatte

Origine read-only `work/baseline-recovery-r001`:17wheel+tokenizer e18copie,
già presenti e congelati. Nuovo output
`/home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/work/baseline-recovery-inspection-r002`
e lock adiacente `baseline-recovery-inspection-r002.lock`,entrambi assenti
all'ingresso previsto. Il parent work esiste. Non creare target/lock/cache a mano.

Esegui ciascun array come una chiamata distinta,dalla radice repository,
argv strutturato,shell=False;nessun comando concatenato. Unica sostituzione della
request è lo SHA stage reale,già valorizzato qui.

```json
["/home/davide/.pyenv/versions/3.12.3/bin/python3", "-I", "-B", "/home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/evidence/implementation-r001/preparation-recovery-inspection-r002/driver.py", "directories", "--stage-sha", "a2032cd82d7791b4d2c55fda6a9fb49615a0430234a7db6c6302304c4bc4ccb2"]
```

```json
["/home/davide/.pyenv/versions/3.12.3/bin/python3", "-I", "-B", "/home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/evidence/implementation-r001/preparation-recovery-inspection-r002/driver.py", "inspect", "--stage-sha", "a2032cd82d7791b4d2c55fda6a9fb49615a0430234a7db6c6302304c4bc4ccb2"]
```

Prima del secondo passo verifica exit0 e receipt directories completa,
PASS_DIRECTORIES_ONLY,stesso stage. Qualsiasi errore => STOP al primo impedimento,
conserva evidenze e consegna. Nessun tentativo con vecchio driver,nuova directory
scelta autonomamente,reset marker,modifica gate o retry sotto la stessa label.
Se il tool restituisce una sessione,raccogli **quel processo** fino all'esito,
non avviare il comando nuovamente. Non aggirare un rifiuto del tool;registra
distintamente rifiuto/timeout/FAIL/runner assente e informa il supervisore.

Directories crea sole directory assegnate e riserva receipt4MiB,lock esclusivo;
stima5s,non deadline sincrona. Inspect figlio120s,preflight30s/verifiche/copie
padre e attese tool aggiuntive. Driver raccoglie/termina solo il Popen creato,
nessun processo/gruppo esterno. Nessuna rete o nuova acquisizione in questi argv.

Host effettivo CPython3.12.3,linuxGNU x86_64,glibc>=2.28,SOABI non debug;
controllo compatibilità non è caricamento dei moduli nativi. ZIP/CRC/RECORD/
METADATA/WHEEL/tag,chiusura attiva/extra,startup/script/.data rimangono obbligatori.
Tetti4096voci/48MiB cumulativi,32MiB/wheel,16MiB/membro,2048directory incluse vuote,
nomi240byte/profondità8,metadata/startup1MiB. Traversal/duplicati/symlink/speciali,
compressioni ignote,firme o schemi .data/sintassi non supportati => STOP;
non allentare limiti per ottenere PASS e non eseguire istruzioni nei pacchetti.

Errori identificano wheel/hash/controllo e osservazione bounded,receipt parziali
e progress marker conservati. L'ultima wheel avviata in caso timeout è attribuzione
di avanzamento,non prova che essa sia difettosa. Errore preflight prima di scritture:
preserva stderr del tool in evidenze proprie,non inventare receipt operativa.
Mancanza/incompletezza report o receipt,anche ENOSPC,resta mancata riuscita.

## Ambiente e budget

Ambiente figlio esatto da request:HOME /home/davide invariata,PATH bootstrap:
/usr/bin:/bin,LANG=LC_ALL=C.UTF-8,TZ=UTC,TMPDIR/cache UV/TIKTOKEN/XDG nel nuovo
target,UV_PYTHON_DOWNLOADS=never,PIP_CONFIG_FILE=/dev/null,PYTHONDONTWRITEBYTECODE=1,
PYTHONNOUSERSITE=1. Non ereditare segreti/proxy/PYTHONPATH/PYTHONHOME/config
implicite. Registra separatamente ambiente padre tool e figlio;non dichiararli
identici. Non modificare configurazioni host per soddisfare i gate.

Budget500MiB,stima incrementale48MiB,totale conservativo preparato153923584byte.
Stop384MiB,monitor obiettivo50ms,riserva esterna16MiB;**non quota atomica**,la
scansione può allungare l'intervallo e i burst restano possibili. Accettato solo
per questi passi con scritture bounded,non per un futuro installer. Radici live
senza link/speciali più addebito storico fisso;non ricopiare o scandire contenuti
di altri progetti. Due symlink sintetici s002 storici non sono input operativi.

Nuovo target envelope24MiB:4riserva fisica,8report+una copia,2log,1receipt,
1overhead,8margine. Log1MiB/stream,report/copia4MiB ciascuno. Nuove evidenze
supervisore/tool entro riserva16MiB;budget include tutta la storia una sola volta.
Nessuna estensione soglie,nuovi privilegi o servizio a pagamento.

## Consegna e stop prima dell'installazione

Scrivi nuove evidenze solo in `evidence/implementation-r001/resume-package-s004/`;
preserva copie report/checkpoint ricevuti. Salva argv/cwd/tempi/exit e tool result,
eventuali rifiuti,disk logici/allocati,hash input/output e limiti. Non duplicare
tutta la storia. Non modificare request/helper/stage/vecchi raw o registri comuni,
sei metadata,prodotto,test/fixture/diagnostici o evidenze supervisore.

Risultati attesi soltanto se realmente ottenuti:driver PASS_DIRECTORIES_ONLY,
PASS_INSPECT_ONLY;17receipt PASS_WHEEL_ONLY;archive-report
PASS_ARCHIVES_AND_CLOSURE_ONLY,complete=true,17wheel e closure PASS_CLOSURE,
label/stageSHA/expectationsSHA corretti,nessuna estrazione/import. Driver conserva
copia `copies/inspect-00.raw`;verifica identità byte/SHA. Nessuno di questi esiti
è anticipato o equivale a baseline recuperata/test ABI o GO codice.

Produci `implementation/stages/impl-r001-stage-package-s004/completion-r001.json`
con manifest/request/hash reali,passi tentati/esiti,receipt/log/report/copia
esistenti e contabilità. Se presente all'ingresso,STOP e chiarisci stato reale;
non sovrascrivere. In caso errore produci impedimento versionato e parziali,
nessun self-hash o file futuro nella completion. Richiesta e stage restano intatti.

Verifica finale s004 MATCH,Git/diff-check,hash origini/copie e assenze persistenti;
non rieseguire il driver per fare verifiche. Aggiorna solo
`implementation/report-r001.md` e `handovers/implementation-r001.md`,stato
**WAITING_FOR_SUPERVISOR_RECEPTION**,percorso esatto completion/impedimento,
processi propri conclusi o stato reale da raccogliere. Scrivi delivery.md con
risultato e limiti,poi consegna **supervisore prompt05**. Nessun servizio in attesa.

Anche dopo PASS devi fermarti:ricezione/review supervisore SHA/startup/scripts/
.data/budget prima di venv/install. Non compilare tu tale review,non usare il
driver s003 per installare,non preparare un GO fittizio o una nuova auto-freeze.

## Obblighi conservati

Baseline6path/tmp assenti,2662record indisponibili;pulizia riferita probabile,
non provata. V0s005 caratterizzazione PASS,suiteFAIL/exit1/67nodeid/201eventi/
62pass5fail/perdite F2/F5/anchor/separatori/bundle/ASCII invariati. Confronti
IMPEDITI,nessuna equivalenza venv/PASS trasferito. Dopo recupero ricevuto serve
baseline-s006/R completa D4,parent/child/TMPDIR/namespace/daemon/blacklist/
socketpair e nuovo S. Originali10moduli+4build input/fixture separati daiP1–P3.
Regex2026baseline incompatibile Marker<2025:nessun pin trasferito al prodotto.

Otto software produzione differiti,uv0.10.10/managedPython3.12.13/setuptools84,
CPU/cu126/no-build/extra invariati;selectorGNU/build20260310/redirect/cap e grafo
universale aperti. D2D3/config-only/flag/IDE/current↔S↔snapshot/S-B-I-E/V1–V9,
due review indipendenti ChatGPT/Claude e arbitrato finale futuri. V7/V8 obbligatorie
costo distinto;V10/V11/pesi/font/inferenza esclusi. Nessun GO codice,cleanup,
modifica sysctl/AppArmor/rete/socket host,privilegi/setuid/profili persistenti,
altri progetti,commit/merge/push/promozione/deploy o invio documenti.
Git manuale utente dopo arbitrato finale;temp ignorata:trasferire i file reali.
