# Implementazione r001 — preparazione corretta installazione baseline package-s006

Agisci esclusivamente come **implementatore**, nuova chat, repository
`/home/davide/workarea/markdown-for-llms`. Nessuna delega, supervisione o review.
Mandato **STATIC_PREPARATION_ONLY**: correggere il modello delle attese dei
file venv in una nuova preparazione e consegnare request package-s006.
Nessun venv,ensurepip,pip/install,uv/download o test applicativo/R in questa chat.
Nessuna ripresa dello stage s005: FAIL e target restano immutabili.

## Letture e identità

Leggi AGENTS,skill manage-implementation-run,protocollo,indici architetturali/
decisioni/roadmap,STATE/HANDOVER comuni,brief A1–A7,piano r003 integrale se non
già letto,arbitrato D1–D5,prompt04/05. Leggi checkpoint autore corrente
`temp/run-a001-fase0-uv/handovers/implementation-r001.md` e request/completion
esattamente indicati. Il ruolo supervisore nei documenti non cambia il tuo ruolo.

Piano SHA256 `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`,
arbitrato `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`,
prompt04 `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035`.
GO piano r003 nei limiti,NO_GO r001/r002 storici,nessun GO codice.

Percorsi seguenti relativi a `temp/run-a001-fase0-uv/`:

- `evidence/supervisor-implementation-r001/package-s005-reception-r001/`:
  decision.md,reception.json,source-readings.json,next-scope.json,transition.json,
  response.json,checks.json;
- `handovers/supervisor-package-s005-reception-r001.md`;
- `snapshots/baseline-install-preparation-context-r002.json`;
- `implementation/stages/impl-r001-stage-package-s005/request.json` e completion-r001.json;
- `evidence/implementation-r001/resume-package-s005/`:delivery.md,analisi FAIL,
  preserved-output-hashes.json,launcher/call/result/log/copie/checks reali;
- `evidence/implementation-r001/preparation-baseline-install-r001/`:preparation,
  prepare_inputs.py,driver.py,install_common.py,seed_pip.py,inventory_installed.py,
  assemble_request.py,manifest/requirements/commands/budget/fonti/test/checks;
- `work/baseline-recovery-install-r001/` e lock adiacente:sola lettura archivistica,
  **non avviare il suo interprete/pip/script,non importare moduli da esso**;
- asset18/copie s003 in `work/baseline-recovery-r001/`,report s004 in
  `work/baseline-recovery-inspection-r002/archive-report.json`,expectations e
  review input ricevuta già identificate nella request s005.

Identità del **nuovo contesto preparatorio r002** in response.json supervisore,
esterna al manifest per evitare self-hash. Prima di scritture verifica SHA/byte
manifest e da radice `python3 scripts/run_context.py verify run-a001-fase0-uv --label baseline-install-preparation-context-r002`.
Assenza/mismatch:STOP,non inventare hash o un freeze. Questo contesto non è lo
stage package-s006:request s006 ancora da preparare. S005 storico soltanto per
sei metadata in transition.json;2488artefatti e1044file del target/lock intatti.
Non richiedere MATCH del worktree s005 contro metadata nuovi.

S005 manifest707662byte/SHA31ade77500ac042f0250fa208246e3ed3f93a5b18d7f06c6b08e0a9380848342;
request2163476byte/SHA2a45920ca2741b8b484674117dcb9f262e4c197f9a1d6c9c67d28236b4e70b19;
completion7894byte/SHA68c7948dcdb9de0496dd342c1e3c228abf466b6db2b28ee2518112f995b6401c.
Git feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto,11tracked modificati/24nuovi. Verifica Git reale;P1–P3 statici
non sono divergenza ignota. Copia report/checkpoint ricevuti prima di aggiornarli.

## FAIL ricevuto e correzione richiesta

Directories PASS/exit0;venv FAIL/exit1 complete=false. Figli venv/seed exit0,
pip24 seeded e PASS_SEED_ONLY,ma postcheck driver fallito. Install17wheel,
pip-check,inventory NOT_EXECUTED. Nessun inventario18dist o ABI/R/V0 provato.
Il FAIL resta FAIL;nessun retry,cleanup,normalizzazione output o receipt promossa.

Causa verificata dal supervisore su fonti pinned: prepare_inputs.py:114 usa
read_text e normalizza i newline;venv.install_scripts legge rb,decode UTF-8,
sostituisce le variabili,encode UTF-8,scrive wb e copia i mode. Activate.ps1
reale è identico al template:9033byte,247CRLF,SHA
3795a060dea7d621320d6d841deb37591fadf7f5592c5cb2286f9867af0e91df.
Atteso8786byte/SHA3a8a32630c8523f31c2e2cbb1be266a7a836320cb024c67e31e5fdf5ff154c69
è la versione LF solo confrontata in memoria. Nessuna modifica del file reale.
Fra i soli venv_files del manifest questo è l'unico mismatch riscontrato;
non dedurre che tutti gli altri gate/installazioni passerebbero.

Prepara modello fedele ai **byte** della fonte pinned per tutti i template
common/posix e le sostituzioni previste,senza normalizzazione newline involontaria.
Non copiare l'hash del target fallito come unica oracle:deriva le attese dal
template identificato e dalla trasformazione documentata. Non escludere
Activate.ps1,non ammettere hash alternativi,non normalizzare i file durante il
confronto e non indebolire l'inventario chiuso per aggirare il FAIL.

Aggiungi regressioni sintetiche stdlib significative:CRLF,LF,misti,newline finale
preservati;template pinned common/posix e contesto/path nuovo;caso reale9033byte,
negativo che rifiuti la mutazione LF8786byte. L'atteso dei test deve essere
indipendente dalla funzione sotto test,non duplicarne la stessa normalizzazione.
Identifica sorgenti finali effettivamente testati con SHA e log/exit,conserva
errori/intermedi. Nessun test deve eseguire venv/pip/installer/socket/rete,
script di attivazione,app/native/plugin. I41sintetici s005 sono storici;
non trasferire il loro PASS alle nuove versioni.

## Pacchetto nuovo e gate invariati

Nuova directory `evidence/implementation-r001/preparation-baseline-install-r002/`.
Nuovo target futuro `work/baseline-recovery-install-r002` e lock adiacente `.lock`,
assenti ora:non crearli in preparazione. Nuova request
`implementation/stages/impl-r001-stage-package-s006/request.json`,label
`impl-r001-stage-package-s006`,fase package,subphase BASELINE_RECOVERY_INSTALL_INPUTS,
WAITING_FOR_STAGE_SNAPSHOT,snapshot_owner supervisor. Se questi path esistono
inaspettatamente,STOP e documenta,non sovrascrivere o scegliere nomi ad hoc.

Versiona gli helper derivati,identifica diff motivato da r001 e rigenera tutte
le attese dipendenti dal nuovo path:pyvenv.cfg,template/script/shebang/entrypoint,
pyc/co_filename/RECORD/direct_url,ambiente/config/assenze,argv e gate di stage.
Non assumere che la sola correzione CRLF basti a spostare il target. Report s004,
17URI/hash e18asset s003,originali10moduli+4build input/fixture restano immutabili.
Requirements devono restare identici come URI/hash;nessun pin/extra cambiato.

Cinque passi futuri distinti directories/venv/install/pip-check/inventory,
nessuno eseguito ora. Mantieni guardie prima di scritture e dei nuovi interpreti,
argv strutturati/shell=False/close_fds,ambiente driver **esatto** come quello
figlio senza eredità shell,lock/marker/receipt esclusivi,predecessori completi,
STOP senza retry,timeout/log/copie bounded e soli processi propri raccolti.
Deadline proposte120s aggregata venv+seed/180s install/60s check/120s inventory,
preflight e copie aggiuntivi;qualunque variazione motivata in request.

Conserva il seed separato --copies --without-pip,lib64 directory reale vuota,
origine/config/site vuoto prima primo interprete,helper ensurepip con -I -B e
temporaneo esclusivo preservato FAIL/spostato copies sul successo. Bundledpip24
2110226byte/SHAba0d021a166865d2265246961bec0152ff124de910c5cc39f1156ce3fa7c69dc;
bootstrap assoluto `/home/davide/.pyenv/versions/3.12.3/bin/python3`,alias python3.12
SHAb7e2da38f91d2b8dd6d08afb6262c9c413d8bd222a25ef831da9e27ef4e0a807.
Nessuna fonte host modificata o upgrade. Audit hook seed non è isolamento rete
del figlio. Nuova guardia supervisor scope deve richiedere autorizzazione futura
con request/budget s006:non leggere il vecchio scope come permesso per r002,
non scrivere tu authorized-scope.json o receipt supervisore.

Precisazione già accolta:resolver interno pip per requisiti diretti fissi/seed
non equivale a nuove dipendenze. Mantieni --no-index --require-hashes
--only-binary=:all: --no-deps --no-cache-dir --no-compile e prefisso --isolated
--require-virtualenv --no-input --disable-pip-version-check. Nessun grafo nuovo,
backend/sdist/build/indice;il nuovo installer richiederà comunque nuovo scope.

-I non disabilita a1_coverage.pth:ramo stdlib eseguito,ramo coverage inattivo
solo con COVERAGE_PROCESS_START/CONFIG escluse e startup/hash prima avvii.
Nessun sitecustomize/usercustomize/startup estraneo o entrypoint/plugin/test
invocato;cinque ELF non caricati. Ambiente HOME invariata,PATH bootstrap:/usr/bin:/bin,
LANG/LC_ALL C.UTF-8,TZ UTC,cache/tmp/config nuovo target,PIP_CONFIG_FILE=/dev/null,
UV_PYTHON_DOWNLOADS=never,PYTHONDONTWRITEBYTECODE=1,PYTHONNOUSERSITE=1 e -B esplicito.
Niente proxy/credenziali/PYTHONPATH/PYTHONHOME/override pip/uv/pytest/coverage
impliciti. Verifica fonti pinned e interazioni isolate/ensurepip;identità host,
stdlib/startup/config inventariate,non valori segreti nei log.

## Budget e freeze futuro

Mandato500MiB invariato;preparazione statica nuova fino16MiB,riserva esterna16MiB,
stop384MiB e misura viva logical/allocated intera run senza doppie somme.
Nessun cleanup o aumento per far passare il gate. Link storici osservati **10**,
già10 nella reception s005:il testo prompt25 diceva8 erroneamente. Lstat senza
seguirli,nuovo target senza link. Usa misure reali aggiornate,non8 hardcoded.

Nuova stima installer deve includere target fallito conservato24.9MB allocati,
nuova storia/snapshot/evidenze,nuovo target112MiB proposti o stima rivista con
parti esplicite e margine. Monitor50ms non quota atomica;stop128MiB target,
384MiB run,RLIMIT_FSIZE32MiB per file,log1MiB/stream,JSON8MiB,riserva4MiB receipt
non danno garanzia fisica della somma. Conserva gate spazio libero516MiB,
limiti e failure,oppure proponi variazione motivata da valutare. Nessun software
operativo viene autorizzato dalla stima o dal presente prompt.

Request contiene lista precisa di input esistenti relativi/non evasivi/senza
symlink e record hash/byte:autorità,nuovi helper/manifests/requirements/fonti,
synthetic-final e prove reali pertinenti,completion/FAIL s005 e inventario
preservato,manifest precedente,originali/fixture. Escludi report/checkpoint
mutabili correnti;usa copie ricevute. Sei metadata supervisore separati
verificabili contro futuro snapshot. Nessun file futuro/S/autorizzazione futura
in lista,nessun self-hash request. Snapshot/receipt nuove spettano al supervisore.

Verifica nuova preparazione AST/sintetici stdlib proporzionati,hash/preservazione
s005 e input s003/s004,contesto r002 MATCH,Git/diff-check e link. Non importare gli
helper operativi per eseguirne effetti o usare la venv fallita per i test.
Errori conservati,nessun test operativo mascherato da sintetico.

## Consegna

Produci preparation.md,diff motivato,fonti e aspettative con prove regressione,
comandi/ambienti/costi/gate concreti,request s006. Aggiorna solo report autore
`implementation/report-r001.md` e checkpoint `handovers/implementation-r001.md`,
con copie d'ingresso,**WAITING_FOR_STAGE_SNAPSHOT**,prossimo ruolo supervisore
con prompt05. Nessuno stato condiviso/changelog/snapshot da implementatore;
puoi consegnare proposta changelog separata. Non attendere servizio automatico.

Confronti baseline IMPEDITI:sei path/tmp assenti,2662record indisponibili,
pulizia probabile riferita non dimostrata. FAIL s003/s005 e V0s005 storico
caratterizzazionePASS/suiteFAIL67nodeid201eventi62pass5fail/perditeF2/F5/anchor/
separatori/bundle/ASCII conservati. Nessuna equivalenza vecchia venv o PASS
trasferito. Dopo recupero ricevuto:baseline-s006 (distinto da package-s006),
R completaD4/nuovoS,parent/child/TMPDIR/namespaces/daemon blacklist/socketpair,
non autorizzati qui. Originali separati daP1–P3.

Otto operazioni produzione differite:managed-acquisition,universal-lock-no-build,
lock-check,entry-deps-without-project,backend-entry-no-build,sdist-and-wheel,
base-sync,canonical-wheel-install. Uv0.10.10/managed3.12.13/setuptools84/CPU-cu126/
no-build/extra invariati;selectorGNU/build20260310/redirect/cap/grafo universale
aperti;regex2026baseline non trasferito a prodotto Marker<2025. D2/D3/config-only/
flag/IDE/current↔S↔snapshot/S-B-I-E/V1–V9 e due review reali indipendenti
ChatGPT/Claude/arbitrato futuri. V7/V8 obbligatorie con costo distinto;
V10/V11/pesi/font/inferenza esclusi,local_marker non collaudato.

Nessun cleanup,modifica sysctl/AppArmor/rete/socket host/privilegi/profili
persistenti/altro progetto,invio documenti,commit/merge/push/promozione/deploy.
Git manuale utente dopo GO finale;temp ignorata:trasferire file reali.
