# Implementazione r001 — pilota preliminare recupero baseline

Agisci esclusivamente come **implementatore**, nuova chat, repository
`/home/davide/workarea/markdown-for-llms`. Usa manage-implementation-run.
Obiettivo: correggere gli strumenti del recupero baseline, verificarli con prove
preliminari offline delimitate e consegnare input stabili per lo stage ufficiale.
Nessuna delega, supervisione o review; nessun aggiornamento di registri comuni.

## Mandato corrente e letture

Questo prompt sostituisce prompt28 per decisione esplicita dell'utente il 2026-10-05.
Leggi AGENTS, protocollo (sezione mandati), ADR0007, indici/roadmap, STATE/HANDOVER
correnti, brief A1–A7, piano r003 integrale se non già letto, arbitrato D1–D5,
[addendum operativo r001](../arbitrations/addendum-operational-protocol-r001.md),
prompt04 e checkpoint `handovers/implementation-r001.md` con request/completion
esatte. Checkpoint e prompt storici non cambiano questo ruolo o mandato.
Leggi prompt28 come specifica tecnica della correzione: STATIC_PREPARATION_ONLY,
vecchio contesto r003 e divieto di ogni nuova venv sono sostituiti qui per il pilota;
restano la riproduzione sintetica, diagnostici e tutti i requisiti tecnici non variati.

Percorsi seguenti relativi a `temp/run-a001-fase0-uv/`:

- `evidence/supervisor-implementation-r001/protocol-simplification-r001/`:
  `pilot-scope.json` (autorità operativa), `reception.json`, `transition.json`,
  `response.json` (identità del nuovo contesto), `checks.json`;
- `snapshots/baseline-recovery-pilot-context-r001.json`;
- `evidence/supervisor-implementation-r001/package-s006-reception-r001/`:
  decisione/fonti/receipt; descrivono il FAIL storico, non autorizzano nuove prove;
- `implementation/stages/impl-r001-stage-package-s006/`: request e completion;
- `evidence/implementation-r001/preparation-baseline-install-r002/`: helper,
  requisiti, modelli byte, budget/fonti e test da rivalidare, sola lettura;
- `work/baseline-recovery-r001/`:18asset s003 e copie; report s004 in
  `work/baseline-recovery-inspection-r002/archive-report.json`, review startup
  identificata dalla request s006, originali10moduli/quattro build input/fixture;
- `work/baseline-recovery-install-r001/` e r002 con lock: preservati, non
  eseguire/importare/riparare. FAIL s003/s005/s006 invariati.

Piano SHA `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`,
arbitrato `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`,
prompt04 `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035`.
GO piano nei limiti; nessun GO codice, nuova review o PASS trasferito.
Git branch feature/run-a001-uv, HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091.
Le modifiche documentali di adozione sono elencate in transition/response: verifica
Git reale contro reception/transition, non i vecchi11tracked delle request storiche.

Ricalcola SHA/byte del manifest rispetto a response.json e da radice esegui:
`python3 scripts/run_context.py verify run-a001-fase0-uv --label baseline-recovery-pilot-context-r001`.
Verifica scope/addendum/input fissi contro il manifest prima di ogni tool operativo,
dopo ciascuno e alla consegna. Mismatch/assente: STOP; nessun auto-freeze o fix helper
per nasconderlo. Le nuove preparazioni e output d'autore non sono nel contesto di
ripresa; ogni versione eseguita richiede il manifest del tentativo descritto sotto.

## Preparazione e correzioni nella stessa chat

Crea `evidence/implementation-r001/preparation-baseline-install-r003/`, se assente.
Deriva nuove versioni dai r002, preservando r002. Riproduci deterministicamente il
meccanismo lstat/rename con fixture sintetica; la causa storica resta inferita
perché il traceback s006 manca. Implementa soltanto la politica live_monitor_policy
esatta in pilot-scope.json, con contatori/diagnostici bounded prima del cleanup.

Solo FileNotFound su temporaneo regolare `pyc_atteso.<id_decimale>` sotto pip della
venv del tentativo mentre il proprio seed è attivo consente nuove scansioni.
Il pyc base deve essere uno dei495 attesi da fonti pinned. Massimo due scansioni
complete aggiuntive per campione, entro deadline figlio. Campione incompleto
registrato, mai inventario o contabilità valida; se non ne ottieni uno completo,
STOP. Nessuna eccezione per parent/root/file fissi, pyc finale, bundled wheel,
link/speciali, PermissionError, cap/timeout o altri errori. Fuori seed e dopo child
scansioni strict. Estendere path/errori/fasi o cambiare protezioni: supervisore.

Conserva byte/mode esatti dei quattro template, Activate.ps1 CRLF9033byte/247CRLF,
SHA3795a060dea7d621320d6d841deb37591fadf7f5592c5cb2286f9867af0e91df,
negativi LF/mode. Rivalida le guardie/regressioni precedenti sui sorgenti finali
aggiungendo i casi significativi monitor e negativi preliminary/official;
identifica hash effettivamente testati prima/dopo, output/exit, errori conservati.
Non basta il precedente53PASS. Nessuna esclusione di file per ottenere PASS.
Un modello di attese errato si corregge soltanto con oracle/fonti pinned
indipendenti: non cambiare hash/pin attesi sulla sola base di un output inatteso.

Il driver ufficiale futuro deve mantenere request/stage/scope supervisore congelati
come gate. Implementa un entrypoint **distinto** per il pilota: scope preliminary,
contesto e scope supervisore congelati più manifest chiuso del tentativo.
Non indebolire verify_stage, non usare --standalone come evidenza ufficiale e non
fabbricare scope del supervisore. I guardiani condivisi possono essere riutilizzati
con input espliciti; rifiuta scope errati prima di lanciare figli.

## Identità e operazioni preliminari

Sono autorizzati al massimo due tentativi, **solo nei path/label della scope**:
`work/baseline-recovery-pilot-r001/attempt-001` e `attempt-002`, lock adiacenti.
Directory/lock assenti alla prima apertura, creati esclusivamente dal passo directories.
Ogni tentativo conserva cinque tool separati: directories→venv→install→pip-check→inventory.
Controlla exit0, receipt completa, predecessore/attempt/context esatti e hash output
prima di proseguire. Sessione in corso: raccoglila, non rilanciare. Arresta al primo
FAIL; passi seguenti NOT_EXECUTED. Un solo tentativo completo riuscito è sufficiente.

Prima di ogni tentativo salva in
`evidence/implementation-r001/pilot-baseline-recovery-r001/attempt-NNN/inputs/`
le versioni reali degli helper eseguibili e dei modelli/commands/budget. Compila
`attempt-inputs.json`: scope preliminary, run/label/target, SHA context e scope,
file con byte/SHA/mode, input originali/host/config derivati dalla request s006 e
verificati (la request è fonte, non nuova autorizzazione), argv/cwd/env esatti,
attese legate al nuovo target, costi/assenze/deadline. I file copiati sono gli
eseguibili effettivi; rigenera i path in commands prima di congelare questo manifest.
Versione e manifest immutabili durante il tentativo, controllati prima/dopo ogni tool.
Nessun self-hash, output futuro o receipt inclusi come input del medesimo manifest.
Ricontrolla integralmente fonti host/stdlib, config e assenze s006 pertinenti;
i metadata tracciati correnti vengono dal nuovo contesto, non dalla vecchia request.

Le receipt preliminari riportano scope preliminary, attempt id, SHA context/scope/
manifest input, argv/cwd/env pubblico, figli/sessioni/timestamp/exit, guardie,
log/copied output e hash. Anche un esito positivo resta PRELIMINARY_ONLY:
nessun R/S/V0 o ricevuta ufficiale S/B/I/E. Prima delle prove reali dimostra con
negativi sintetici che helper/input mutati, scope ufficiale usato nel pilota e
receipt di altro tentativo sono rifiutati prima del figlio operativo.

Operazioni e guardie tecniche restano quelle del mandato s006, adattate soltanto
ai path del tentativo e all'identità preliminary:

1. directories: lock/target/directory/riserva4MiB esclusivi e tokenizer locale
   esatto, nessun GET/tokenizzazione;
2. venv: bootstrap assoluto `/home/davide/.pyenv/versions/3.12.3/bin/python3 -I -B
   -m venv --copies --without-pip`; controlla origine/binari/cfg/site vuoto prima
   del nuovo interprete, poi seed separato bundledpip24 con adattamento process-local
   -I -B e tmp esclusivo come seed r002. Nessun upgrade/host source mutato;
3. install: nuovo python -I -B -m pip --isolated --require-virtualenv --no-input
   --disable-pip-version-check install --no-index --require-hashes --only-binary=:all:
   --no-deps --no-cache-dir --no-compile -r requirements congelato del tentativo;
4. pip-check: stesso prefisso pip con check dopo le guardie;
5. inventory: bootstrap -I -B con inventario chiuso18distribuzioni(17+pip24),
   RECORD/direct_url/script/495pyc/origine/tokenizer, copia identica fino8MiB,
   senza import applicativi/native/plugin o esecuzione entry point.

Requirements17URI/hash identico r002:3898byte,
SHA5c7a72dfbd59fb28b1375deeef30e46761d3a45b3bceaf1e9a97c5ba593352f5.
Bundledpip24.0:2110226byte/SHAba0d021a166865d2265246961bec0152ff124de910c5cc39f1156ce3fa7c69dc;
bootstrap aliaspython3.12 SHAb7e2da38f91d2b8dd6d08afb6262c9c413d8bd222a25ef831da9e27ef4e0a807.
Mantieni startup/origin e hash a1_coverage.pth prima di ogni nuovo avvio:
-I non disattiva .pth; COVERAGE_PROCESS_START e COVERAGE_PROCESS_CONFIG assenti.
Resolver interno pinned per soli URI fissi/seed ammesso, nessun grafo aggiuntivo,
backend/build/sdist/indice remoto. Nessuna ABI applicativa provata da pip-check.

Ricostruisci integralmente environment_child della request s006, cambiando solo
TMPDIR, XDG_CACHE_HOME, XDG_CONFIG_HOME, UV_CACHE_DIR e TIKTOKEN_CACHE_DIR verso
il nuovo target. Ambiente driver identico ai figli; non fondere os.environ e non
lanciare direttamente da shell aggiungendo PWD/SHLVL. Launcher bootstrap -I -B,
subprocess argv strutturati, env=dict(esatto), shell=False, close_fds=True.
HOME/PATH/LANG/LC_ALL/TZ e altre variabili rimangono esatte. No proxy/credenziali/
PYTHONPATH/PYTHONHOME/coverage/override. Salva argv reali e output senza segreti.

## Budget e arresti

Scope definisce run500MiB/stop384MiB, nuovo lavoro cumulativo64MiB inclusa prep16MiB,
riserva112MiB per ufficiale +16MiB esterni. Misura lstat run intera e delta pilot
max(logical,allocated), contando prep/evidenze/copie/entrambi tentativi una volta.
I due target falliti storici sono già nel totale. Non seguire14link storici;
nuovi target zero link. Cap target64MiB/4096file/1024directory, free516MiB,
perfile32MiB/stream1MiB/JSON8MiB/receipt4MiB/monitor50ms invariati o più stretti.
Riserva ufficiale è una stima di ammissione futura, non autorizzazione installer.

Prima della prima scrittura di preparazione registra baseline logica/allocata del
nuovo lavoro (solo evidenze del supervisore nuove già esistenti fuori delta autore).
Prima di ogni passo verifica totale vivo + allowance pilot residua +112MiB+16MiB
sotto384MiB e stima parti del lavoro residuo entro allowance. Anche evidenze di
consegna rientrano. Monitor resta non atomico, nessuna garanzia fisica della somma.
Deadline venv+seed120s/install180s/check60s/inventory120s, controlli e copie aggiuntivi;
non interrompere tool creando orfani. Raccogli solo gruppo/sessioni figli propri.

Al primo errore preserva target/lock/tmp/log/receipt/parziali e raccogli i processi.
Solo difetto ordinario con causa documentata e correzione motivata può portare al
secondo target nella stessa chat, dopo regressioni/input nuovi e budget ammesso.
Nessun reset/cleanup/retry dello stesso target. Errori di integrità/input/host,
cap/timeout/ENOSPC/PermissionError, rifiuto sandbox, nuovi costi/dipendenze/privilegi,
estensione della politica monitor o altri confini: STOP e consegna supervisore.
Nessun aggiramento del rifiuto automatico, kill per nome o polling supervisore.

## Consegna per freeze ufficiale

Se il pilota riesce, completa preparazione statica r003 e request ufficiale
`implementation/stages/impl-r001-stage-package-s007/request.json`,
WAITING_FOR_STAGE_SNAPSHOT, owner supervisor, target riservato
`work/baseline-recovery-install-r003` e lock **ancora assenti**.
Modelli/path/config/argv/deps/fonti devono essere rigenerati per quel target;
prove sintetiche sui sorgenti finali del driver ufficiale e delta motivato dal pilota.
Scope ufficiale futura PENDING: non inventarla e non avviare il target ufficiale.
Lista precisa di input esistenti: autorità+addendum+scope pilota, helper finali/
politica/commands/budget/requirements/test/fonti, originali/fixture, asset/report,
FAIL preservati e manifest/ricevute/inventario preliminari prodotti. Nessun output
futuro, copie massive o report/checkpoint mutabili; conserva copie d'ingresso.
Changelog proposto separato, metadata supervisore correnti distinti dai nuovi input.

Se pilota non riesce o non ammesso, consegna WAITING_FOR_SUPERVISOR_RECEPTION,
esiti reali FAIL/IMPEDITA/NOT_EXECUTED, nessuna request dichiarata pronta artificialmente.
In entrambi i casi scrivi delivery.md e completion-preliminary.json nella directory
pilot d'evidenze, aggiorna solo implementation/report-r001.md e proprio checkpoint.
Riporta passaggi di ruolo, tempi reali, budget logico/allocato, esiti per step e
identità, preservazione input/target storici, verify contesto/Git/diff-check,
processi/sessioni conclusi o limiti osservativi. Nessun tempo/risparmio inventato.
Prossimo ruolo supervisore con **prompt30**, che applica addendum e prompt05.

R completa D4/nuovo S/baseline-s006/V0, otto operazioni produzione, managed/lock/
grafo, D2D3/config-only/flag/IDE/S-B-I-E/V1–V9 e review codice/arbitrato futuri.
Baseline/confronti ancora IMPEDITI,6path/tmp/2662record indisponibili; V0 storico
62pass5fail/perdite preservate. V7/V8 obbligatorie costo distinto; V10/V11/pesi/
font/inferenza esclusi. Nessun GO codice, host/sysctl/AppArmor/rete/socket/
privilegi/profili persistenti/altro progetto/invio remoto/cleanup/Git/deploy.
Temp ignorata: trasferire file reali. Git manuale utente dopo arbitrato finale.
