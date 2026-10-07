# Implementazione r001 — preparare monitor live corretto, package-s007

Agisci esclusivamente come **implementatore**, nuova chat, repository
`/home/davide/workarea/markdown-for-llms`. Nessuna delega, supervisione o review.
Mandato **STATIC_PREPARATION_ONLY**: nuova preparazione r003, riproduzione
sintetica e correzione del monitor budget live, nuova request package-s007.
Nessun venv/seed/ensurepip/pip/install/uv/download, test applicativo o R ora.
FAIL e target s005/s006 restano immutabili; nessun retry o prosecuzione.

## Letture e contesto

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

- `evidence/supervisor-implementation-r001/package-s006-reception-r001/`:
  decision.md,reception.json,source-readings.json,next-scope.json,transition.json,
  response.json,checks.json;
- `handovers/supervisor-package-s006-reception-r001.md`;
- `snapshots/baseline-install-preparation-context-r003.json`;
- `implementation/stages/impl-r001-stage-package-s006/` request/completion;
- `evidence/implementation-r001/resume-package-s006/`:delivery.md,stop-checks,
  preserved-output-hashes,receipt/log/copie,call/result/session/process/delivery checks;
- `evidence/implementation-r001/preparation-baseline-install-r002/`:helper,
  template_bytes.py,manifest/requirements/commands/budget,fonti e53test finali;
- target falliti `work/baseline-recovery-install-r001/` e r002 con lock adiacenti:
  **solo letture archivistiche,non avviare interpreti/pip/script,non importarli**;
- asset18/copie s003 in `work/baseline-recovery-r001/`,report s004 in
  `work/baseline-recovery-inspection-r002/archive-report.json`,expectations e
  review input ricevuta già identificate nella request s006;
- originali10moduli+4build input/fixture separati daP1–P3,immutabili.

Il nuovo contesto preparatorio r003 ha SHA/byte in response.json supervisore,
esterna al manifest per evitare self-hash. Prima di scritture ricalcola questi
valori e da radice esegui
`python3 scripts/run_context.py verify run-a001-fase0-uv --label baseline-install-preparation-context-r003`.
Assenza/mismatch:STOP,non inventare hash o auto-freeze. Questo non è stage
package-s007:request futura ancora da consegnare. S006 storico soltanto per
sei metadata in transition.json;3758artefatti e695file r002/1044r001 intatti.
Non pretendere MATCH del vecchio worktree s006 contro metadata nuovi.

S006 manifest1087917byte/SHA722f3072e8a6adc1273797ad55ca95e4a9e59c1e9b537d71e5a0f437986a8f6d;
request2721273byte/SHA99f755702d842896a7aebf59cf1ae89cd1d21a4771d1f643f1b6fa8602215bca;
completion9239byte/SHA112623e28860adb6f8196d3776fde0d51695f118761ccd7541efeb8eb3849e90.
Git feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto,11tracked/24nuovi preesistenti. Verifica Git reale;P1–P3 statici
non sono divergenza ignota. Copia report/checkpoint ricevuti prima di aggiornarli.

## Osservazione e limite causale

S006 directoriesPASS/exit0;venvFAIL/exit1 complete=false. Creazione venv exit0,
seed raccolto exit-9 dal cleanup proprio del driver. FileNotFoundError nel
driver su `pip/_internal/__pycache__/wheel_builder.cpython-312.pyc.133843463992976`.
Install17wheel/check/inventory NOT_EXECUTED. Nessun PASS_SEED_ONLY osservato,
nessun inventario18dist;tmp/pip24-seed/bundled ancora presente,nessuna copies/
pip24-seed di successo.695file includono lock,694file/103directory nel target.

Fonti pinned:run_child chiama budget durante il figlio;count_tree enumera e
poi lstat;py_compile usa importlib._bootstrap_external._write_atomic che crea
nome .pyc.id e os.replace. Nome/errore compatibili con race di scansione budget
su temporaneo scomparso,**ma traceback storico assente:causa inferita**.
Non dichiarare seed timeout/cap/ENOSPC/rifiuto sandbox o importlib corrotto.
Il seed è stato fermato dal driver;non è attestata la sua conclusione naturale.

Non eseguire r002 per riprodurre. Prepara una riproduzione **sintetica
stdlib deterministica** su fixture nuove che faccia sparire/rinominare un
file dopo enumerazione e prima di lstat,mostrando il difetto originale e il
comportamento della nuova implementazione. La dimostrazione sintetica prova
il meccanismo;non sostituisce un traceback della vecchia esecuzione.

## Correzione confinata al monitor attivo

Distingui esplicitamente il campionamento live mentre un figlio proprio è
attivo dalle verifiche strict di admission/preflight/postcheck/inventario/
input immutabili. Non usare catch globale FileNotFoundError o un budget
fittizio precedente. Non togliere monitor50ms,cap o scansioni per evitare FAIL.

Proponi una politica concreta e revisionabile per discendenti **volatili
dichiarati del solo nuovo target**:ad esempio temporanei pyc.id del pip pinned,
coerenti con lista495pyc attesi,ed eventuali rinomine seed previste dalle fonti.
Confina fase/path/tipo di errore e registra transienti,campioni incompleti,
numero/durata delle nuove scansioni e stima usata. Non escludere definitivamente
file dal conteggio. Campione incompleto non è inventario stabile né quota fisica.
Nuove scansioni locali devono essere bounded dalla deadline e da limiti
concreti;esaurimento/incertezza che impedisce il gate resta STOP,non loop infinito.
La politica finale richiede accettazione esplicita del supervisore nella
nuova scope s007 (campo suggerito live_monitor_transient_policy_accepted,
legato a request/budget/politica congelati). Non scrivere tu tale scope.

Restano FAIL:input immutabile/fisso mancante,root inattesa scomparsa,errore
fuori perimetro volatile,PermissionError/I/O/corruzione,symlink/speciale,
cap/timeout/inventario/receipt incompleto. Nessuna tolleranza in check_record,
walk_files/startup/origin/closed inventory o conteggi a figlio concluso.
Target zero link sempre;non seguirli neppure durante campionamento. Le14link
storiche/sintetiche sono dati dichiarati esterni al target,non eccezioni nuove.

Aggiungi regressioni significative su sparizione atomica prevista,posizione
non ammessa,root/fisso mancante,PermissionError,symlink/speciale,sforamento
budget/target/file/directory,limite di scansioni/deadline,ritorno alla scansione
strict dopo child,correttezza totale senza doppie somme,un campione non spacciato
per completo. Usa mock/barriere o sole mutazioni stdlib di fixture nella nuova
preparazione. Eventuali thread sintetici propri raccolti;nessun subprocess
operativo/socket/rete/venv/pip o script attivazione nei test. Test devono
verificare esiti e limiti effettivi,non duplicare il codice implementato.

Prepara diagnostici eccezioni bounded prima del cleanup:tipo,path e frame utili
(file/funzione/linea),senza locals,env o segreti;errori di log non devono
nascondere l'errore primario né lasciare processi propri pendenti. Receipt FAIL
rimane incompleta e collection/finally resta confinata al gruppo proprio.
Nessun nuovo collaudo installer o stress host autorizzato ora.

## Nuovo pacchetto e attese

Nuova directory `evidence/implementation-r001/preparation-baseline-install-r003/`.
Nuovo target futuro `work/baseline-recovery-install-r003` e lock adiacente .lock
assenti:non crearli in preparazione. Nuova request
`implementation/stages/impl-r001-stage-package-s007/request.json`,label
`impl-r001-stage-package-s007`,fase package,subphase BASELINE_RECOVERY_INSTALL_INPUTS,
WAITING_FOR_STAGE_SNAPSHOT,snapshot_owner supervisor. Se path già esistenti
inaspettatamente,STOP e documenta,non sovrascrivere o scegliere nomi ad hoc.

Versiona helper derivati/diff motivato r002→r003;rigenera attese path per
pyvenv.cfg/template/script/shebang/entrypoint,pyc co_filename/RECORD/direct_url,
ambiente/config/assenze/argv/gate. Requirements17URI/hash restano identici,
asset/report/originali/fixture immutabili. Nessun RECORD/pyc dei target falliti
riusato. Non escludere file inattesi per passare.

Conserva correzione template read_bytes/decodeUTF8/sostituzioni ordinate/encode:
nessuna normalizzazione newline,Activate.ps19033byte/247CRLF,
SHA3795a060dea7d621320d6d841deb37591fadf7f5592c5cb2286f9867af0e91df,
mode fonte e quattro template common/posix nel confronto chiuso. Negativo
LF8786byte e mode mutato restano. Rivalida53guardie/regressioni precedenti
su sorgenti finali e nuove prove del monitor;non trasferire53PASS storici.
Identifica hash prima/dopo dei sorgenti realmente testati e log/exit;
conserva eventuali errori/intermedi. Oracle byte indipendenti/fonti pinned.

## Cinque passi futuri e gate conservati

Directories/venv/install/pip-check/inventory proposti come tool distinti,
nessuno eseguito ora. Argv strutturati/shell=False/close_fds,ambiente driver
esatto come figli senza os.environ ereditato,lock/marker/receipt esclusivi,
predecessori completi e stesso SHAstage,snapshot/request/host/config/assenze
prima scritture e nuovi interpreti,STOP senza retry,log/copie bounded.
Deadline proposte120s aggregata venv+seed/180s install/60s check/120s inventory,
preflight/postcheck/copie aggiuntivi;variazioni motivare nella request.
Il nuovo campionamento deve rispettare deadline e raccolta dei soli figli propri.

Venv --copies --without-pip,lib64 directory reale vuota,origine/config/site
vuoto prima primo interprete,seed separato ensurepip con soli adattamenti nel
processo per -I -B/temp esclusivo preservato FAIL/spostato copies successo.
Bundledpip24.0 soltanto2110226byte/SHA
ba0d021a166865d2265246961bec0152ff124de910c5cc39f1156ce3fa7c69dc;
bootstrap `/home/davide/.pyenv/versions/3.12.3/bin/python3`,aliaspython3.12
SHAb7e2da38f91d2b8dd6d08afb6262c9c413d8bd222a25ef831da9e27ef4e0a807.
Nessun host source cambiato/upgrade,fallback. Audit hook seed non isola rete
figlio. Nuova scope deve rifiutare levecchie s005/s006,legarsi agli SHA request/
budget/politica s007,non inventare accettazioni supervisore.

Pip interno per requisiti diretti fissi/seed già chiarito:conserva --isolated
--require-virtualenv --no-input --disable-pip-version-check e install
--no-index --require-hashes --only-binary=:all: --no-deps --no-cache-dir
--no-compile. Nessuna nuova dipendenza/grafo/backend/sdist/build/indice.
-I non disabilita a1_coverage.pth:ramo stdlib eseguito,coverage inattivo con
COVERAGE_PROCESS_START/CONFIG escluse e hash/startup prima nuovi avvii.
Nessun app/native/plugin/entrypoint/CLI dotenv/completion shell invocato.

Ambiente ricostruito HOME invariata,PATH bootstrap:/usr/bin:/bin,LANG/LC_ALL
C.UTF-8,TZUTC,tmp/cache/config nuovo target,PIP_CONFIG_FILE=/dev/null,
UV_PYTHON_DOWNLOADS=never,PYTHONDONTWRITEBYTECODE=1,PYTHONNOUSERSITE=1,-B esplicito;
no proxy/credenziali/PYTHONPATH/PYTHONHOME/override pip/uv/pytest/coverage.
Fonti host/stdlib/startup/config inventariate,nessun valore segreto nei log.

## Budget, request e consegna

Mandato500MiB invariato;nuova preparazione fino16MiB più16MiB riserva esterna,
stop384MiB,lstat intera run senza doppie somme/seguire link. Entrambi target
falliti sono già nel totale vivo;nuova stima installer li include una volta,
piùstoria/snapshot/evidenze/request e nuovo target futuro con parti/margine
espliciti. Non trasferire vecchia stima senza misura aggiornata. Nessun cleanup
per il gate o software autorizzato dalla sola stima. Link attuali14,nuove
fixture eventuali dichiarate con provenienza,nuovo targetzero link.

Mantieni gate516MiB libero,128MiB target/4096file1024directory,32MiB perfile,
1MiB/stream,8MiB/JSON,riserva fisica4MiB receipt,monitor50ms. Non quota atomica
négaranzia della somma controburst. Nuova politica deve dichiarare burst,
limiti/duration delle scansioni incomplete ed effetti sull'ammissione,senza
allentare in silenzio limiti oetichettare completa una scansione live.

Request listaesatta di input esistenti relativi/regolari/non evasivi/senza
symlink:autorità,nuovi helper/modelli/politica/manifest/requirements/fonti,
prove sintetichedella versione finale,completion/FAIL s006 eparziali identificati,
s005preservato,manifestprecedente/originali/fixture. Report/checkpoint correnti
mutabili esclusi,usa copie ricevute;sei metadata supervisore separati. Nessun
outputfuturo/S/scope/receipt futura incluso,request senza selfhash.

Verifica AST,sintetici stdlib proporzionati,hash/preservazione s005/s006,
asset s003/report s004,contesto r003 MATCH,Git/diff-check/link e budget finale.
Produci preparation.md,diff motivato,diagnostici/prove/politica/argv/costi
reviewable e request s007. Aggiorna solo report `implementation/report-r001.md`
e checkpoint `handovers/implementation-r001.md`,copie d'ingresso preservate,
**WAITING_FOR_STAGE_SNAPSHOT**,prossimo ruolo supervisore prompt05. Nessuno
stato comune/changelog/snapshot da implementatore;proposta changelog separata.
Nessun servizio automatico da attendere dopo consegna.

Baseline/confronti IMPEDITI:sei path/tmp assenti/2662record indisponibili,
pulizia probabile riferita non dimostrata. FAIL s003/s005/s006,V0s005 storico
caratterizzazionePASS/suiteFAIL67nodeid201eventi62pass5fail/perditeF2/F5/anchor/
separatori/bundle/ASCII invariati;nessuna equivalenza venv/PASS trasferito.
Dopo recupero ricevuto baseline-s006 distinta da package-s007,R completaD4/
nuovoS,parent-child/TMPDIR/namespaces/daemon blacklist/socketpair,non autorizzati qui.

Otto operazioni produzione differite:managed-acquisition,universal-lock-no-build,
lock-check,entry-deps-without-project,backend-entry-no-build,sdist-and-wheel,
base-sync,canonical-wheel-install. Uv0.10.10/managed3.12.13/setuptools84/CPU-cu126/
no-build/extra invariati;selectorGNU/build20260310/redirect/cap/grafo aperti,
regex2026baseline non pinprodotto Marker<2025. D2D3/config-only/flag/IDE/
current↔S↔snapshot/S-B-I-E/V1–V9,due review reali ChatGPT/Claude/arbitrato futuri.
V7V8 obbligatorie/costo distinto;V10V11/pesi/font/inferenza esclusi,
local_marker non collaudato. Nessun GO codice/cleanup/sysctl/AppArmor/rete/socket
host/privilegi/profili persistenti/altroprogetto/invio documenti/Git/deploy.
Git manuale utente dopo arbitrato finale;temp ignorata:trasferire file reali.
