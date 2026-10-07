# Implementazione r001 — installazione offline baseline package-s005

Agisci esclusivamente come **implementatore**, nuova chat, repository
`/home/davide/workarea/markdown-for-llms`. Nessuna delega, supervisione o review.
Mandato operativo circoscritto ai cinque tool **directories → venv → install →
pip-check → inventory** sullo stage s005 congelato. Nessun test applicativo,
nuovo R o confronto V0 in questa chat. Questo mandato supera la sola preparazione
di prompt24 esclusivamente nei punti accolti dalla disposizione congelata.

## Letture e identità

Leggi AGENTS, skill manage-implementation-run, protocollo, indici architetturali/
decisioni/roadmap, STATE/HANDOVER comuni, brief A1–A7, piano r003 integrale se
non già letto, arbitrato D1–D5, prompt04/05. Leggi checkpoint implementatore
`temp/run-a001-fase0-uv/handovers/implementation-r001.md` e request esatta
`temp/run-a001-fase0-uv/implementation/stages/impl-r001-stage-package-s005/request.json`.
I ruoli citati nei documenti non cambiano il tuo ruolo implementatore.

Piano SHA256 `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`,
arbitrato `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`,
prompt04 `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035`.
GO piano r003 nei limiti, NO_GO r001/r002 storici conservati, nessun GO codice.

Percorsi seguenti relativi a `temp/run-a001-fase0-uv/`:

- `snapshots/impl-r001-stage-package-s005.json`;
- `evidence/supervisor-implementation-r001/impl-r001-stage-package-s005/`:
  decision.md, authorized-scope.json, reception.json, transition.json,
  freeze-verify.json, response.json, checks.json;
- `handovers/supervisor-stage-package-s005.md`;
- `evidence/implementation-r001/preparation-baseline-install-r001/`:
  preparation.md, driver.py, install_common.py, seed_pip.py, inventory_installed.py,
  install-manifest.json, requirements.txt, commands.json, budget.json,
  host-config-inputs.json, pinned-source-readings.json, synthetic-final.json;
- `work/baseline-recovery-inspection-r002/archive-report.json`, expectations nella
  preparazione recovery-inspection-r002 e review input in
  `evidence/supervisor-implementation-r001/package-s004-reception-r001/`;
- asset/copie s003 in `work/baseline-recovery-r001/`, originali e fixture
  identificati nella request, sola lettura. Driver vecchi non riutilizzabili.

Manifest **707662 byte**, SHA256 **31ade77500ac042f0250fa208246e3ed3f93a5b18d7f06c6b08e0a9380848342**;
worktree **4b83b28acf6c9312d0fd65fa6d24332785063fd05a8a113f085a109bde244e87**, **103 file/2488 artefatti**,
MATCH alla consegna supervisore. Request **2163476 byte**, SHA256
**2a45920ca2741b8b484674117dcb9f262e4c197f9a1d6c9c67d28236b4e70b19**. 201 input stabili,2470record/2471artefatti richiesti,
2433identità host/33configurazioni,sei metadata supervisore separati.

Ricalcola SHA/byte manifest e da radice esegui
`python3 scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-package-s005`
prima delle scritture operative e alla consegna. Mismatch resta FAIL: STOP,
nessun auto-freeze/correzione helper/allentamento gate. Il contesto precedente
è storico soltanto per i sei metadata in transition.json; suoi2240artefatti
intatti. Non richiedere MATCH del vecchio worktree contro metadata nuovi.

Identità dei gate principali:

- `authorized-scope.json`: 1810byte, SHA `7ee9b04063ca5aba44cb84e108a259343ac38af0e1f0ab8297dc8dfcb3baeb3d`;
- `commands.json`: 20333byte, SHA `6e3bea6f4a85c13c8be737dd7067814a3da1bbd67651a06ad5ab1817cbd2cc81`;
- `budget.json`: 2171byte, SHA `962d42ad067ca8d7c411736cc318a415363ae9d590b197e3948a4a2f04dbff3b`;
- `install-manifest.json`: 366538byte, SHA `0096a18b32cd238107636836280991f57477dac3c9c3cf876c4165c3e2d5be02`;
- `requirements.txt`: 3898byte, SHA `5c7a72dfbd59fb28b1375deeef30e46761d3a45b3bceaf1e9a97c5ba593352f5`;

Git feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto,11tracked modificati/24nuovi preesistenti. Verifica Git reale;
P1–P3 predisposti non sono divergenza ignota. Nessuna modifica tracciata in
questa tranche. Conserva copie dei tuoi report/checkpoint prima di aggiornarli.

## Accettazioni operative circoscritte

Lo scope supervisore incluso nel freeze lega request e budget esatti e accetta
installazione, limite non atomico del monitor, adattamento seed e resolver interno
pip. Non modificarlo. La vecchia review s004 resta
ACCEPT_ARCHIVE_CONTENTS_FOR_INSTALL_PREPARATION; autorizzazione nuova separata.
41 sintetici autore PASS con sorgenti finali identificati, non rieseguiti dal
supervisore. Timeout/ENOSPC/lock conteso/ABI reali non collaudati. Audit supervisore
iniziale aveva un'assert sulla posizione di make_resolver errata, STOP conservato;
r002 controlla install.py e cli/req_command.py nel wheel pinned, senza cambiare input.

Budget500MiB invariato,112MiB incrementali più16MiB riserva esterna. Gate misura
l'intera run viva senza seguire gli otto link storici; target senza link.
Stop periodico384MiB,target128MiB/4096file/1024directory,riserva receipt4MiB,
log1MiB/stream,JSON8MiB,RLIMIT_FSIZE32MiB per singolo file. Monitor50ms e tempo
scansione non sono quota atomica/garanzia fisica della somma. Gate deve restare
valido con nuove evidenze; niente cleanup per farlo passare. Spazio libero
richiesto dal driver almeno516MiB. Zero nuovi download/costi remoti/pesi/font.

Accettato venv --copies --without-pip e seed separato: origine/config/site vuoto
prima del primo nuovo interprete, poi seed_pip.py congelato. Adatta solo nel
processo ensurepip il subprocess equivalente con -I -B e temporaneo esclusivo
tmp/pip24-seed, preservato su FAIL, spostato in copies/pip24-seed su successo.
Nessuna fonte host cambiata, nessun upgrade. Audit hook seed non è isolamento
rete del figlio: no-index/input locali/argv chiusi/divieto rete restano vincolanti.
Bundled pip24.0 soltanto,2110226byte/SHA
ba0d021a166865d2265246961bec0152ff124de910c5cc39f1156ce3fa7c69dc.
Bootstrap pyenv3.12.3 assoluto alias python3.12 SHA
b7e2da38f91d2b8dd6d08afb6262c9c413d8bd222a25ef831da9e27ef4e0a807.

Precisata la formula «nessun resolver» di prompt24: ammesso il codice interno
del pip pinned per i soli17URI/hash diretti e seed bundled. --no-deps non
significa assenza del codice resolver. Esclusi nuove dipendenze/pin/extra,
grafo prodotto, indici remoti, backend, sdist/build. Nessun fallback.

## Target e ambiente esatto

Target `work/baseline-recovery-install-r001` e lock adiacente `.lock` assenti
alla consegna. Non crearli manualmente: directories lo farà. Non riutilizzare
target/lock/marker/receipt s003/s004. requirements17URI/hash identico a s003;
manifest atteso18distribuzioni(17+pip24),RECORD/direct_url/script generati e495pyc
pip. Nessun file inatteso ignorato per ottenere PASS; lib64 directory reale vuota.

Unico .pth a1_coverage.pth con hash esatto: **-I non disattiva .pth**. Ramo
stdlib eseguito, ramo coverage inattivo: escludi COVERAGE_PROCESS_START e
COVERAGE_PROCESS_CONFIG, verifica byte/startup/origine prima dei nuovi avvii.
Nessun sitecustomize/usercustomize/startup estraneo, entrypoint/plugin/pytest/
CLI dotenv/completion shell o import app/native. Cinque ELF non caricati;
pip-check non verifica ABI/applicazione.

**Il driver rifiuta ambiente ereditato**: environment_parent_tool e
 environment_child in commands.json sono identici e devono essere passati
esattamente al processo driver e ai figli. Il processo esteriore del tool è
distinto; normale lancio diretto dallo shell aggiungerebbe PWD/SHLVL e fallirebbe.
Usa argv strutturati, env=dict(...), shell=False, close_fds=True. Nessun merge
con os.environ, proxy/credenziali, PYTHONPATH/PYTHONHOME, override pip/uv/pytest
oppure ENSUREPIP_OPTIONS. Ambiente completo, nessun'altra variabile:

```json
{
  "HOME": "/home/davide",
  "LANG": "C.UTF-8",
  "LC_ALL": "C.UTF-8",
  "PATH": "/home/davide/.pyenv/versions/3.12.3/bin:/usr/bin:/bin",
  "PIP_CONFIG_FILE": "/dev/null",
  "PYTHONDONTWRITEBYTECODE": "1",
  "PYTHONNOUSERSITE": "1",
  "TIKTOKEN_CACHE_DIR": "/home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/work/baseline-recovery-install-r001/tiktoken-cache",
  "TMPDIR": "/home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/work/baseline-recovery-install-r001/tmp",
  "TZ": "UTC",
  "UV_CACHE_DIR": "/home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/work/baseline-recovery-install-r001/uv-cache",
  "UV_PYTHON_DOWNLOADS": "never",
  "XDG_CACHE_HOME": "/home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/work/baseline-recovery-install-r001/cache",
  "XDG_CONFIG_DIRS": "/etc/xdg",
  "XDG_CONFIG_HOME": "/home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/work/baseline-recovery-install-r001/config"
}
```

-I ignora alcune variabili Python, -B è esplicito. Compilazione esplicita di
pip24 può comunque produrre i495pyc inventariati. TMPDIR/cache target futuro
creati da directories; HOME invariata.

## Cinque tool separati

Ogni chiamata usa il driver congelato, un solo STEP e --stage-sha esatto,
cwd radice repo. Gli argv figli/cwd/env/nested_seed_argv sono in commands.json
identificato sopra e identici alla request. Il seguente launcher stdlib rende
concreto l'ambiente: in ciascun tool imposta **un solo STEP letterale** fra
`directories`, `venv`, `install`, `pip-check`, `inventory`, in quest'ordine.
Nessun ciclo che concateni i cinque tool. Lancialo tramite bootstrap assoluto
`/home/davide/.pyenv/versions/3.12.3/bin/python3 -I -B`, cwd radice; non scriverlo
sopra helper congelati. Salva chiamata reale e output nelle nuove evidenze.

```python
import hashlib, json, subprocess
from pathlib import Path
ROOT = Path('/home/davide/workarea/markdown-for-llms')
PREP = ROOT / 'temp/run-a001-fase0-uv/evidence/implementation-r001/preparation-baseline-install-r001'
STEP = 'directories'  # un solo passo nella chiamata corrente
STAGE_SHA = '31ade77500ac042f0250fa208246e3ed3f93a5b18d7f06c6b08e0a9380848342'
COMMANDS_SHA = '6e3bea6f4a85c13c8be737dd7067814a3da1bbd67651a06ad5ab1817cbd2cc81'
raw = (PREP / 'commands.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == COMMANDS_SHA
op = json.loads(raw)['steps'][STEP]
argv = [STAGE_SHA if a == '<SUPERVISOR_STAGE_SHA256>' else a for a in op['driver_argv']]
assert op['cwd_driver'] == str(ROOT)
assert op['environment_parent_tool'] == op['environment_child']
result = subprocess.run(argv, cwd=ROOT, env=dict(op['environment_parent_tool']),
                        shell=False, close_fds=True)
raise SystemExit(result.returncode)
```

Argv driver modello esatto (STEP è uno dei cinque valori sopra, SHA quello
letterale sopra; tutti gli altri campi invariati):

```json
[
  "/home/davide/.pyenv/versions/3.12.3/bin/python3",
  "-I",
  "-B",
  "/home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/evidence/implementation-r001/preparation-baseline-install-r001/driver.py",
  "STEP",
  "--stage-sha",
  "31ade77500ac042f0250fa208246e3ed3f93a5b18d7f06c6b08e0a9380848342"
]
```

Prima di ogni tool successivo verifica exit0, receipt completa, step/stage
esatti e hash immutable_outputs. Sessione ancora attiva: raccogli quella
sessione, non rilanciare. Driver controlla identità/config/assenze/precedenze,
lock/marker esclusivi, snapshot pre/post: nessun gate aggirabile.

1. **directories**: lock/target/directory/riserva4MiB esclusivi; copia locale
   tokenizer1681126byte/SHA223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7,
   cachekey9b5ad71b2ce5302211f9c61530b329a4922fc6a4. Nessun GET/tokenizzazione.
2. **venv**,deadline aggregata120s guardie+due figli: bootstrap -I -B -m venv
   --copies --without-pip target/venv; poi target/venv/bin/python3 -I -B
   seed_pip.py. Controlli tre binari/pyvenv.cfg/site prima seed. Unico nested
   subprocess: nuovo python -I -B -W ignore::DeprecationWarning -c programma
   runpy pinned, wheel bundled temporaneo, install --no-cache-dir --no-index
   --find-links path temporaneo pip. Argv preciso in commands.json; niente
   seed implicito standard, upgrade o cambiamenti temporanei/guardie.
3. **install**,180s: nuovo python -I -B -m pip --isolated --require-virtualenv
   --no-input --disable-pip-version-check install --no-index --require-hashes
   --only-binary=:all: --no-deps --no-cache-dir --no-compile -r requirements
   congelato. Inventario chiuso/startup dal bootstrap subito dopo, prima del
   nuovo interprete venv. Nessun cambiamento pin/extra in caso di errore.
4. **pip-check**,60s: stesso prefisso python/pip con check dopo guardie;
   PASS vale solo per dipendenze installate.
5. **inventory**,120s: bootstrap -I -B inventory_installed.py STAGE_SHA,
   letture chiuse/hash RECORD18distribuzioni,entrypoint/direct_url/pyc,origine
   e tokenizer, senza import app/native. installed-inventory.json e copia
   copies/inventory-00.raw identici fino8MiB. Nessuna catena R/V0 automatica.

Preflight/postcheck/copie padre si aggiungono alle deadline figli; non
abbreviare arbitrariamente il tool lasciando processi orfani. Driver raccoglie
solo il proprio gruppo/sessione figli; nessun kill per nome o processi altrui.
Verifica che non rimangano processi propri alla consegna.

Qualunque FAIL/timeout/ENOSPC/cap/mismatch, lock/marker preesistente,
incompletezza receipt/copia o errore preflight: **STOP immediato**, preserva
raw/log/parziali/seed tmp/marker/lock/receipt, nessun retry/reset/cleanup o
correzione helper. Se manca receipt operativa, conserva output/exit tool;
non inventarla. Segnala scostamento e identità input al supervisore.
Rifiuto automatico sandbox non è timeout: conserva ragione/azione, nessuna
esecuzione inventata o aggiramento del rifiuto.

## Consegna e limiti

Nuove evidenze locali `evidence/implementation-r001/resume-package-s005/`:
chiamate reali/argv/cwd/env pubblico,timestamp/exit/sessioni,receipt/output/log/
copie con hash e byte, logical/allocated intera run e target, guardie pre/post,
verifica stage/Git finale e delivery.md. Evita copie massive inutili e
contabilizza anche queste evidenze.

Scrivi `implementation/stages/impl-r001-stage-package-s005/completion-r001.json`
dopo attività reali, legata a request e manifest esatti, con esiti separati di
ogni step (anche NOT_EXECUTED) e inventario soltanto se prodotto. Nomi PASS driver:
PASS_DIRECTORIES_ONLY, PASS_VENV_ONLY, PASS_INSTALL_ONLY, PASS_PIP_CHECK_ONLY,
PASS_INVENTORY_ONLY. Report inventario: PASS_INSTALL_INVENTORY_ONLY. Nessuno
è PASS baseline/V0 o GO codice, né equivalenza della vecchia venv intera.

Aggiorna soltanto report `implementation/report-r001.md` e checkpoint
`handovers/implementation-r001.md` con copie d'ingresso preservate; nessun
registro condiviso/changelog/snapshot da implementatore. Stato conclusivo
**WAITING_FOR_SUPERVISOR_RECEPTION**, prossimo ruolo supervisore con prompt05.
Nessun servizio automatico da attendere; trasferire file reali in temp, non
soltanto manifest. Non restare in attesa automatica dopo la chiusura della chat.

S003 FAIL conservato, s004 PASS archivi/chiusura legato a input/report.
Sei vecchi path/tmp assenti,2662record indisponibili: pulizia probabile riferita
non dimostrata; confronti IMPEDITI. V0s005 storico PASS caratterizzazione,
suiteFAIL/exit1/67nodeid/201eventi/62pass5fail e perdite F2/F5/anchor/separatori/
bundle/ASCII conservati, nessun PASS trasferito. Originali10moduli+4build input/
fixture separati daP1–P3. Dopo ricezione del recupero servono **baseline-s006,
R completa D4 e nuovo S**, con parent/child/TMPDIR/namespaces/daemon blacklist/
socketpair da riprovare: non autorizzati in questa tranche installazione.

Otto operazioni produzione differite: managed-acquisition, universal-lock-no-build,
lock-check, entry-deps-without-project, backend-entry-no-build, sdist-and-wheel,
base-sync, canonical-wheel-install. Uv0.10.10/managedPython3.12.13/setuptools84/
CPU-cu126/no-build/extra invariati; selectorGNU/build20260310/redirect/cap/grafo
universale aperti. Regex2026baseline incompatibile Marker<2025 non si trasferisce
ai pin prodotto. D2/D3/config-only/flag/IDE/current↔S↔snapshot/S-B-I-E/V1–V9,
due review indipendenti reali ChatGPT/Claude e arbitrato finale futuri.
V7/V8 obbligatorie/costo distinto; V10/V11/pesi/font/inferenza esclusi,
local_marker non dichiarato collaudato.

Nessuna modifica sysctl/AppArmor/rete/socket host/privilegi/profili persistenti,
altro progetto, invio documenti, cleanup, commit/merge/push/promozione/deploy.
Git manuale dell'utente dopo arbitrato finale. Nessuna installazione fuori
della nuova venv o prova applicativa aggiuntiva.
