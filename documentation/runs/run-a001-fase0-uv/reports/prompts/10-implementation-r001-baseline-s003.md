# Ripresa implementativa r001 — baseline s003 — run-a001-fase0-uv

Agisci come **implementatore** nel repository
`/home/davide/workarea/markdown-for-llms`, riprendendo il checkpoint reale
`handovers/implementation-r001.md`. Preparazione s003 ricevuta, verificata e
congelata dal supervisore: **IMPLEMENTATION / STAGE_SNAPSHOT_READY**.
Continua [prompt04](04-implementation-r001.md) e la
[disposizione D1/D4 TMPDIR — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
GO piano r003, nessun GO codice. R s003 NON_ESEGUITA, S NON_GENERATO,
V0 NON_ESEGUITA; propagazione effettiva TMPDIR ancora da provare.

Chat distinta dal supervisore; niente subagenti, review simulate, auto-snapshot
o registri comuni. Il supervisore non è un servizio continuo. Prompt08 e gli
impedimenti s001/s002 sono storia, non comandi per riusare output esistenti.

## Letture e identità

1. AGENTS, CHANGELOG, PROJECT-CONTEXT/HANDOVER globali e STATE/HANDOVER della run.
2. Skill manage-implementation-run, protocollo e skill verify-conversion-fidelity.
   Brief A1–A7, piano r003 e arbitrato r003 integralmente se chat fresca;
   altrimenti riconferma identità e D1–D5. NO_GO r001/r002 e 24 disposizioni
   antecedenti conservati, nessuna nuova r004 o review piano.
3. Prompt04 e prompt09 preparazione completata; report/checkpoint propri,
   request s003 e receipt negative s002 con equivalenza limitata di scelta.
4. [Response supervisore — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
   reception.json, baseline-input-audit.json, transition.json e resume-commands.json
   nella stessa cartella; [checkpoint supervisore — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
5. Manifest s003; in evidence/implementation-r001/preparation-baseline-s003/:
   host-inventory/host-read-receipt, input-inventory, baseline-inputs, due target,
   wrapper.diff, pure-argv-checks, target-transition, preparation/handoff-checks.
   Provenienza F1–F6, copie originali e cinque diagnostici visibili. Leggi
   strutturalmente gli inventari grandi, recupera intervalli troncati, non tutta temp.

Percorsi senza prefisso relativi a `temp/run-a001-fase0-uv/`.

| Oggetto | SHA-256 |
| --- | --- |
| plans/plan-r003.md | `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b` |
| arbitrations/arbitration-plan-r003.md | `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d` |
| prompts/04-implementation-r001.md | `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035` |
| implementation/stages/impl-r001-stage-baseline-s003/request.json | `009d07bd0e0c52a99b4eba8ab63bca0b56921fbc80c5ccb241d4a20687d600b4` |
| snapshots/impl-r001-stage-baseline-s003.json | `f518b327c4aebe8b0a3c29ec9c5aa153e95c78d28fb63970b0040be0f5b1d32a` |
| evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s003/resume-commands.json | `19a1f81f7ad555e64983104e1e06e6dd55eac496b31b06240f5a407fb69e1959` |
| evidence/supervisor-implementation-r001/runner-recovery-r002/decision.md | `fcb7d0558a9b1481c08b0a28e11b85aa200f5422e7b7d45e1aad1c3fb6f4ba24` |
| evidence/supervisor-implementation-r001/runner-recovery-r002/proposed-commands-baseline-s003.json | `087459cd4190f79fc960c49242e4622197d27a76a9d5c23cae91faa31ce7c70e` |

Wrapper visibile `scripts/diagnostics/run-a001-fase0-uv/run_offline.py` SHA
`a937e0067f81f9d09c2fec49f4b47de00eb9a8b254b64d75caed1029b3c71018`.
Stage **impl-r001-stage-baseline-s003 MATCH**, **96 file +181 artefatti**;
worktree `b16467caaf15963b987007b569b4e7c3f56f2b4776ae0c344aa4f46e9b679ed4`.
Branch feature/run-a001-uv, HEAD/dev/merge-base
66ba82200e5def5a4db76f9bafccb0731b506091, indice vuoto. Sei metadata documentali
modificati e 17 file preparatori nuovi; nessun prodotto/lock/pyproject migrato.
Nessun reset/cambio branch/commit/merge/push/promozione o deploy.

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
git diff --cached --name-only
python3 scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-baseline-s003
```

Verifica hash esatti e MATCH s003 prima/dopo ogni passo. Recupero r002 storico
per wrapper/CHANGELOG alla ricezione, poi sei metadata del supervisore; 161
artefatti intatti come 90/109/120/58/96/109 antecedenti. S001/s002 ora storici
anche per il delta wrapper identificato: niente nuovo MATCH attribuito alle
prove vecchie. Manifest e output originali non riscritti. Hash dei report/
checkpoint storici legati alle versioni prima-s003 e copie ricevute; file propri
correnti mutabili. Altre divergenze non spiegate → consegna al supervisore.

## Preparazione riconfermata e perimetro

232 file/assenze della request (218 file e 14 assenze), 181 artefatti confinati;
2.891 file distinti e 24 assenze riconfermati dal supervisore. Inventario attuale
2.828 file; 1.939/1.940 tecnici e 825 file uv cache invariati. Unico delta tecnico:
36 byte e un argomento `--env=TMPDIR=<target['tmpdir']>` prima di `--` nel
prefisso Firejail comune. Ramo unshare e ogni altro byte wrapper invariati;
gate/blacklist/cleanup/clean_env/check_runner/S producer/driver V0 conservati.
Quattro verifiche pure argv dell'autore, primo errore quote AST conservato:
nessuna verifica runtime e nessun nuovo test della suite applicativa.

Nuovi target 1.380 file ciascuno, 1.379 invariati e wrapper aggiornato; guardie
operative/deps/cache/daemon identici, riferimenti di provenienza e candidato
aggiornati. 74 record baseline-inputs e dieci originali/quattro build input
uguali a HEAD/copie. Host UID/GID1000 da receipt reale dell'autore, sola lettura
require_escalated exit0; namespace/policy/stat/binari/deps riconfermati.
Non è un'osservazione host o prova R del supervisore. Compatibilità R **NON_PROVATA**.

Conserva `/tmp/a001-uv-baseline-pi8cvs6x`: venv/tmp/workspace/uv-cache/tiktoken-cache.
Bootstrap `/home/davide/.pyenv/versions/3.12.3/bin/python3`, baseline
`/tmp/a001-uv-baseline-pi8cvs6x/venv/bin/python`, Python3.12.3; 17 dipendenze
leggere più pip e cache pubblica cl100k_base già presenti. Nessun download,
install/sync/build/fetch per questo stage. F6 token/chunk/overlap da misurare;
i sette mock antecedenti non sono R/V0/fedeltà.

**Firejail esistente è l'unico candidato attivo s003**. Unshare s002 realmente
inadeguato D4 abilita la scelta del fallback soltanto con stesso host/UID/
binario/ramo unshare/inventario daemon riconfermati. Non riavviare il primario
identico, non trasferire PASS da Firejail s002. Se tali condizioni cambiano,
riconsegna lo scostamento; nessun altro candidato/host implicito.

Per ogni passo usa `exec_command` con **sandbox_permissions="require_escalated"**,
CWD della radice e la justification specifica in step_tool_profiles, senza
prefix_rule ampia. Wrapper host ordinario UID1000, programma solo nel runner
completo. Ogni chiamata è soggetta alla review automatica del tool: lettura o
freeze non approvano in anticipo sonde o conversioni. Non usare il confine
use_default che ha prodotto EPERM s001, né cambiare canale dopo un rifiuto.

I may_execute_now=false dei JSON congelati rappresentano la preparazione:
non modificarli. Response e questo prompt ammettono la sequenza dopo i gate.
resume-commands conserva argv/cwd/output/timeout/prerequisiti e rinomina soltanto
i flag storici come preparation_may_execute_now. Nessuna esecuzione in blocco.

Nessun sudo/nuovo privilegio/setuid, sysctl/AppArmor/rete/permesso/socket host
cambiati, profilo persistente, aggiornamento Firejail o API daemon. Path noti,
alias/canonici e sintetico proprio: sole connect/close diagnostiche, zero
send/recv/API/operazioni Docker. Sintetico esclusivo/cleanup e IP documentali
soltanto nel percorso D4; socketpair positivo reale, namespace figli e tutto
§R restano obbligatori. Osserva namespace attuali, senza riparare l'host per
riprodurre un ID storico. Nessun invio remoto/fallback terminale implicito.

## Sequenza concreta

Quattro argv integrali in [resume-commands.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
identici a request/proposta. Il wrapper costruisce il prefisso reale Firejail
con `--env=TMPDIR` dal target e tutte le blacklist prima di `--`; non usare un
placeholder o una sonda abbreviata TMPDIR.

1. Riconferma MATCH/hash/assenze esterne e output futuri assenti. Ricontrolla
   spazio nel profilo host: 709.898.240 byte erano liberi all'inventario, non
   sono una misura corrente. Stime output 50 MB, rete0, R/S2 minuti, V0/audit30
   minuti: distingui stima e osservazione, niente acquisizione implicita.
2. **R-bootstrap Firejail completo**, target bootstrap s003, output
   `evidence/implementation-r001/runner-bootstrap-r-firejail-s003/`.
   Conserva parametri tool/justification/profilo, risultato della review, sessione,
   exit/stdout/stderr/durata e receipt anche se negato/incompleto.
3. PASS soltanto con exit0 e receipt/inside/runner PASS: namespace distinto host
   e coerente padre/figlio, nessun egress/rotte esterne, IP documentali negati,
   socketpair positivo, daemon/sintetico negati padre/figlio e sintetico positivo
   host prima/dopo; prerequisiti/origini/deps/binari/cache/FD corrispondenti.
   **TMPDIR esatto dal target in padre e figlio prima del programma**, TIKTOKEN
   esatto e variabili vietate assenti. Cleanup e hash/MATCH pre/post obbligatori.
   Path daemon assenti non provano blacklist; netns IP solo non confina UNIX.
4. Bootstrap completo PASS → **R-baseline Firejail completo**, venv e target
   baseline s003, output `runner-baseline-r-firejail-s003/`. Nessun PASS ereditato
   dal bootstrap per ambiente baseline; tutti gate ripetuti.
5. Entrambi R PASS → **S-baseline**, wrapper ripete R/preflight nella medesima
   invocation del producer. Copia originali come --repo, clone come --host-repo,
   baseline-inputs e snapshot s003. Output esclusivo
   `implementation/stages/impl-r001-stage-baseline-s003/sources.json`;
   wrapper output `runner-baseline-sources-firejail-s003/`. Schema1/payload/ID
   canonico, hash esterno/log; confronto sorgenti attuali/copia/files dello
   snapshot e mapping dieci originali. Questo S baseline è distinto dal prodotto.
6. S corrente → **V0-originals**, con R/preflight/comando nella medesima
   invocation. Driver confronta S/current/snapshot/input/receipt R corrente
   prima di collection/import. Runner output `runner-baseline-v0-firejail-s003/`,
   contenuti `evidence/implementation-r001/baseline-results-s003/`, workspace
   `/tmp/a001-uv-baseline-pi8cvs6x/workspace/v0-s003`. Niente install/build/fetch.
7. Prima/dopo ciascun passo: verify s003/hash esterni, exit/namespace/processi
   e ricevute. Non ritoccare input congelati o output già prodotti per farli
   passare. Gate incompleto → IMPEDITA; mismatch hash/S/receipt/snapshot → FAIL.

Esempio per **un solo passo**: passalo al tool nel profilo require_escalated,
con CWD/justification del passo. Scegli il valore step solo dopo gate reali.
Il codice verifica le identità e alcuni prerequisiti delle receipt; resta
obbligatoria la lettura di tutti i gate sopra e la review automatica del tool.

```bash
/home/davide/.pyenv/versions/3.12.3/bin/python3 -I - R-bootstrap <<'PY'
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
root = Path('/home/davide/workarea/markdown-for-llms')
path = root / 'temp/run-a001-fase0-uv/evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s003/resume-commands.json'
assert hashlib.sha256(path.read_bytes()).hexdigest() == '19a1f81f7ad555e64983104e1e06e6dd55eac496b31b06240f5a407fb69e1959'
data = json.loads(path.read_text())
step = sys.argv[1]
assert step in ('R-bootstrap', 'R-baseline', 'S-baseline', 'V0-originals')
assert os.getuid() == os.getgid() == 1000
commands = {x['step']: x for x in data['profiles']['firejail']['commands']}
item = commands[step]
snapshot = root / data['snapshot']['path']
assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == data['snapshot_sha256']
assert hashlib.sha256((root / data['wrapper']['path']).read_bytes()).hexdigest() == data['wrapper']['sha256']
subprocess.run([sys.executable, '-I', str(root / 'scripts/run_context.py'),
                'verify', data['run_id'], '--label', data['label']], cwd=root, check=True)
for prior in (['R-bootstrap'] if step == 'R-baseline' else
              ['R-bootstrap', 'R-baseline'] if step == 'S-baseline' else
              ['R-bootstrap', 'R-baseline', 'S-baseline'] if step == 'V0-originals' else []):
    command = commands[prior]
    receipt = json.loads(Path(command['wrapper_receipt']).read_text())
    runner = json.loads(Path(command['R_receipt']).read_text())
    inside = json.loads((Path(command['output_directory']) / 'inside.json').read_text())
    assert receipt['status'] == runner['status'] == inside['status'] == 'PASS'
    assert receipt['wrapper_sha256'] == data['wrapper']['sha256']
    assert receipt['inputs_unchanged'] and receipt['temporary_socket_cleaned']
    target_path = Path(command['argv'][command['argv'].index('--target') + 1])
    assert receipt['target_sha256'] == hashlib.sha256(target_path.read_bytes()).hexdigest()
    target = json.loads(target_path.read_text())
    assert runner['same_namespace']
    assert all(runner[side]['environment']['TMPDIR'] == target['tmpdir'] for side in ('parent', 'child'))
    for field in ('snapshot_before', 'snapshot_after'):
        assert receipt[field]['exit_code'] == 0 and receipt[field]['sha256'] == data['snapshot_sha256']
assert not Path(item['output_directory']).exists()
assert shutil.disk_usage('/tmp/a001-uv-baseline-pi8cvs6x').free >= 50_000_000
if step == 'V0-originals':
    assert (root / 'temp/run-a001-fase0-uv/implementation/stages/impl-r001-stage-baseline-s003/sources.json').is_file()
print(json.dumps({'step': step, 'depends_on': item['depends_on'],
                  'argv': item['argv'], 'profile': 'require_escalated'}), flush=True)
proc = subprocess.run(item['argv'], cwd=item['cwd'], shell=False, timeout=item['timeout_seconds'])
raise SystemExit(proc.returncode)
PY
```

Yield breve e sessioni conservate, aggiornamenti entro60 secondi. Rifiuto tool:
registra azione/motivo ed IMPEDITA e torna al supervisore; nessun candidato o
terminale/canale alternativo per aggirarlo. Runner inadeguato/gate incompleto:
IMPEDITA, niente baseline fuori runner, nessuna modifica host. Nessun retry
sugli stessi output o errore nascosto. Hash/S mismatch FAIL prima applicazione.

## Audit V0, invalidazioni e consegna

Apri output/diff/report **integrali** F1–F6: Unicode, numeri, formule LaTeX,
codice, link/riferimenti, ordine e campi deterministici validation/summary.
F4 tabella/formula e failed=0; F5 riferimento/hash PNG e perdita eventuale
della copia asset. Registra difetti legacy senza correggerli durante baseline;
JSON/presenza file non chiudono fedeltà/bundle. Difetto orchestratore CWD
esterna conservato nella caratterizzazione.

F6: custom/1000/100 da validated, almeno due chunk, contenuti/frontmatter/
indici/confini/metadati e overlap misurato. Complemento sliding stesso testo/
parametri: almeno due chunk, intersezione >0. 1200/80 e1600/120 osservabili
identificati, non attribuire V5; semantic legacy può avere overlap zero.
Audit manuale prima di V0 conclusa. Perdita nuova/incerta → FAIL e incroci
soltanto con codice/interprete/deps/constraints/tokenizer identificati.

legacy_suite: quattro file legacy espliciti, argv/nodeid/collection/pass/fail/
skip/errori/exit conservati, exit distinto dalla caratterizzazione automatica.
Non presentarla come suite completa né assumere67/6 storici. Conftest attuale
inserisce la radice ancora originale, non verifica la futura wheel.

Tuning F6 o nuovo input/interprete/cache/deps/diagnostico/retry → s003
esplorativo identificato e request **impl-r001-stage-baseline-s004** o successiva,
nuovo freeze D1 e ripetizioni pertinenti. Nessuna sovrascrittura S/receipt/
snapshot o PASS della fixture nuova senza prove. P2/D2/negativi receipt reali
restano obblighi futuri, nessun mock li chiude.

Dopo R/S/V0 effettivi **e audit concluso**, conserva chiusura/hash/limiti baseline
e prosegui prompt04 P1–P3 verso request stage package. Cambiamenti prodotto
rendono storico il worktree s003: nuovo stage/S prima build, poi tests/image/
final D1/P2. Non trasferire PASS baseline al prodotto. Nessuna fase dalla roadmap.

Aggiorna report/checkpoint propri con esiti reali, hash/input/output/costi,
namespace/processi/limiti e prossima request; consegna via
[prompt05](05-supervisor-stage-r001.md) per package, s004 o impedimento.
Niente auto-snapshot/polling o STATE/HANDOVER/eventi/indici/ADR comuni. Changelog
del lavoro reale solo alla consegna **dopo l'ultimo verify s003**, identificando
la conseguente transizione storica. Preserva prima le versioni proprie attuali.

V7/V8 future **obbligatorie**, perimetro e costo pesante distinti. V10/V11/pesi/
font/inferenza esclusi, nessun Marker normale/deploy/invio remoto. Due review
indipendenti codice e arbitrato finale futuri; Git manuale utente. Temp e
venv/cache esterne non viaggiano con Git: preservare il lavoro effettivo.
