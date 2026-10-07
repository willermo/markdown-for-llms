# Ripresa implementativa r001 — baseline s005 — run-a001-fase0-uv

Agisci come **implementatore** in una nuova chat nel repository
`/home/davide/workarea/markdown-for-llms`, riprendendo il checkpoint reale
`handovers/implementation-r001.md`. Il supervisore ha ricevuto, verificato e
congelato la preparazione s005: **IMPLEMENTATION / STAGE_SNAPSHOT_READY**.
GO piano r003 invariato; nessun GO codice. **R s005 NON_ESEGUITA, S NON_GENERATO,
V0/collection/suite NON_ESEGUITE.** R/S s004 positivi storici; V0 s004 resta
FAIL/exit2 e suite FAIL/exit1 (67 nodeid,201 eventi,62 pass/5 fail osservati).
Impedimenti s001/s002 e NO_GO r001/r002 conservati, nessun PASS ereditato.

Mandato soltanto **R-bootstrap → R-baseline → S s005 → V0/audit s005**, poi
consegna al supervisore con prompt05. P1–P3 attendono baseline adeguata e mandato
successivo. Prompt04 origine della migrazione; prompt13 preparazione completata.
IMPL-V0-004 accolta come correzione del diagnostico nel perimetro del piano,
nessuna r004 del piano. Non avviare packaging/installazioni/lock/Docker o fasi
dalla roadmap. Chat distinta, nessun subagente, review simulata o auto-snapshot.
Non aggiornare STATE/HANDOVER/eventi/indici/arbitrati condivisi. Supervisore non
è un servizio continuo: consegna una richiesta concreta e checkpoint completo.

## Letture e identità

1. AGENTS, CHANGELOG, PROJECT-CONTEXT/HANDOVER globali e STATE/HANDOVER della run;
   manage-implementation-run, protocollo e verify-conversion-fidelity.
2. Brief A1–A7, piano r003 e arbitrato r003 integralmente per la chat fresca;
   D1–D5 e le 24 disposizioni antecedenti restano vincolanti.
3. Prompt04, prompt13, report/checkpoint implementatore, request s005;
   evidence/supervisor-implementation-r001/baseline-recovery-r002/decision.md,
   sources.json e response.json della stessa cartella;
   implementation/stages/impl-r001-stage-baseline-s004/impediment-r004.json. Nessuna reinterpretazione come GO V0.
4. In evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s005/:
   response.json, reception.json, decision.md, sources.json, transition.json,
   metadata-transition.json, snapshot-verify.json, resume-commands.json,
   control-errors.json e checks-final.json; checkpoint
   handovers/supervisor-stage-baseline-s005.md. Non rieseguire helper supervisore.
5. Manifest s005; in evidence/implementation-r001/preparation-baseline-s005/:
   preparation.md, input-inventory/baseline-inputs, due runner-target,
   host-inventory/host-read-receipt/final-host-observation e receipt,
   dependency-metadata, target-transition, diff, pure-tests-01 e log,
   real-s004-pure-audit e replay, internal-suite-commands, handoff-checks.
   Inventari grandi in lettura strutturata. Leggi i cinque diagnostici,
   fixture/provenienza, copie originali e sorgenti pertinenti.

Percorsi senza prefisso relativi a `temp/run-a001-fase0-uv/`; diagnostici/test
sono relativi alla radice. Non caricare tutta temp; recupera intervalli troncati.
Riepiloghi, hash e test puri non sostituiscono le prove operative dei contenuti.

| Oggetto | SHA-256 |
| --- | --- |
| plans/plan-r003.md | `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b` |
| arbitrations/arbitration-plan-r003.md | `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d` |
| prompts/04-implementation-r001.md | `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035` |
| prompts/13-implementation-r001-prepare-baseline-s005.md | `bc1c3a04499dfcc37c6b7b346f4d130fe8ea36b52859f2aee02f3d2b76445577` |
| implementation/stages/impl-r001-stage-baseline-s005/request.json | `2d82e003935a2936fd1c200027f3d860ea6bdf5e21887b3584864c489929a324` |
| snapshots/impl-r001-stage-baseline-s005.json | `aba4fc260a9e3de766bd5a5022bac60f9703881f128a443c61671015fc627ed0` |
| evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s005/resume-commands.json | `ac31c626ed173672e5aa297e4170bc5a6dfb4cdd9d58158599488559dbbc7f87` |
| snapshots/baseline-recovery-context-r002.json | `81d6f349860bf797c0dcad3a828cf370398cd370025ffeefead2e9187f7ccff3` |
| evidence/implementation-r001/preparation-baseline-s005/internal-suite-commands.json | `bf6ba2de47cea0689a365fe2e9aca1de60b2af5296d74d8f3da47bd798a4aee9` |
| scripts/diagnostics/run-a001-fase0-uv/run_offline.py | `a937e0067f81f9d09c2fec49f4b47de00eb9a8b254b64d75caed1029b3c71018` |
| scripts/diagnostics/run-a001-fase0-uv/check_runner.py | `086f7dbb12798ca990a04dec575ae72e5066cc32721ee522aa26e41644adf1ed` |
| scripts/diagnostics/run-a001-fase0-uv/make_source_manifest.py | `7dc068734a5fd00b63a652995402727264b9d8e909ffa1b3c93841e5948ce44d` |
| scripts/diagnostics/run-a001-fase0-uv/run_baseline.py | `750e120edca98f00c543e497218d8f56cb7cfac56d90859292f83753f3ad2e44` |
| tests/unit/test_uv_diagnostics.py | `4ee4c198de5fb148efad0de2dc7f8480e14d02611d3691ae54b16ac779e14d97` |

Stage **impl-r001-stage-baseline-s005 MATCH:96 file +903 artefatti**, manifest262.858 byte;
worktree `2213067cf2a0d74c07c44dbfa4d717fb93fa43f8ccdba4cfffa1fcff32fffc71`. Branch `feature/run-a001-uv`, HEAD/dev/merge-base
`66ba82200e5def5a4db76f9bafccb0731b506091`, indice vuoto. Sei metadata modificati
più17 file preparatori nuovi; nessun pyproject/lock/interprete prodotto migrato.

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
git diff --cached --name-only
python3 scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-baseline-s005
```

Conferma hash/MATCH prima e dopo ogni passo. Request992 file/20 assenze, tutti
immutati; sei metadata supervisore esclusi dalla request ma inclusi nei96 file
snapshot.903 artefatti comprendono request senza self-hash, piano/arbitrato/
prompt04/13, inventari/fixture/copie/evidenze già prodotte. Nuovi S/receipt/output
assenti; risposta/prompt14/checkpoint/checks successivi fuori dal freeze.
S sarà prodotto dopo e citerà questo manifest, senza autoreferenzialità.

Recovery-r002 storico alla ricezione soltanto per CHANGELOG e due diagnostici
assegnati, poi sei metadata prima del freeze.718 artefatti recovery e455 s004,
219 output dell'impedimento s004 e antecedenti invariati. Vecchi hash del driver
si confrontano con la copia prima del delta, CHANGELOG storico con la copia
ricevuta s004; non trasferire MATCH al nuovo worktree o riscrivere receipt/S/
manifest. Report/checkpoint autore mutabili esclusi, copie ricevute conservate.

## Preparazione verificata e limiti

Audit supervisore di sola lettura/stat/hash/AST e dati ricevuti:3.722 file
distinti, inventario autore3.714,24 assenze tecniche e36 future. Due target
da1.381 file:1.379 invariati, soli driver/test aggiornati; provenienza aggiornata,
parametri runtime/guardie/host/deps/cache invariati. Quattro argv esterni e due
interni verificati, nuovi path/cache s005; nessun runtime del supervisore.
23 test stdlib puri autore PASS, non rieseguiti dal supervisore; gate precedenti
F4/sliding e observer/preflight/producer/wrapper immutati secondo lettura/AST.
Tre errori dei soli lettori supervisore sono conservati e risolti; nessun input
tecnico o receipt storica cambiato. Identità/autore/assenza delega sono dichiarazioni.

IMPL-V0-004: il confronto puro separa gli override soltanto dopo validazione
esatta di fase/argv/invocation, repo/CWD fidati, cache e verbosità effettive;
config comune non ignorata e presenza distinta da null. Il replay puro dei raw
s004 riproduce il vecchio errore e completa col nuovo confronto67nodeid/201eventi/
62pass/5fail osservati, conservando suiteFAIL/exit1 e V0s004FAIL/exit2. Questa
preparazione non è la nuova prova V0s005 e non completa la receipt fallita.
Comparator circoscritto alla baseline senza config file, non generale.

Preserva `/tmp/a001-uv-baseline-pi8cvs6x` con venv/tmp/workspace/uv-cache/tiktoken-cache.
Bootstrap `/home/davide/.pyenv/versions/3.12.3/bin/python3`; baseline
`/tmp/a001-uv-baseline-pi8cvs6x/venv/bin/python`, Python 3.12.3, 17 dipendenze
leggere più pip, pytest 9.1.1 e tiktoken 0.14.0 già presenti. Cache pubblica
cl100k_base hash `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`.
Nessun fetch/download/install/sync/build; niente pesi/font/inferenza.

**Unico candidato attivo: Firejail esistente 0.9.72**, stesso host/clone UID1000.
Il negativo unshare s002 è equivalente solo per la scelta fallback, con stesso
binario/ramo/host/path inventariati; non riavviare il primario identico.
I PASS R s004 non ammettono il nuovo driver: tutti i gate R vanno ripetuti.
Se host/input/guardie cambiano, consegna lo scostamento; nessun altro candidato.

Per ciascuno dei quattro comandi esterni usa `exec_command` con
**sandbox_permissions="require_escalated"**, CWD della radice e justification
specifica in step_tool_profiles di resume-commands. Ogni comando è soggetto
alla review automatica del tool, senza prefix_rule ampia; freeze/lettura non
sono una sua approvazione anticipata. Non usare il confine use_default già
EPERM s001 o un terminale/host diverso per aggirare un rifiuto.

I may_execute_now=false dei JSON congelati descrivono la preparazione e restano
immutati. La risposta e questo prompt ammettono la sequenza soltanto dopo i
gate. Resume-commands preserva gli argv/cwd/output/timeout della request e
rinomina soltanto i campi di ammissione storici; non è un'autorizzazione ampia.

Nessun sudo, nuovo privilegio/setuid/profilo persistente o cambiamento
sysctl/AppArmor/rete/permessi/socket/policy host. Wrapper host UID1000,
programma soltanto nel runner D4 completo: path daemon noti/alias/canonici
solo connect/close senza send/recv/API; socket sintetico proprio esclusivo
positivo host prima/dopo e negativo nel runner; socketpair positivo reale.
IP isolato non dimostra confinamento UNIX. Path assenti non provano blacklist;
namespace numerici storici non sono un'identità da riprodurre. Limiti /proc/
canali inventariati da dichiarare, nessuna garanzia generale di sandbox ostile.

## Sequenza concreta, un passo alla volta

Gli argv integrali e parametri tool sono in
[resume-commands.json](../../evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s005/resume-commands.json).
Quattro comandi esterni identici a request/proposta, con soli path s005 rispetto
a s004. Il prefisso Firejail costruito dal wrapper conserva `--env=TMPDIR`
esatto dal target e tutte le blacklist prima di `--`; niente sonde abbreviate.

1. Verifica identità/Git/MATCH e tutti gli input esterni, i futuri output/S
   assenti e cache collection/run nuove. Controlla spazio nel profilo host:
   320806912 byte osservati nel read supervisore 2026-10-04T07:29:37.467128+00:00, non una
   misura futura. Stime: rete/acquisizioni zero, output 50 MB, R/S due minuti,
   V0/audit 35 minuti. Manuale audit distinto dal timeout wrapper 1.200 secondi.
   Se spazio/venv/cache mancano, registra impedimento; nessuna pulizia implicita.
2. **R-bootstrap** completo con target bootstrap s005, output
   `evidence/implementation-r001/runner-bootstrap-r-firejail-s005/`.
   Conserva tool parameters/justification/profilo/review, sessione, exit,
   stdout/stderr/durata e receipt anche se negato o incompleto.
3. PASS solo con exit0 e wrapper/inside/runner PASS: netns distinto host e
   coerente padre/figlio, IP/egress/rotte esterne negati, socketpair positivo,
   daemon e sintetico negati padre/figlio; sintetico positivo host pre/post.
   Origini/binari/interprete/deps/cache/prerequisiti/FD corrispondenti,
   TMPDIR e TIKTOKEN esatti padre/figlio prima del programma, variabili vietate
   assenti, cleanup e hash/MATCH pre/post. Leggi tutti i gate, non solo status.
4. Bootstrap PASS → **R-baseline** completo con venv/target baseline s005,
   output `runner-baseline-r-firejail-s005/`. Nessun PASS ereditato dal bootstrap.
5. Due R complete PASS → **S-baseline**, wrapper ripete R/preflight nella stessa
   invocation del producer. Copia baseline-originals come --repo, clone corrente
   come --host-repo, baseline-inputs e snapshot s005. Output esclusivo
   `implementation/stages/impl-r001-stage-baseline-s005/sources.json`, wrapper
   `runner-baseline-sources-firejail-s005/`. Envelope schema1; ID SHA256 del
   payload JSON sort_keys=True, ensure_ascii=True, separators=(',',':'),
   timestamp escluso. Confronta dieci originali/copie/HEAD/sorgenti correnti/
   files e artifacts del manifest; osserva hash esterno/ID e input toolchain.
   S baseline distinto dal prodotto, non chiude B/I/E/negativi D2 futuri.
6. S corrente → **V0-originals**, R/preflight/comando nella medesima invocation,
   output runner `runner-baseline-v0-firejail-s005/`, contenuti
   `evidence/implementation-r001/baseline-results-s005/`, workspace
   `/tmp/a001-uv-baseline-pi8cvs6x/workspace/v0-s005`. Driver verifica S/correnti/
   snapshot/receipt R prima di import o collection. Non eseguire la suite fuori
   dal wrapper: collection e run sono i due processi figli interni a questa V0.
7. Prima/dopo ogni passo verifica MATCH e inventari/hash, exit/namespace/
   processi, input unchanged e cleanup. Gate incompleto → IMPEDITA;
   mismatch di S/hash/receipt/snapshot → FAIL prima dell'applicazione.
   Non ritoccare input o output congelati/prodotti per ottenere PASS.

Esempio eseguibile per **un solo passo** nel profilo require_escalated, con
justification del passo. Scegli step solo dopo i gate reali; non lanciare tutti
e quattro in blocco. Il codice controlla identità e alcuni prerequisiti; resta
obbligatoria la lettura di tutti i gate D4 e la review automatica per comando.

```bash
/home/davide/.pyenv/versions/3.12.3/bin/python3 -I - R-bootstrap <<'PY'
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
root = Path('/home/davide/workarea/markdown-for-llms')
path = root / 'temp/run-a001-fase0-uv/evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s005/resume-commands.json'
assert hashlib.sha256(path.read_bytes()).hexdigest() == 'ac31c626ed173672e5aa297e4170bc5a6dfb4cdd9d58158599488559dbbc7f87'
data = json.loads(path.read_text())
step = sys.argv[1]
assert step in ('R-bootstrap', 'R-baseline', 'S-baseline', 'V0-originals')
assert os.getuid() == os.getgid() == 1000
commands = {x['step']: x for x in data['profiles']['firejail']['commands']}
item = commands[step]
snapshot = root / data['snapshot']['path']
assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == data['snapshot_sha256']
for key in ('request', 'wrapper', 'baseline_driver'):
    record = data[key]
    assert hashlib.sha256((root / record['path']).read_bytes()).hexdigest() == record['sha256']
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
source_path = root / 'temp/run-a001-fase0-uv/implementation/stages/impl-r001-stage-baseline-s005/sources.json'
if step == 'S-baseline':
    assert not source_path.exists()
if step == 'V0-originals':
    source = json.loads(source_path.read_text())
    canonical = json.dumps(source['payload'], sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode('utf-8')
    assert source['schema'] == 1 and source['id'] == 'sha256:' + hashlib.sha256(canonical).hexdigest()
    assert source['payload']['snapshot']['label'] == data['label']
    assert source['payload']['snapshot']['sha256'] == data['snapshot_sha256']
    assert not (root / 'temp/run-a001-fase0-uv/evidence/implementation-r001/baseline-results-s005').exists()
    assert not Path('/tmp/a001-uv-baseline-pi8cvs6x/workspace/v0-s005').exists()
print(json.dumps({'step': step, 'depends_on': item['depends_on'],
                  'argv': item['argv'], 'profile': 'require_escalated'}), flush=True)
proc = subprocess.run(item['argv'], cwd=item['cwd'], shell=False, timeout=item['timeout_seconds'])
raise SystemExit(proc.returncode)
PY
```

Yield breve e sessioni conservate, aggiornamenti entro 60 secondi. Rifiuto
tool: azione/motivo/review registrati e IMPEDITA al supervisore; nessun altro
canale per aggirarlo. Runner inadeguato → IMPEDITA, niente baseline esterna.
S/receipt mismatch → FAIL. Nessun retry sugli stessi output, errore nascosto
o modifica host. Conserva receipt partial e cleanup anche quando il programma
non parte; osserva processi owned e stato finale, senza dichiarare assenza
di processi sulla sola mancanza di un file.

## Audit contenuti e provenienza V0

Apri output, diff e report **integrali** F1–F6: testo, Unicode, numeri,
formule LaTeX, codice, link/riferimenti, ordine, validation/summary e asset.
Hash/presenza JSON non chiudono fedeltà. Registra le perdite legacy F2/F5/
anchor/separatori senza correggerle; PNG originale e perdita di copia/riferimento
distinti. Non chiamare completo il bundle legacy incompleto. Distingui original
→ converted → cleaned → validated → chunked e la natura dei tre confronti.

IMPL-V0-001: F4 due identità [1], letterali o escaped, nei contesti richiamo e
bibliografia e in ordine fra le due sezioni; tabella A/123,45 e B/-7, formula,
Unicode e failed=0. Non usare un unescape globale o normalizzare l'output.
Il gate nuovo deve osservare questi contenuti nella nuova esecuzione.

IMPL-V0-002: F6 semantic custom/1000/100 da validated, almeno due chunk,
frontmatter/indici/confini/metadati e contenuto ordinato, tre config 1000/100,
1200/80,1600/120 identificate senza attribuire V5. Overlap semantic zero legacy
è da inventariare. Sliding complementare sul medesimo sorgente F6 ASCII:
unica encode originale, finestre decode osservate e source ID reali; verifica
ID comuni, testo integrale delle slice, margini/offset, intersezioni di testo
e copertura continua del sorgente. Mantieni la vecchia misura ritokenizzata
separata dal gate; observer delegano senza fabbricare dati o risultati.
16 chunk/15 intersezioni da 100 ID e 326–349 caratteri erano dati s004,
non PASS s005 da copiare. Limite ASCII dichiarato; nessun claim generale Unicode.

SUP-V0-003: leggi
[internal-suite-commands.json](../../evidence/implementation-r001/preparation-baseline-s005/internal-suite-commands.json)
e confronta argv interni/config/osservazioni effettive. Stessi quattro file:
tests/unit/test_cleaning.py, test_validation.py, test_chunking.py e
tests/integration/test_pipeline.py. Non aggiungere i test diagnostici al conto
legacy, non eseguire discovery dalla radice con i due test_pipeline.py.

- Collection e run: due processi distinti con interprete baseline `-I`, nel
  medesimo runner della V0 e CWD esterna legacy-suite. Preflight S/R in ciascun
  figlio prima di pytest/conftest. Cache `pytest-cache-collection` e
  `pytest-cache-run` assolute, assenti prima e inventariate dopo; niente riuso
  cache s004 né cache del clone. Collection exit0, run exit1 atteso distinto.
- Collection `--collect-only -q --verbosity=-1`, run `-v -rA --verbosity=1`
  e `-o verbosity_test_cases=1`, entrambi `-o cache_dir=<cache assoluta>`.
  Non modificare pytest.ini/addopts né selezione/assert/esiti. Conserva rootpath,
  config file/hash/ini, effective argv/addopts, verbosity globale/test_cases,
  pytest/plugin/versioni e cache path effettivi. L'elenco storico atteso dei 67
  nodeid è un riferimento di identità, non prova dei futuri pass.
- Osserva tutti i nodeid raccolti ed eventi setup/call/teardown, exit, cause e
  captured output. Zero duplicati, extra, deselected, skip/xfail/errori; ogni
  nodeid ha eventi e setup/teardown positivi. 67 raccolti/eseguiti, 62 PASS
  e cinque FAIL call **osservati**, cache fresh coerente; niente complemento
  inferito dei soli fail. Conserva collection/run stdout/stderr e i due JSON
  observed, anche se la riconciliazione fallisce o è interrotta.
- Cinque FAIL attesi in TestPipelineIntegration: test_end_to_end_without_conversion,
  test_pipeline_statistics_tracking, test_concurrent_processing,
  test_pipeline_logging, test_configuration_integration. Cause lookup
  `clean_markdown.py` da CWD esterna (ENOENT/AssertionError) da confermare
  individualmente. Cause/conti diversi o provenienza incompleta → V0 FAIL;
  niente xfail/skip/forzature o correzione del prodotto per rientrare nel conto.


IMPL-V0-004 nella **nuova** esecuzione: tutti i campi config obbligatori presenti,
null esplicito solo per configfile/configfile_record assenti in questa baseline;
rootdir clone, selezione quattro file, addopts=[], versione/distribuzioni plugin
stabili, inicfg comune vuoto. Ogni fase ammette soltanto cache_dir assoluta derivata
dalla CWD reale v0-s005/legacy-suite; soltanto run anche verbosity_test_cases="1".
Verifica valori effettivi/ini e argv integrali senza extra/dup/conflitti, cache
prima/dopo e verbosità (-1,-1)/(1,1). Repo/CWD attesi derivati dall'invocazione,
non dai JSON observed. Campo mancante non equivale a null; configurazione nuova,
addopts/inicfg comune nuovi o versione diversa invalidano questa baseline.
Preserva raw completi, nomi plugin numerici non confrontati fra processi;
confronta distribuzioni/versioni con S/R/deps identificati. I due override
prescritti non autorizzano a ignorare gli altri campi o a copiare il replay s004.

**La suite rimane FAIL/exit1 quando registra i cinque fallimenti.** Un eventuale
V0 characterization PASS richiede la caratterizzazione completa prevista e
l'audit manuale dei contenuti, senza chiamare PASS la suite o la migrazione.
Conftest ancora riferisce il clone originale, non la futura distribuzione.
Le API/cache lette staticamente e i 23 test puri non sostituiscono queste prove.
Perdita nuova/incerta → FAIL; incroci solo con input/tool/interprete/deps/
constraints/tokenizer identificati, senza acquisizioni implicite.

## Invalidazioni e consegna al supervisore

Qualsiasi tuning, nuovo input/fixture/interprete/deps/cache/diagnostico/retry
dopo freeze richiede request **impl-r001-stage-baseline-s006** o successiva,
nuovo freeze D1 e ripetizioni pertinenti. Non correggere qui un difetto di
driver scoperto dal runtime, non riusare S/receipt s004 o s005 discordanti.
Preserva output falliti ed esplicita il blocco; niente GO condizionale ambiguo.

Aggiorna solo report/checkpoint propri con esiti reali, hash/ID/input/output/
conti/costi, processi/namespace/cleanup/limiti e audit integrale, preservandone
prima le versioni ricevute. Consegna al supervisore con
[prompt05](05-supervisor-stage-r001.md), sia in caso di baseline adeguata sia
per FAIL/IMPEDITA o nuova preparazione. **P1–P3 restano in attesa in questa
chat**; non aprire lo stage package automaticamente.

Niente auto-snapshot/polling, STATE/HANDOVER/eventi/indici/ADR comuni. Changelog
del lavoro reale solo alla consegna dopo l'ultimo verify s005, documentando
che la sola modifica dichiarata lo rende storico. Supervisor farà la nuova
transizione; non riscrivere il manifest. S/B/I/E e negativi D2 reali restano
obblighi futuri, i mock non li chiudono.

V7/V8 future **obbligatorie**, costo/perimetro pesante separati. V10/V11,
pesi/font/inferenza, Marker normale, deploy/invio remoto non autorizzati.
Due review indipendenti codice e arbitrato finale futuri; commit/merge/push/
promozioni eseguiti dall'utente, mai da te. Temp e venv/cache esterne non
viaggiano con Git: preserva il lavoro effettivo.
