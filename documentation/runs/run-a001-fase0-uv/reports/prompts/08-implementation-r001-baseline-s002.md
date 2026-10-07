# Ripresa implementativa r001 — baseline s002 — run-a001-fase0-uv

Agisci come **implementatore** nel repository
`/home/davide/workarea/markdown-for-llms`, riprendendo il checkpoint reale
`handovers/implementation-r001.md`. Preparazione s002 ricevuta e congelata dal
supervisore: stato comune **IMPLEMENTATION / STAGE_SNAPSHOT_READY**.
Continua [04-implementation-r001.md](04-implementation-r001.md), nel perimetro
della [decisione D1/D4 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
GO sul piano r003, nessun GO sul codice. R s002 NON_ESEGUITA, S NON_GENERATO,
V0 NON_ESEGUITA. I due impedimenti s001 restano evidenze storiche.

Non impersonare supervisore/revisori, creare subagenti, produrre snapshot o
modificare registri comuni. Il supervisore non è un servizio continuo.

## Letture e identità di ingresso

1. AGENTS.md, documentation/CHANGELOG.md, temp/PROJECT-CONTEXT.md,
   temp/HANDOVER.md e STATE/HANDOVER della run.
2. Skill manage-implementation-run, protocollo run-lifecycle e skill
   verify-conversion-fidelity. Brief A1–A7, piano r003 e arbitrato r003
   integralmente se chat fresca; altrimenti riconferma identità e D1–D5.
   NO_GO r001/r002 e tutte le disposizioni antecedenti conservati.
3. Prompt04, checkpoint/report propri, request s002; prompt07 come preparazione
   completata. Prompt06 e receipt s001 come antecedenti, senza ripeterli.
4. [Risposta del supervisore — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
   reception.json, transition.json e resume-commands.json nella stessa cartella;
   [checkpoint supervisore — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
5. Manifest s002; inventari in evidence/implementation-r001/preparation-baseline-s002/:
   host-inventory, host-read-receipt, baseline-inputs, due runner-target,
   target-transition e checks. Provenienza fixture, originali e cinque diagnostici
   del repository. Leggi strutturalmente i grandi inventari, senza tutta temp.

I percorsi senza prefisso sono relativi a `temp/run-a001-fase0-uv/`.

| Oggetto | SHA-256 |
| --- | --- |
| plans/plan-r003.md | `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b` |
| arbitrations/arbitration-plan-r003.md | `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d` |
| prompts/04-implementation-r001.md | `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035` |
| implementation/stages/impl-r001-stage-baseline-s002/request.json | `985cfb1c8c5b9430b23ac7b90f5d2ba3d41386cfc0707f2d06c69745ba07e19d` |
| snapshots/impl-r001-stage-baseline-s002.json | `c73d1547ad79ce19cd2b481b6f252ceb1cb21148a15c9fdf70768a664c0fc34f` |
| evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s002/resume-commands.json | `1a3a35f9d6c7c9a7f58024c5752d13ff8cd0a9dc8e1b2b6ebacd218e90769c6e` |
| evidence/supervisor-implementation-r001/runner-recovery-r001/decision.md | `5ce20ce29c2d7f15335042d0f92525486091cadfca9261fd49f2b65c4cede862` |
| evidence/supervisor-implementation-r001/runner-recovery-r001/proposed-commands-baseline-s002.json | `4be03c4beebf8e70fc9355ef920916e71d16c9ea498e67abaa06ef1ef5e139bc` |
| evidence/implementation-r001/preparation-baseline-s002/host-inventory.json | `aa4f41b9609c3f3cf4b52a685147de1ad1c1dbf4e80b5cbb3d9651ef311fdcdc` |
| evidence/implementation-r001/preparation-baseline-s002/host-read-receipt.json | `cad167ea7db8f5375059bf5ca2873039569a840d10bc2dd40abab5ea4f49e435` |

Stage **impl-r001-stage-baseline-s002**, **96 file +109 artefatti**, MATCH alla
consegna. Worktree SHA-256
`7a20d1d70827e6bf152d1ee2e0b5ffe5e5daaebfa5f917e4e19da0dbda1869b2`.
Branch `feature/run-a001-uv`; HEAD/dev/merge-base
`66ba82200e5def5a4db76f9bafccb0731b506091`; indice vuoto. Sei documenti
tracciati modificati e 17 file preparatori nuovi, nessun sorgente prodotto,
lock o pyproject implementato. Non resettare, cambiare branch o integrare Git.

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
git diff --cached --name-only
python3 scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-baseline-s002
```

Richiesti hash esatti e **MATCH s002** prima di ogni prova e dopo. Alla ricezione
runner-recovery-context-r001 era STALE solo per il changelog della preparazione;
prima del freeze il supervisore ha aggiornato sei metadata. I suoi 96 artefatti
sono invariati; conservati anche i 90/109/120/58 dei contesti precedenti.
transition.json distingue le transizioni. Non pretendere MATCH dei worktree
storici, rigenerare manifest o ignorare altre divergenze. I vecchi hash di
report/checkpoint nelle receipt si riferiscono alle versioni conservate:
report-r001-before-baseline-s002.md, implementation-r001-before-baseline-s002.md
e copie ricevute dal supervisore; i due file propri correnti sono mutabili.

## Preparazione riconfermata e perimetro operativo

Il supervisore ha verificato 160 input/assenze, 109 artefatti confinati,
2.819 file distinti e 24 assenze. Dieci moduli originali e quattro input build
sono identici a HEAD e alle copie. 1.940 file tecnici, 825 file uv cache e
1.380 file per target riconfermati per byte/hash; guardie e dipendenze invariati.
Le proprietà host provengono dalla receipt reale di sola lettura dell'autore,
exit 0 nel profilo individuato: UID/GID1000, namespace/policy/stat/binari e
origini registrati. Il supervisore non ha sostituito quelle proprietà con
lo stat del sandbox, né eseguito R o nuove sonde. Compatibilità **NON_PROVATA**.

Conserva `/tmp/a001-uv-baseline-pi8cvs6x`: venv, tmp, workspace, uv-cache,
tiktoken-cache. Bootstrap assoluto
`/home/davide/.pyenv/versions/3.12.3/bin/python3`; baseline
`/tmp/a001-uv-baseline-pi8cvs6x/venv/bin/python`. Python 3.12.3, 17 dipendenze
leggere più pip, cl100k_base già preparati; nessun nuovo download/install/sync.
Fixture F1–F6 sintetiche, F6 da 160 paragrafi: token/multichunk/overlap da
misurare. I sette test mock/sintetici antecedenti non provano R, V0 o fedeltà.

**Confine D4 per ciascun passo:** tool `exec_command` con
`sandbox_permissions: "require_escalated"`, wrapper host ordinario UID1000 sullo
stesso clone; applicazione soltanto nel runner unshare/Firejail completo.
Usa la justification dello specifico passo in `step_tool_profiles`, senza
prefix_rule ampia. Ogni chiamata rimane soggetta alla review automatica del tool.
L'approvazione della precedente lettura non approva sonde o conversioni.
Il freeze rende disponibile la sequenza nel mandato D1/D4, non è un'approvazione
automatica dei comandi da parte del sistema.

I `may_execute_now=false` dei JSON congelati descrivono la fase preparatoria
prima dello snapshot: non modificarli. La nuova response e questo prompt
ammettono i passi dopo i rispettivi gate; resume-commands conserva quegli argv,
CWD, timeout/output/prerequisiti e rinomina solo il flag storico di preparazione.
Non lanciare la sequenza nel profilo `use_default` che ha prodotto EPERM s001.

Nessun sudo, nuovo setuid/privilegio, cambiamento sysctl/AppArmor/rete/socket,
profilo persistente o operazione daemon. Firejail 0.9.72 esistente soltanto come
alternativa completa. Path daemon/alias/canonici inventariati: connect/close
diagnostico senza send/recv/API o comandi Docker; socket sintetico esclusivo
proprio e cleanup verificato. Nessun invio remoto o fallback terminale/altro host.
Gli ID namespace osservati sono storici: rileva quelli della prova, senza
tentare di riparare la macchina per riprodurre un numero precedente.

## Sequenza concreta e receipt

Gli **otto argv integrali** sono in
[resume-commands.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
uguali a `request.profiles` e al JSON proposto, per unshare e Firejail.
Non eseguire entrambi i profili in blocco o ricomporre comandi con eval.
Il placeholder descrittivo Firejail non è un argv: il wrapper costruisce la
blacklist per socket noti/alias/canonici e sintetico proprio.

1. Riconferma identità/MATCH, hash e assenze degli input esterni, directory/cache
   conservati e output futuri assenti. Ricontrolla spazio: 671.911.936 byte erano
   liberi alla lettura host, non sono una misura corrente. Stima 50 MB incrementali,
   rete zero, R/S due minuti, V0/audit 30 minuti; stime, nessuna prova eseguita.
2. Avvia **solo R-bootstrap unshare**, nel profilo tool D4 sopra, con argv della
   request, nuovo target bootstrap s002 e output
   `evidence/implementation-r001/runner-bootstrap-r-unshare-s002/`.
   Conserva comando tool completo, justification/profilo/risultato della review,
   exit/stdout/stderr/durata e receipt, anche se negato o incompleto.
3. PASS richiede exit 0 e receipt/inside/runner tutti PASS: namespace IP diverso
   host e stesso padre/figlio, nessuna egress/rotta esterna, socketpair positivo,
   path daemon/sintetico negati dentro e sintetico positivo host prima/dopo,
   cache/temp/Pandoc/file/dipendenze/origini corrispondenti, cleanup verificato.
   Netns IP da solo non confina UNIX su path. Se unshare è negato/inadeguato dal
   runner, conserva l'esito e prova **R-bootstrap Firejail** con il proprio argv
   e output. Un rifiuto della review automatica del tool richiede invece ritorno
   al supervisore, senza usare l'alternativa per aggirarlo.
4. Candidato con bootstrap completo PASS: **R-baseline dello stesso candidato**,
   target s002 baseline e venv, output `runner-baseline-r-<candidato>-s002/`.
   Tutti i gate reali. Prima di S/V0, se la catena unshare è inadeguata, Firejail
   deve avere entrambi i propri R; non eredita il PASS bootstrap di unshare.
5. Dopo entrambi R PASS: **S-baseline** dello stesso candidato. Il wrapper ripete
   R/preflight nella stessa invocazione del produttore. Copia originali come
   --repo, clone come --host-repo, nuovo baseline-inputs s002 e snapshot già
   congelato. Output esclusivo
   `implementation/stages/impl-r001-stage-baseline-s002/sources.json`.
   Conserva schema 1/payload, ID canonico, hash esterno e log; verifica sorgenti
   correnti/copia/artifacts del manifest e mapping esplicito ai dieci originali.
   Questo S baseline è distinto dal futuro prodotto migrato.
6. **V0-originals** dello stesso candidato: runner R/preflight ripetuti nella
   stessa invocazione; prima di collection/import driver confronta S corrente,
   input/snapshot e receipt R corrente. Runner output
   `runner-baseline-v0-<candidato>-s002/`; contenuti `baseline-results-s002/`;
   workspace esterno `/tmp/a001-uv-baseline-pi8cvs6x/workspace/v0-s002`.
   Nessuna installazione/build/sync/fetch durante la prova.
7. Prima/dopo ogni passo: verify s002 e hash esterni, output/namespace effettivi,
   exit e processi. Osservazioni mancanti non provano equivalenza. Non ritoccare
   request, inventari, diagnostici, fixture, receipt o snapshot congelati.

Per eseguire **un solo passo**, il seguente comando usa argv JSON verificati
come dati e `subprocess.run(..., shell=False)`. Passalo a `exec_command` nel
profilo **require_escalated**, CWD della radice, con la justification del passo.
Il codice non sostituisce il controllo delle receipt dei prerequisiti né la
review del tool. Cambia solo candidato/step ai valori ammessi dopo i gate.

```bash
/home/davide/.pyenv/versions/3.12.3/bin/python3 -I - unshare R-bootstrap <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path('/home/davide/workarea/markdown-for-llms')
path = root / 'temp/run-a001-fase0-uv/evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s002/resume-commands.json'
assert hashlib.sha256(path.read_bytes()).hexdigest() == '1a3a35f9d6c7c9a7f58024c5752d13ff8cd0a9dc8e1b2b6ebacd218e90769c6e'
data = json.loads(path.read_text())
candidate, step = sys.argv[1:]
assert candidate in ('unshare', 'firejail')
assert step in ('R-bootstrap', 'R-baseline', 'S-baseline', 'V0-originals')
item = next(c for c in data['profiles'][candidate]['commands'] if c['step'] == step)
snapshot = root / 'temp/run-a001-fase0-uv/snapshots/impl-r001-stage-baseline-s002.json'
assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == data['snapshot_sha256']
subprocess.run([sys.executable, '-I', str(root / 'scripts/run_context.py'),
                'verify', data['run_id'], '--label', data['label']], cwd=root, check=True)
assert not Path(item['output_directory']).exists()
print(json.dumps({'candidate': candidate, 'step': step, 'depends_on': item['depends_on'],
                  'argv': item['argv'], 'profile': 'require_escalated'}), flush=True)
proc = subprocess.run(item['argv'], cwd=item['cwd'], shell=False, timeout=item['timeout_seconds'])
raise SystemExit(proc.returncode)
PY
```

Usa yield breve del tool e conserva sessione/processi: evita attese che impediscano
aggiornamenti oltre 60 secondi. Non rilanciare un output esistente. Per ogni
comando registra i parametri tool e l'output effettivo separatamente dalle
receipt del wrapper. Rifiuto automatico: registra azione/motivo e IMPEDITA,
spiega la causa nella consegna e ritorna al supervisore, nessun altro canale.
Entrambi runner negati/inadeguati, gate incompleto o perimetro diverso necessario:
**IMPEDITA**, nessuna baseline fuori runner. Mismatch hash/S/snapshot: **FAIL**
prima dell'applicazione, nessun fetch/fallback o riparazione degli input.

## Audit V0, invalidazioni e consegna

Apri output/diff/report integrali F1–F6. Controlla testo Unicode, numeri, formule
LaTeX, codice, collegamenti e riferimenti, ordine, campi deterministici validation
e summary. F4 tabella/formula e failed=0; F5 riferimento/hash PNG e perdita
eventuale della copia asset. Registra difetti legacy senza correggerli qui;
JSON/metadati da soli non chiudono fedeltà o bundle. Difetto orchestratore da
CWD esterna conservato nella caratterizzazione.

F6: custom/1000/100 da validated, almeno due chunk, sequenza/contenuti/frontmatter/
indici/confini/metadati e overlap misurato. Complemento sliding sullo stesso
testo/parametri: almeno due chunk, intersezione >0. 1200/80 e 1600/120 ulteriori
osservabili identificati, non attribuire V5. Audit manuale prima di chiudere V0;
overlap semantic legacy può essere zero. Perdita nuova/incerta → FAIL e incroci
solo con codice/interprete/dipendenze/constraints/tokenizer identificati.

`legacy_suite` mantiene exit distinto dalla caratterizzazione automatica:
quattro file legacy espliciti, nodeid/raccolta/pass/fail/skip/errori e argv reali.
Non chiamarla suite completa o assumere risultati storici 67/6. Conftest oggi
inserisce la radice con moduli ancora originali: non verifica la futura wheel.

Se F6 richiede tuning o cambia un input/interprete/cache/dipendenza/diagnostico,
mantieni s002 esplorativo, prepara request **impl-r001-stage-baseline-s003**,
invalidazioni esplicite e ripetizioni dopo nuovo freeze del supervisore. Non
sovrascrivere S/receipt/snapshot o attribuire PASS alla fixture nuova. Anche
un retry su output già esistente richiede consegna identificata. P2/D2 e i
negativi reali receipt restano obblighi futuri: mock preparatori non li chiudono.

Dopo R/S/V0 effettivi e audit concluso, conserva chiusura/hash/limiti della
baseline e prosegui prompt04 P1–P3 verso richiesta stage package. Modifiche al
prodotto rendono storico il worktree s002; nuovo stage e S prima di build, poi
tests/image/final secondo D1/P2. Non trasferire PASS baseline al prodotto nuovo.
Non avviare fasi dalla sola roadmap o altre prove con perimetro non identificato.

Aggiorna report/checkpoint propri con esiti reali, input/output/hash, costi,
namespace/processi, limiti e prossima richiesta. Consegna via
[05-supervisor-stage-r001.md](05-supervisor-stage-r001.md) per s003, package o
impedimento. Nessun polling/auto-snapshot o modifica STATE/HANDOVER comuni,
eventi/indici/ADR. Changelog del lavoro realizzato soltanto alla nuova consegna,
dopo l'ultima prova/verify s002: la modifica rende storico il verify globale.

V7/V8 future **obbligatorie**, costo pesante distinto e target esplicito.
V10/V11/pesi/font/inferenza esclusi; nessun Marker normale, deploy o invio remoto.
Due nuove review indipendenti del codice e arbitrato finale ancora necessari.
Commit/merge/push/promozione dell'utente. Temp ignorata e venv/cache esterne
non viaggiano con Git: preservare e trasferire il lavoro effettivo.
