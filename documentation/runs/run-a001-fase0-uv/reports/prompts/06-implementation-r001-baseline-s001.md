# Ripresa implementativa r001 — baseline s001 — run-a001-fase0-uv

Agisci come **implementatore** nel repository
`/home/davide/workarea/markdown-for-llms`, riprendendo la consegna reale di
`handovers/implementation-r001.md`. Il supervisore ha ricevuto e congelato gli
input richiesti: **STAGE_SNAPSHOT_READY**, stato condiviso **IMPLEMENTATION**.
Questo prompt continua [04-implementation-r001.md](04-implementation-r001.md):
GO sul piano r003, nessun GO sul codice. Non impersonare supervisore/revisori,
non creare subagenti, non produrre snapshot o modificare registri condivisi.

## Letture e identità

1. AGENTS.md, documentation/CHANGELOG.md, temp/PROJECT-CONTEXT.md,
   temp/HANDOVER.md e STATE/HANDOVER della run.
2. Skill manage-implementation-run, protocollo run-lifecycle e skill
   verify-conversion-fidelity. Brief A1–A7; piano r003 e arbitrato r003 integrali
   se questa è una chat fresca, altrimenti riconfermarne identità e D1–D5.
   NO_GO r001/r002 e tutte le disposizioni antecedenti restano vincolanti.
3. Prompt 04, checkpoint proprio e report parziale; richiesta
   `implementation/stages/impl-r001-stage-baseline-s001/request.json`.
4. [Risposta del supervisore — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
   reception.json, static-findings.json e transition.json nella stessa cartella;
   [checkpoint di supervisione — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
5. Manifest dello stage, inventari di baseline/target/tokenizer, provenance delle
   fixture e cinque diagnostici visibili nel repository. Non caricare tutta temp.

I percorsi senza prefisso sono in `temp/run-a001-fase0-uv/`.

| Oggetto | Identità SHA-256 |
| --- | --- |
| plans/plan-r003.md | `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b` |
| arbitrations/arbitration-plan-r003.md | `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d` |
| prompts/04-implementation-r001.md | `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035` |
| implementation/stages/impl-r001-stage-baseline-s001/request.json | `39eefbd38a2d2448210fef013d3fdb00e0cdf4abf8e1b6c62497c320568c1763` |
| snapshots/impl-r001-stage-baseline-s001.json | `ba686193b00dc068510167608a6ddb439e415210c5fb02e6cd14a6d4117a4c96` |
| evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s001/resume-commands.json | `8414bb2fdd3877bff2dcc6ebd8b65534ebd9ff6b7d385bd6bc635171b39ea012` |

Stage **impl-r001-stage-baseline-s001**, 96 file +58 artefatti.
Worktree SHA-256 `d2ca720b15c08bc1e65d27d488070d37d80c75a4de7971d3da08deb3bbefea3b`.
Branch `feature/run-a001-uv`, HEAD/dev/merge-base
`66ba82200e5def5a4db76f9bafccb0731b506091`; indice vuoto.
Sei documenti tracciati di supervisione modificati e 17 file nuovi preparatori.
Non resettare né integrare la patch preesistente.

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
git diff --cached --name-only
python3 scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-baseline-s001
```

Richiesto **MATCH** dello stage nuovo e hash esatti sopra, prima delle prove.
`implementation-context-r001` è storico: all'ingresso implementatore era MATCH,
poi CHANGELOG +17 file nuovi; ora sei metadata ulteriori del supervisore,
documentati in transition.json. I suoi 120 artefatti sono invariati, come i
90/109 dei contesti review/arbitrato. Non pretendere MATCH dei worktree storici,
non ignorare divergenze ulteriori e non rigenerare i manifest.

## Preparazione già ricevuta

Dieci sorgenti originali e quattro input legacy sono byte-identici a HEAD e
alle copie baseline; 109 input/assenze e 1.942 file distinti degli inventari
riconfermati dal supervisore. Fixture F1–F6 sintetiche, F6 da 160 paragrafi:
token count, multi-chunk e overlap **non misurati**. Sette test preparatori
mock/sintetici documentati nei log dell'autore, non rieseguiti dal supervisore.
R/V0 **NON_ESEGUITI**, S **NON_GENERATO**, nessun PASS trasferibile da quei test.

Conservare `/tmp/a001-uv-baseline-pi8cvs6x`: venv/bin/python, tmp, workspace,
uv-cache e tiktoken-cache. Bootstrap assoluto
`/home/davide/.pyenv/versions/3.12.3/bin/python3`; baseline assoluta
`/tmp/a001-uv-baseline-pi8cvs6x/venv/bin/python`.
Le 17 dipendenze leggere da wheel, pip e vocabolario cl100k_base sono pronti;
inventari/lock preparatorio/tokenizer identificati, nessuna nuova acquisizione
necessaria per questo stage. Non ricreare ambienti o cache di nascosto.

## Sequenza concreta: R bootstrap → R baseline → S → V0

Gli **otto argv integrali**, CWD, timeout, output e prerequisiti sono in
[resume-commands.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
copiati senza modifiche da `request.profiles`. Usa solo queste quattro chiavi
`R-bootstrap`, `R-baseline`, `S-baseline`, `V0-originals` per ciascun candidato.
Non eseguire entrambi i profili in blocco. Il wrapper aggiunge internamente
blacklist del socket sintetico proprio; il placeholder descrittivo del
runner_prefix Firejail non è un argv da eseguire.

1. Riconferma hash/assenze e conservazione delle directory esterne. Avvia
   **solo R-bootstrap unshare**: wrapper stdlib assoluto con bootstrap Python,
   target runner-target-bootstrap.json, snapshot s001 e output nuovo
   `evidence/implementation-r001/runner-bootstrap-r-unshare-s001/`.
   Conserva exit/stdout/stderr/receipt anche su negazione della policy.
2. Per PASS servono exit 0 del wrapper e receipt/inside/runner **tutti PASS**,
   namespace IP distinto host e coerente padre/figlio, niente route/egress,
   socketpair positivo, daemon inventariati e socket sintetico negati dentro,
   sintetico raggiungibile fuori prima/dopo, prerequisiti/hash corretti e
   cleanup. Non assumere che unshare confini i socket UNIX su path.
   Se unshare è negato/inadeguato, registra il motivo e prova **R-bootstrap
   firejail** con l'altro argv e output proprio. Usa solo Firejail esistente
   0.9.72, noprofile/net=none/blacklist, senza nuovi profili o privilegi host.
3. Un candidato completo PASS ammette **R-baseline dello stesso candidato**:
   interprete della venv e runner-target-baseline.json; output separato
   `runner-baseline-r-<candidato>-s001/`. Anche qui tutti i gate reali.
   Se la catena di unshare risulta inadeguata prima di S/V0, l'alternativa
   deve avere entrambi i propri R, non ereditare il PASS bootstrap altrui.
4. Dopo entrambi R PASS, esegui **S-baseline** del candidato selezionato.
   Il wrapper ripete R nella medesima invocazione del produttore S. Il
   produttore usa la copia baseline come --repo, il clone come --host-repo,
   inventario baseline-inputs.json e snapshot già esistente. Output esclusivo:
   `implementation/stages/impl-r001-stage-baseline-s001/sources.json`.
   Conserva S.id, SHA esterno, schema/payload e log. Verifica legame dei dieci
   moduli agli artifacts della copia congelata, non ai futuri moduli migrati.
5. Esegui **V0-originals** dello stesso candidato: wrapper con target baseline,
   R reale nella stessa invocazione, driver run_baseline.py e S corrente.
   Runner output `runner-baseline-v0-<candidato>-s001/`; output contenuti
   `baseline-results-s001/`; workspace esterno `workspace/v0-s001`.
   Prima di import/collection il driver confronta S, input correnti, snapshot
   e R del namespace corrente. Nessuna installazione/build/sync/fetch nei test.
6. Verifica stage MATCH e input esterni prima/dopo ogni passaggio. Conserva
   argomenti effettivi, namespace osservati e figli troppo brevi non osservati;
   un'osservazione mancante non diventa prova di equivalenza. Non riscrivere
   request, inventari, fixture, diagnostici o artefatti già congelati.

Esempio di esecuzione **di un solo passo**, da radice, usando gli argv JSON
come dati. Cambia soltanto candidato/step ai valori sopra, dopo aver controllato
i prerequisiti; non usare eval o una shell per ricomporre la stringa:

```bash
python3 - unshare R-bootstrap <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
root = Path('/home/davide/workarea/markdown-for-llms')
data_path = root / 'temp/run-a001-fase0-uv/evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s001/resume-commands.json'
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == '8414bb2fdd3877bff2dcc6ebd8b65534ebd9ff6b7d385bd6bc635171b39ea012'
data = json.loads(data_path.read_text())
candidate, step = sys.argv[1:]
item = next(c for c in data['profiles'][candidate]['commands'] if c['step'] == step)
subprocess.run(['python3', 'scripts/run_context.py', 'verify', data['run_id'], '--label', data['label']], cwd=root, check=True)
assert not Path(item['output_directory']).exists()
print(json.dumps({'step': step, 'candidate': candidate, 'depends_on': item['depends_on'], 'argv': item['argv']}), flush=True)
proc = subprocess.run(item['argv'], cwd=item['cwd'], shell=False, timeout=item['timeout_seconds'])
raise SystemExit(proc.returncode)
PY
```

Prima di selezionare un passo successivo, apri le receipt dei prerequisiti:
l'esempio non sostituisce quel controllo. Mantieni i processi e output
identificati; usa l'esecuzione dei tool con yield breve e aggiorna il checkpoint
se interrotta. Non riavviare su output esistenti per ottenere un nuovo PASS.

Perimetro s001: Linux di questo clone, UID host 1000, binari inventariati;
nessuna modifica sysctl/AppArmor/setuid/rete/socket o dati reali dell'host.
Connect/close ai path daemon noti solo diagnostico, zero richieste operative;
sonde IP documentali solo dopo i gate di isolamento. Policy dei tool rispettata,
nessun aggiramento implicito. Il sandbox preparatorio negava anche socketpair:
non alleggerire quella prova. Se entrambi i candidati sono negati/inadeguati,
conserva le evidenze e consegna **IMPEDITA** al supervisore; V0 non parte fuori
runner. Un perimetro diverso va identificato dal supervisore prima dell'uso.

Costi dichiarati: rete stage 0 byte, nessun download; output incrementali stimati
50 MB, R/S circa 2 minuti e V0/audit circa 30 minuti, stime non esiti. Verifica
spazio disponibile: la misura nella richiesta è storica. Nessun canale remoto
implicito o a pagamento. Non cambiare target per far passare il diagnostico.

## Audit V0 e gestione delle invalidazioni

Il PASS automatico del driver riguarda la **caratterizzazione dei contenuti**;
`legacy_suite` conserva un exit distinto. Non dichiarare suite passata quando
fallisce. Registra raccolta/nodeid/pass/fail/skip ed errori; i quattro file
legacy espliciti non sono la suite completa né i 67/6 test storici.
Conftest della suite legacy inserisce la radice: oggi i dieci moduli lì sono
ancora originali; questa esecuzione non verifica la futura wheel.

Apri gli output effettivi, diff e report completi F1–F6: testo Unicode/numeri,
LaTeX/codice/riferimenti, ordine, campi deterministici validation e summary,
F4 tabella/formula e failed=0, F5 riferimenti/hash PNG e perdita eventuale di
copia asset. Registra i difetti legacy senza correggerli in questa baseline.
I JSON/metadati da soli non chiudono fedeltà o bundle degli asset.

F6: CLI custom/1000/100 da validated, almeno due chunk; conserva sequenza,
contenuto/frontmatter/indici/confini/metadati e overlap misurato. Complemento
sliding sullo stesso testo e parametri: almeno due chunk e intersezione >0.
1200/80 e 1600/120 sono ulteriori osservabili identificati. Audit manuale dei
risultati prima di chiamare chiusa V0; l'overlap semantic legacy può essere zero.

Se un input cambia o F6 richiede tuning, mantieni s001 come prova esplorativa e
prepara una nuova richiesta **impl-r001-stage-baseline-s002**, con input e
invalidazioni aggiornati. Non sovrascrivere s001, S, receipt o log; nessun
PASS definitivo attribuito alla fixture nuova finché il supervisore non
congela s002 e sono rifatte le prove pertinenti. Diagnostici da correggere,
dipendenze/interprete/cache diversi o tentativi da ripetere richiedono una
consegna identificata, senza ritoccare gli inventari congelati.

Mismatch dello stage o S/input: **FAIL** prima dell'applicazione; runner
indisponibile: **IMPEDITA**. Nessun GO condizionale o fallback non confinato.
Il produttore S è preparatorio: completare P2/D2 e tutti i negativi receipt
nella successiva implementazione; i test mock non li chiudono operativamente.

## Continuazione e consegna

Dopo R/S/V0 effettivamente caratterizzati e conservati, prosegui il mandato
del prompt 04, P1–P3 e preparazione dello stage package. Prima di cambiare
input registra la chiusura e identità della baseline: le modifiche successive
rendono storico il worktree s001; conserva le prove, non chiamarlo MATCH.
Nuovi sorgenti/packaging richiedono stage package e S nuovi prima di build,
poi tests/image/final secondo D1/P2. Incroci ambiente solo se divergono i
contenuti, senza alterare la baseline originale o il lock di produzione.

Consegna report/checkpoint propri aggiornati con esiti reali, processi ancora
attivi, costi, input/output/hash, limiti e prossima richiesta di stage. Se serve
s002 o il runner è IMPEDITA, torna al supervisore con
[05-supervisor-stage-r001.md](05-supervisor-stage-r001.md); niente servizio
continuo, poll o auto-snapshot. Non modificare STATE/HANDOVER comuni,
eventi/indici/metadata ADR o gli originali storici. Changelog solo per il lavoro
realizzato, nel momento di una nuova consegna: la sua modifica invalida il
verify globale s001, quindi prima termina le prove che lo richiedono MATCH.

V7/V8 restano future **obbligatorie**, con costo pesante distinto e target
esplicito. V10/V11, pesi/font/inferenza non autorizzati; nessun avvio Marker
normale, deploy o invio remoto implicito. Dopo il report finale serviranno due
nuove review indipendenti del codice e arbitrato; nessun GO finale in questo
freeze. Commit/merge/push/promozione sono dell'utente. Temp e ambienti esterni
non viaggiano con Git: trasferire contesto e lavoro quando cambia workspace.
