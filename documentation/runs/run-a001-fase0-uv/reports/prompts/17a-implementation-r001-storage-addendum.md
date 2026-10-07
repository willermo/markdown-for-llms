# Addendum a prompt17 — nuova location operativa autorizzata dall’utente

Per la **chat implementatrice già avviata** su run-a001-fase0-uv.
Leggi questo file insieme al prompt17 originale e alla
[disposizione storage](../../evidence/supervisor-handover-package-s002-r001/storage-decision.md).
L’utente ha riferito probabile cancellazione della baseline nella pulizia di /tmp
e richiesto temp del progetto per i nuovi dati operativi. Nessun recupero già eseguito.

La sola preparazione s002 continua con gli stessi22URL/5SHA/17pin/513hash e cap:
usa il nuovo [scope effettivo](../../evidence/supervisor-handover-package-s002-r001/storage-scope-r001.json).
Sostituisci negli helper/manifest/argv/env/assenze/inventari/proposte non ancora
congelati .cache/uv-package-s002 con
**temp/run-a001-fase0-uv/work/package-s002** (sidecars,baseline-json,tmp,uv-cache).
Target baseline futuro: **temp/run-a001-fase0-uv/work/baseline-recovery-r001**.
Usa path assoluti derivati dalla radice verificata dove richiesto da argv/TMPDIR;
nei manifest usa root-relative confinati. Parent work può essere creato insieme
al passo futuro directory esclusive, non nella preparazione; verifica tutti
gli antenati, git check-ignore e filesystem/spazio. Output ordinali, nessun symlink.

Non creare ora quei target, non acquisire HTTP/software/venv, non cambiare
prodotto/test/fixture/diagnostici esistenti o .venv/.venv-python canoniche.
Preserva .cache/uv-package-s001 e vecchi receipt/report/snapshot. Se avevi già
preparato helper/request dei vecchi path, conserva la versione e produci una
correzione esplicita con hash nuovi; nessun rewrite retroattivo di consegna.

Questo addendum, scope/disposizione e il nuovo contesto
supervisor-handover-package-s002-r001 vanno inclusi fra gli input della request D1.
SHA/worktree esatti nella receipt.json del supervisore nella medesima cartella,
scritta dopo lo snapshot senza autoreferenzialità. Prompt17/vecchio scope restano
antecedenti; il contesto package-metadata-preparation-context-r001 può ora essere
STALE **solo per CHANGELOG/worktree_sha256** della transizione descritta. Verifica
confrontando1343artefatti immutati e tutti file tecnici, poi nuovo contesto MATCH;
non ignorare divergenze aggiuntive. Nessuna modifica tracciata propria è autorizzata.

Checkpoint/report autore propri aggiornati a WAITING_FOR_STAGE_SNAPSHOT con
request reale; nuovo supervisore via prompt18/prompt05 riceve e congela solo dopo
verifica. Non aggiornare stato comune o simulare ricezione dell’addendum.
Al resto applica integralmente prompt17/A1–A7/D1–D5; zero software, runtime
applicativo, cleanup, commit/merge/push/deploy o delega nella preparazione.
