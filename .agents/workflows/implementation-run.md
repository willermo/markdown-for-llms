# Workflow — Ciclo supervisionato delle run

La procedura autorevole è il [protocollo delle run](../../documentation/development/run-lifecycle.md),
adottato con [ADR 0007](../../documentation/decisions/0007-supervised-development-runs.md).
Usare la [skill dedicata](../skills/manage-implementation-run/SKILL.md) per scegliere
il ruolo e recuperare il contesto minimo.

Sequenza: analisi → supervisione → piano → due review indipendenti → arbitrato →
implementazione → due review indipendenti → arbitrato → integrazione Git manuale →
eventuale redeploy e archivio. NO_GO sul piano riapre pianificazione; NO_GO sul codice
richiede pianificazione dei fix e un nuovo giro. Non simulare il lavoro delle chat mancanti.

Il contesto parte da `temp/HANDOVER.md`. Se manca, recuperare l'eventuale archivio
permanente o inizializzare una nuova run con `scripts/run_context.py` secondo il
protocollo. Una run creata non equivale a una fase approvata o completata.
