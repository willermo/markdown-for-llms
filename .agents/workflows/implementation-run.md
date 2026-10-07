# Workflow — Ciclo supervisionato delle run

La procedura autorevole è il [protocollo delle run](../../documentation/development/run-lifecycle.md),
adottato con [ADR 0007](../../documentation/decisions/0007-supervised-development-runs.md).
Usare la [skill dedicata](../skills/manage-implementation-run/SKILL.md) per scegliere
il ruolo e recuperare il contesto minimo.

Sequenza: analisi → supervisione → piano → due review indipendenti → arbitrato →
implementazione → due review indipendenti → arbitrato → integrazione Git manuale →
eventuale redeploy e archivio. Correzioni locali del piano diventano errata verificate; NO_GO sul codice
consegna subito il prompt di fix nella stessa run. Nuovo piano soltanto per cambi
sostanziali. Implementazione/fix possono usare la stessa chat; i due reviewer sono
nuove chat a ogni giro, prima sul delta completo e poi su chiusure/regressioni. Non simulare il lavoro delle chat mancanti.

L'implementazione comprende fix, aggiornamento autonomo degli input delle prove
e loro ripetizione fino alla consegna del risultato. Non aggiungere passaggi al
supervisore per freeze tecnici o errori ordinari. Il supervisore congela il
risultato consegnato per le review, arbitra e prepara chiusura o mandato di fix.

Il contesto parte da `temp/HANDOVER.md`. Se manca, recuperare l'eventuale archivio
permanente o inizializzare una nuova run con `scripts/run_context.py` secondo il
protocollo. Una run creata non equivale a una fase approvata o completata.
