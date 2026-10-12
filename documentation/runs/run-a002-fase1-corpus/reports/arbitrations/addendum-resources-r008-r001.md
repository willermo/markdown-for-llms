# Addendum risorse a002 r008 r001 — disposizione24h

2026-10-10 Europe/Rome. **DISPOSED / implementazione locale r008 autorizzata**,
nessun GO del prodotto. Fonte: [risposta originale utente](../notes/user-wall-extension-r008-2026-10-10.md).
Supervisore Codex/OpenAI, skill manage-implementation-run e
verify-conversion-fidelity. Si supera una volta il solo limite temporale
del prompt24/27 e degli arbitrati precedenti, senza riscriverne gli esiti.
Prevale sul PENDING del [prompt27](../prompts/27-implementation-r008-fix-scanner-progress-and-inline-domains.md)
e sulla [proposta12h](proposed-resources-r008.md), conservata immutata.

## Scelta, perimetro e tempo

L'utente conferma la finestra estesa e accetta24h se12h rischiano esaurimento.
Il supervisore dispone **24h**, considerando sette giri di fix, nuove prove,
attese di due chat indipendenti e gli eventuali recuperi nella stessa chat.
Le ultime due review sono già state solo statiche per superamento dello stop:
il rischio è concreto. Non occorre usare tutte24h e nessuna estensione ulteriore
è implicita. La preparazione di questo addendum consuma la stessa finestra.

| Voce | UTC | Europe/Rome |
| --- | --- | --- |
| Conferma osservata dal supervisore | 2026-10-10 09:37:31 | 2026-10-10 11:37:31 |
| Stop nuovi workload (+21h) | 2026-10-11 06:37:31 | 2026-10-11 08:37:31 |
| Deadline consegna (+24h) | 2026-10-11 09:37:31 | 2026-10-11 11:37:31 |

Timestamp piattaforma non disponibile: ancoraggio al primo clock effettivo
dopo ricezione, non alla fine di preparazione o al prossimo avvio. Riserva3h
fra stop e deadline; tutto incluso, anche attese/review/log/arbitrato.
Stima operativa, non durata misurata:3h fix/prove,2h due review,1h registri/
arbitrato,6h attese,9h margine per recuperi,3h riserva =24h. È un'unica
disposizione per risultato e fix locali, non cap ripetuti per singoli comandi.

Entry originale **2026-10-08T07:06:54.369068Z** preservata. Wall trascorso
alla conferma osservata **181836.630932s**; nuovo limite cumulativo
**268236.630932s** (~74.510175h),
stop cumulativo **257436.630932s**. I contatori non
si azzerano. Quota52h, vecchio stop/deadline, review solo statiche, receipt
FAIL/STALE/NONPASS e lacune rimangono fatti storici, non PASS retroattivi.

## Limiti invariati e autonomia

CPU21600s, stopCPU19800s; nuovi6GiB, stop5GiB; RAMoperativa8GiB,
stop sampled6GiB. Riserve1800CPU-s/1GiB oltre allo stop. R≤900s/comando,
1MiB/stream e32MiB/file, €0 servizi, rete/acquisizioni/invio0, Firejail/R,
S→B→I e stesso managed3.12.13/package dei figli invariati. Gli stop temporali,
CPU o disco valgono al primo limite raggiunto. Prima dei workload aggiornare
ledger/misure e preflight sugli input realmente consumati; CPU lower e
disco retained storici non sono un saldo certificato. Nessun incremento
implicito di queste quote. Il caso loop ha timeout breve e reap verificato.

Mandato [prompt27](../prompts/27-implementation-r008-fix-scanner-progress-and-inline-domains.md):
F35–F38 e fix ordinari nella stessa chat implementatrice, prove dipendenti,
rescore/invarianza64 senza motori/V, consegna completa r008, nuovo comune,
due NUOVE chat reviewer indipendenti e arbitrato reale. NO_GO r007 resta
in vigore tecnicamente. Corpus/gold/raw/V13/A03 e L04 distinta preservati;
nessuna fase2/ML/provider/GPU/campagna a001/Git automatico/nuova pulizia.
Gli strumenti reversibili ordinari sono delegati, nessun freeze intermedio;
EPERM/EACCES si diagnostica/escalation tool mantenendo isolamento, rispettando
un rifiuto effettivo. Nessuna nuova conferma per ciascun comando autorizzato.

Receipt numerica e fonte in [resources-disposition.json](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-resources-r008-r001/resources-disposition.json).
