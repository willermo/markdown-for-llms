# Proposta di disposizione temporale r008 — PENDING

2026-10-10. **Proposta, non autorizzazione**. F35–F38 sono definiti nel
[mandato concreto](../prompts/27-implementation-r008-fix-scanner-progress-and-inline-domains.md)
e nell'[arbitrato NO_GO r007](arbitration-implementation-r007.md).

La quota unica52h è arrivata allo stop nuovi workload alle10:06:54 Europe/Rome
del10ottobre, deadline13:06:54. Il [prompt24](../prompts/24-implementation-r007-fix-markdown-context-boundaries.md)
dispone «Nessun reset/incremento/cap minimo ripetuto». La riserva serve per
consegna e registri, non per un nuovo ciclo di workload. Nessuna nuova prova
o implementazione r008 eseguita dal supervisore.

Proposta unica per risultato e fix locali: **12 ore dalla conferma effettiva
dell'utente**, incluse preparazione, implementazione, prove, attesa e due
review, arbitrato e consegna. Nuovo stop workload a conferma+9h; deadline
a conferma+12h; riserva3h. Stima di pianificazione, non tempo misurato:
3h fix/prove dipendenti,2h due review e ricezione,1h registri/arbitrato,
3h margine prima dello stop,3h riserva di consegna. Il margine copre fix
ordinari e ripetizioni invalidate; nessun cap minimo per singola sonda.

**Entry originale e contatori cumulativi non si azzerano**: dopo conferma
il supervisore registra timestamp reale, wall trascorso e nuovo limite
totale dalla entry originale = wall trascorso alla conferma+12h. L'intervallo
già trascorso, anche di attesa, resta nel ledger. Non si retrodatano prove
né si trasforma l'attuale stop in un PASS.

CPU21600s, nuovi6GiB, RAMoperativa8GiB/stop sampled6GiB e stopCPU19800s/
disco5GiB restano cumulativi originali, con riserve1800CPU-s/1GiB. R≤900s,
1MiB/stream,32MiB/file, €0 servizi, rete/acquisizioni/invio0 e isolamento
invariati. CPU nota è lower-bound e disco retained non saldo certificato:
ledger e input attuali da ricontrollare prima dei workload, senza incremento
implicito di questi limiti. Il caso loop usa timeout breve e reap verificato.

La disposizione effettiva verrà registrata dal supervisore con risposta
originale e orari concreti in un addendum versionato. Silenzio/tempo trascorso,
snapshot tecnico, questa proposta o prompt27 non valgono come conferma.
Fino ad allora: **AWAITING_WALL_DISPOSITION**, nessun nuovo workload r008.
