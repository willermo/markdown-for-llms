# R021 — Fix mirato dopo le review r002 e correzione del tempo

2026-10-07, supervisore. Si applica con
[arbitrato r002](arbitration-implementation-r002.md) e
[prompt59](../prompts/59-implementation-r001-fix-launcher-and-close-review-r002.md).
Supera lo stato di attesa delle review di R020; non modifica requisiti del piano
r003, isolamento o prove storiche.

## Contabilità corretta

CLA-I006 accolto: cumulativo storico osservabile **almeno 8321,418589 s**,
sforamento **almeno 41,418589 s** sul cap 8280 (41,42 s arrotondati).
La riscrittura finale successiva alla misura non è conservata. Il limite
superiore storico è ignoto; quota storica NON PASS. R020, report-r019 e receipt
originali restano intatti e sono corretti per il dato corrente dal
[ledger — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Ultimo workload prodotto noto 8251,006318 s entro il cap. Nessuna nuova prova
richiesta per sanare o cancellare un costo storico.

## Autorizzazione prospettica del supervisore

**+600 s core, nuovo cap cumulativo 8880 s**, per il solo risultato di prompt59.
Prima di partire addebita un ingresso pari al massimo tra **8380 s** e un
eventuale maggior costo storico ricostruito. La soglia 8380 è una carica
prudenziale per l'ammissione nuova, distinta dal limite inferiore storico;
non attesta che tutta l'attività passata sia stata misurata.
Da tale ingresso conteggia preparazione, correzioni, prove, verifiche e consegna.
All'ingresso minimo rimangono **500 s**; riserva almeno 60 s per raccogliere
processi e chiudere documenti/receipt. Singolo figlio massimo 180 s e sempre
entro il residuo reale, inclusa raccolta. Nessun reset o trasferimento di costi.
L'attesa fra chat/review non è lavoro implementativo; i nuovi intervalli attivi
sono esplicitamente identificati. Ogni riscrittura finale deve avere il comando
conservato e una misura comprensiva della sua attività, senza dati esatti fittizi.

Storage **352 MiB incrementali**, Hentry553541632, pool1GiB, stop896MiB e
riserva16MiB invariati. Radici/esclusioni e guardie di spazio dello scope vigente
restano obbligatorie; includere review/supervisione nel ledger. La ricezione del
presente arbitrato ha circa13MiB residui: rimisurare prima del lavoro, non
trattare questo numero come nuova quota. Output compatti e nuove versioni;
nessun cleanup di baseline, FAIL o oggetti altrui.

Nessuna nuova acquisizione, rete, installazione di dipendenze, build, Docker,
inferenza/modelli/font/GPU, servizio o modifica persistente dei privilegi.
B56/immagine r007 rimangono attribuite ai propri input. Il fix riguarda diagnostico
R, test mirati e guide non consumate dalla build; non modificare README,
pyproject/lock, dieci moduli, Compose o driver/test Docker in questo mandato.
Mezzi reversibili, nuovi snapshot tecnici, nuovi binding S/I necessari e
sonde reali isolate per a001/seconda run sono delegati all'implementatore.

Scope macchina leggibile:
[authorized-scope](../../evidence/supervisor-arbitration-implementation-r002/authorized-scope.json).
EPERM/EACCES: diagnosi ed eventuale escalation tramite tool, mantenendo R;
solo rifiuto effettivo o nuovo confine sostanziale richiede supervisione.
Consegna completa report-r020; poi due review nuove mirate. Nessuna attesa freeze,
nuova pianificazione o operazione Git automatica.
