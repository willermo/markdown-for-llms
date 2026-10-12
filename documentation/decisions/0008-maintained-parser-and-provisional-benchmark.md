# ADR0008 — Parser mantenuto e benchmark provvisorio

- Data: 2026-10-12
- Stato decisione: **accettata** come direzione e condizione della consegna ridotta
- Stato implementazione: **non iniziata**; r014 usa ancora il reader esistente
- Origine: accettazione esplicita dell’utente, run a002, dopo14 revisioni autore
- Sostituisce: nessun ADR; integra ADR0001 e applica un’eccezione locale aADR0007

## Contesto

Il reader del benchmark ha progressivamente ricostruito sintassi Markdown,
metadata, code e HTML con decisioni non sempre coerenti. Fix locali hanno
chiuso casi reali ma lasciato classi non convalidate. R014 Claude NO_GO statico,
review OpenAI incompleta per costo amministrativo: nessun arbitrato tecnico
finale positivo. L’utente accetta la consegna sperimentale e richiede che il
problema venga ripreso e risolto, dando priorità a un prodotto valutabile.

## Decisione

Adottare un parser mantenuto per la sintassi del profilo Markdown, con versione,
estensioni e configurazione riproducibili; adattare l’AST al modello osservato
invece di continuare a espandere la grammatica propria. Pandoc, già locale
3.1.3 e disponibile via JSON AST nelle prove a002, è il candidato iniziale
scelto per la verifica dell’adattatore. Nessuna nuova dipendenza o installazione
è attuata qui; l’idoneità a coordinate/provenienza e contenuti va dimostrata.
Se l’AST non offre dati sufficienti, il contratto deve dichiarare UNAVAILABLE
per quelle misure o valutare un parser mantenuto con source-map adeguata;
non inventare coordinate né riprodurre l’intera grammatica per colmare il gap.

Separare parsing sintattico, misure di fedeltà e controlli di confinamento/
risorse. L’AST del convertitore non è un oracolo indipendente della bontà
della sua conversione: gold, sorgenti e valutazione umana restano necessari.
Nessun COMPLETE del reader sperimentale autorizza rendering attivo o accessi.

Chiudere a002 con **accettazione utente del perimetro ridotto**, senza GO
tecnico retroattivo. Le doppie review restano per le implementazioni future.
Riaprire obbligatoriamente il debito dopo la prima applicazione valutabile
(uscita fase3), risolverlo prima degli automatismi di attestazione e del
rilascio. Non blocca il confronto esplorativo fase1 svolto con limiti e
revisione umana; non autorizza saltare le dipendenze della roadmap.

## Alternative considerate

- Continuare fix locali del reader: risultati parziali provati, costo dei giri
  e assunzioni condivise non accettabili come strategia principale.
- Eliminare ogni verifica: impedirebbe confronti motivati e diagnosi di perdite.
- Usare solo AST del motore come gold: confonderebbe sintassi con correttezza.
- Parser mantenuto più riferimenti indipendenti: direzione accettata; source-map,
  profilo e prestazioni da convalidare in un intervento distinto.

## Conseguenze e verifica

Conservare raw, asset, riferimenti, revisioni ed errori. Classificare separatamente
difetti riprodotti, tracce statiche e ipotesi AST. Chiudere il debito solo con
prove su originali/varianti di liste,div,fence,metadata,formule,HTML e risorse;
source-map verificata dove dichiarata, nessuna risorsa silenziosamente omessa,
controlli positivi di code/valori inerti e negativi di HTML reale. Rerun sul
corpus e sui test invalidati, classificando ogni delta rispetto ai dati storici;
due nuove review indipendenti e arbitrato sull’adattatore. Nessuna garanzia
di perfezione su Markdown arbitrario e nessuna sanatoria dei vecchi FAIL.

I budget degli esperimenti devono essere dimensionati al lavoro previsto;
la qualità non si giudica dal consumo CPU. Evitare lookup ripetuti degli
inventari: indicizzare una volta per path/hash, distinguere prove nuove da
ricevute, non rieseguire tutta la storia per ogni sonda. Limiti reali di
risorse e isolamento restano espliciti, nessuna quota aggiunta qui.

## Collegamenti

- [Fedeltà](0001-document-fidelity.md), [run supervisionate](0007-supervised-development-runs.md).
- [Chiusura a002](../runs/run-a002-fase1-corpus/manifest.md).
- [Registro del debito e criteri](../benchmarks/a002-residual-debt.md).
- [Roadmap](../roadmap.md), [prossimo incremento](../development/next-run-a003-engine-comparison.md).
- [Pandoc: reader→AST→writer e JSON](https://pandoc.org/filters.html), consultato2026-10-12:
  documenta capacità dell’interfaccia, non prova equivalenza completa di r014.
