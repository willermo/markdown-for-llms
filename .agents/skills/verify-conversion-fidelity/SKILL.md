---
name: verify-conversion-fidelity
description: "Verifica la preservazione dei contenuti quando si cambiano o confrontano parser, OCR, cleaning, formule, immagini, ordine di lettura e serializer Markdown del convertitore. Usare per regressioni di conversione e benchmark dei motori; non per sole modifiche editoriali o confronti prestazionali senza contenuti."
---

# Verificare la fedeltà della conversione

1. Leggere [ADR 0001](../../../documentation/decisions/0001-document-fidelity.md),
   la fase interessata della [roadmap](../../../documentation/roadmap.md) e i contratti
   effettivamente implementati. Non assumere che l'architettura prevista esista già.
2. Selezionare il campione più piccolo che espone il rischio: per esempio indici
   LaTeX, una pagina a due colonne, una tabella o immagini con didascalie. Usare
   fixture sintetiche o redistribuibili; non aggiungere documenti privati al Git.
3. Confrontare sorgente, rappresentazione intermedia se disponibile e output reale.
   Verificare testo/Unicode, omissioni, numeri, ordine dei blocchi, formule, note,
   celle e asset pertinenti. Un file presente, un voto del validatore o un LaTeX
   renderizzabile non dimostrano correttezza del contenuto.
4. Riprodurre il difetto prima della modifica quando possibile e dimostrare il
   risultato dopo. Per stato/storage includere il caso pertinente: stesso nome,
   riesecuzione, output incompleto o interruzione. Non ampliare indiscriminatamente
   la suite se non riguarda il cambiamento.
5. Per confronti tra motori seguire il [workflow benchmark](../../workflows/converter-benchmark.md).
   Registrare versioni, configurazione effettiva e hardware; separare mock da prove
   reali. Non avviare implicitamente chiamate a pagamento o download pesanti.
6. Riportare errori e limiti osservati, inclusi quelli non risolti. Non normalizzare
   via regex simboli o spazi significativi soltanto per far coincidere i risultati.

Consegnare evidenze prima/dopo e verifiche pertinenti. Quando mancano sorgente di
riferimento, modello o hardware, delimitare la conclusione alle prove eseguite.
