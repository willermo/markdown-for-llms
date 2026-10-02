# ADR 0001 — Fedeltà del documento e derivati opzionali

- Data: 2026-10-02
- Stato decisione: accettata
- Stato implementazione: da realizzare nella nuova pipeline
- Origine: obiettivo espresso dall'utente e impostazione generale accettata

## Contesto e decisione

Il risultato principale è un documento Markdown completo, accompagnato dagli asset
e da un manifest di conversione. Preservare testo, matematica, tabelle, immagini,
didascalie, note e riferimenti, rispettando la struttura logica e l'ordine di lettura.
L'impaginazione originale a colonne non deve essere riprodotta come colonne nel testo.

Conservare il sorgente e rappresentare internamente blocchi, gerarchie e provenienza.
L'esatta rappresentazione intermedia sarà scelta dopo il confronto dei motori.
Le correzioni OCR/AI producono revisioni tracciabili; le incertezze restano visibili.
Non promettere identità universale ottenuta automaticamente da scansioni ambigue.

Chunking e preparazione RAG sono derivati opzionali, disabilitati nel percorso base.
Il dialetto Markdown e le regole di esportazione saranno formalizzati con fixture
prima di dichiarare interoperabilità con altri renderer o formati.

## Alternative e conseguenze

Una pipeline basata soltanto su stringhe Markdown è più semplice, ma perde informazioni
necessarie a revisione, ordine di lettura e provenienza. L'attuale cleaning globale
non costituisce un modello riutilizzabile di correzione conservativa.

Le verifiche devono misurare omissioni, alterazioni, formule, riferimenti e asset;
Markdown sintatticamente valido o formule renderizzabili non provano fedeltà semantica.
La ricostruzione di grafici o diagrammi resta un derivato distinto dall'immagine originale.

Riferimenti: [draft](../architecture/preliminary-draft.md), [roadmap](../roadmap.md).
