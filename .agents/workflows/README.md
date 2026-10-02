# Workflow del repository

Procedure da eseguire nel contesto del lavoro autorizzato. Questa cartella è una
convenzione documentale, non una cartella di automazioni riconosciuta dal client.

| Workflow | Risultato |
| --- | --- |
| [Decisione architetturale](architecture-change.md) | ADR, motivazione e collegamenti aggiornati |
| [Incremento di sviluppo](implement-feature.md) | Modifica circoscritta, verificata e documentata |
| [Confronto dei convertitori](converter-benchmark.md) | Evidenze riproducibili e limiti dei candidati |
| [Documentazione](documentation.md) | Pagina Diátaxis aderente al prodotto disponibile |
| [Integrazione e rilascio](release.md) | Versione verificata lungo il percorso feature → dev → main |

La futura CI eseguirà controlli automatici distinti da queste procedure. Le skill
locali le richiamano senza richiedere MCP o installazioni globali.
