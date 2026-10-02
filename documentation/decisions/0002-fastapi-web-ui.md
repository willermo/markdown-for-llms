# ADR 0002 — FastAPI, worker separato e Web UI

- Data: 2026-10-02
- Stato decisione: accettata
- Stato implementazione: da realizzare; framework frontend non scelto
- Origine: preferenza esplicita FastAPI e richiesta GUI/Web UI dell'utente

## Contesto e decisione

Adottare FastAPI per le API della nuova applicazione. I servizi applicativi devono
essere richiamabili anche dalla CLI. Eseguire OCR e conversioni lunghe in un worker
separato dal processo che risponde alle richieste HTTP, con stato persistente dei job.
Tecnologia della coda e schema preciso dei job restano da progettare.

Fornire una Web UI utilizzabile nel browser su Linux, macOS e Windows: caricamento,
avanzamento, archivio, anteprima Markdown e asset, confronto col sorgente, revisione
delle proposte OCR e download. Introdurre il percorso minimo nella prima versione
che integra API, worker e storage; completare la revisione nelle fasi successive.

FastAPI genera documentazione delle API con OpenAPI e Swagger UI; queste capacità
supportano gli sviluppatori. L'interfaccia per convertire e revisionare documenti
richiede un frontend dedicato. [Fonte FastAPI](https://fastapi.tiangolo.com/features/).

## Alternative e conseguenze

Una GUI desktop imporrebbe distribuzioni e aggiornamenti aggiuntivi. Una Web UI
servita insieme all'applicazione risponde al requisito locale e multipiattaforma.
React con TypeScript è un candidato, non una dipendenza già approvata o installata.
La scelta sarà motivata da anteprima, confronto, manutenzione e costo di sviluppo.

La velocità delle API non determina quella dell'OCR: misurare separatamente risposta
HTTP, attesa in coda e inferenza. Le librerie AI restano dietro adattatori, senza
legare la logica di conversione al framework HTTP.

Dettagli: [progetto Web UI](../architecture/web-ui.md), [roadmap](../roadmap.md).
