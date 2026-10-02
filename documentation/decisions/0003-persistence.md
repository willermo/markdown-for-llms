# ADR 0003 — SQLite e storage su filesystem

- Data: 2026-10-02
- Stato decisione: accettata per la prima versione locale
- Stato implementazione: da realizzare
- Origine: preferenza SQLite dell'utente e impostazione generale accettata

## Contesto e decisione

Usare SQLite per documenti, esecuzioni, stati dei job, configurazioni effettive prive
di segreti, problemi rilevati, revisioni e riferimenti agli artefatti. Conservare file
sorgente, immagini, risultati e cache su filesystem, separando lo storage durevole
dalle cache eliminabili. I nomi di file forniti dall'utente non sono identificatori.

Identificare i sorgenti tramite hash e assegnare identità distinte alle esecuzioni.
La riusabilità di un risultato dipende anche da configurazione e versioni di motori,
modelli e pipeline. Una nuova esecuzione non sovrascrive la precedente.

Pubblicare un risultato soltanto dopo scrittura e controlli degli artefatti. Progettare
esplicitamente recupero da crash e riconciliazione tra DB e filesystem: non esiste
una transazione unica implicita che li comprenda entrambi. Prevedere migrazioni,
backup coerenti e verifica di ripristino.

## Alternative e conseguenze

SQLite riduce i servizi da amministrare nella distribuzione su singolo host. Usare
transazioni brevi e misurare contesa tra API e worker. PostgreSQL diventa una scelta
da riesaminare per più host o concorrenza di scrittura elevata; non è un requisito
della prima versione. DB e storage locali devono risiedere su volumi persistenti.

La [documentazione SQLite](https://sqlite.org/whentouse.html) chiarisce i limiti
di concorrenza e l'opportunità di un database client/server in altri scenari.
Collegamento operativo: [roadmap](../roadmap.md).
