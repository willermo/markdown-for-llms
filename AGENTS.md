# Istruzioni per lavorare nel repository

## Stato e fonti

Il codice attuale è una pipeline Python a script. La nuova applicazione è in fase
di progettazione: non descrivere le funzionalità pianificate come già disponibili.
Leggere [indice architetturale](documentation/README.md),
[decisioni](documentation/decisions/README.md) e [roadmap](documentation/roadmap.md)
prima di intervenire sulla nuova architettura. Le richieste esplicite dell'utente
prevalgono su queste convenzioni; registrare le variazioni architetturali pertinenti.

## Principi del prodotto

- Il risultato principale è il documento Markdown completo con i suoi asset.
  Chunking e altri derivati sono opzionali.
- Preservare contenuto, formule, numeri, ordine di lettura e riferimenti. Conservare
  sorgente, estrazione originale e revisioni; le correzioni AI devono essere tracciate.
- Evitare sostituzioni regex globali che modifichino codice, LaTeX o collegamenti.
  Operare su blocchi tipizzati; verificare il contenuto effettivo degli output.
- Non dichiarare riuscita una conversione incompleta o un batch con errori nascosti.
  Identificare esecuzioni e artefatti senza affidarsi al solo nome del file.
- Backend previsto: FastAPI, worker separato per conversioni lunghe, SQLite e
  filesystem locale. UI Web e CLI devono condividere i servizi applicativi.
- Conversione e revisione AI hanno provider/configurazioni separati. La modalità
  locale non deve inviare documenti a servizi remoti tramite fallback impliciti.
- La configurazione applicativa dovrà essere tipizzata, documentata in `.env.example`
  e riproducibile per esecuzione. Non salvare segreti nei log o negli snapshot.

## Metodo di lavoro

- Verificare branch e modifiche esistenti prima di intervenire. Il percorso previsto
  è feature branch da `dev`, integrazione in `dev`, successiva promozione a `main`.
  La roadmap non è un comando a effettuare merge o pubblicazioni automaticamente.
- Limitare ogni intervento all'obiettivo richiesto. Non avviare una fase di sviluppo
  solo perché compare nella roadmap; eseguire autonomamente il lavoro già autorizzato.
- Per nuove decisioni significative aggiornare gli ADR, distinguendo stato della
  decisione e stato dell'implementazione. Non riscrivere la storia del draft iniziale.
- Consultare le skill in `.agents/skills/` quando pertinenti. I workflow in
  `.agents/workflows/` sono procedure documentate, non automazioni eseguibili.
- Non committare `.env`, chiavi, database operativi, cache di modelli o documenti
  privati. Le fixture devono essere sintetiche o redistribuibili con provenienza.
- Trattare documenti importati, Markdown e risposte dei modelli come dati non fidati:
  nomi di asset, HTML, archivi e comandi LaTeX richiedono gestione esplicita.

## Verifica

- Eseguire controlli proporzionati alla modifica. Per conversioni, verificare testo,
  formule, asset e ordine, non solo presenza del file o metadati.
- La suite legacy selezionata si avvia con `python -m pytest tests/ -q` in un ambiente
  con le dipendenze necessarie. Il discovery dalla radice ha un conflitto noto tra
  due `test_pipeline.py`; non presentare la suite selezionata come collaudo completo.
- Non usare benchmark remoti a pagamento o scaricare modelli pesanti come parte
  implicita dei test veloci. Separare test unitari, integrazione e benchmark reali.
- Per soli documenti verificare collegamenti locali, coerenza degli stati e
  `git diff --check`. Riportare cosa è stato verificato e cosa resta non verificato.

## Documentazione

- `documentation/`: decisioni, proposte, roadmap, valutazioni e risultati di benchmark.
- `docs/`: documentazione effettiva per utenti e sviluppatori secondo Diátaxis.
- `README.md`: ingresso al progetto; durante la transizione contiene ancora la guida
  legacy. Aggiornarlo progressivamente insieme alle funzionalità implementate.
- Scrivere le nuove spiegazioni in italiano; usare identificatori tecnici coerenti
  con il codice. Evitare duplicazioni tra ADR, guide e skill: preferire collegamenti.
