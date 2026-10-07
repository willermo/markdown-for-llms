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

- Per le run di implementazione seguire il [ciclo supervisionato](documentation/development/run-lifecycle.md)
  e la skill `.agents/skills/manage-implementation-run/SKILL.md`. Leggere prima
  `temp/HANDOVER.md` e lo stato della run indicata, se presenti. In assenza di `temp/`,
  ricreare la struttura con `scripts/run_context.py`; non inventare review o esiti.
- Ogni run parte da un feature branch di `dev`. Supervisione, pianificazione,
  implementazione e le due revisioni indipendenti hanno prompt, report e responsabilità
  separati. ChatGPT e Claude sono i revisori previsti; sostituzioni vanno registrate.
- Il mandato copre un risultato completo e i suoi fix nel perimetro autorizzato.
  Errori ordinari, test rossi e difetti degli strumenti si correggono nella stessa
  chat; scostamenti sostanziali tornano al supervisore. I fix locali non riaprono
  pianificazione e doppia review del piano. La delega è per obiettivo: sono ammesse
  anche soluzioni tecniche reversibili non enumerate nel prompt, se rispettano
  requisiti, costi e confini espliciti. Fonti pubbliche già configurate e relativi
  redirect con provenienza verificata rientrano nelle acquisizioni autorizzate;
  non sono nuove fonti per il solo cambio di host. Registrare scelta, motivo,
  rischi e prove per i reviewer. Le review del codice restano due,
  indipendenti e in chat nuove, con criteri comuni e perimetro mirato nei giri di fix.
- Timeout, livello di log, percorsi di nuovi output e launcher propri sono scelte operative entro
  i limiti del mandato: correggerli e riprovare dopo diagnosi nella stessa chat,
  registrando la chiamata reale e conservando il fallimento. Un argv congelato
  identifica la versione consegnata; non rende ogni suo parametro un requisito.
  Distinguere input di confronto protetti, input della prova e parametri adattabili
  secondo il protocollo. Non aumentare budget o indebolire isolamento per correggere
  un errore ordinario; non attribuire prove vecchie a input modificati.
- Un errore di permessi durante l'esecuzione (es. EPERM/EACCES) non equivale a
  un rifiuto del sistema di approvazione. Diagnosticare e, se necessario e
  disponibile, richiedere l'escalation prevista dagli strumenti per l'azione
  già autorizzata, mantenendo i controlli di isolamento del progetto. Fermarsi
  al rifiuto effettivo dell'approvazione o se il recupero cambia costi, privilegi
  persistenti o confini; non aggirare una decisione negativa. Regola completa
  nel [protocollo](documentation/development/run-lifecycle.md#restrizioni-del-sandbox-e-approvazione-degli-strumenti).
- Solo il supervisore aggiorna stato condiviso e arbitrati; gli altri ruoli scrivono
  report e checkpoint propri. Un `GO` vale per gli artefatti identificati nel report.
- L'implementatore completa il piano approvato e i fix senza consegne intermedie
  per errori ordinari o freeze delle prove. Identifica autonomamente gli input
  effettivi, crea gli eventuali snapshot di esecuzione e ripete le prove invalidate.
  Questi snapshot attribuiscono le prove e non sono approvazioni. Alla consegna
  completa il supervisore crea lo snapshot comune per le due review, arbitra e
  prepara chiusura oppure prompt di fix. Solo limiti sostanziali reali richiedono
  un intervento anticipato; continuare intanto il lavoro indipendente autorizzato.
- L'utente esegue commit, merge, eventuale push e promozione a `main`: fornire comandi
  contestualizzati dopo il `GO`, senza eseguirli automaticamente. Un deploy richiede
  un perimetro operativo esplicito. Questa preferenza prevale sui workflow precedenti.
- Prima di passare a una nuova chat aggiornare il checkpoint del proprio ruolo.
  Archiviare ciò che serve in futuro prima della pulizia selettiva di `temp/`.
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
- Per il progetto usare la venv collegata al managed locale CPython 3.12.13,
  con origine verificata e package installato nello stesso interprete dei figli.
  Dopo modifiche ricostruire sdist/wheel, reinstallare la wheel canonica e confrontare
  i dieci hash e gli input S/B/I prima di avviare comandi con --no-sync; il solo sync
  non basta come prova. Procedure e limiti in [guida uv](docs/how-to/ambiente-uv.md).
- La suite veloce è `"$RUN_DEV_PY" -I -B -m pytest tests/ --ignore=tests/api
  --ignore=tests/packaging --ignore=tests/docker -q`, dopo preflight installazione,
  cache tokenizer e Pandoc pronti e runner R PASS. La [reference](docs/reference/toolchain-legacy.md)
  descrive API/packaging/Docker/governance e discovery. I due smoke radice sono
  esclusi esplicitamente. La suite selezionata non è collaudo completo; i risultati
  legacy storici non si trasferiscono al codice nuovo. La migrazione non è ancora
  convalidata; non installare o scaricare per riparare una collection fallita.
- Non usare benchmark remoti a pagamento o scaricare modelli pesanti come parte
  implicita dei test veloci. Separare test unitari, integrazione e benchmark reali.
- Per soli documenti verificare collegamenti locali, coerenza degli stati e
  `git diff --check`. Riportare cosa è stato verificato e cosa resta non verificato.

## Documentazione

- `documentation/`: decisioni, proposte, roadmap, valutazioni e risultati di benchmark.
- Aggiornare [documentation/CHANGELOG.md](documentation/CHANGELOG.md) a ogni avanzamento
  significativo: data, fase/run, risultato, verifiche, limiti e prossimo passo. Separare
  lavoro pianificato, realizzato e integrato; registrare commit, merge e deploy solo
  dopo verifica, conservando la cronologia precedente.
- `docs/`: documentazione effettiva per utenti e sviluppatori secondo Diátaxis.
- `README.md`: ingresso al progetto e collegamenti alle guide della pipeline legacy.
  Gli esiti transitori delle run restano nei registri di sviluppo.
- Scrivere le nuove spiegazioni in italiano; usare identificatori tecnici coerenti
  con il codice. Evitare duplicazioni tra ADR, guide e skill: preferire collegamenti.
