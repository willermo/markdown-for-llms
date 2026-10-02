# Debito tecnico della pipeline legacy

Questa sintesi conserva i principali esiti della ricognizione precedente alla nuova
architettura. Base esaminata: commit `8fa5be9`. Non è una nuova certificazione del
codice né un elenco di difetti già risolti. Serve a evitare di trasferire gli stessi
problemi nella riscrittura e a individuare verifiche di regressione utili.

| Area | Problema osservato | Risposta prevista |
| --- | --- | --- |
| Integrità | Cleaning globale può alterare indici, codice, formule e spazi significativi; alcune opzioni non hanno effetto | Blocchi tipizzati e confronto dei contenuti, fase 2–4 |
| Configurazione | Precedenze incoerenti tra CLI, JSON, preset ed environment; errori possono ricadere silenziosamente sui default | Schema tipizzato, errori espliciti e snapshot effettivo, fase 2 |
| Identità | Collisioni tra file con stesso stem; presenza di un output può far saltare nuovi input | Identità per contenuto ed esecuzioni distinte, fase 2 |
| Riesecuzione | Output e chunk obsoleti rimangono mescolati a quelli nuovi | Manifest e pubblicazione atomica degli artefatti, fase 2 |
| Stato | Batch parzialmente falliti possono apparire riusciti; alcuni comandi terminano con successo senza risultati | Stati per documento/job, errori e codici di uscita coerenti, fase 2–3 |
| Validazione | Diverse soglie/configurazioni non applicate; punteggi non dimostrano fedeltà | Metriche sul corpus e problemi con provenienza, fase 1 e 4 |
| Chunking | Limiti token, overlap e strategie non sempre rispettati; offset errati possono produrre corpi vuoti | Derivato separato e test sul contenuto, fase 7 |
| Unicode | Tagli e normalizzazioni possono perdere o corrompere contenuto | Round trip e fixture multilingua, fase 2–4 |
| Formati | Supporto dichiarato più ampio del supporto effettivo dei percorsi Pandoc/Marker | Matrice di capacità verificata, fase 1 e 4 |
| Provider | Parametri ricevuti dal server non sempre applicati; immagini/provenienza perse | Contratti adattatori e test di integrazione, fase 2–4 |
| Job remoti | Polling e retry possono ripresentare lavori; timeout non equivalgono a fallimento certo | Conservare ID remoto, distinguere invio/attesa/recupero, fase 4 |
| Server | Elaborazione sincrona blocca endpoint async; health può essere positivo con motore non pronto | Worker separato, readiness distinta e limiti upload, fase 2–3 |
| Input e asset | Percorsi di asset non confinati e URL di polling non vincolati al provider | Risoluzione sicura, validazione archivi/URL e credenziali confinate, fase 2–4 |
| Packaging | Package costruito con moduli mancanti/entry point incoerenti; dipendenze da directory corrente | Package installabile, entry point verificati e lock delle dipendenze, fase 2 |
| Container | Dipendenze non bloccate e configurazione/mount non sempre effettivi | Immagini riproducibili e smoke test dei profili, fase 6 |
| Osservabilità | Log e monitor descrivono stati/metriche non sempre coerenti | Eventi e stato strutturati per esecuzione, fase 2–3 |
| Test e guide | Test non coprono perdita reale di contenuto; esempi e script obsoleti | Corpus, test di percorso completo e documentazione verificata |

Nella ricognizione la suite selezionata `tests/` ha riportato **67 test passati** in
un ambiente Python 3.12 predisposto per l'audit. Il discovery dalla radice presentava
un conflitto fra i due file `test_pipeline.py`. Non erano stati eseguiti benchmark
con modelli Marker reali o conversioni cloud. Questi risultati storici non sostituiscono
i controlli delle future modifiche e non attestano la correttezza delle conversioni.

Riutilizzare selettivamente fixture, logica e adattatori soltanto dopo confronto con
i nuovi contratti. Prima della rimozione del legacy documentare compatibilità,
migrazione degli output e comportamento delle CLI: vedere [roadmap](roadmap.md).
