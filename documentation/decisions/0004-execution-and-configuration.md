# ADR 0004 — Configurazione e profili di esecuzione

- Data: 2026-10-02
- Stato decisione: accettata come indirizzo architetturale
- Stato implementazione: I1 Linux CPU r004 GO, integrato in dev a f8bf44c e push verificato il 2026-10-17; altri profili da verificare
- Origine: requisiti locale/remoto, CPU/GPU, Docker e `.env` dell'utente

## Decisione

Supportare elaborazione locale e provider remoti tramite adattatori sostituibili.
Separare il provider di conversione dal provider di revisione AI. La modalità locale
non attiva automaticamente una chiamata cloud. Conservare modello, versione e
parametri effettivamente usati, senza includere chiavi o credenziali.

Il file `.env` sarà il punto di configurazione manuale dell'applicazione, con un
`.env.example` commentato e validazione tipizzata all'avvio. Eventuali opzioni per
singola conversione nella UI/CLI/API saranno validate e persistite nel DB. Precedenza
e valori predefiniti devono essere univoci e documentati; la politica esatta sarà
definita nella fase di fondazione. Non aggiungere un secondo file JSON obbligatorio.

Prevedere profili CPU, GPU NVIDIA e servizio di inferenza nativo ove necessario,
con download espliciti, versioni bloccate, cache persistente e uso offline dopo
preparazione dei modelli. Un profilo non garantisce supporto di ogni modello.

## Conseguenze e alternative

Docker è la distribuzione principale proposta per API, UI e worker. L'accesso alla
GPU richiede un percorso specifico per piattaforma: verificare Linux/NVIDIA e
Windows/WSL2; su Mac valutare inferenza nativa per usare Metal. Non assumere che
il medesimo container esponga tutte le GPU. Le combinazioni supportate saranno
pubblicate dopo prove reali, insieme a requisiti RAM/VRAM e prestazioni.

Marker resta un candidato insieme a Docling e parser nativi. Nessun motore viene
selezionato sulla sola base di una demo. Riferimenti e candidati sono nel
draft (archived: `documentation/architecture/preliminary-draft.md`, tag `pre-docs-2026-10-18`); criteri nella [roadmap](../roadmap.md).

## Precisazione 2026-10-17 — Profilo iniziale

[ADR0009](0009-first-usable-product.md) seleziona Marker 1.10.2 CPU per il
primo prodotto sperimentale Linux locale; adapter provato con C06/C07 e
nuova C06 reale r003, GO I1 r004. La candidatura generale sopra resta storia; non è promessa di
supporto degli altri profili o di fedeltà del contenuto.
