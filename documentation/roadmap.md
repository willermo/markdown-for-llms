# Roadmap del convertitore

Aggiornata al 2026-10-02. Le fasi descrivono il percorso fino al prodotto; non sono una
stima temporale né un'autorizzazione ad avviare ogni attività in anticipo. Dipendenze,
hardware e corpus determineranno dimensione e durata degli incrementi.

## Quadro delle fasi

| Fase | Risultato verificabile | Dipendenze | Stato |
| --- | --- | --- | --- |
| 0 | Governance, decisioni e documentazione iniziale | Nessuna | Completata con questo intervento |
| 1 | Corpus, benchmark e scelta motivata dei motori | Campioni e profili hardware | Da iniziare |
| 2 | Package applicativo, contratti, configurazione, DB e worker | Risultati essenziali della fase 1 | Da iniziare |
| 3 | Prima applicazione completa con FastAPI e Web UI minima | Fase 2 e scelta frontend | Da iniziare |
| 4 | Formati richiesti, PDF scientifici, OCR e asset | Fasi 1–3 | Da iniziare |
| 5 | Revisione OCR/AI e confronto sorgente/risultato | Provenienza e asset della fase 4 | Da iniziare |
| 6 | Distribuzione e collaudo CPU/GPU/remoto sui sistemi previsti | Fasi 3–5; prove preliminari già in fase 1 | Da iniziare |
| 7 | Esportazioni e derivati opzionali | Documento e revisioni stabili | Da iniziare |
| 8 | Migrazione, documentazione completa e rilascio | Criteri delle fasi richieste soddisfatti | Da iniziare |

## Fase 0 — Base progettuale

Consegne: branch `feature/document-converter-v2` da `dev`, draft iniziale, ADR,
requisiti Web UI, roadmap, `AGENTS.md`, skill e workflow locali, struttura `docs/`
Diátaxis e valutazione MCP. Il codice applicativo resta legacy.

Completamento: collegamenti locali e skill validati, nessuna decisione tecnica ancora
aperta presentata come implementata. Il README rimanda ai nuovi documenti; la sua
riscrittura completa accompagna lo sviluppo.

## Fase 1 — Corpus e scelta dei motori

Costruire un corpus piccolo ma rappresentativo, con riferimenti verificati a mano:
PDF testuali a una e più colonne, scansioni IT/EN, pagine miste, formule inline e
multilinea, matrici, note, bibliografie, tabelle e grafici. Includere TXT, DOCX, EPUB,
RTF e progetti LaTeX con inclusioni e immagini. Specificare diritti e provenienza;
tenere i documenti privati fuori dal repository.

Confrontare parser nativi/Pandoc, Docling e Marker aggiornato. Valutare un provider
remoto nel perimetro di dati e costo concordato. Per ciascun candidato registrare
versioni del codice e dei pesi, licenze, hardware, configurazione, tempo, memoria,
costo e fallimenti. Eseguire una prova CPU e una verifica preliminare dei percorsi
GPU previsti prima di basare l'architettura su un modello.

Misurare accuratezza del testo rispetto alla trascrizione, omissioni, ordine dei
blocchi, formule e simboli, celle delle tabelle, completezza degli asset e riferimenti.
Concordare soglie per classe di documento sul corpus osservato; nessuna media unica
deve nascondere una perdita di formule o pagine. I punteggi di confidenza del motore
sono segnali, non il riferimento corretto.

**Uscita:** report riproducibile, matrice formati/capacità/limiti, ADR sui motori e
prima scelta del modello documentale. Nessun vincitore predeterminato. I punti ancora
incerti devono avere un esperimento successivo circoscritto.

## Fase 2 — Fondazioni affidabili

Creare un package Python installabile con dipendenze riproducibili, separando dominio,
servizi, adattatori, API e worker. Definire documento strutturato, asset, manifest,
revisioni e profilo Markdown. Introdurre ID persistenti e provenienza per blocco;
evitare di accoppiare tutti i componenti allo schema interno di un singolo motore.

Implementare configurazione tipizzata e `.env.example` commentato, con precedenza
documentata e snapshot privo di segreti. Implementare SQLite con migrazioni, storage
confinato, hash del sorgente, esecuzioni distinte e invalidazione del riuso basata
su configurazione/versioni. Progettare job persistenti, tentativi, annullamento e
recupero dopo crash; scegliere il meccanismo di coda proporzionato al singolo host.

Preparare CI e controlli sul package, sui contratti e sui casi di regressione pertinenti.
I test veloci non scaricano modelli né usano credenziali reali. Conservare distinta
la suite legacy durante la transizione, correggendone discovery e packaging quando
si interviene su tali componenti.

**Uscita:** installazione da ambiente pulito; documenti omonimi e riesecuzioni non
collidono; errori persistiti; un riavvio non perde la coda; DB e artefatti sono
riconciliabili dopo scrittura interrotta; le opzioni effettive sono verificabili.

## Fase 3 — Primo percorso utilizzabile con Web UI

Esporre API FastAPI per upload, avvio, stato, archivio e download, con schema OpenAPI.
Il worker esegue una conversione reale di un formato semplice, inizialmente TXT e/o
DOCX. La CLI richiama gli stessi servizi. Scegliere il frontend con un ADR e fornire
UI per caricamento, coda, archivio, anteprima e download del bundle Markdown.

Servire UI e API con una configurazione locale semplice. Distinguere stato del job
e qualità/revisione del documento, mostrare gli errori e impedire download che
appaiano completi quando mancano artefatti. Conservare la configurazione delle
conversioni anche dopo riavvio. Introdurre subito la documentazione Diátaxis relativa
alle sole funzionalità effettivamente disponibili.

**Uscita:** prova nel browser dall'upload al download; ricaricamento e riavvio
mantengono archivio e stato; API reattiva durante conversione; batch con un file
errato mostra successo/errore per documento. Questo è il primo incremento di prodotto.

## Fase 4 — Conversione dei documenti scientifici e formati nativi

Completare gli adattatori per PDF testuale, scansioni e PDF misti con OCR IT/EN,
ordine di lettura a colonne, formule LaTeX, figure e didascalie. Verificare TXT,
DOCX, EPUB, RTF e bundle LaTeX con risorse associate. Documentare ciò che non è
rappresentabile fedelmente nel profilo Markdown e il relativo fallback esplicito.

Estrarre immagini o renderizzare regioni quando necessario; conservare grafici con
assi e legenda. Controllare tabelle, note, numerazione e riferimenti. Integrare il
primo provider remoto e separare invio, polling e recupero: un timeout non autorizza
a duplicare automaticamente un job. Vincolare URL e credenziali al provider previsto.

Portare in UI pagine sorgenti, formule, immagini e problemi individuati. Proteggere
percorsi di asset, upload e archivi; limitare risorse e confinare eventuali processi
LaTeX. Non perdere contenuti durante il rendering del risultato.

**Uscita:** corpus misurato rispetto alla fase 1, soglie concordate rispettate o
problemi espliciti; ogni immagine referenziata esiste; ordine e formule controllati;
nessun fallback cloud implicito; la matrice di supporto corrisponde a prove reali.

## Fase 5 — Revisione assistita e qualità

Integrare revisione AI opzionale tramite provider locale/remoto distinto dal motore
di conversione. Fornire contesto e ritagli, raccogliere proposte limitate a blocchi
identificabili e conservare originale, proposta, motivazione, decisione e revisione.
Iniziare con accettazione manuale; l'automatismo richiede una decisione successiva
supportata da misure di correzioni giuste, mancate e dannose.

Completare confronto affiancato, navigazione dei problemi, accettazione/rifiuto,
riapertura delle revisioni e selezione del risultato da esportare. Separare errori
OCR da refusi già presenti nel sorgente e descrizioni generate da contenuti originali.

**Uscita:** ogni modifica è rintracciabile e reversibile tramite revisioni; matematica,
numeri e negazioni hanno casi di prova specifici; il contesto inviato al provider è
coerente con il profilo scelto; un'indisponibilità AI non distrugge la conversione base.

## Fase 6 — Distribuzione e funzionamento locale

Produrre immagini e configurazioni Docker riproducibili per API/UI/worker, volumi
persistenti e cache dei modelli separata dai dati. Formalizzare profili CPU e NVIDIA;
collaudare Linux, Windows/WSL2 e Mac. Per Apple Silicon provare il servizio di inferenza
nativo quando serve Metal, documentando avvio e collegamento dal container.

Gestire download, controllo delle versioni, spazio richiesto e offline dopo preparazione.
Pubblicare RAM/VRAM e prestazioni misurate; verificare architetture CPU supportate.
Completare limiti upload, readiness, log senza segreti, backup e ripristino. Se l'uso
richiede accesso di rete o più utenti, completare autenticazione e isolamento coerenti
con quel perimetro prima di dichiararlo supportato.

**Uscita:** avvio su macchine/profili dichiarati, test offline reale, riavvio con
persistenza, backup ripristinato e limiti documentati. Dove manca hardware per una
prova, indicare «non verificato», senza trasformarlo in supporto certificato.

## Fase 7 — Esportazioni e derivati

Verificare esportazione HTML/PDF e, secondo priorità, DOCX dal Markdown con asset,
formule e note. Verificare il bundle anche fuori dall'applicazione. Documentare le
differenze di impaginazione e le estensioni richieste ai renderer.

Se richiesto per l'uso concreto, aggiungere chunking semantico/RAG dal documento
revisionato: budget del tokenizer destinatario, metadati di provenienza, formule e
tabelle non spezzate, gestione esplicita dei blocchi oltre limite. L'indice vettoriale
e una chat sui documenti non sono necessari al convertitore base.

Valutare Mermaid o ricostruzione di grafici come derivati chiaramente etichettati,
conservando sempre la figura originale e verificando eventuali dati ricostruiti.

**Uscita:** export verificati e limiti dichiarati. RAG e ricostruzioni restano estensioni
opzionali: la loro mancata attivazione non impedisce il rilascio del convertitore.

## Fase 8 — Migrazione e rilascio

Definire cosa importare degli output legacy e cosa riconvertire perché privo di
provenienza. Non fabbricare versioni del motore o parametri che non sono disponibili.
Documentare compatibilità delle CLI e dismissione dei vecchi script.

Riscrivere il README come ingresso sintetico al nuovo prodotto. Completare `docs/`:
tutorial iniziale, guide di conversione/revisione/backup, reference di configurazione,
API e formati, spiegazioni di fedeltà e architettura, percorso per gli sviluppatori.
Verificare comandi, esempi e collegamenti contro la versione da distribuire.

**Uscita:** criteri concordati soddisfatti, limiti noti pubblicati, prove CI e di
conversione disponibili, migrazione e ripristino provati. Integrare il feature branch
in `dev` dopo revisione; promuovere a `main` la versione verificata secondo la decisione
di rilascio. Nessun merge automatico fa parte dell'impostazione iniziale.

## Definizione del risultato finale

Una persona può installare e configurare il prodotto, convertire i formati dichiarati,
consultare sorgente e problemi, correggere l'OCR con tracciabilità, recuperare una
conversione dall'archivio e scaricare Markdown completo con asset. I percorsi locale
CPU, GPU e API remota hanno capacità e limiti documentati. Test e benchmark dimostrano
fedeltà sul corpus di riferimento, senza promettere perfezione su qualsiasi documento.
