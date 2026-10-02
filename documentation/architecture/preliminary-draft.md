# Draft preliminare — Evoluzione del convertitore

Data di archiviazione: 2026-10-02. Stato: **proposta iniziale**.

Questo documento trasferisce in forma documentale la proposta discussa prima della
richiesta di impostare il repository, riorganizzandone il contenuto per argomento.
Le precisazioni successive, inclusa la scelta esplicita di FastAPI, sono registrate
negli [ADR](../decisions/README.md). Non descrive funzionalità già implementate.

## Direzione proposta

Proporrei una nuova versione dell'applicazione, mantenendo Python e riutilizzando
motori di conversione esistenti. Riscriverei il nucleo che gestisce documenti,
configurazione, controlli e risultati. I requisiti richiedono una gestione della
fedeltà e della provenienza del contenuto che l'attuale progetto non possiede.

Il documento completo deve essere il risultato principale: conversioni verificabili,
correzioni tracciate e motori sostituibili. Conservare ciò che è conveniente dopo
averne verificato comportamento, aggiornabilità e licenza.

## Fedeltà: cosa significa e cosa può essere garantito

Preservare testo, formule, dati, immagini, didascalie, note, bibliografia e riferimenti.
La struttura assimilabile è quella logica: titoli, sezioni, paragrafi, ordine di
lettura e relazioni tra elementi. Font, paginazione e colonne appartengono invece
all'impaginazione originale, che rimane consultabile nel sorgente conservato.

Nessun sistema può garantire automaticamente contenuti sempre identici partendo da
scansioni ambigue o deteriorate. Il prodotto deve conservare le evidenze, individuare
le incertezze e consentire revisione, senza presentare un risultato dubbio come
conversione sicuramente fedele.

## Il ruolo del chunking

Il chunking non serve a produrre un buon Markdown e non deve essere obbligatorio.
La disponibilità di contesti LLM più grandi permette in alcuni casi l'uso del documento
intero, ma non elimina i problemi di costo, latenza e ricerca su collezioni ampie.
Il recupero di porzioni pertinenti continua a essere utile in un sistema RAG;
si veda il lavoro su [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval).

Proposta: esportazione RAG separata, disabilitata per default, derivata dal documento
revisionato. I chunk devono rispettare blocchi semantici, formule, tabelle e provenienza,
con limiti misurati usando il tokenizer del modello destinatario.

## Rappresentazione intermedia

Il solo Markdown non basta a conservare tutte le informazioni necessarie alla
conversione. Occorre un documento strutturato con identificatori stabili, tipi di
blocco, gerarchie, pagine, coordinate ove disponibili e riferimenti agli asset.
Valutare [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/)
prima di introdurre un modello proprietario equivalente.

```text
Sorgente → analisi del formato e delle pagine → parser / OCR / matematica
         → documento strutturato + asset → controlli e revisione
         → Markdown completo → esportazioni PDF / HTML / DOCX
                             → esportazione RAG opzionale
```

Una prima proposta è un profilo Markdown compatibile con Pandoc, con matematica
`$...$` e `$$...$$`, note, tabelle e riferimenti relativi agli asset. Precisare e
testare il profilo: il comportamento della matematica dipende dal formato di uscita,
come documentato nel [manuale Pandoc](https://pandoc.org/MANUAL.html#math).

Un pacchetto esportato potrebbe contenere:

```text
documento/
  documento.md
  assets/
    figura-001.png
    figura-002.svg
  conversione.json
```

Il manifest descrive provenienza e conversione senza segreti. Sorgenti, estrazioni
grezze, revisioni e verifiche restano nell'archivio interno; il bundle finale deve
essere utilizzabile indipendentemente dal DB applicativo.

## Documenti scientifici

| Elemento | Trattamento proposto |
| --- | --- |
| Pagine a più colonne | Ricostruire l'ordine logico dei blocchi, separando intestazioni e note |
| Formule | Conservare LaTeX nativo; per PDF/scansioni usare riconoscimento dedicato, mantenendo numerazione e riferimenti |
| Immagini | Estrarre l'originale quando possibile, altrimenti un ritaglio fedele; collegare didascalie |
| Grafici | Conservare un'immagine leggibile con assi, legenda e annotazioni |
| Diagrammi | Eventuale derivato Mermaid, mantenendo la rappresentazione originale |
| Tabelle complesse | Conservare struttura e dati; prevedere una rappresentazione più ricca quando Markdown semplice non basta |

Estrarre un grafico è diverso dal ricostruirne i dati. Nei PDF una figura può essere
composta da oggetti vettoriali: renderizzare la regione può preservarla meglio di
un'estrazione incompleta. Ricostruire curve, valori o diagrammi con AI richiede una
verifica separata e non deve sostituire silenziosamente l'evidenza originale.

Un riferimento Markdown a un'immagine non fornisce i pixel a un LLM testuale. Gli
utilizzi multimodali devono inviare anche gli asset; eventuali descrizioni generate
devono essere distinte dalle didascalie presenti nel documento.

## Secondo passaggio OCR con AI

L'idea è utile se il secondo passaggio produce correzioni circoscritte e verificabili.
Fornire testo riconosciuto, contesto e ritaglio della sorgente; ottenere proposte
strutturate con blocco, testo originale, sostituzione e motivazione. Per esempio,
una lettura `rnatrice` potrebbe essere proposta come `matrice`.

Partire da suggerimenti sottoposti a revisione. Prestare particolare attenzione a
formule, numeri, nomi propri, negazioni e unità di misura. Un refuso dell'autore non
è automaticamente un errore OCR: conservarlo o annotarlo separatamente.

Il giudizio del modello su sé stesso non è una misura sufficiente. Affiancare controlli
su copertura delle pagine, asset, riferimenti e differenze tra revisioni. Una formula
che viene renderizzata correttamente può comunque essere matematicamente sbagliata.

## Motori da confrontare

I difetti del wrapper attuale non dimostrano che Marker sia il motore sbagliato.
Confrontare versioni aggiornate su documenti rappresentativi prima della scelta.

| Candidato | Ruolo da valutare |
| --- | --- |
| Pandoc | Parser dei formati strutturati come DOCX, EPUB, RTF e LaTeX; esportazioni |
| Docling | PDF, OCR, struttura e possibile rappresentazione intermedia |
| Marker aggiornato | Conversione di PDF scientifici, matematica e immagini |
| Datalab o Mistral OCR | Primo provider remoto per casi difficili o hardware limitato |

Fonti di partenza: [Pandoc](https://pandoc.org/MANUAL.html),
[Docling](https://github.com/docling-project/docling),
[riconoscimento formule Docling](https://docling-project.github.io/docling/usage/enrichments/#formula-understanding),
[Marker](https://github.com/datalab-to/marker),
[release Marker](https://github.com/datalab-to/marker/releases),
[Mistral OCR](https://docs.mistral.ai/studio/document-processing/basic_ocr).
Supporto, licenze del codice e dei pesi, requisiti e costi vanno verificati per le
versioni effettivamente scelte; questa tabella non è un benchmark né una promessa
di supporto universale da parte di ciascun motore.

Usare i parser strutturali quando esiste una sorgente nativa. Un progetto LaTeX può
richiedere file inclusi, macro, bibliografia e immagini: prevedere bundle di sorgenti.
Nei PDF misti decidere OCR e riconoscimento per pagina o regione, evitando passaggi
inutili sulle parti già leggibili.

## Locale, hardware e Docker

Prevedere un profilo CPU, un profilo Linux/NVIDIA e uno Windows/NVIDIA tramite WSL2.
Per Mac Apple Silicon valutare applicazione in Docker e inferenza nativa sull'host
per sfruttare Metal. La portabilità dei container non rende identico l'accesso GPU.

Riferimenti da verificare durante la distribuzione:
[GPU in Docker Desktop](https://docs.docker.com/desktop/features/gpu/),
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
[Docker Model Runner](https://docs.docker.com/ai/model-runner/).
Un runtime come Model Runner deve comunque supportare i modelli richiesti dal motore.

Il profilo hardware può coinvolgere più modelli: layout, OCR, matematica e revisione.
Gestire download, versioni, cache persistente e preparazione per uso offline. Misurare
RAM, VRAM e tempi; non promettere che ogni modello sia praticabile su ogni macchina CPU.

## Persistenza e applicazione

SQLite è una scelta iniziale ragionevole per una distribuzione locale su singolo host.
Conservare nel DB documenti, esecuzioni, configurazioni, problemi e revisioni; tenere
i file nello storage. Identificare documenti e risultati tramite hash e ID, evitando
collisioni tra file omonimi e riuso di output diventati obsoleti.

Le esecuzioni sono distinte e riproducibili, con versioni e configurazioni registrate.
Un risultato si pubblica dopo aver completato scritture e controlli. Per deployment
su più host o forte concorrenza di scrittura si rivaluterà il DB, coerentemente con
le [indicazioni SQLite](https://sqlite.org/whentouse.html).

Proporrei un backend Python con API, un worker di conversione separato, una Web UI
e una CLI che condividano lo stesso nucleo applicativo. La UI deve includere upload,
stato dei job, archivio e versioni, download, confronto sorgente/risultato, problemi
per blocco, approvazione delle correzioni e anteprima di matematica, tabelle e immagini.

[Docling Serve](https://docling-project.github.io/docling/usage/api_server/deployment/)
può accelerare prove sui motori; archivio persistente e revisione restano responsabilità
dell'applicazione. Una demo di conversione non copre da sola questi requisiti.

## Configurazione

Un `.env` commentato e comprensibile deve coprire storage, modalità locale/remota,
hardware, lingue, provider, credenziali, revisione e worker. Validare i valori all'avvio.
Le opzioni per una singola conversione possono essere scelte nella UI e persistite
nel DB senza imporre un secondo file manuale. Separare conversione e revisione AI.

## Sequenza inizialmente proposta

1. Corpus rappresentativo e confronto misurato dei motori.
2. Nucleo affidabile: documento strutturato, asset, Markdown completo, SQLite e job.
3. Web UI, revisione OCR/AI e verifiche della qualità.
4. Esportazioni PDF/DOCX e derivati opzionali RAG o diagrammi.

La [roadmap](../roadmap.md) dettaglia ora questa sequenza e anticipa una UI minima.
Restano da conoscere hardware e RAM/GPU, quantità e dimensioni dei documenti,
frequenza dei batch e uso personale o condiviso: vedere le
[questioni aperte](../open-questions.md).
