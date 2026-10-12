# Roadmap del convertitore

Aggiornata al 2026-10-12. Le fasi descrivono il percorso fino al prodotto; non sono una
stima temporale né un'autorizzazione ad avviare ogni attività in anticipo. Dipendenze,
hardware e corpus determineranno dimensione e durata degli incrementi.

## Quadro delle fasi

| Fase | Risultato verificabile | Dipendenze | Stato |
| --- | --- | --- | --- |
| 0 | Governance, decisioni e documentazione iniziale | Nessuna | Bootstrap integrato in dev a `66ba822` |
| 0.1 | Migrazione Python a uv e packaging riproducibile | Bootstrap governance integrato in dev; ciclo run a001 | GO finale implementazione r003 dopo GO/GO; deviazioni temporali NON PASS conservate, archivio verificato; integrata manualmente in dev a `b90c6d4` il 2026-10-07, albero identico alla feature `8479f97`; push verificato a `80c31da` |
| 1 | Corpus, benchmark e scelta motivata dei motori | Fase0.1, campioni e profili hardware | A002 chiusa con perimetro ridotto/debito accettato; parser futuro ADR0008. Fase1 ancora aperta: prossimo confronto motori con valutazione umana; nessuna scelta definitiva |
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

## Fase 0.1 — Toolchain Python riproducibile

Decisione [ADR 0006](decisions/0006-python-toolchain-uv.md), run iniziale
`run-a001-fase0-uv`. Bootstrap integrato manualmente in `dev` a `66ba822`;
branch `feature/run-a001-uv` verificato con quella base, come registrato nel
[changelog](CHANGELOG.md). Applicare il [ciclo supervisionato](development/run-lifecycle.md):
piano, due review, arbitrato, implementazione, due review e arbitrato finale.

Introdurre pyproject, lockfile, versione Python e ambiente locale; separare dipendenze
runtime/dev, correggere discovery del package e entry point, allineare istruzioni e
build riproducibili. Escludere `.venv/` e `temp/` dai contesti Docker. Valutare il
perimetro delle dipendenze Marker senza scegliere prematuramente il motore definitivo.
Conservare il comportamento applicativo salvo correzioni necessarie al packaging.

**Uscita:** ambiente ricreabile dal lockfile, package importabile e comandi verificati
fuori dalla directory dei sorgenti, controlli legacy pertinenti eseguiti, limiti
espliciti, GO finale sullo snapshot e integrazione manuale. Questa fase anticipa
la parte di packaging della fase 2; il nuovo dominio applicativo resta in fase 2.

## Fase 1 — Corpus e scelta dei motori

**Decisione corrente2026-10-12:** a002 chiusa per [accettazione utente della
consegna sperimentale](runs/run-a002-fase1-corpus/manifest.md); primo incremento
concluso nel perimetro ridotto, fase1 complessiva non conclusa. [Debito](benchmarks/a002-residual-debt.md)
aperto con ripresa obbligatoria; [ADR0008](decisions/0008-maintained-parser-and-provisional-benchmark.md)
adotta parser mantenuto da implementare. Segue [confronto a003](development/next-run-a003-engine-comparison.md)
con raw/gold e valutazione umana, metriche provvisorie dove non convalidate.
Storia sotto conservata: NO_GO e review incompleta non diventanoGO. Nessuna
integrazione Git o benchmark ML/disponibilità hardware inventati.


Incremento iniziale autorizzato il 2026-10-07: [run a002](development/next-run-a002-corpus.md),
corpus di riferimento e misurazione della fedeltà. Piano r001 e review indipendenti
ricevuti; errata E01–E08 verificate e arbitrato r002 GO del 2026-10-08, mandato
completo con quota52h disposta. Consegna completa ricevuta:13riferimenti
VERIFIED/A03 chiuso nelle receipt, confronto originale/Markdown e risposta V
documentale reale. Due review r001 NO_GO/NO_GO e arbitrato locale conservati; consegna r002 completa
ricevuta con rettifica celle/numeri/WER e nuova baseline. Due nuove review r002
NO_GO/NO_GO e arbitrato locale conservati. Due nuove review r003 NO_GO/NO_GO e arbitrato locale conservati. Consegna
r004 completa e due nuove review NO_GO ricevute, arbitrato locale conservato.
Consegna r005 e due nuove review NO_GO ricevute, arbitrato locale conservato.
Consegna r006 e due review NO_GO ricevute, arbitrato locale conservato.
Consegna r007 e due review NO_GO statiche arbitrate, storia conservata.
Consegna completa r008 e due review NO_GO ricevute, arbitrate tre cause:
guardia HTML interrotta, falso tag incompleto e perdita link dopo stripping.
F35/F36 chiusi nei domini verificati,F37/F38 parziali; F39–F42 per
consolidamento locale in nuova chat autrice e nuova supervisione richieste
dall’utente. Non nuovo piano/parser universale. Consegna r009 completa ora ricevuta: guardia indipendente, semantica lessicale
condivisa e dominio/mapping producer;308/308matrice,1869fast+310subtest,70exitCLI
e64derivati invarianti dichiarati dall'autore. Ricezione statica dei byte e
legami; due review r009 NO_GO reali ora arbitrate, F40/F41 chiusi nei domini revisionati. F43/F44 bloccanti ereditati: tag reali persi nel fallback e falsi unsafe dai code letterali dopo HTML inline. F45 interazioni/documenti e limiteF36 non bloccante; mandato34 nella chat autrice per r010, poi nuovo comune e due NUOVE review alla consegna completa. Quota24h cumulativa invariata,stop11ott
08:37:31 Europe/Rome/deadline11:37:31;V13/A03 e64misure preservati,nessun GO.

Consegna completa r014 F58–F62 ricevuta: 157/157 contratti (159 input distinti), suite3156+310subtest, quattro controprove private,12 exit CLI e64 ricalcoli. Chiusure da giudicare, NO_GO r013 conservato; nuovo comune r014 e due NUOVE chat indipendenti ChatGPT/Claude. Zero delta numerici/gate,16 diagnosi EPUB eotto metadata TeX attribuiti. Nessuna deadline; limiti econtatori invariati, review mirate entro residuo corrente. Nessun GO codice o integrazione.

Aggiornamento precedente2026-10-11: Consegna completa r013 ricevuta per review: F54–F57 dichiarati realizzati, reader observations-v13-ordered-guards-fences. 301 attesi (285+16 metadata multilinea),2997 test+310 subtest,102 exit CLI e64 ricalcoli con19 campi numerici/gate invariati. Censimento per fonte/callsite,AST Pandoc locale e due controprove su copie private ricevuti. Nuovo comune e prompt44/45 per due NUOVE chat indipendenti ChatGPT/Claude,poi arbitrato reale. Chiusure da giudicare;NO_GO r012 conservato,nessun GO codice. Nessuna deadline;limiti non temporali e contatori invariati.

Aggiornamento precedente2026-10-11: Due review r012 reali NO_GO ricevute e arbitrate. F54:attributi Markdown attivi bypassano la sessione della prima unsafe e interrompono le guardie successive;F55:ownership errata del fence tilde dopo testo,con COMPLETE senza guardia/asset. Regressioni r012,rami ereditati della stessa classe inclusi senza parser universale. F56:censimento tuttefonti/callsite,AST Pandoc locale e due controprove prima della consegna;F57:documenti/coverage privata,non bloccante. Prompt43 completo nella chat autrice,stessa run/incremento. Nessuna deadline,altri limiti/contatori invariati;nessun GO codice.

Aggiornamento precedente2026-10-11: Consegna completa r012 ricevuta per review: F50–F53 dichiarati implementati, reader observations-v12-shared-atoms. 350/350 attesi, regressioni153/303/308,2688 test+310subtest,134exitCLI e64ricalcoli con zero delta su19campi numerici/gate. Nuovo comune e prompt41/42 per due NUOVE chat indipendenti ChatGPT/Claude,poi arbitrato reale. Chiusure da giudicare,NO_GO r011 preservato,nessun GO codice. Nessuna deadline della run; limiti CPU/disco/RAM/rete/R e contatori invariati.

Aggiornamento precedente2026-10-11: Disposizione utente: nessuna deadline, durata massima o stop temporale della run. NO_GO r011 tecnico invariato. Prompt40 r012 F50–F53 eseguibile nella chat autrice r009/r010/r011: consolidamento delle decisioni lessicali duplicate, copertura guardia esplicita e generazione finita indipendente prima della consegna. Limiti CPU/disco/RAM/rete/R invariati, contatori e storia senza reset.

Aggiornamento precedente2026-10-11: Due review r011 reali statiche NO_GO ricevute e arbitrate: F50 bound math delle label accettate con escape+LF (alta,regressione r010 non chiusa),F51 raw fence nella quote esentata dal fallback (media,regressione r011),F52 raw inline nella label saltato e prima diagnosi persa (media,ereditato r010). F53 invarianti/prove/documenti; F46 chiuso nei domini esaminati,F47/F48/F49 parziali,F36 preservato. Prompt40 r012 completo nella chat autrice r009/r010/r011, esecuzione in attesa di disposizione temporale esplicita. Stop2026-10-11T06:37:31Z passato,deadline09:37:31Z=11:37:31Roma. Quota24h da10ott09:37:31Z ed entry08ott07:06:54.369068Z senza reset; nessuna risposta temporale assunta. CPU21600/stop19800s,disco6/stop5GiB,RAM8/stop sampled6GiB,riserva1800CPU-s/1GiB,R≤900s/stream1MiB/file32MiB,rete/acquisizioni/invii0/€0. Anche amministrazione inclusa e ferma alla deadline reale.

Aggiornamento precedente2026-10-11: Consegna r011 completa ricevuta: reader v11/F46–F49 dichiarati verificati,153/153matrice,303/303r010 e308/308r009,2327fast+310subtest,74exitCLI e64misure senza delta numerico/gate. Byte/prove ricevuti; nuovo comune e prompt38/39 per due NUOVE review ChatGPT/Claude,poi arbitrato reale. Chiusure da giudicare,NO_GO r010 preservato,nessun GO codice. Stop nuovi workload11ott08:37:31Europe/Rome superato all’ingresso08:55:55; riserva amministrativa fino deadline11:37:31. Review statiche soltanto,nuove sonde R ferme senza disposizione temporale esplicita. Richiesta estensione presentata,nessuna risposta assunta. CPU/disco/RAM/rete0/€0 e contatori senza reset.

Aggiornamento precedente2026-10-10: Due review r010 reali NO_GO arbitrate: F46 metadata inline ereditato e variante H2 regressiva; F47 falso unsafe nella math della label e F48 apertura reale saltata dentro owner HTML, regressioni r010 bloccanti. F49 invarianti finite/documenti, F36 e quattro osservazioni non bloccanti conservati. Mandato37 r011 nella chat autrice r009/r010, poi nuovo comune e due NUOVE review alla consegna completa; quota24h invariata, nessun GO codice. F43/F44 chiusure originali preservate ma invarianti complessive parziali; F45parziale,F40/F41nel perimetro,V13/A03/raw/limiti invariati.

Aggiornamento precedente2026-10-10: Consegna r010 completa ricevuta: reader v10/F43–F45 dichiarati verificati,303/303matrice e308/308regressioni r009,2174veloci+310subtest,82exitCLI e64misure invarianti. Byte e legami verificati dal supervisore, nuovo comune r010 e prompt35/36 per due NUOVE review indipendenti, poi arbitrato. NO_GO r009 e limiteF36 preservati, nessun GO codice;quota24h/entry/contatori invariati. F40/F41 chiusi nel perimetro r009, F43–F45 da giudicare;stage-s009 e prove finali legate ai byte,nessuna nuovaV/motore.

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

**Obbligo di follow-up:** all’uscita di questa fase aprire la run dedicata al
parser mantenuto e risolvere il [debito a002](benchmarks/a002-residual-debt.md).
La prima app resta valutabile manualmente; il reader sperimentale non è una
barriera di sicurezza o una certificazione automatica. Debito da chiudere
prima di automatismi di attestazione e rilascio, con nuove prove e review.


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
