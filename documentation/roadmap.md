# Roadmap del convertitore

Nota17ott2026: riferimenti operativi eliminati o già indisponibili sono
resi testo; esiti storici invariati. [Disponibilità delle prove](development/cleanup-2026-10-17.md).

Aggiornata al 2026-10-17. Le fasi descrivono il percorso fino al prodotto; non sono una
stima temporale né un'autorizzazione ad avviare ogni attività in anticipo. Dipendenze,
hardware e corpus determineranno dimensione e durata degli incrementi.

## Priorità vigente — prodotto utilizzabile, 2026-10-16

L'utente autorizza il percorso verso le funzionalità della roadmap, iniziando
con conversione reale, Markdown/asset consultabili, CLI e Web UI. Si accetta
un prodotto sperimentale con qualità non certificata e limiti dichiarati.
Parser/trace e validazione avanzata non sono più prerequisiti delle fasi2–8
per tale uso; debito e vecchi NO_GO restano aperti. Nessun GO tecnico a003
retroattivo o adozione del fork. [Disposizione vigente ADR0008](decisions/0008-maintained-parser-and-provisional-benchmark.md).

Primo incremento: verificare l'utilità degli output PDF già disponibili,
selezionare una configurazione motivata e realizzare un percorso completo
con le sole fondazioni necessarie. Poi estendere formati, revisione,
distribuzione ed esportazione per incrementi verificabili. La pianificazione
produrrà sequenza e criteri, senza promettere tutte le piattaforme o i profili
hardware prima di prove. Fase1 resta parziale; serve evidenza d'uso del motore,
non una certificazione del comparatore. Risorse gratuite autorizzate,
€0/remoti0 invariati; nessun deploy pubblico o integrazione Git automatica.

Gestione degli errori, presenza/identità degli artefatti, conservazione del
raw e confinamento rimangono obblighi. Distinguere job terminato, integrità
del bundle e fedeltà non verificata; non nascondere asset mancanti noti o
introdurre cleaning distruttivo. Il prodotto iniziale non dipende dal trace
sperimentale; le sue prove restano in temp senza ulteriori campagne automatiche.

**Debito tecnico riconosciuto:** reader/comparatore a002/a003, parser/trace e
validazione avanzata restano aperti e da consolidare appena tempo/risorse lo
consentiranno. Riesame al primo prodotto utilizzabile e quando una funzione ne
richiede le proprietà; nessuna promessa di chiusura già realizzata.
[Registro e responsabilità](development/technical-debt.md).

Aggiornamento17ott2026: due review del piano ricevute, GO ChatGPT e NO_GO
locale Claude. Blocco del contratto risolto con errata vincolanti su HTML,
ancore e download unchecked; chiarimenti su immagini, configurazione e CLI.
[ADR0009](decisions/0009-first-usable-product.md): scelte I1 accettate,
All’approvazione del piano P01–P14 erano ancora da eseguire.
Cancellazione selettiva dall’archivio da pianificare dopo I1; nessuna
cancellazione automatica per quota.

Consegna I1 ricevuta17ott2026: codice applicativo e guide presenti, prove
realistiche PDF/CLI/browser/ZIP/crash attribuite alla build008. Ricevute
3194test e378subtest PASS ricontate; nessuna suite rieseguita in supervisione.
Due review indipendenti sul delta completo, poi arbitrato: nessun GO al
codice, rilascio o integrazione già eseguiti. Validazione avanzata differita
invariata; P14 utente PENDING non sostituito da giudizi dell’agente.

Arbitrato successivo17ott2026: entrambe le review r001 NO_GO; il percorso
utile è corroborato, ma servono correzioni operative di quota/temporanei e
letture transitorie. R002 autorizzata con fix locali e regressioni, senza
nuovo piano o riapertura del certificatore. Nessuna cancellazione generale
dell'archivio aggiunta; capacità e limiti amministrativi devono essere gestibili
senza perdere i risultati conservati.

Consegna r002 ricevuta17ott2026: fix dichiarati sui dieci ID e prove nuove di
quota/crash/recupero,3215test/378subtest nelle ricevute canoniche004. C06/C07
sono replay verificati per le nuove prove applicative, non nuove inferenze.
Seguono due review indipendenti mirate e arbitrato; nessun GO al codice o
certificazione di qualità già ottenuti.

Arbitrato r00217ott2026: NO_GO locale per falsi stati su errori recuperabili
e durabilità non ripristinata dopo fsync fallito. R003 autorizzata con prove
per causa e chiusure non invalidate conservate. Una C06 reale sulla build
finale verificherà la compatibilità del monitor corrente con Marker; nessuna
nuova campagna del parser o modifica delle priorità del prodotto.

Consegna r003 ricevuta17ott2026: fix dei sette ID dichiarati, nuova C06
integrale col monitor finale e browser sul risultato reale; canonica003,
suite004 con3231test/409subtest nelle ricevute. Le chiusure passano a due
nuove review indipendenti mirate; nessun GO al codice, integrazione o
certificazione automatica. Debito e limiti del prodotto invariati.

Arbitrato r00317ott2026: GO Claude e NO_GO ChatGPT valutati sul contratto,
NO_GO locale per pubblicazione già persistita prima del rename fallito.
R004 autorizzata con sei ID consolidati in quattro cause: recupero/lease,
manifest rileggibili, documento guasto e diagnosi passiva. Chiusure e C06 reale
non invalidate conservate; niente nuovo piano o campagne ML non necessarie.

Consegna r004 ricevuta17ott2026: chiusure dichiarate dei sei ID, nuova
canonica004/suite005 con3240test/440subtest nelle receipt, browser e recupero
stesso ID dopo errore prima rename. Passa a due review indipendenti mirate,
nessun GO al codice o integrazione. Replay C06 esplicito e prova ML r003
conservata; nessun nuovo benchmark o cambiamento del debito avanzato.

Arbitrato r00417ott2026: **GO tecnico I1 dopo due review indipendenti GO/GO**.
Sei ID r003 chiusi, tre residui bassi riconosciuti nel registro del debito;
nessun nuovo ciclo di fix automatico. Prodotto sperimentale pronto per
integrazione manuale e primo uso, qualità unverified e P14 PENDING.
[Sintesi e limiti](runs/run-a004-fase2-prodotto-utilizzabile/manifest.md),
[istruzioni Git](runs/run-a004-fase2-prodotto-utilizzabile/manual-integration.md).
Le fasi2/3 hanno il primo percorso funzionante; non sono completate in tutte
le funzionalità previste. Nessun merge/push/deploy ancora registrato.

Pulizia pre-commit17ott2026 completata su richiesta utente: circa16,17GiB
di spazio libero recuperato, runtime/pesi e prove finali conservati.
[Disponibilità e limiti](development/cleanup-2026-10-17.md); niente nuova
certificazione, integrazione o cambiamento dei cinque ID del debito.

## Quadro delle fasi

| Fase | Risultato verificabile | Dipendenze | Stato |
| --- | --- | --- | --- |
| 0 | Governance, decisioni e documentazione iniziale | Nessuna | Bootstrap integrato in dev a `66ba822` |
| 0.1 | Migrazione Python a uv e packaging riproducibile | Bootstrap governance integrato in dev; ciclo run a001 | GO finale implementazione r003 dopo GO/GO; deviazioni temporali NON PASS conservate, archivio verificato; integrata manualmente in dev a `b90c6d4` il 2026-10-07, albero identico alla feature `8479f97`; push verificato a `80c31da` |
| 1 | Corpus, benchmark e scelta motivata dei motori | Fase0.1, campioni e profili hardware | Risultati parziali a003 conservati, debito a002 aperto; trace r005 non revisionato e validazione avanzata differita. Non blocca il prodotto sperimentale autorizzato |
| 2 | Package applicativo, contratti, configurazione, DB e worker | Evidenza operativa dei motori e dati locali | Fondazioni I1 r004 GO dopo GO/GO e arbitrato; integrazione manuale pendente, residui bassi tracciati |
| 3 | Prima applicazione completa con FastAPI e Web UI minima | Fondazioni necessarie e frontend | Primo percorso CLI/API/Web UI I1 r004 GO, browser e replay esplicito provati; C06 reale r003, qualità unverified e P14 PENDING. Funzioni avanzate successive |
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

Aggiornamento corrente 2026-10-15: [a003](runs/run-a003-fase1-engine-comparison/manifest.md)
r007 NO_GO arbitrato; riesame mirato del metodo e nuova pianificazione da svolgere.
Risultati del campione conservati, firma umana pendente, fase1 aperta.

Storia di preparazione 2026-10-12: commit amministrativo registri/24log a002
osservato a `9f702a2`, padre7caff33; base locale dev verificata,nessuna query
remota/prova di push. [Preparazione a003](development/a003-engine-comparison-brief.md)
con sette casi proposti e [ricognizione statica](development/a003-local-reconnaissance.md).
Allora branch manuale,piano e budget workload pendenti; nessuna conversione o scelta
di motore ancora eseguita. Fase1 aperta,nessuna nuova autorizzazione fase2.
Il riepilogo di ricezione a002 sotto precede il commit amministrativo.

Integrazione a002 verificata12ott2026 a `7caff336119b6d7d78978204142cfdc16d620fff`, contenuti
identici; registri/log da committare separatamente. Ricevuta (riferimento storico non disponibile).
Push/main/deploy non eseguiti; la chiusura con debito non completa la fase1.

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

**Follow-up aggiornato il2026-10-16:** il debito del validatore resta aperto,
ma l'uscita fase3 non impone la ripresa immediata del parser. L'utente può usare
il prodotto sperimentale e distribuire nel tempo il lavoro di validazione.
Nessuna attestazione automatica di fedeltà/completezza semantica senza prove;
controlli operativi, sicurezza dei file e limiti esposti restano necessari.

Esporre API FastAPI per upload, avvio, stato, archivio e download, con schema OpenAPI.
Il worker esegue una conversione reale anche su un PDF rappresentativo,
con TXT/DOCX come ulteriori controlli quando pertinenti. La CLI richiama gli stessi servizi. Scegliere il frontend con un ADR e fornire
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

Per disposizione16ott2026 è ammessa una consegna/rilascio sperimentale locale
con validazione avanzata differita e limiti espliciti, senza promessa di
certificazione o deploy pubblico. Le review delle funzioni operative restano.

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
