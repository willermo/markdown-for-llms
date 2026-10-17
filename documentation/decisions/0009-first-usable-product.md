# ADR 0009 — Primo prodotto locale: Marker CPU e interfaccia inerte

- Data: 2026-10-17
- Stato decisione: accettata per il primo incremento sperimentale
- Stato implementazione: I1 r004 GO dopo review indipendenti GO/GO e arbitrato il 2026-10-17; integrazione manuale pendente, residui bassi tracciati
- Origine: priorità utente del 16 ottobre, piano a004 r001 e arbitrato delle
  review indipendenti ChatGPT/Claude, con errata locali verificate
- Sostituisce / sostituito da: nessuno; precisa ADR0002/0003/0004/0008

## Contesto

Serve un percorso completo PDF→Markdown/asset, CLI e browser utilizzabile
prima del consolidamento del validatore. Gli output a003 osservati mostrano
utilità per prosa, tabella e figura C06, insieme a perdite matematiche;
non dimostrano fedeltà generale, OCR scientifico affidabile o tutte le pagine
dei PDF misti. Il [debito riconosciuto](../development/technical-debt.md)
rimane aperto e non diventa una dipendenza del primo prodotto.

## Decisione

Per I1 adottare Marker 1.10.2, Surya 0.17.1 e torch 2.7.1+cpu dietro un
adattatore diretto, in figlio confinato del worker separato. Primo perimetro
da provare: Linux x86_64, CPU, singolo utente, loopback, nessun invio remoto.
Pin e modelli conservati richiedono verifica mirata prima dell'uso. Non
redistribuire automaticamente modelli né promettere altre piattaforme.

FastAPI serve API e UI HTML/CSS/JavaScript senza framework SPA o CDN. Servizi
applicativi condivisi con CLI; SQLite con claim/lease e recupero esplicito,
filesystem locale con pubblicazione e riconciliazione. Il browser mostra
Markdown tramite testo, galleria raster verificata e download; rendering
scientifico e confronto visuale avanzato arriveranno in incrementi successivi.

Usare `markdown-it-py` per il controllo operativo dei riferimenti del profilo,
con versione, configurazione e plugin necessari congelati nell'implementazione.
La scelta facilita l'integrazione Python senza aggiungere il fork Pandoc.
Non fornisce il modello osservato, la source-map o le garanzie del futuro
validatore dell'ADR0008; non viene dichiarata equivalenza di dialetto.
HTML noto letto con tokenizer esistente, mai eseguito o riserializzato.

Distinguere esecuzione, stato del bundle e qualità `unverified`. `ready`
controlla inventario/hash e riferimenti nel profilo. `unchecked` consente
download ordinario con etichetta e limiti quando tutti i file dichiarati
sono integri e non vi sono guasti noti; asset mancante noto, manomissione o
path non sicuro continuano a bloccarlo. Ancore interne non sono file mancanti;
HTML senza risorse non rende da solo il risultato `unchecked`.
Questi stati non certificano la completezza degli asset rispetto al PDF.

Validare le immagini per la galleria in un processo distinto senza importare
Marker, con limiti espliciti prima del decode: il pin disabilita il tetto
globale Pillow nel proprio processo. Configurazione, schema e identità dello
storage sono controllati da API, worker e CLI. I1 non include cancellazione
dall'archivio: il limite va documentato e la gestione pianificata in seguito.

## Alternative considerate

Docling resta recupero futuro esplicito: i campioni non mostrano un vantaggio
universale per formule e segni. Il server legacy non soddisfa il nuovo
contratto di worker/immagini. Pandoc non converte PDF; il trace sperimentale
non è approvato come dipendenza. React/TypeScript resta valutabile quando
confronto e revisione ne giustifichino il costo; per I1 la UI testuale basta.
Il parser avanzato resta un progetto distinto, con gold indipendenti.

## Conseguenze e verifica

Implementare il percorso completo e verificarlo da wheel fuori dal checkout,
con PDF reale C06, CLI, browser effettivo, ZIP esterno, C07, crash/riavvio,
errori e sonde operative P01–P14. Aggiungere le forme normali del pin:
span/ancore, sup/sub, br, tabelle HTML e immagini HTML locali/esterne.
Le review del piano sono statiche: nessuna prova runtime è già trasferita.

Nessuna promessa di matematica preservata, supporto hardware non provato,
validazione automatica o prodotto già disponibile. Riesaminare il motore
se il nuovo percorso non conserva l'utilità osservata; nessun cleaning
distruttivo o fallback cloud per occultare perdite. Due review indipendenti
del codice precedono il GO della consegna; Git rimane manuale dell'utente.

## Collegamenti

[Roadmap](../roadmap.md), [risultati osservati a003](../benchmarks/a003-engine-comparison-r002.md),
[ADR0008 vigente](0008-maintained-parser-and-provisional-benchmark.md).
Piano, errata e arbitrato a004 sono contesto operativo in `temp/`, non bundle
permanenti. Fonti primarie del pin, lette come testo e identificate nelle
prove temporanee: `marker/renderers/markdown.py`, `html.py`,
`schema/blocks/reference.py`, `schema/text/span.py` e `line.py` della 1.10.2.
Le capacità sono riscontrate nel codice e nei raw conservati; l'esecuzione
del nuovo adattatore resta da provare.

## Avanzamento 2026-10-17 — Consegna da revisionare

I1 ricevuto con nuovo package, CLI/API/UI e worker; prove autoriali di C06/C07,
browser, ZIP e recupero crash disponibili in temp. La ricezione verifica
identità della consegna, wheel/installato e ZIP, senza rieseguire inferenze.
Due review indipendenti del codice ancora da completare; i criteri sopra
sono ora oggetto di verifica, non attestazioni già approvate. Qualità
unverified e P14 utente PENDING; debito del validatore invariato.

Arbitrato successivo dello stesso giorno: NO_GO r001 per gestione quota,
temporanei e invalidazione su errori transitori. Fix locali r002 autorizzati;
scelte architetturali e debito invariati. La verifica della configurazione
non impone di rendere inapribile l'archivio a ogni cambio di quota/timeout:
prevedere modifica amministrativa compatibile o migrazione mirata verificata,
senza bypass di identità/radici/schema o alterazione dei manifest storici.

Consegna r002 ricevuta nello stesso giorno: capacità condivisa, cache ZIP
singola e configurazione amministrativa realizzate con prove attribuite;
chiusure dei dieci ID ancora da giudicare nelle due review indipendenti.
C06/C07 r002 sono replay dichiarati; le inferenze restano attribuite a r001.
Architettura accettata invariata, nessun GO al codice o chiusura del debito.

Arbitrato r002 successivo: NO_GO su classificazione positiva della corruzione
e ripristino della durabilità nel recupero. R003 corregge questi confini e
residui locali, mantenendo le chiusure acquisite. Scelte architetturali
invariate; C06 reale finale richiesta per il monitor corrente, non per
certificare il motore. Nessun GO generale o riapertura del debito avanzato.

R003 poi ricevuta nello stesso giorno: barriera durevole comune e confini
espliciti realizzati con prove nuove, C06 reale intera sul monitor finale e
browser. Chiusure dei sette ID dichiarate dall'autore, review indipendenti e
arbitrato ancora pendenti. Scelte architetturali e debito avanzato invariati;
nessun GO di prodotto o qualità già ottenuto.

Arbitrato r003: un GO e un NO_GO ricevuti, NO_GO locale accolto sul risultato
completo già publishing ma non ancora rinominato. R004 completa recupero e
simmetria manifest/reader, con residui di disponibilità dell'archivio.
Architettura e chiusure non invalidate conservate; C06 reale r003 non deve
essere ripetuta se le dipendenze ML/monitor restano uguali. Nessun GO al codice.

R004 poi ricevuta nello stesso giorno con prove di recupero prima/dopo rename,
lease, serializzazione del manifest e lettura passiva. Canonica004 e browser
con replay C06 dichiarato; la C06 ML r003 resta prova storica attribuita.
Sei ID dichiarati chiusi dall'autore, review indipendenti e arbitrato pendenti;
architettura e debito invariati, nessun GO al codice o integrazione.

Arbitrato r004 successivo: **GO tecnico I1 r004004**, sei ID r003 chiusi.
Tre rilievi bassi accolti e rinviati nel [registro del debito](../development/technical-debt.md),
senza dichiararli risolti o ampliare il supporto. Scelte invariate;
[sintesi permanente](../runs/run-a004-fase2-prodotto-utilizzabile/manifest.md)
e comandi Git manuali predisposti. Implementazione presente e provata nel
worktree, non ancora integrata. Qualità unverified/P14 PENDING, nessun GO
retroattivo al validatore o certificazione degli asset rispetto al PDF.
