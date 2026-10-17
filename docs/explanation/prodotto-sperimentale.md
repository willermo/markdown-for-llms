# Che cosa significa «bundle disponibile»

Il prodotto sperimentale separa tre domande. `state` descrive l’esecuzione
tecnica. `bundle` descrive file e riferimenti controllabili. `quality` rimane
sempre `unverified`: il motore può omettere o alterare contenuti senza segnalarlo.

`ready` significa che inventario e digest sono coerenti e i riferimenti del
profilo non hanno guasti noti. `unchecked` segnala forme fuori profilo; permette
lo ZIP solo quando tutti i file dichiarati sono integri. Un riferimento locale
riconosciuto e mancante produce `incomplete`; un file alterato o un path non
sicuro produce `invalid`. Questi guasti prevalgono sul fuori profilo.

Il controllo usa markdown-it-py4.0.0, CommonMark con tabelle e dollarmath di
mdit-py-plugins0.5.0. Code e math non sono risorse. I link vuoti si riferiscono al documento corrente; `./img.png` verifica il
file locale senza riscrivere il raw, anche con `./` iniziale ripetuto.
Destinazioni locali non-file come `.`, `./` e `docs/` sono `unchecked`, con
motivo esplicito; non sono considerate path ostili. Le ancore `#frammento` non
richiedono file; l’esistenza dell’ancora non è attestata. `html.parser` distingue
HTML inerte da `a[href]` e `img[src]`. Srcset, CSS/style, poster, data e altri
contesti non supportati restano `unchecked`, con numero totale non determinabile.
Le destinazioni esterne vengono classificate e mai scaricate. Questo lettore
operativo non sostituisce il validatore e il debito a002/a003 differito.

Nel C06 sintetico di due pagine la prima prova reale conserva prosa, numeri
`0012`, `−7`, `123,45`, `51,08`, `92,06`, tabella/cella vuota e figura con assi
e legenda. La formula `E = mc^2` rimane leggibile; frazione, somma e matrice
perdono struttura, mentre l’aligned viene separato. La nota rimane testo.
Queste osservazioni non certificano altri PDF né l’OCR scientifico.

La prova preliminare C06 ha richiesto circa89s, di cui10,3s osservati per avvio/
caricamento e77,2s per conversione/serializzazione, con circa7,85GiB di RSS
aggregata campionata. Non sono upper bound o velocità per pagina. Il limite
450s può interrompere PDF ammessi più lunghi; le due pagine sintetiche non
rappresentano l’intero carico reale.

Il browser mostra testo inerte tramite `textContent`, conserva i link come
caratteri e usa soltanto immagini locali verificate. Il validatore raster gira
in un figlio separato perché Marker disabilita il limite globale dei pixel
Pillow nel proprio processo. Un’immagine rifiutata dalla galleria resta
conservata nel bundle se l’integrità tecnica lo consente.

SQLite registra la coda; il worker detiene un lock di proprietà e usa token,
lease e identità del processo. La pubblicazione attraversa uno stato esplicito
`publishing`: il rename filesystem non è una transazione SQLite. Il riavvio
riconcilia i risultati completi ancora nei tentativi o già rinominati,
ripetendo la barriera di sincronizzazione
e distingue i parziali interrotti. Un rifiuto di capacità o un errore I/O
lascia la pubblicazione recuperabile sullo stesso ID. Non esiste
retry automatico. Nuovi tentativi non sovrascrivono originali o risultati.

I1 non include cancellazione dell’archivio, altri formati, rendering Markdown/
formule, confronto PDF interattivo, correzioni manuali/AI, cloud, multiutente,
GPU o distribuzione pubblica. Le funzioni successive rimangono nella roadmap,
separate da questo percorso. [ADR0009](../../documentation/decisions/0009-first-usable-product.md),
[debito riconosciuto](../../documentation/development/technical-debt.md).
