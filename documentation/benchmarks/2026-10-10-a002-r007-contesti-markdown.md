# A002 r007 — contesti Markdown e confini condivisi

Consegna datata 2026-10-10 (Europe/Rome), creazione UTC 2026-10-09T23:39:19.198652+00:00, run
`run-a002-fase1-corpus`. Rettifica il [report r006](2026-10-09-a002-r006-confini-lessicali-html.md)
e le sue affermazioni globali sui trasporti; non riscrive risultati o fallimenti
storici. [Dati r007](2026-10-10-a002-r007-contesti-markdown-data.json): 64 misure derivate nuove dei medesimi
raw r001/r002, zero conversioni native di campagna e nuova V0. Le fixture dei test
legacy possono invocare Pandoc; non sono nuove conversioni del corpus.

| Fix | Prima, su codice r006 ricevuto | Dopo, su reader v7 |
| --- | --- | --- |
| F31 | Lookahead code/math/tag oltre il proprio blocco nascondeva commenti/CDATA; trasporto riprendeva nel corpo | Lookahead limitato al producer; intervalli realmente riconosciuti trasmessi ai consumer, corpo inerte anche dopo merge/prefix/cella/label |
| F32 | Autolink installava un falso stack HTML; codice seguente perdeva contesto | Stesso atomo autolink in scanner/inline/top-level; codice letterale resta opaco, blocchi/offset/ref conservati |
| F33 | Metadata e comandi già consumati inline aprivano falsi token globali | Atomi finiti condivisi per metadata/command/code/link/span/heading; raw TeX non supportato resta unknown locale |
| F34 | Report r006 generalizzava la correttezza locale a tutti i confini | Nuove prove prima/dopo, domini espliciti, costi e legami ispezionabili; E9 e FAIL storici conservati |

## Domini verificati

| Dominio del producer | Scanner e confine | Consumer e prova discriminante |
| --- | --- | --- |
| Paragrafo | Blank o marker del successivo producer; soft line lecite dentro il dominio | inline sul proprio valore; barriera prima del token inerte reale, code/math non attingono dal corpo |
| Heading | Singola riga sorgente prima del merge; metadata finale soltanto nel suo host | peel_attrs/inline con intervalli sorgente; title letterale conserva blocchi seguenti |
| Lista, quote, nota | Continuazioni effettive 4 spazi, prefix >, indent/tab; niente attraversamento del prossimo producer | read_parts mappa gli intervalli prima di rimuovere i prefissi; codice soft-line quote resta letterale |
| Pipe cell e caption | Delimitatori riconosciuti fuori dagli stessi atomi code/link/command/metadata; caption nel proprio dominio | pipe_row/table_contents e offset nel raw; token chiuso o EOF intero, nessuna formula/asset dal corpo |
| Label immagine/link | Walk bilanciato condiviso, codice locale e token opachi interi; lookahead visibile prima del newline | sub_inert rimappa le barriere nel testo label; destinazione e titolo distinti; anche EOF senza destinazione conserva il token |
| HTML vero/raw fence | Tag quotato nel dominio; raw HTML usa il parser locale, latex gli atomi finiti | Comment/CDATA completi; code inline/fence Markdown letterali restano inerti |

Gli intervalli sono controlli privati derivati dal sorgente, non testo normalizzato
o nuovi campi gold. Solo un token riconosciuto fuori dagli atomi può estendere
l'unità Markdown; l'estensione non allarga retroattivamente il lookahead dei
consumer. I byte interni restano un token localizzato. Il profilo resta finito,
non promette HTML universale e non usa whitelist delle stringhe dei reviewer.

1112 test mirati PASS, 16 F04 deselected; regressioni F13–F30 incluse.
21 casi GPT lessicali, 15 metadata, 14 trasporti e 15 Claude Y/K hanno nuove copie
con confronti sui file; i conteggi prima/dopo sono nei dati, compresa la variante
esplorativa raw LaTeX che non fonda un requisito aggiuntivo. 12 casi ulteriori
controllano quote e valuta prima del token nei domini heading/cella/caption/label/
nota. I test propri aggiungono EOF, Unicode e pipe negli atomi. Gli originali dei
reviewer e tutte le versioni intermedie sono protetti.

37 casi CLI, 74 exit OS reali, writer sintetico esplicito: positivi run0/evaluate5,
gold PROVISIONAL, automatic PASS e qualità INDETERMINATE; negativi audit INCOMPLETE,
run2/evaluate2/score3, missing pagina2 ancora FAIL quando il testo è nel commento.
H01–H08 R PASS; H12± R FAIL deliberati, expected0/harness1/wrapper2 immutati.
L'atteso score0 del vecchio E9 è errato per gold PROVISIONAL: le copie conservano
unmet E9, nessun PASS retroattivo; le CLI positive verificano davvero evaluate5.
[Report implementatore](../../temp/run-a002-fase1-corpus/reports/implementation-r007.md)
identifica gate finali, suite veloce, byte degli helper e fallimenti conservati.

## Misure, provenienza e limiti

Il checker indipendente ricalcola distanze globali char/token e decomposizione
condizionata, controlla 19 campi, denominatori, gate e l'intero albero raw.
Differenze numeriche/gate attuali rispetto a r006: 0.
I file mostrano ciascun set32 con {'FAIL': 28, 'INDETERMINATE': 0, 'PASS': 4}
e {'FAIL': 28, 'INDETERMINATE': 0, 'PASS': 4}; nessun assert imponeva i vecchi conteggi.
final corrente distinto da conversion_final/run e old_*r006, identità originali
immutate. F16 rimane verificabile. Scorer/alignment/audit/safety/R/adapters/registry/
writer/config/PATH e gold non cambiano; nessuna nuova V o conversione del corpus.

F26 resta invariato: misura completa e conteggi condizionati dipendono dal matching
strutturale; text_lower_bound garantisce soltanto la distanza del flusso globale
specificato. EXACT_GLOBAL_DISTANCE è esatto in quel dominio, TRIVIAL_ZERO non è
una misura completa zero e non occulta missing/pagine/ordine/numeri/celle/asset.
Il corpo inerte non può soddisfare un blocco visibile o produrre aggiunte note.

F27/F30 e F28 locale preservati; F28 globale/F29 corretti nei domini F31–F33;
F13/F14/F09 conservano unsupported localizzati. F23/F24/F26/F19/F20/F05/F15
preservati. V13/A03 reale invariata; L04 formule output UNAVAILABLE distinta,
L01 PDF/OCR/L02 compile TeX NOT_MEASURED, L05 CPU lower/RSS sampled/cacheOS/hardware
non certificati, Chrome NONPASS e gap dei12 helper storici non sanati. R non è
sandbox universale FS. Nessun GO autore; nuovo snapshot comune e due nuove review
indipendenti mirate spettano al supervisore.
