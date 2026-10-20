# Debito a002 — misuratore sperimentale e parser mantenuto

Disponibilità17ott2026: alcune prove e copie storiche sono state rimosse
dalla pulizia autorizzata; i riferimenti successivi descrivono la consegna
storica. Corpus/gold e casi decisivi restano; nessuna nuova firma umana
o chiusura del debito. Dettagli (archived: `documentation/development/cleanup-2026-10-17.md`, tag `pre-docs-2026-10-18`).

2026-10-16. **APERTO, risoluzione differita per priorità al prodotto utilizzabile**.
[ADR0008](../decisions/0008-maintained-parser-and-provisional-benchmark.md),
chiusura e prove (archived: `documentation/runs/run-a002-fase1-corpus/manifest.md`, tag `pre-docs-2026-10-18`).
Decisione accettata distinta dall’implementazione dell’adattatore, che non è iniziata; il nucleo sperimentale è verificato separatamente.

## Responsabile e ripresa vigente

Responsabile di programmazione/verifica: supervisore del progetto. Disposizione
utente16ott2026: il debito resta aperto, ma non blocca il prodotto sperimentale
con conversione/Web UI e limiti dichiarati. Risoluzione dilazionabile dopo un
prodotto utilizzabile; niente ripresa obbligatoria automatica all'uscita fase3.
Il trigger precedente è superato da [ADR0008 aggiornato](../decisions/0008-maintained-parser-and-provisional-benchmark.md).
Prima di introdurre funzioni che attestino automaticamente proprietà dipendenti
da questi ID, servono ancora implementazione, prove e review pertinenti.
Nessuna vecchia review trasformata in GO. Trace r005 consegnato/non revisionato,
non dipendenza del primo prodotto; controlli operativi e guardie separati.
Le disposizioni successive sotto conservano la cronologia del percorso precedente.

### Debito riconosciuto — precisazione16ott2026

I cinque ID restano debito tecnico da consolidare appena avremo modo e tempo;
la priorità prodotto non autorizza cancellazione o etichette di completamento.
[Registro trasversale](../development/technical-debt.md) include anche comparatore
a003 e trace non revisionato. Pulizia dei soli ambienti/build ricostruibili:
prove salvate e criteri conservati; per ripresa del trace occorre ricostruire
l'ambiente Haskell, senza pretendere MATCH dei vecchi snapshot incompleti.

### Ripresa anticipata autorizzata — 2026-10-15

Per disposizione esplicita dell'utente, il lavoro viene anticipato nel riesame
del metodo a003, senza attendere l'uscita fase3. Stato del debito: **APERTO,
ripresa sperimentale sui ruoli sintattici**. Il piano r002 deve coprire fattibilità del nucleo
mantenuto condiviso, adattamento specifico del modello osservato a002 e criteri
di risoluzione dei cinque ID, insieme all'adattatore del comparatore a003.
Le prove comuni si riusano; gli esiti di chiusura rimangono distinti.

L'apertura originariamente prevista di una run dedicata non è un prerequisito
per questa ripresa autorizzata in a003. Implementazione e prove restano future;
chiusura per ID solo dopo nuove review e arbitrato. Nessuna modifica retroattiva
agli esiti r014 o ai limiti sperimentali esterni alla sintassi.

Ricezione piano r002, 2026-10-15: NOT_VIABLE preliminare dei profili provati
anche per a002. Il pianificatore riporta nuove premesse sintattiche e controprove
del reader sui casi D01–03; sono consegne da verificare nelle due review, non
chiusure o arbitrati già svolti. Le classificazioni storiche della tabella sotto
rimangono attribuite alla chiusura r014. Stato a003 (archived: `documentation/runs/run-a003-fase1-engine-comparison/manifest.md`, tag `pre-docs-2026-10-18`).

Arbitrato del piano, 2026-10-16: diagnosi circoscritta accolta, errata
richieste. Controprovata su P/C l'omissione dall'AST di HTML/immagini nella
cella eccedente di una tabella: caso obbligatorio D03/D04. Nessun nuovo
adattatore o falso COMPLETE nei caller già dimostrato da questa sola sonda;
risorse e residui devono avere prova esplicita. Tutti i cinque ID restano aperti.

Verifica delle errata, 2026-10-16: accolta la prova di insufficienza del
wrapper AST per residui/risorse e del testimone di soli valori. È disposto
un esperimento di provenienza nei punti originali del reader con contabilità
del raw ed equivalenza AST. Nessun nuovo adattatore/audit o stato COMPLETE
convalidato; D01–05 restano aperti, la chiusura storica ridotta non cambia.

Ricezione del trace, 2026-10-16: la visita osservativa del parser originale
riporta risorse nelle celle scartate nei14controlli e conserva provenienza nei
rientri provati. Le due review successive hanno portato a NO_GO del nucleo
per trim e verificatore; nucleo r002 poi approvato con due GO e arbitrato.
Prototipo ruoli r003 poi arbitrato NO_GO: false origini HTML possono occultare
risorse senza diagnostica; D-A002-04 INSUFFICIENT fino al fix r004 disposto.
Due review r004 poi arbitrate NO_GO: estensioni HTML errate e certificati
respinti su input ammessi; sufficienza per il prototipo non confermata.
Mandato r005 sui termini nativi, derivate incomplete e origini esclusive,
con visita riferita a P/TagSoup e guardie separate dal modello di browser.
Note e chiusure non invalidate preservate, nessuna chiusura D01–05.
Caller/modello completo ancora assenti;
il gate positivo non chiude D01–05 e non è un audit del prodotto.

## Elementi aperti e stato della prova

| ID | Residuo | Stato della prova e impatto | Criterio di chiusura |
| --- | --- | --- | --- |
| D-A002-01 | Lista→heading/div/quote/nota e stato del paragrafo, CLA-R014-I001 | Traccia statica deterministica; combinazione Pandoc non eseguita. Possibile falsoCOMPLETE/asset omesso nel modello e falsoUNSAFE; non fetch/esecuzione dimostrati. | AST pinned degli input decisivi, poi prova dell’adattatore su originali/continuazioni/positivi e negativi; se premessa smentita registrare respinto con prova. |
| D-A002-02 | Div non chiuso,tokenmetadata `+`/`._c`,ATX indentata, CLA-I002/O001/O002 | Ipotesi condizionate, non difetti runtime confermati. Non promuovere automaticamente a blocchi o bug risolti. | Confermare/respingere premesse AST; proprietà degli intervalli opachi e guardia su HTML reale verificate. |
| D-A002-03 | Permissività span in pipe_cuts, ipotesi ChatGPT r014 | Review incompleta, nessun finding convalidato. | Input raggiungibile, requisito e impatto attribuiti; decisione esplicita, nessun fix senza prova. |
| D-A002-04 | Affidabilità globale di parser/guardia/misura e coordinate | Le chiusure precedenti valgono nei domini provati; non certificazione generale di COMPLETE/UNSAFE. | Adattatore parser mantenuto, profilo/versione fissi, risorse/testo/formule/numeri/ordine/asset e source-map dove dichiarata; fuori profilo esplicito, niente riuscita silenziosa. |
| D-A002-05 | Review tecnica r014 incompleta e costo amministrativo | GPT41,042049CPU-s propri,35,510746 nel lookup ripetuto; Claude6,44 misurati. Nessuna coppiaGO/GO. | Non fabbricare review retroattiva: nuove review e arbitrato della futura implementazione; inventari indicizzati/prove proporzionate. |

I limiti di profilo F36B1–B4/A8/backtick-run/rawhtml5-htmlspace non confermati/
timingcommenti/readerfallbackID restano dichiarati: definirne il trattamento
nel futuro contratto; non pretendere parser universale o cambiare la gravità
per ottenereGO. L01PDF-OCR eL02compile nonmisurati,L04outputUNAVAILABLE,
L05lower/sampled/hardware,ChromeNONPASS/helper12gap/H15/E9 sono limiti degli
esperimenti, non tutti difetti risolvibili con l’AST. Confronti motori/hardware
appartengono ai rispettivi incrementi della roadmap.

## Cosa è utilizzabile adesso

Corpus/gold e raw con asset conservati, provenienza e risultati finiti sono
utilizzabili per ispezione/esplorazione. Metriche e giudizi delle classi non
convalidate sono provvisori; non fondare ranking definitivo o garanzie di
sicurezza su di essi. Nessuna modifica al software nella chiusura: audit/CLI
continuano tecnicamente a produrre gli stati attuali, questa etichetta limita
le conclusioni dell’utente, non corregge quegli stati.

## Verifica della futura soluzione

1. Fissare profilo parser e verificare AST delle premesse residue; preservare
   i casi smentiti e i controlli CodeInline/CodeBlock realmente inerti.
2. Adattare l’AST e la provenienza; guardie di file/processi indipendenti dal
   giudizio di fedeltà. Nessuna perdita del Markdown grezzo o dei suoi asset.
3. Confrontare contro gold indipendenti, casi originali dei reviewer e varianti
   finite; test invalidati e corpus ricevuto, delta classificati e revisione
   umana di numeri/formule/asset/ordine. Coordinate solo se verificate.
4. Due nuove review indipendenti e arbitrato con criteri espliciti; aggiornare
   questo registro con ID/prova/byte/versione. Solo allora stato RISOLTO.

Non usare l’AST dello stesso motore come certificazione della sua conversione.
Non trasferire suite3156+310 o64ricalcoli storici alla nuova implementazione.
