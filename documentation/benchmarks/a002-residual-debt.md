# Debito a002 — misuratore sperimentale e parser mantenuto

2026-10-12. **APERTO, accettato per uso esplorativo; risoluzione obbligatoria**.
[ADR0008](../decisions/0008-maintained-parser-and-provisional-benchmark.md),
[chiusura e prove](../runs/run-a002-fase1-corpus/manifest.md).
Decisione accettata distinta dall’implementazione, che non è iniziata.

## Responsabile e ripresa

Responsabile di programmazione/verifica: supervisore del progetto; autore della
futura run parser per implementazione, due reviewer indipendenti per verifica.
Trigger obbligatorio: completamento del primo percorso valutabile nel browser
(uscita fase3), con apertura di run dedicata. Prima di automatizzare attestazioni
di completezza/sicurezza o di dichiarare rilascio, questo debito deve essere
chiuso tramite prove e arbitrato. Nessuna deadline nuova: trigger di prodotto,
priorità di correttezza, non rinvio indefinito a discrezione di una chat futura.

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
