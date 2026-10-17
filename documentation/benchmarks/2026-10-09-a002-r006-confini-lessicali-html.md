# A002 r006 — confini lessicali HTML e contenuti inerti

Nota17ott2026: riferimenti operativi eliminati o già indisponibili sono
resi testo; esiti storici invariati. [Disponibilità delle prove](../development/cleanup-2026-10-17.md).

Misure derivate e prove reali del 2026-10-09 UTC, run `run-a002-fase1-corpus`.
Questo report rettifica F25 del [report r005](2026-10-09-a002-r005-diagnosi-html-e-domini.md)
senza modificarlo. Dati r006 (riferimento storico non disponibile):
64 nuovi legami raw→osservazione→audit→score→final, su conversioni r001 e
baseline003 r002 storiche. Zero motori, render o nuove verifiche umane.

Il reader è `observations-v6-html-lexical-boundaries`; scorer, alignment,
normalizzazioni e denominatori restano invariati. Il profilo `html-tokens-v1`
continua a classificare commenti, CDATA, dichiarazioni e PI come unsupported:
audit INCOMPLETE, run/evaluate2. Il contenuto dei commenti e CDATA è inerte,
non contribuisce a prosa, formule, riferimenti o asset e non soddisfa blocchi
attesi visibili. Raw, coordinate Unicode/byte e occorrenze restano localizzati.

## Rettifica e prove

| Fix | Difetto ricevuto e prova prima | Risultato corrente e prova dopo |
| --- | --- | --- |
| F27 | `</>` consumato senza callback; paragrafo/cella/raw HTML fence potevano essere COMPLETE/PASS automatico | Token esatto diagnosticato in ogni trasporto; INCOMPLETE, run2/evaluate2; code e commenti restano inerti |
| F28 | CDATA troncata al primo `>`, corpo ripreso inline con testo o asset inventati | Intero `]]>` o EOF; nessuna reference/asset/apertura dal corpo, noto `Known words.` conservato |
| F29 | Blank e marcatori Markdown spezzavano commenti aperti; corpo diventava blocchi o risorse | Unità lessicali fino alla chiusura effettiva o EOF; missing/page loss restano FAIL anche se il testo è nel commento |
| F30 | Nodo senza foglie sempre etichettato unmatched close | Riepilogo neutro; reason specifiche, raw, offset e severità delle occorrenze conservati |

Le 36 riproduzioni GPT avevano 8 attesi non rispettati, le 15 Claude altri 8;
ora tutti gli attesi sono rispettati. Dieci varianti dei trasporti mostravano
troncamenti in enfasi, caption/label, EOF, attributi e celle; quattro ulteriori
casi per formule/link/raw LaTeX mostravano delimitatori di enfasi anticipati.
Le correzioni attraversano i produttori reali, senza whitelist dei payload.

936 test mirati PASS, 16 F04 deselected. 23 casi CLI con writer sintetico
esplicito verificano 46 exit OS reali: quattro positivi run0/evaluate5 con gold
PROVISIONAL, automatic PASS e qualità INDETERMINATE; 19 negativi
INCOMPLETE/run2/evaluate2/score3. Queste sonde non sono conversioni native né V.
H01–H08 R PASS; H12± restano R FAIL deliberati, expected0/harness1/wrapper2.
Il [report implementatore](../../temp/run-a002-fase1-corpus/reports/implementation-r006.md)
registra stage, S/B/I/R, byte, errori conservati, suite veloce e chiusura finale.

## Misure preservate e limiti

Il checker indipendente ricalcola decomposizione condizionata e distanze globali
char/token, verifica denominatori, gate, 19 campi e l'intero albero raw.
I 64 record hanno zero differenze numeriche/gate rispetto a r005; ciascun set
contiene effettivamente 4 PASS e 28 FAIL. Identità/hash dei derivati sono nuovi,
conversioni e raw originali invariati. F16 rimane verificabile nei dati.

F26 resta chiuso: CER/WER completi e conteggi condizionati dipendono dal matching
strutturale; `text_lower_bound` garantisce soltanto la distanza del flusso globale
dichiarato. EXACT_GLOBAL_DISTANCE è esatto in quel dominio; TRIVIAL_ZERO non è
una misura completa zero né una prova di fedeltà. Non sommare diagnosi strutturali
come bound. Gli esempi r005 DOCX62/60, EPUB54/53, D1 e D2 restano invariati.

Le chiusure normali `-->`, `--!>` e abrupt seguono il ramo effettivo del tokenizer
managed 3.12.13; CDATA usa `]]>`. Code inline/fence letterale, destinazioni/titoli
e formule conservano i propri confini; una raw HTML fence EOF rimane unsupported.
È un profilo finito, senza HTML universale, fetch o comandi dai documenti.

La review storica non mostrava un bypass URI di F27 né un falso PASS per i CDATA/
commenti già INCOMPLETE: in quei casi erano errati trasporto, conteggi, risorse e
missing. Report/dati precedenti e fallimenti delle prove restano conservati.
Corpus/sorgenti/gold/render/asset/packet e V13/A03 invariati; nuova M/V0.
L04 formule output UNAVAILABLE, L01 PDF/OCR e L02 compile TeX NOT_MEASURED;
L05 CPU lower-bound/RSS sampled/cache OS/hardware, Chrome NONPASS e gap dei
12 helper storici non sanati. Nessun GO autore o nuova scelta di motori.
