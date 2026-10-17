# A002 r005 — diagnosi HTML, heading e domini testuali

Nota17ott2026: riferimenti operativi eliminati o già indisponibili sono
resi testo; esiti storici invariati. [Disponibilità delle prove](../development/cleanup-2026-10-17.md).

Misure derivate reali del 2026-10-09 UTC, run `run-a002-fase1-corpus`.
Conversioni storiche r001 e baseline003 r002 conservate; nessun motore o nuovo
render/V. Questo report rettifica e completa il [report r004](2026-10-09-a002-r004-profili-html-e-bound.md)
senza riscriverlo. Dati r005 (riferimento storico non disponibile),
con tutti i64 record e i legami raw→osservazione→audit→score→final.

Reader `observations-v5-html-diagnostics`, scorer invariato
`metrics-v4-conditional-and-bounds`. Le diagnosi di contenitori HTML non chiusi
sopravvivono alla proiezione dei wrapper; raw/coordinate e tipo restano disponibili
ai consumatori, senza duplicare testo o risorse. Un heading HTML supportato non
riceve più il prefisso ATX del Markdown; h1–h6 sono misurabili anche in wrapper e
fence raw. Gli host heading/code HTML applicano il profilo inline finito ai
blocchi annidati. Commenti, dichiarazioni e processing instruction sono token
unsupported localizzati, mai prosa o risorse attive estratte dal loro contenuto.
Markdown code/fence letterale, `<` letterale, formule e URI conservano i confini.

## Risultati reali e rettifica del dominio

Nuove osservazioni/audit/score `r005b` su due set da32raw, con gli stessi input,
gold rev5 e receipt V13 derivate. Ogni set ha effettivamente4PASS/28FAIL qualità;
non è un conteggio imposto dal checker. Il batch conserva i FAIL. Zero differenze
numeriche/gate nei campi storici confrontati con r004; cambiano identità/hash delle
osservazioni e i metadati ATX. F16 resta verificabile con decomposizione per ID,
offset, testo, categorie e denominatori nei dati. Il checker r005 ricalcola le
distanze globali char/token con matrice propria, senza importare reader/scorer.

| Formato | WER strutturale completo / conteggio condizionato | Distanza globale a parole sullo stesso denominatore |
| --- | --- | --- |
| TXT | 0/89 | 0/89 |
| DOCX | 11/182 (caption unmatched2+9) | 11/182 |
| EPUB | 8/186 (omissioni) | 8/186 |
| RTF | 19/162 =11aligned+3omissione titolo+5inserzioni | 13/162 |
| TeX | completo UNAVAILABLE, condizionato1/152 | TRIVIAL_ZERO0/152 |

`text_lower_bound` limita **soltanto la distanza del flusso globale dichiarato**.
Non è un limite inferiore di CER/WER completi o condizionati sotto matching
strutturale e può superarli per ordine e separatori. DOCX:62char globali contro
60strutturali; EPUB54contro53. D1 (due paragrafi invertiti) conserva CER strutturale0
ma distanza globale38char/10parole (CER globale≈0,667), insieme al FAIL di ordine.
D2 (omissione) ha26char strutturali e27globali per il separatore LF. La garanzia
r004 era corretta nel dominio globale; questa è una precisazione documentale,
non un nuovo calcolo o un cambio di alignment/normalizzazione/denominatori.

EXACT_GLOBAL_DISTANCE è la distanza minima esatta del flusso misurabile.
TRIVIAL_ZERO esprime solo la nonnegatività per completamenti arbitrari dell'unknown,
anche zero testo; non è una misura completa0 né prova di fedeltà. Missing/pagine,
ordine, numeri, celle e asset restano decisioni autonome. Non confrontare il bound
con CER/WER strutturali come se condividessero le corrispondenze.

## Prove e limiti

652 test mirati PASS,16F04 deselected; riproduzioni GPT16 e Claude26 dopo i fix
rispettano gli attesi correnti. Nove casi CLI sintetici con writer stub dichiarato
verificano18exit operativi run/evaluate e presenza receipt/final, senza conversioni
native: heading positivi misurabili, aggiunteFAIL3, markup non chiuso/missing/unsafe
INCOMPLETE2. H01–H08 PASS, H12± RFAIL deliberati/expected0/harness1/wrapper2.
Il [report implementatore](../../temp/run-a002-fase1-corpus/reports/implementation-r005.md)
lega ogni risultato a stage, manifest, byte consumati, snapshot e receipt reali.
Suite veloce e chiusura della consegna sono registrate nel report implementatore.

Corpus/sorgenti/gold/render/asset/packet/V13 e64raw invariati. Nessuna nuova M/V
umana, L04 formule output UNAVAILABLE distinta dalla V sui riferimenti;
L01PDF/OCR e L02compileTeX NOT_MEASURED. L05CPU lower-bound/RSS sampled/cacheOS,
hardware locale, ChromeNONPASS e12helper storici non recuperati restano limiti.
Nessuna nuova misura di altri motori/ML/remoto, nessun vincitore o GO autore.
I report r001–r004 e i fallimenti delle prove restano immutati.
