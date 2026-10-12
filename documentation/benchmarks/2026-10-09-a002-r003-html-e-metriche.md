# A002 r003 — HTML, disponibilità testuale e composizione WER

Data: 2026-10-09 Europe/Rome; misure derivate eseguite il 9 ottobre UTC.
Report dell'implementatore, in attesa di due nuove review indipendenti e
arbitrato. Nessun GO autore né scelta di motore. [Dati per tutti i tentativi](2026-10-09-a002-r003-html-e-metriche-data.json).

I fix locali F13–F16 correggono il benchmark consegnato r002 dopo NO_GO/NO_GO:
formule e unknown dentro span/a HTML non raggiungevano lo scorer; immagini
inline/figure potevano sfuggire all'audit; unknown testuale estraneo lasciava
CER/WER complete AVAILABLE0. La composizione strutturale del WER era omessa.
Reader `observations-v3-html-transport`, scorer `metrics-v3-text-domain` e
[protocollo](../../scripts/benchmarks/PROTOCOL.md) qualificano il nuovo contratto.

Il ponte HTML conserva tutte le foglie, raw, coordinate, segmenti e offset,
anche annidati e nei fence. Risorse di ogni frammento passivo supportato o
fallback sono estratte/validate oppure rendono INCOMPLETE; URI vietate danno
UNSAFE_REFERENCE prima di leggere il target. Attributi duplicati sono unsupported/INCOMPLETE, senza perdere il primo target.
Code letterale rimane inerte.
MathML con unica annotation TeX conserva AST/raw; altro MathML resta unknown.
Non è un parser universale HTML/CommonMark. Nessun browser o Pandoc è stato
eseguito sui payload pericolosi.

CER/WER complete diventano UNAVAILABLE/None per unknown testuale osservato
unmatched, con ragione, ID, offset e scope; il denominatore atteso rimane
completo. I segmenti noti sono lower-bound: Known words. più tre parole note
dà3/2; comando unknown estraneo dà completo UNAVAILABLE, senza inventare edit.
Nel caso C1 del reviewer restano3 parole note/30 attese e completo None;
FAIL noto precede unavailable. Formula/code e celle/caption fuori dal dominio
testuale intero hanno controlli locali distinti. Non si annullano page loss o
missing con unknown estraneo e non cambia la regola locale di attribuzione.

## Rettifica e identità delle conversioni

Nuove observe/audit/score su **64 raw** conservati:32 della rettifica storica
r002 e32 della baseline003 r002. Nuovi ID derivati, engine_runs0; raw, vecchie
metriche/report e26141 input protetti invariati. La baseline003 resta una
conversione **r002 del 2026-10-08 UTC**, con misure derivate r003; nessuna nuova
baseline di conversione è dichiarata. Ogni set produce effettivamente4PASS/
28FAIL, senza imporre tali conteggi come attesi del workload.

Tutti64 raw effettivi non contengono HTML; solo gli8 TeX hanno l'unknown o0013
già attribuito localmente. Questo scan accompagna il rescore completo, non lo
sostituisce. Un checker stdlib separato verifica dai file reali aritmetica,
denominatori, candidate di ruolo e identità dei valori/gate/alignment r002.
Numeri, celle, CER/WER, ordine, esiti e FAIL noti rimangono uguali; nuove
provenienze/offset, disponibilità motivata e decomposizione cambiano i byte.

Questo report integra e rettifica l'attribuzione di
[r002](2026-10-09-a002-r002-rettifica-e-baseline.md) e dei tre report r001
[nativo](2026-10-08-a002-native.md),
[riferimento verificato](2026-10-08-a002-verified-reference.md),
[confronto umano](2026-10-08-a002-human-comparison.md), che rimangono immutati.
Le rettifiche numeriche r002 sono confermate sui raw; non si descrive ogni
conteggio strutturale come parola materialmente inventata o persa dal motore.

## Composizione strutturale verificata

Valori per tentativo, identici nelle quattro ripetizioni di ciascun percorso
e in entrambi i set. I dati contengono tutti64 tentativi con ID e descrittori.

| Percorso | WER | Edit aligned | Omissioni attese | Inserzioni unmatched |
| --- | --- | --- | --- | --- |
| direct-txt | 0/89 | 0 | 0 | 0 |
| direct-docx / legacy-docx | 11/182 | 0 | 0 | 11 |
| direct-epub / legacy-epub | 8/186 | 0 | 8 | 0 |
| direct-rtf / legacy-rtf | 19/162 | 11 | 3 | 5 |
| direct-tex | UNAVAILABLE; noto1/152 | 1 | 0 | 0 |

DOCX11 è interamente da due caption osservate come paragrafi unmatched:
«Celle misurate»2 parole e «Figura 1 — Tre regioni e assi del laboratorio.»9.
Il testo nei blocchi aligned è esatto nel dominio dichiarato. La caption può
anche fallire il proprio gate: non si sommano questi gate come perdite distinte.

RTF8 conteggi strutturali sono titolo declassato da heading a paragraph:
3 omissioni+3 inserzioni, e caption tabella2 inserzioni;11 edit restano nei
blocchi aligned. I dati mostrano testi effettivi, tipi/ruoli, block/observation
ID, offset, denominatori, candidate di ruolo esatte e lower-bound. Nessun
matching arbitrario, nuova normalizzazione o modifica dell'allineamento per
ridurre WER. «Candidate» è diagnostica testuale esatta, senza inferenza causale
universale; non cambia la metrica.

## Prove e limiti

373 test mirati passati su nuovi S/B/I, wheel canonica/diecihash/origine e R;
10 contratti dipendenti H01–H08/H12 con attesi congelati e R expected-exit0.
H06 dimostra UNSAFE_REFERENCE/run2/evaluate2; i due H12 sono R FAIL deliberati.
Le riproduzioni proprie prima/dopo conservano i payload originali dei reviewer.
Suite veloce conclusiva, receipt, input byte e bilancio finale sono documentati
nel [report completo della run](../../temp/run-a002-fase1-corpus/reports/implementation-r003.md).
Il collaudo è parziale: API/packaging/Docker e due smoke radice sono esclusi.

Corpus rev5, sorgenti/PDF/render/asset/include/referenceMarkdown/packet e13 V
derivate invariati; zero nuove firme umane. L04 output formule UNAVAILABLE;
L01 PDF/OCR, L02 compilazione TeX e L05 hardware non misurati. Gap helper dei
12 raw storici, Chrome renderer NONPASS e limiti RSS sampled/CPU lower-bound/
cache OS restano dichiarati. Risorse/R, adapter, validator, PATH, writer,
registry e motori non cambiati; F03/H15/F04 e conversioni non rieseguiti.

Quota originale52h, attese incluse, senza reset; network/acquisizioni/invio
documenti0, servizi nuovi0€. Nessun download, servizio, fase2, Git di scrittura
o aggiornamento dei registri comuni. Supervisore: nuovo snapshot comune,
due nuove review mirate indipendenti e arbitrato; archiviare evidenze e byte
prima della futura pulizia di temp. Commit/merge/push restano manuali dopo GO.
