# Confronto documentale e fix didascalie a002 — 2026-10-08

Il corpus revisione 4 ripristina nei sorgenti DOCX, RTF e nell'include TeX le
didascalie già previste dalla specifica manuale. Il nuovo confronto offre
**13 documenti completi affiancati**, con formule MathML leggibili, celle vere,
figure/note/riferimenti e .md scaricabile. La verifica umana resta PENDING;
nessuna risposta completa reale ricevuta, nessun documento VERIFIED.

Questo supplemento aggiorna il perimetro del
[report storico su revisione 3](2026-10-08-a002-native.md), conservato immutato.
Il vecchio M291PASS non confrontava alcune didascalie: non certifica quelle
proprietà. Le perdite di caption calcolate su quei tre sorgenti incoerenti
non vanno attribuite al motore. I vecchi raw/metriche/packet restano evidenze
storiche sugli input realmente consumati; non diventano prove della revisione 4.
Il mandato06 e l'addendum del supervisore sostituiscono il vecchio lavoro
umano a 215 righe, senza una nuova pianificazione o review simulata.

## Canonici e controlli indipendenti

Testi, numeri, AST manuali, denominatori, nove pagine PDF, dodici render fissi
e 1664 parole restano invariati. Cambiano canonical/gold DOCX e RTF; per TeX
cambiano `sections/one.tex` e gold, mentre il main mantiene lo stesso SHA.
Sette altri canonical/gold e tutti i controlli sono byte-identici. Revisione3
è preservata integralmente nella run; nessuna taratura su output Pandoc.

Due generazioni013/014 indipendenti, entrambe R PASS, hanno canonici/PNG
byte-identici. Le sole differenze PDF LibreOffice sono quelle metadata ammesse
e conservate senza riscrittura. M-gold006 **403 required PASS**, M-controls004
**89 required PASS**. M confronta caption/celle/figure/target codificati,
fonte matematica manuale, campi PDF/locatori e due estrattori indipendenti;
raster Unicode e forma visibile richiedono ancora V. Il render DOCX ora
contiene «Celle misurate», confermato dagli estrattori e da agent_visual.

Il [contratto sui canonici effettivi](../../temp/run-a002-fase1-corpus/evidence/implementation-r001/caption-source-contract-001.json)
è R PASS: tutti e tre i vecchi sorgenti hanno **celle PASS, caption FAIL**;
i nuovi hanno entrambi PASS. Nessuna presenza delle celle nasconde l'omissione.

## Tentativi dipendenti effettivi

Campagna `a002-native-caption-r003`, scope `dependent-rerun`, congelata prima
dei motori: cinque percorsi × quattro tentativi, **20 record validi**. Tutti
SUCCESS/COMPLETE, tutti score3/qualityFAIL; cinque gruppi e summary R PASS.
Non sono 32 nuovi tentativi. TXT e due percorsi EPUB rimangono nella campagna002
storica: input/gold invariati, nessun nuovo workload attribuito loro.

| Percorso | Tentativi | Numeri esatti | Formule AST | Celle esatte | Figure+caption | Riferimenti | CER | WER | Primo s | Mediana successive s [range] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| direct-docx | 4 | 8/13 | 2/2 | 0/9 | 0/1 | 2/3 | 0.04959 | 0.00000 | 0.0431 | 0.0435 [0.0433–0.0436] |
| direct-rtf | 4 | 3/8 | 0/0 | 0/6 | 0/0 | 1/2 | 0.06816 | 0.08642 | 0.0425 | 0.0428 [0.0426–0.0429] |
| direct-tex | 4 | 4/9 | 5/5 | 0/9 | 1/1 | 1/3 | 0.01646 | 0.01974 | 0.0425 | 0.0426 [0.0425–0.0428] |
| legacy-docx | 4 | 8/13 | 2/2 | 0/9 | 0/1 | 2/3 | 0.04959 | 0.00000 | 0.6229 | 0.6029 [0.5958–0.6041] |
| legacy-rtf | 4 | 3/8 | 0/0 | 0/6 | 0/0 | 1/2 | 0.06816 | 0.08642 | 0.6331 | 0.6317 [0.6110–0.6675] |

Conteggi e metriche sono uguali nelle quattro ripetizioni di ogni percorso.
CER/WER e associazioni restano distinti: WER0 non sana numeri/celle o figure
non associate. A10 operativo è soddisfatto da quattro DOCX per ciascuno dei
due percorsi con figura realmente referenziata e contenuto verificato; il
conteggio «figure+caption» misura invece la fedeltà completa dell'associazione.
Le celle mostrano perdite di shape/contenuto e i riferimenti/anchor hanno diff
espliciti. Queste sono misure strutturali sui nuovi input, gold ancora PROVISIONAL;
non giudizi matematici di semantica o tipografia degli output.

Tempi Pandoc diretti e child legacy comprendono perimetri differenti. Ogni
processo/workspace è nuovo, page cache OS non controllata. wait4 CPU e sampled
RSS10ms hanno i limiti dichiarati nel protocollo, non un hard cap aggregato.
[Fatti e receipt dei20 tentativi](../../temp/run-a002-fase1-corpus/evidence/implementation-r001/caption-fix-results-001.json).

L04output-packet002 conserva36 occorrenze/72 giudizi potenziali/9 casi distinti,
non assegnati; il suo R PASS attesta inventario/composizione, nessun giudizio
output. L01PDF/OCR e L02compilazione TeX restano NOT_MEASURED.

## Confronto e risposte v2

[C02 pronto](../../temp/run-a002-fase1-corpus/evidence/implementation-r001/V-comparison-001/C02-docx-it.html),
[indice delle13 coppie](../../temp/run-a002-fase1-corpus/evidence/implementation-r001/V-comparison-001/index.html).
Il riferimento deriva dai blocchi manuali, non da observations/scorer o da
Pandoc. A sinistra pagine reali/capitoli actual ordinati oppure sorgenti
TXT/TeX/controlli con asset reali. TeX non compilato è dichiarato. La vista a
destra e il .md comprendono tutto il documento, note/anchor e link locali
risolti. Nell'interfaccia principale non compaiono b-ID, AST, locator o hash.

Checker indipendente: **13 documenti, 1906 controlli PASS**, R PASS su stage039.
Controlla anche le unità sorgenti e le pagine originali, il .md completo e
l'effettiva matematica MathML, celle vuote, caption, figure/note e ordine.
Il pair digest lega sorgenti/include/render/asset, HTML/.md/mappa e CSS;
la sola revisione amministrativa globale è esclusa per conservare coppie
con byte identici. Il packet rimane immutato dopo la composizione.

Una risposta naturale per documento basta: «C02: corrisponde», «C02: manca …»,
«C02: non riesco a valutarlo». Receipt schema2 con testo originale/hash,
autore/ora ricezione realmente attribuiti con limiti, coppia/revisione/ambito,
decisione e differenze. Nessuna firma atomica o timestamp per proprietà.
L'importatore riclassifica il testo originale; risposte parziali conservano
quelle valide sugli altri documenti, anche se una risposta successiva è STALE
per un altro documento. Vaghezza PENDING, dubbi AMBIGUOUS, differenze REJECTED,
hash diverso STALE. M required PASS e V documentale esplicita completa
ACCEPTED sono entrambi necessari per VERIFIED. Le API schema1 restano storiche.

Stima prospettica: C02 5–8 minuti, tutte le 13 coppie 60–90 minuti. Nessuna nuova
finestra concordata o ora di inizio dichiarata; 260 min restano accordo storico
sul vecchio workflow, nessun tempo umano svolto attribuito.

## Prove, attribuzione e limiti

Stage038 S/B/I/preflight/inventario per packet e motori; stage039 ricostruito/
reinstallato per il fix finale dell'importatore e del checker. Suite mirata
**151 PASS in 6.31 s**, suite veloce prescritta **292 PASS + 310 subtest in 17.04 s**,
entrambe R PASS stage039. Sono esclusi API/packaging/Docker e i due smoke radice;
non è collaudo completo o convalida della migrazione. I 20 contratti runner
precedenti rimangono storici; nessun criterioR modificato da questo fix.

Sono preservati due FAIL ordinari di test: literal_text TXT mancante nel
serializer (40pass/1fail), poi un disallineamento del nome immagine EPUB nel
checker originale (150pass/1fail). Correzioni/test ripetuti con nuovi stage;
una label snapshot amministrativa uppercase respinta prima del workload è
conservata e ritentata con label valida. Nessuna richiesta di freeze al supervisore.

Chrome locale ha prodotto due screenshot effettivi C02/C05 per riscontro
agent_visual, commandexit0; **R renderer FAIL per namespace Chrome annidati**.
Sandbox browser non disabilitato; screenshot sono presentazione diagnostica,
non V o prova semantica. Anche i due capitoli EPUB riusati preservano il loro
renderer R NON PASS. Composizione/checker R PASS non sanano questi limiti.

Gli stage usano keeper e dipendenze già locali, con wheel ricostruita e
reinstallata per ultima. Nessuna acquisizione di tool/modelli, invio documenti
a provider o costo remoto. Quota unica e ledger non azzerati; CPU nota è lower bound, disco
retained inode-dedup non misura gross/peak. Le prove native restano attribuite
alla wheel/stage038 effettivi: i dieci helper nativi/config, sorgenti/gold e
origini prodotto sono invariati rispetto al fix importer stage039. Non si
ripetono motori per modifiche della sola raccolta V. Il supplemento è scritto
dopo le prove; la receipt del delta documentale distingue tale avanzamento
editoriale dalla preflight usata dai workload.

A03 rimane PENDING per tredici giudizi reali. Questa è una consegna del fix
tecnico e un checkpoint implementatore, non il risultato completo della run.
Dopo V reale questa chat importa le risposte e chiude A03 oppure corregge i
difetti nel mandato. Il supervisore prepara poi lo snapshot comune e due
nuove review indipendenti ChatGPT/Claude. Nessuna modifica ai registri comuni,
nessun commit/merge/push o fase2.
