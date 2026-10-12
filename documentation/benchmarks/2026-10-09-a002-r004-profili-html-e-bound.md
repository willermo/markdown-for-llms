# A002 r004 — profili HTML e misure testuali distinte

Misure derivate il 2026-10-09 UTC, implementazione locale ancora soggetta a due
nuove review e arbitrato. Nessun GO autore, nuova conversione o V umana.
[Dati dei64 tentativi](2026-10-09-a002-r004-profili-html-e-bound-data.json).
Questa rettifica integra il [report r003](2026-10-09-a002-r003-html-e-metriche.md)
senza modificarlo, e mantiene separati i risultati di
[r002](2026-10-09-a002-r002-rettifica-e-baseline.md),
[riferimenti verificati r001](2026-10-08-a002-verified-reference.md),
[confronto umano r001](2026-10-08-a002-human-comparison.md) e
[baseline nativa r001](2026-10-08-a002-native.md).

## Copertura e prove

F17–F21 correggono la perdita di foglie nelle proiezioni table/figure e nel
trasporto inline, le caption aggiunte con atteso assente/vuoto, l'attribuzione
impropria di bound ai conteggi strutturali e il `<` letterale classificato come
risorsa. Reader `observations-v4-html-coverage`, scorer
`metrics-v4-conditional-and-bounds`. I profili finiti inline/table/figure e la
politica di copertura sono nel [protocollo](../../scripts/benchmarks/PROTOCOL.md).

Block HTML inline conserva tutte le foglie note e diventa unsupported esplicito.
Table ammette caption unica, righe/celle finite e whitespace; un ramo ulteriore
è conservato interamente come residuo, senza scartare le sue risorse. Figure
ammette img primaria e figcaption; testo ulteriore è residuo. Caption duplicate
non selezionano l'ultima. Formule supportate aggiunte falliscono, unknown non
ottiene PASS, asset mancanti danno INCOMPLETE/exit2. URI evasive erano già
respinte prima dell'apertura: non è attribuito un bypass storico inesistente.
Code/fence letterali restano inerti, `<` letterale in prosa resta testo fedele.

475 test mirati PASS, 16 unitari F04 esclusi da questa selezione; nessuna nuova
campagna F04/F03/H15. Sono118 casi nuovi F17–F21 e357 precedenti pertinenti.
Dieci contratti H01–H08/H12 con esiti attesi: otto R PASS, due H12 R FAIL
**deliberati**, harness1/wrapper2 contro R expected0 fissato prima. Riproduzioni
originali su copie proprie prima/dopo sotto catene S/B/I/R nuove valide; corpus,
gold, packet, sorgenti/render/asset,13 receipt V derivate e64 raw protetti.
Il [report implementatore](../../temp/run-a002-fase1-corpus/reports/implementation-r004.md)
riporta anche suite veloce finale, costi, prove/fallimenti e identità degli stage.
La suite selezionata non è un collaudo completo: API/packaging/Docker, due smoke
radice e ML sono esclusi; non si scarica o installa per riparare collection.

## Rettifica delle misure testuali

Il numeratore noto r003 non era sempre un bound: con unknown intercalato,
l'allineamento può rifiutare il nodo e contare sia omissione sia inserzione
dello stesso contenuto visibile. Gold `Known words.` contro
`Known \foo{x}words.` produce conteggio condizionato4/2 ma ammette un
completamento con distanza0; L1 produce20/30 e ammette il completamento1/30.
Il FAIL per missing strutturale segue F08 e rimane distinto da questa diagnosi.

Ora si pubblicano tre misure con regole e corrispondenze esplicite:

- CER/WER completi sotto il matching strutturale: None/UNAVAILABLE quando il
  testo non è misurabile, denominatori pieni, stesso matching/normalizzazioni.
- `conditional_text_counts` e `WER_decomposition` v2: conteggi condizionati al
  matching; alias `known_word_edits` compatibile, `guaranteed_bound=false`.
  Entry/categorie conservano ID, offset, testi e somme. Non sono una stima
  certa di parole materialmente perse/inventate, né bound garantiti.
- `text_lower_bound`, `global-text-completion-bound-v1`: distanza minima globale
  del flusso misurabile, indipendente dal matching; zero garantito su qualunque
  completamento del testo non misurabile o limite di calcolo globale. Char
  stream con LF fra proiezioni testuali non vuote, token stream concatenato;
  table/figure/formula/anchor esclusi, code letterale incluso, denominatori
  attesi invariati (separatori LF non aggiunti al denominatore).

Ogni completamento unknown può emettere zero o qualunque testo e cambiare i
confini dei token. La distanza è nonnegativa: zero non supera alcun completamento.
Nel caso misurabile Levenshtein è il minimo sulle corrispondenze di edit del
flusso fissato. Questa è la garanzia, distinta dai10 casi/42 completamenti
sintetici confrontati con una matrice di edit indipendente (caratteri e parole).
Il bound zero non è una misura completa0 e non modifica l'attribuzione locale,
i missing/page loss, l'ordine, i numeri, le celle o gli asset. C1 conserva3/30
come conteggio strutturale e FAIL, con bound globale0 dove il flusso è incerto.
Inserzione interamente nota `Additional invented content.` conserva3/2 e FAIL.

## 64 misure correnti, zero nuovi motori

I due set da32 sono quelli storici r001 e baseline003 r002. Raw byte-identici,
identità di conversione originali e helper storici preservati; osservazioni,
audit e score hanno nuovi ID r004a. Conversioni baseline003 del2026-10-08 UTC,
misure r004 del2026-10-09 UTC. Ogni set ha effettivamente4 PASS e28 FAIL:
non era un assert da mantenere. Il checker stdlib separato conferma somme,
denominatori, legami e zero differenze numeriche/gate nei campi storici comparati.

| Formato | WER strutturale completo | Conteggio condizionato | Bound globale garantito | Composizione strutturale |
| --- | --- | --- | --- | --- |
| TXT |0/89|0/89|0/89|Nessun edit|
| DOCX diretto/legacy |11/182|11/182|11/182|Caption2+9 unmatched; aligned0|
| EPUB diretto/legacy |8/186|8/186|8/186|8 omissioni|
| RTF diretto/legacy |19/162|19/162|13/162|11 aligned +3 omissione titolo +5 inserzioni (titolo3/caption2)|
| TeX diretto |UNAVAILABLE|1/152|0/152|1 aligned; unknown locale attribuito|

Il valore globale RTF elimina la duplicazione da corrispondenze strutturali;
non sana altri FAIL. TeX1/152 è ora qualificato come condizionato, senza
attribuirgli una garanzia che non è provata. Le caption di tabelle/figure hanno
un dominio locale distinto dal CER/WER della prosa: assente/vuota equivalente
solo come testo, aggiunta nota non vuota FAIL3. Le stesse perdite possono attivare
più gate: non sommare WER, caption, numeri e blocchi come errori indipendenti.

Le quantità per fenomeno rimangono: corpo celle DOCX9/9, EPUB9/9, RTF6/6,
TeX9/9; numeri TXT9/9, DOCX12/13, EPUB11/13, RTF5/8, TeX8/9. Formule AST
DOCX2/2, EPUB2/2, TeX5/5; figure DOCX0/1, EPUB1/1, TeX1/1. Riferimenti
DOCX2/3, EPUB1/5, RTF1/2, TeX0/3. Questi confronti automatici non sono V
sugli output: L04 rimane UNAVAILABLE anche con AST esatto e Vgold già ricevuta.

## Provenienza e limiti

Tre pyc managed encodings modificati durante la review r003 sono conservati e
identificati; causa/autore non accertati. Audit statico dei campi eseguibili
contro sorgenti controllati, manifest package/startup attuali vincolati a
snapshot r004, nuovi S/B/I, wheel canonica reinstallata/dieci hash e preflight
prima degli import. Nessun ripristino dei byte o dei vecchi descrittori.
Input/helper/config/test fuori wheel fisicamente preservati prima/dopo,
R nello stesso managed CPython3.12.13 dei figli, Firejail con rete isolata.

Quota originale52h, attese incluse, nessun reset; CPU nota lower-bound, RSS
campionato e disco conservato con supplementi, cache OS non controllata.
L01 PDF/OCR e L02 TeX compile NOT_MEASURED; L05 hardware non esteso; gap12 helper
raw storici e Chrome renderer NONPASS non sanati. Nessuna rete, acquisizione,
invio documenti o servizio nuovo. Archiviare prove e byte prima di pulire temp.
Registri comuni e changelog spettano al supervisore dopo la ricezione.
