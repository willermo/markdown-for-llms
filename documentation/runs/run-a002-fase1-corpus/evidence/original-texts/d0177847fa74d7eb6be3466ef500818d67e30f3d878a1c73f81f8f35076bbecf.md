# Review indipendente implementazione a002 r007 — ChatGPT

Agisci esclusivamente come **reviewer ChatGPT** in una **NUOVA chat**,
distinta da autore, supervisore e reviewer dei giri precedenti. Registra il
provider effettivo ed eventuali sostituzioni senza impersonare quello previsto.
Non leggere la review concorrente **r007**, le sue evidenze/checkpoint o
l'eventuale arbitrato r007. Non avviare agenti o review delegate.

Output esclusivamente tuoi, sotto `temp/run-a002-fase1-corpus/`:

- Report: `reviews/review-implementation-r007-chatgpt.md`.
- Evidenze: `evidence/review-implementation-r007-chatgpt/`.
- Checkpoint: `handovers/review-implementation-r007-chatgpt.md`.
- Nuovi ID: `GPT-R007-I001`, progressivi, distinti dai giri precedenti.
- Prompt origine: `prompts/25-review-implementation-r007-chatgpt.md`.
- Per eventuali prove proprie, nuovi snapshot tecnici **solo** con label
  `review-r007-chatgpt-*`, sotto `snapshots/`, creati con il manager stdlib.
  Sono output propri; non rigenerare o modificare alcuno snapshot ricevuto.

## Oggetto comune e identità

Run `run-a002-fase1-corpus`, **review dei fix dell'implementazione r007**,
stato IMPLEMENTATION_REVIEW / FIX_REVIEW_PROMPTS_READY. Consegna completa
ricevuta: F31–F34 completati secondo autore, chiusure/regressioni da valutare.
Nessuna review/arbitrato/GO r007 preesistente. Non è una review del piano.

- Snapshot: `temp/run-a002-fase1-corpus/snapshots/implementation-r007.json`.
- SHA256 snapshot: `6a5235582225134b364818c96d0e3029597d505d39db936b16e0bec039de7ab4`.
- Impronta worktree: `296a3cc7b90044cb6abb2f38fcf2c5d030820e34f03d361a45d078c5c329b535`.
- 322 file di worktree e 65669 artefatti espliciti. I 20 PDF del
  corpus sono versionabili e inclusi nell'impronta generale.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto alla ricezione.
- Piano r001 SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`;
  errata SHA256 `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.
- Mandato: piano/errata/GO piano, prompt05/06/09/12/15/18/21 e addendum V, poi
  `arbitrations/arbitration-implementation-r006.md` e prompt24/F31–F34.
- Autore: `reports/implementation-r007.md` e `handovers/implementation-r007.md`;
  prevalgono le sezioni finali **Consegna completa** e **Chiusura amministrativa finale**.

Percorsi relativi della run sotto `temp/run-a002-fase1-corpus/`. Delta autore
**7 file**: scripts/benchmarks/observations.py e PROTOCOL.md,
tests/unit/test_conversion_fix_r007.py, docs/how-to/benchmark-fedelta.md,
docs/reference/benchmark-fedelta.md, nuovi report/dati permanenti r007.
metrics.py è byte-identico a r006; audit/safety/adapter/R/resources/PATH/
writer/registry/config/lock/pin/corpus sono invariati. Quattro documenti
generali del supervisore sono distinti in transition.json, senza fix prodotto.
Il delta include nuovi file non committati: `git diff base...HEAD` non basta.

Temp e documentation/runs sono esclusi dall'impronta generale, gli input
pertinenti sono vincolati come artefatti espliciti. Inventario
`evidence/supervisor-implementation-review-r007/review-inputs.json` per
navigare, senza obbligo di leggere tutti i log. Include prove/byte ricevuti,
report/CP autore, arbitrato/mandato, review r006 e prove r007. Registri vivi,
nuovi output reviewer, prompt25/26 e checkpoint corrente supervisore sono
esclusi: non alterano il comune. L'esclusione non consente di mutare input
ricevuti o ignorare codice e prove pertinenti.

## Letture e verifiche iniziali

1. AGENTS.md, temp/HANDOVER, parte iniziale PROJECT-CONTEXT, STATE/HANDOVER/
   brief e questo prompt; `handovers/supervisor-implementation-review-r007.md`.
2. Skill manage-implementation-run e verify-conversion-fidelity, protocollo
   run-lifecycle, template review, arbitrato r006 e prompt24 **integrali**.
3. **Entrambi i report r006 sono ammessi**: reviews/review-implementation-r006-chatgpt.md
   e reviews/review-implementation-r006-claude.md, checkpoint e sonde citate;
   sono input precedenti, non report concorrenti r007. Giri anteriori solo
   per requisiti/regressioni pertinenti, senza campagne a001 o nuova pianificazione.
4. Report e CP finale autore r007, final-checks002, PROTOCOL, due guide,
   `documentation/benchmarks/2026-10-10-a002-r007-contesti-markdown.md`
   e relativo `-data.json`, receipt di ricezione/transizione del supervisore.
5. Intero delta di codice/test e dipendenze raggiunte: source_scope_end/inline_scope_end/code_end/math_end/command_at/metadata_at/AUTOLINK/
   html_lexeme/markdown_lexemes/markdown_units/barriere/label/link/pipe/trim/HTMLTree/offset, audit e safety, raccolta risorse, scorer
   e disponibilità CER/WER, domini/counts/decomposizione, test r007 e regressioni.
   Lettura senza import implicito del prodotto o ML.
6. Piano/errata/ADR0001/0004/roadmap fase1/workflow benchmark/guida uv nei
   requisiti pertinenti. Non riaprire integralmente piano o fix già chiusi.

Verifica branch/HEAD/dev/merge-base/indice/delta, hash snapshot/piano/errata.
Esegui **prima e dopo** la review, registrando argv/cwd/output/exit:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r007
```

È manager stdlib del contesto, non test del prodotto. Non correggere input
o rigenerare il comune. STALE/identità diversa invalidano la review del
precedente oggetto: conserva il fatto e segnala il prossimo passo senza GO.
Vecchi snapshot/stage legano prove storiche e non devono tutti essere MATCH.

## Prove ricevute e limiti

Supervisore: 60801 confronti hash,58941protetti invariati,
146input/copie consegna prima-dopo,73copie storiche,205runtime S005→S006
equivalenti,10origini installate e206input esterni corrispondenti.141helper/
test invariati secondo equivalenza autore;3813package/startup e ambiente
chiuso controllati dall'autore. Quattro nuovi registri generali supervisore
rendono storico S006 integrale; nessun preflight vecchio trasferito.
Nessuna suite/build/R/CLI/conversione/rescore/V eseguita dal supervisore.

Prove sotto evidence/implementation-r007 e work/implementation-r007:

- Ingresso fix-r006 MATCH prima del primo managed; startup guard/package3813,
  dieci moduli/installazione/archivi ricevuti. Tre pyc encodings correnti
  preservati, nessun restore o causa storica nuova.73 byte copie ricevute,
  protected-descriptors58941. Input-store e input-bytes per workload legano
  helper/config/test/reader e versioni intermedie **prima** delle esecuzioni;
  copie146delivery002 sono di consegna, non dichiarate tutte per-workload.
- Prima: GPT21/14/15 con8/9/8unmet, ClaudeY/K15 con10unmet. Dopo finali
  after-004-gpt/transport/attributes:21/14/15 conformi; ClaudeY/K15/15,
  domains-after00412/12. ClaudeB16 conserva solo E9 atteso storico errato:
  gold PROVISIONAL impone evaluate5,non0. Copie proprie delle sonde autore,
  originali reviewer r006 immutati; nessuna nuova review o V simulata.
- R-r007-tests004:1112PASS/16F04deselected,176 nuovi. Suite finale S006:
  R-r007-tests-fast0011269PASS+310subtest in34,41s, percorso tests/ con ignore
  API/packaging/Docker e smoke radice esclusi. Collaudo selezionato parziale,
  non campagna completa né acquisizione per riparare collection.
- cli-after00137casi e cli-after-e9-0021caso:38casi/76exit OS, writer
  **sintetico dichiarato**,zero motori o nuova V.10positivi principali più
  E9 run0/evaluate5, gate automaticoPASS/qualitàINDETERMINATE per gold
  PROVISIONAL;27negativi INCOMPLETE/run2/evaluate2/score3bundle. Run/final/
  config/raw/attesi precedono figli reali,non semplice regola evaluate derivata.
- contracts00210conformi,H01–H08 R PASS,H12±RFAIL deliberati con expected0
  fisso/harness1/wrapper2. final-checks00224receipt22PASS+2FAIL, binding
  prima-dopo/reap/socket e own_pending0. R PASS distinto dagli assert.
- Codice finale S005, input documentali finali S006: S/B/I, wheel canonica
  reinstallata,diecehash/origini/preflight e R gate nuovi. Equivalence002:
  205runtime/3813package/141helper-test identici,PROTOCOL solo delta benchmark.
  S/I binding,shebang/prefix e receipt nuove,non byte-identità inventata.
  Valutare attribuzione di prove funzionali S005 a byte finali S006 realmente
  consumati; i gate generali S006 sono nuovi,non PASS S005 trasferito.
- Rescore-manifest002/rescore00264nuoviIDr007b:engine_runs0/nuovaV0,
  due set32 ciascuno4PASS28FAIL,raw/source/gold/V13 e conversioni originarie
  preservati. Checker002 indipendente:19campi numerici/gate/somme/denominatori/
  full edit matrix condizionale e globale,intero rawtree/catena/versioni,
  zero differenze da r006. final deriva attuale,conversion_final/run storico,
  old_*r006. Verificare legami/file,non copiare il numero64.
- Fallimenti entryKeyError/helpergate erroneo/domain001 whitespace/S003
  tests0021104PASS8FAIL/S004tests0031110PASS2FAIL/launcher escape/whitelist
  amministrativa001 conservati; nuovi helper/versioni e tests004 validi.
  Due veri EOF-label e assert schema propri distinti. Nessun FAIL/E9/STALE
  riclassificato; report/CP finale prevale sulle sezioni WIP storiche.

## Criteri tecnici identici dei due reviewer

Review mirata F31–F34, **intero delta7 e dipendenze raggiunte**, compresi i
nuovi primitivi condivisi/barriere/label e producer. F27/F30 e F28locale
già chiusi da preservare; F28globale/F29/residui F13/F14/F09 da valutare
tramite questi fix. F23/F24/F26/F19/F20/F05/F15 e F16sui64 preservati nei
domini. Nessuna nuova review del piano o riapertura indiscriminata di a001.

| Criterio | Verifiche sostanziali |
| --- | --- |
| C01 F31 domini e confini | source_scope_end/inline_scope_end/PARAGRAPH_BOUNDARY/code_end/math_end/tag_end e producer effettivi limitano lookahead agli stessi domini? Code/math/tag incompleti o chiusi in altro blocco non saltano commenti/CDATA successivi né riprendono dentro fence; dollar amount,softline,fence validi e quote/list/note/heading/cella/caption mantengono il proprio comportamento finito. GPT8varianti eCLA Y3–Y9/K1–K5 chiusi. Token effettivamente aperto resta intero sino a chiusura/EOF,corpo zero prosa/formula/ref/asset/apertura anche se immagine esistente/URI unsafe; missing q/pagina2 autonomo e testo esterno una volta. Non basta INCOMPLETE. |
| C02 F32 autolink | AUTOLINK identico nei domini scanner/inline/HTML producer,nessun falso stack. Y1/Y2/Y10 e varianti http/https/mailto con code comment/PI/CDATA/declaration,link destination/title/escape: stessi blocchi/tipi/ordine/testo/offset del positivo,COMPLETE senza token spurio,refs conservati senza fetch. HTML davvero aperto/commento reale/ref attiva restano discriminanti. |
| C03 F33 metadata e raw TeX | metadata_at/command_at/attrs/peel_attrs/destination/title/code/span/heading e inline consumano gli stessi intervalli finiti. Aperture HTML nelle foglie/valori sicuri non estendono blocchi esterni; quote HTML e attrs Markdown distinti. Ref/eqref/label preservati,foo unknown locale,types/ordine/offset Unicode/celle/ref/anchor corretti. HTML vero in label/prosa continua analizzato,active attrs/URI unsafe attivi rifiutati. Nessun rifiuto generico o supporto TeX universale per aggirare il difetto. |
| C04 trasporti/barriere/regressioni | Matrice dominio→scanner→consumer→confine coerente nell'intero delta. Barriere sorgente calcolate prima di merge e rimappate dopo prefix stripping/nesting: label senza destination,EOF/opener a cavallo di softline,celle/pipe cuts/caption/figure,enfasi/quote/note/liste/heading/raw html-latex fence. Nessuna chiusura code/math presa dal corpo inerte,nessun raw/text/offset/line duplicato o occultato. Prosa normale,Unicode,literal<,codice e delimitatori in titolo/destinazione restano leciti. Positivi/mutazioni coprono domini e interazioni finite,non solo payload o numero di test; profili/limiti32MiB/profondità/safety invariati. |
| C05 chiusure precedenti e misure | F27 `</>` diagnosticato nei trasporti e inerte nel codice,F28CDATA locale intera,F30reason neutra e occorrenze specifiche mantenute. Unknown non wildcard per missing/page/order/aggiunte/numeri/celle/asset/ref. Scorer byte-identico,F26 counts condizionati/decomposizione distinti dal bound globale garantito nel proprio dominio; può superare CER/WER strutturali,TRIVIAL_ZERO non misura completa0. Denominatori/AST/gold/config non adattati per far passare il reader. |
| C06 F34 documenti e decisioni | PROTOCOL/due guide e report/dati nuovi descrivono domini reali/limiti/prove finali,rettificano affermazioni r006 senza riscriverlo. Scelte reversibili motivate e assegnazione dei fallimenti:EOF-label reale vs assert/launcher propri,sonde PROVISIONAL vs V reale,stub CLI vs motori. Nessun PASS retroattivo o firma umana agente. Unsupported del subset visibile/inert/locale senza occultare foglie e seguito. |
| C07 64misure e V | Catena/hash/rawtree/versioni/ID observe→audit→score→final attuali e conversion_final/run/old_*storici coerenti.4PASS28FAIL per set verificati da file,zero motori/V.TXT0/89,DOCX11/182caption,EPUB8/186,RTFcondizionato19/162 vs globale13/162,TeX completoNone/condizionato1/152/globale0; somme/denominatori reali senza sommare gate come perdite indipendenti. V13/A03 riferimenti non firma output,L04 UNAVAILABLE distinta. |
| C08 prova finale/toolchain | Byte reader v7 legati a tests004/probe004/domains004/CLI001-E9-002/contratti002/rescore002/S005 e suiteS006; nuovi S/B/I/diecihash/gate,same interpreter/R/reap/socket.58941protetti/146delivery/73copie/205runtime/141helper e binding nuovi.22PASS+2FAIL H12 conservati. Copy146delivery non tutte prima di ogni workload; input-bytes per-workload e intermedi fisici verificabili. Equivalenza circoscritta,preflight S006 generale storico dopo documenti supervisore,nessun vecchio PASS trasferito o restore pyc. |
| C09 risorse/chiusura/ciclo | Wall attuale distinto da ledger autore storico40,625h,quota52h include attese/recensioni,nessun reset. Stop nuovi workload49h/10:06:54 Europe/Rome,deadline13:06:54; nessuna nuova sonda dopo stop. CPUlower/RSSsampled/disco conservato non upper/certificazioni. L01/L02/L04/L05,ChromeNONPASS/helper12gap/fallimenti preservati,ruoli/Git manuale e confini rispettati,nessuna fase2/motore scelto. |

Soluzioni reversibili non enumerate possono chiudere il requisito. Sintetici
ammessi possono esporre difetti assenti dai64raw; rarità non deroga.
Non imporre HTML/Markdown/algebra/TeX universali,nuovo schema o campagne
estranee. Rilievi fondati su input raggiungibile,impatto,requisito,attribuzione
e correzione minima; nessun GO da numero di test o assenza dell'altro report.

## Vincolo temporale corrente prevalente

Ricezione supervisore2026-10-10T08:00:36.516257Z,wall48,895h. Le attese dal
report autore sono incluse. **Stop nuovi workload già prossimo o trascorso
alla lettura:10:06:54 Europe/Rome (08:06:54Z),deadline13:06:54 (11:06:54Z)
oggi10ottobre**. Verifica l'ora reale prima di ogni eventuale sonda; la
facoltà sotto descritta non autorizza R/test/CLI dopo lo stop né nuova quota.
Dopo lo stop completa letture statiche/hash/documenti/report e review nella
riserva, dichiarando i limiti delle prove. Una review statica è ammissibile;
se un rilievo decisivo non può essere concluso,registra l'impedimento reale,
non un GO automatico. La deadline e gli altri limiti restano vincolanti.

## Autonomia e limiti delle prove proprie

Consentiti letture statiche, hash/link/aritmetica, manager stdlib e propri
report/evidenze/checkpoint. Piccole sonde sintetiche reader/audit/scorer,
incluso percorso CLI pertinente con writer sintetico, sono consentite se
necessarie a decidere un rilievo: copie proprie, attesi fissati prima, R
corrente e gate validi. Non rieseguire suite generale/test campaign, build,
reinstallazione, baseline motori/conversioni, render/browser, M/V, download/
modelli/remoto/a pagamento o Git di scrittura. Nessun fix prodotto/test/corpus/
gold canonici/input altrui e nessuna firma umana agente.

Prima del primo managed registra startup/package input e origine locale
3.12.13, archivi/installazione reale/dieci hash e nuovi controlli S/B/I/
preflight sulle dipendenze consumate. Manifest/launcher/target e **nuovi
snapshot tecnici propri** con le label sopra sono autorizzati anche se
l'helper risolve snapshot sotto la directory comune; vincola helper/input
prima del workload, output vivi esclusi. Non modificare vecchi/common e
non chiedere freeze intermedio al supervisore. Un S integrale s006 ora
stale per quattro documenti amministrativi non diventa PASS: conserva il
FAIL, diagnostica e prepara nuovi input reali circoscritti. Non eseguire
con gate rossi, trasferire receipt vecchie o ripristinare pyc per nascondere
delta. Input indipendenti invariati non richiedono nuove campagne a001.

R/no-network/same interpreter, shell=False/registry/confini preservati;
R≤900s per comando/1MiB per stream/32MiB per file, sonde piccole e motivate.
EPERM/EACCES: diagnosi ed eventuale escalation tool della stessa operazione
autorizzata mantenendo isolamento; fermati al rifiuto effettivo, senza aggirarlo.
Statico/runtime/prove autore lette distinti; fallimento prima del workload
non dimostra il risultato dedotto e prove dell'autore non diventano proprie.

Quota unica52h dall'entry2026-10-08T07:06:54.369068Z, attese incluse,
CPU21600s/disco6GiB/RAM8GiB/stop sampled6GiB; stop nuovi workload49h/
19800CPU-s/5GiB, riserva3h/1800CPU-s/1GiB. Stop nuovi workload
2026-10-10T08:06:54.369068Z, deadline11:06:54.369068Z. Nessun reset/incremento.
Ledger autore chiusura146250,572159s (~40,625h), CPU10400,457453s lower,
retained2086483026B/RSSsampled740855808B **storici**, non saldo attuale.
Registra costi/tempi propri senza upper/hardware inventati. Nessuna rete,
acquisizione/document-send/costo esterno implicito; fonti primarie attuali
solo se indispensabili per nuove versioni/capacità, senza acquisizioni.

## Report ed esito

Usa il template review. Dichiara provider/modello/chatID effettivi se esposti,
prompt/revisione/snapshot/hash/impronta/base/delta/indipendenza, verify prima/
dopo, criteri controllati, argv/input/prove e limiti. **GO** o **NO_GO** reale
sul comune identificato; GO non certifica altri motori o L04. Oggetto invalido:
dichiara impedimento fattuale e prossimo passo, nessun GO dall'assenza di difetti.
Ogni rilievo: ID, severità/blocco, posizione, input/percorso/requisito/attribuzione,
impatto, evidenza statica/runtime, correzione minima e criterio di chiusura.
Valuta F31–F34 e regressioni pertinenti, compresi i residui F13/F14/F09;
non un voto medio e non GO per numero di test o assenza dell'altro report.

Non modificare STATE/HANDOVER/eventi/indici/changelog/checkpoint supervisore,
snapshot ricevuti o file concorrenti. Nel checkpoint: input/output/esito,
costi/processi propri pendenti e prossimo ruolo **supervisore per ricezione
dei due report r007 e arbitrato**. Concludi in chat con esito e percorsi.
