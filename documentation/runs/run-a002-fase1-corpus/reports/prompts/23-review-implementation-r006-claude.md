# Review indipendente implementazione a002 r006 — Claude

Agisci esclusivamente come **reviewer Claude** in una **NUOVA chat**,
distinta da autore, supervisore e reviewer dei giri precedenti. Registra il
provider effettivo ed eventuali sostituzioni senza impersonare quello previsto.
Non leggere la review concorrente **r006**, le sue evidenze/checkpoint o
l'eventuale arbitrato r006. Non avviare agenti o review delegate.

Output esclusivamente tuoi, sotto `temp/run-a002-fase1-corpus/`:

- Report: `reviews/review-implementation-r006-claude.md`.
- Evidenze: `evidence/review-implementation-r006-claude/`.
- Checkpoint: `handovers/review-implementation-r006-claude.md`.
- Nuovi ID: `CLA-R006-I001`, progressivi, distinti dai giri precedenti.
- Prompt origine: `prompts/23-review-implementation-r006-claude.md`.
- Per eventuali prove proprie, nuovi snapshot tecnici **solo** con label
  `review-r006-claude-*`, sotto `snapshots/`, creati con il manager stdlib.
  Sono output propri; non rigenerare o modificare alcuno snapshot ricevuto.

## Oggetto comune e identità

Run `run-a002-fase1-corpus`, **review dei fix dell'implementazione r006**,
stato IMPLEMENTATION_REVIEW / FIX_REVIEW_PROMPTS_READY. Consegna completa
ricevuta: F27–F30 completati secondo autore, chiusure/regressioni da valutare.
Nessuna review/arbitrato/GO r006 preesistente. Non è una review del piano.

- Snapshot: `temp/run-a002-fase1-corpus/snapshots/implementation-r006.json`.
- SHA256 snapshot: `633503036862eab4cd21443311aa25cfa5b1e5d73fc970840a2c56bfeed8398d`.
- Impronta worktree: `8eefc61f4380b11ac1f77fd8764c7658778cb40085e83533ac94e81684375158`.
- 319 file di worktree e 58212 artefatti espliciti. I 20 PDF del
  corpus sono versionabili e inclusi nell'impronta generale.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto alla ricezione.
- Piano r001 SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`;
  errata SHA256 `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.
- Mandato: piano/errata/GO piano, prompt05/06/09/12/15/18 e addendum V, poi
  `arbitrations/arbitration-implementation-r005.md` e prompt21/F27–F30.
- Autore: `reports/implementation-r006.md` e `handovers/implementation-r006.md`;
  prevalgono le sezioni finali **Consegna completa** e **Chiusura amministrativa finale002**.

Percorsi relativi della run sotto `temp/run-a002-fase1-corpus/`. Delta autore
**7 file**: scripts/benchmarks/observations.py e PROTOCOL.md,
tests/unit/test_conversion_fix_r006.py, docs/how-to/benchmark-fedelta.md,
docs/reference/benchmark-fedelta.md, nuovi report/dati permanenti r006.
metrics.py è byte-identico a r005; audit/safety/adapter/R/resources/PATH/
writer/registry/config/lock/pin/corpus sono invariati. Quattro documenti
generali del supervisore sono distinti in transition.json, senza fix prodotto.
Il delta include nuovi file non committati: `git diff base...HEAD` non basta.

Temp e documentation/runs sono esclusi dall'impronta generale, gli input
pertinenti sono vincolati come artefatti espliciti. Inventario
`evidence/supervisor-implementation-review-r006/review-inputs.json` per
navigare, senza obbligo di leggere tutti i log. Include prove/byte ricevuti,
report/CP autore, arbitrato/mandato, review r005 e prove r006. Registri vivi,
nuovi output reviewer, prompt22/23 e checkpoint corrente supervisore sono
esclusi: non alterano il comune. L'esclusione non consente di mutare input
ricevuti o ignorare codice e prove pertinenti.

## Letture e verifiche iniziali

1. AGENTS.md, temp/HANDOVER, parte iniziale PROJECT-CONTEXT, STATE/HANDOVER/
   brief e questo prompt; `handovers/supervisor-implementation-review-r006.md`.
2. Skill manage-implementation-run e verify-conversion-fidelity, protocollo
   run-lifecycle, template review, arbitrato r005 e prompt21 **integrali**.
3. **Entrambi i report r005 sono ammessi**: reviews/review-implementation-r005-chatgpt.md
   e reviews/review-implementation-r005-claude.md, checkpoint e sonde citate;
   sono input precedenti, non report concorrenti r006. Giri anteriori solo
   per requisiti/regressioni pertinenti, senza campagne a001 o nuova pianificazione.
4. Report e CP finale autore r006, final-checks001, PROTOCOL, due guide,
   `documentation/benchmarks/2026-10-09-a002-r006-confini-lessicali-html.md`
   e relativo `-data.json`, receipt di ricezione/transizione del supervisore.
5. Intero delta di codice/test e dipendenze raggiunte: html_lexeme/markdown_lexemes/markdown_units/
   inline_delimiter/link/pipe/trim/HTMLTree/visit/node/offset, audit e safety, raccolta risorse, scorer
   e disponibilità CER/WER, domini/counts/decomposizione, test r006 e regressioni.
   Lettura senza import implicito del prodotto o ML.
6. Piano/errata/ADR0001/0004/roadmap fase1/workflow benchmark/guida uv nei
   requisiti pertinenti. Non riaprire integralmente piano o fix già chiusi.

Verifica branch/HEAD/dev/merge-base/indice/delta, hash snapshot/piano/errata.
Esegui **prima e dopo** la review, registrando argv/cwd/output/exit:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r006
```

È manager stdlib del contesto, non test del prodotto. Non correggere input
o rigenerare il comune. STALE/identità diversa invalidano la review del
precedente oggetto: conserva il fatto e segnala il prossimo passo senza GO.
Vecchi snapshot/stage legano prove storiche e non devono tutti essere MATCH.

## Prove consegnate e limiti

Ricezione statica supervisore:50027 confronti hash,48306protetti invariati,
125input/copie prima-dopo consegna,72copie ricevute,205runtime S004→S005
equivalenti e10origini installate coincidenti coi sorgenti.122helper/test
invariati secondo equivalenza autore,125copie correnti controllate qui;
3813package/startup e ambiente chiuso verificati dall'autore.206input esterni
S005 corrispondono alla ricezione,zero differenze. Nessuna nuova build/suite/
R/conversione/rescore/V del supervisore; questa ricezione non è GO semantico.

Tre pyc managed ricevuti preservati, causa storica non attribuita; Chrome
NONPASS distinti. I quattro registri generali supervisore modificati dopo
ricezione rendono storico il preflight integrale S005: per eventuali nuove
sonde usa input/gate attuali, senza trasferire vecchi PASS o mutare receipt.
Equivalenza semantica non significa stessa identità di stage/shebang/prefix.

Prove sotto evidence/implementation-r006 e work/implementation-r006:

- entry-verify/entry-git/received-bytes/protected-descriptors, package-delta
  zero e managed-startup-guard-host, received-installation/entry-target-inputs
  e snapshot managed-entry001: prima del primo managed, origine/diecihash
  e input effettivi. Versioni intermedie/helper/test/config congelati con
  input-bytes-*-before.json/content store,125delivery non prima di ogni workload.
- Prima: work/before-gpt/probe-summary.json36 (28met/8unmet),
  before-claude-summary.json15 (7controlli/8Xunmet), cli-before001.
  Finali: work/after-gpt-003/probe-summary.json36 e after-claude-summary003.json15,
  tutti gli attesi rispettati. Le copie dei probe sono proprie, originali
  reviewer immutati; verifiche sintetiche ricevute sostituite dichiaratamente
  da goldPROVISIONAL nelle copie, nessuna V umana inventata.
- boundary-before001/boundary-after003:10varianti del trasporto prima unmet,
  dopo rispettate. known-transport-before/after:4foglie formule/link/LaTeX
  con delimitatori di enfasi, prima unmet e dopo preservate una volta.
  `$a*b$` resta AST unsupported: preservare raw non certifica algebra universale.
- R-r006-tests-003:936mirati (284nuovi+652precedenti),16F04deselected,14,09s.
  R-r006-tests-fast001:1093PASS+310subtest in41,47s. Suite prescritta tests/
  con ignore API/packaging/Docker,smoke radice esclusi; collaudo parziale,
  nessuna collection riparata con download o campagna F03/H15/F04 implicita.
- cli-after004:23casi/46exitOS reali, manifest congelato, writer **stub
  sintetico**,non motore nativo. Quattro positivi run0/evaluate5, gatePASS/
  qualitàINDETERMINATE per goldPROVISIONAL;19negativi INCOMPLETE/run2/
  evaluate2/score3,run.json/final presenti. Score3 bundle e evaluate2 hanno
  precedenze diverse, non correggerle per far coincidere gli exit.
- contracts002:10contratti conformi,H01–H08 R PASS,H12±RFAIL voluti,
  expected0 fisso/harness1/wrapper2. final-checks001:21receipt finali19PASS+
  2FAIL,binding prima/dopo/reap/socket/own_pending0. R PASS dei raccoglitori
  non è qualità PASS; un contratto che attende R FAIL non rinomina la receipt.
- S004 funzionale finale e S005 corrente: nuovi S/B/I,canonical reinstall/
  diecihash/preflight/R,managed3.12.13 stesso interprete figli. semantic-
  equivalence002:205runtime/3813package/122helper-test/protetti invariati,
  PROTOCOL solo delta benchmark; S/I binding e shebang/prefix sono **nuovi**,
  non byte-identità. Gate S005 precede suite finale. Valuta dipendenze reali
  della riattribuzione S004,non PASS integrale vecchio trasferito a S005.
- Rescore-manifest002/rescore002:64nuoviIDr006b,engine_runs0/nuovaV0, due set32
  ciascuno4PASS/28FAIL, raw/source/gold/V13 e conversioni originarie preservati.
  baseline003 resta conversione r002 con misure derivate r006. derived-check002
  confronta19campi numerici/gate,somme/denominatori/edit char-token globali e
  condizionati,catena e intero raw tree:zero differenze da r005. Nei nuovi dati
  final è il derivato attuale; conversion_final/run e old_* mantengono le
  identità storiche. Controllare questi legami,non solo il numero64.
- Fallimenti test001817PASS+3assert errati sui denominatori, CLI002exit46
  corretti ma atteso score5 sbagliato, eq001binding di stage confrontati
  letteralmente, copia PROTOCOL e documentcheck001 di chiusura sono preservati.
  Nuovi stage/helper/prove002/003/004 chiudono gli errori; **chiusura finale002**
  prevale sul footer001 storico. Nessun FAIL vecchio riclassificato o input
  modificato attribuito alla vecchia receipt. Gap12helper non sanato.

## Criteri tecnici identici dei due reviewer

Review mirata F27–F30, **intero delta7 e dipendenze toccate**, incluse le nuove
funzioni lessicali/splitter e i trasporti ordinari. F23/F24/F26 già chiusi,
F22originali chiusi con residuoF27,F25parzialeF28/F29,F13/F14/F09 localizzati.
F19/F20/F05/F15 e F16sui64 da preservare nei loro domini. Non riaprire altri
fix senza una dipendenza/difetto raggiungibile,non una nuova review del piano.

| Criterio | Verifiche sostanziali |
| --- | --- |
| C01 F27 errore lessicale | `</>` fuori dai domini inerti in paragrafo/cella/wrapper/caption/inline/raw fence: raw/intervallo/coordinate e diagnostica consumata,INCOMPLETE/run2/evaluate2,niente COMPLETE/PASS. Ramo tokenizer senza callback coperto nel subset finito,non semplice massimo end. Prosa/ref/math una volta; positivi bilanciati/literal< e code/comment body inerti discriminanti. |
| C02 F28 CDATA | Intero token sino a `]]>` o EOF,anche con `>` interno/multipli/Unicode,in blocco/inline/cella/caption/residuo. Raw/end/offset char-byte-line e avanzamento coerenti nei produttori/trasporti; corpo non diventa prosa/formula/ref/asset o apertura tentata. Known words resta il testo noto esterno. Unsupported visibile raggiunge audit/scorer,non supporto semantico CDATA; commenti/PI/doctype/code invariati. |
| C03 F29 confini Markdown | Commento aperto fuori codice non si spezza su blank/ATX/lista/fence/quote/nota/:::,in prosa e HTML wrapper,chiusura normale/--!>/abrupt/EOF coerente col tokenizer locale. Un token diagnostico inerte con intervallo intero,nessun blocco/foglia/ref/asset dal corpo,neppure URI unsafe o immagine presente. Zero aperture da esso,reason unsupported coerente; paragrafo/pagina visibilmente omessi rimangono missing/FAIL. Literal inline/fence/code positivi,prosa esterna e ordine/coordinate fisiche corretti. |
| C04 contratto e regressioni dei trasporti | html_lexeme/markdown_lexemes/markdown_units/inline_delimiter condividono confini finiti senza sostituire byte/newline; prefissi heading/list/quote/note,pipe row/caption/figure,link label/destinazione/titolo/attributi,math/rawTeX/code/enfasi e trim EOF rispettano foglie e contesti. Il corpo inerte non è ritokenizzato; normali contenuti leciti non vengono assorbiti,rifiutati o duplicati. Delimitatori interni a link/formula/HTML attrs/codice non cambiano il dominio. Positivi/mutazioni nuovi coprono la classe e intero delta,non sole stringhe reviewer o numero di test. Limiti32MiB/profondità/confini/safety preservati. |
| C05 F30 diagnostica | Nodo senza foglie ha reason neutra/coerente con occorrenze,non unmatched close per qualsiasi errore; raw/tipo/offset/severità/stati specifici invariati. Wrapper vuoto,close spaiata e diagnostiche multiple. Nessuna duplicazione della prosa per trasportare diagnosi. |
| C06 misura e domini | Unknown non wildcard globale: missing/page/order/aggiunte certe/numeri/celle/math/asset/ref restano autonomi,denominatori attesi invariati. CER/WER completi None dove non misurabili; counts condizionati/decomp e bound globale distinti. F26 chiuso: bound non limita CER/WER strutturali e può superarli,TRIVIAL_ZERO non completo0/fedeltà. Normalizzazioni/allineamento/AST/gold/config non adattati al risultato. |
| C07 64misure/pubblicazione/V | Catena/hash/raw/versioni/ID osservazioni→audit→score→final correnti e conversion_final/run/old_* storici coerenti.4PASS28FAIL per set verificati da file reali,zero motori/V. TXT0/89,DOCX11/182caption,EPUB8/186,RTFcondizionato19/162 contro globale13/162,TeXcompletoNone/condizionato1/152/globale0. Checker indipendente circoscritto e sum/denominatori veri,gate non sommati come perdite indipendenti. Rettifica F25 report/guide r006 senza riscrivere r005. V13/A03 sui riferimenti non firma output,L04UNAVAILABLE distinta. |
| C08 toolchain e prova finale | Ultimi byte reader v6 legati a test003/probe003/boundary003/known-after/CLI004/contratti002/rescore002/S004 e suiteS005.48306protetti/125delivery/72copie,205runtime equivalenti/122helper-test e binding stage nuovi,nuovi S/B/I/diecihash/gate reali,same interpreter/R/reap/socket.19PASS+2FAIL voluti preservati,vecchi stage/fail/footer non riattribuiti. Helper fuori wheel vincolati prima-dopo,copy125delivery non prima di ogni workload; suite/API/Docker/ML non collaudo universale. |
| C09 risorse e consegna | Quota52h/entry/attese/riserva invariati,costi nuovi registrati;CPUlower/RSSsampled/retained conservativo non upper o certificazione. Chiusura002 autorevole,fallimenti/ChromeNONPASS/helper12gap/L01-L02-L04-L05 conservati. Report/guide/checkpoint usabili,ruoli/Git manuali preservati,nessuna fase2 o scelta motore prematura. |

Una soluzione reversibile non enumerata è ammissibile se chiude il requisito.
Un sintetico può esporre un blocco pur assente dai64raw; rarità non è deroga.
Unsupported è consentito nel subset finito se visibile e inerte,senza occultare
foglie/risorse/errori o PASS spurio. Non richiedere HTML/algebra universale,
nuovo schema o campagne estranee. Collega requisito/input ammesso/percorso/
impatto/attribuzione e fix minimo; non inventare rilievi o usare un voto medio.

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
non chiedere freeze intermedio al supervisore. Un S integrale s005 ora
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
Ledger autore chiusura140540,315652s (~39,039h), CPU9277,600452s lower,
retained1861397431B/RSSsampled740855808B **storici**, non saldo attuale.
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
Valuta F27–F30 e regressioni pertinenti, compresi i residui F13/F14/F09;
non un voto medio e non GO per numero di test o assenza dell'altro report.

Non modificare STATE/HANDOVER/eventi/indici/changelog/checkpoint supervisore,
snapshot ricevuti o file concorrenti. Nel checkpoint: input/output/esito,
costi/processi propri pendenti e prossimo ruolo **supervisore per ricezione
dei due report r006 e arbitrato**. Concludi in chat con esito e percorsi.
