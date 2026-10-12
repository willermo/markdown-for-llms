# Review indipendente implementazione a002 r005 — ChatGPT

Agisci esclusivamente come **reviewer ChatGPT** in una **NUOVA chat**,
distinta da autore, supervisore e reviewer dei giri precedenti. Registra il
provider effettivo ed eventuali sostituzioni senza impersonare quello previsto.
Non leggere la review concorrente **r005**, le sue evidenze/checkpoint o
l'eventuale arbitrato r005. Non avviare agenti o review delegate.

Output esclusivamente tuoi, sotto `temp/run-a002-fase1-corpus/`:

- Report: `reviews/review-implementation-r005-chatgpt.md`.
- Evidenze: `evidence/review-implementation-r005-chatgpt/`.
- Checkpoint: `handovers/review-implementation-r005-chatgpt.md`.
- Nuovi ID: `GPT-R005-I001`, progressivi, distinti dai giri precedenti.
- Prompt origine: `prompts/19-review-implementation-r005-chatgpt.md`.
- Per eventuali prove proprie, nuovi snapshot tecnici **solo** con label
  `review-r005-chatgpt-*`, sotto `snapshots/`, creati con il manager stdlib.
  Sono output propri; non rigenerare o modificare alcuno snapshot ricevuto.

## Oggetto comune e identità

Run `run-a002-fase1-corpus`, **review dei fix dell'implementazione r005**,
stato IMPLEMENTATION_REVIEW / FIX_REVIEW_PROMPTS_READY. Consegna completa
ricevuta: F22–F26 completati secondo autore, chiusure/regressioni da valutare.
Nessuna review/arbitrato/GO r005 preesistente. Non è una review del piano.

- Snapshot: `temp/run-a002-fase1-corpus/snapshots/implementation-r005.json`.
- SHA256 snapshot: `b4fe0240533ad92e0acb65c355c01991c6d15f83abab2f4781e7d28f9653433e`.
- Impronta worktree: `0a99ca81113185b6ed48263b798e72bc2bf542c0a92d64d2bd2fb72033ccdd34`.
- 316 file di worktree e 47654 artefatti espliciti. I 20 PDF del
  corpus sono versionabili e inclusi nell'impronta generale.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto alla ricezione.
- Piano r001 SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`;
  errata SHA256 `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.
- Mandato: piano/errata/GO piano, prompt05/06/09/12/15 e addendum V, poi
  `arbitrations/arbitration-implementation-r004.md` e prompt18/F22–F26.
- Autore: `reports/implementation-r005.md` e `handovers/implementation-r005.md`;
  prevalgono le sezioni finali **Consegna completa** e **Chiusura amministrativa**.

Percorsi relativi della run sotto `temp/run-a002-fase1-corpus/`. Delta autore
**7 file**: scripts/benchmarks/observations.py e PROTOCOL.md,
tests/unit/test_conversion_fix_r005.py, docs/how-to/benchmark-fedelta.md,
docs/reference/benchmark-fedelta.md, nuovi report/dati permanenti r005.
metrics.py è byte-identico a r004; audit/safety/adapter/R/resources/PATH/
writer/registry/config/lock/pin/corpus sono invariati. Quattro documenti
generali del supervisore sono distinti in transition.json, senza fix prodotto.
Il delta include nuovi file non committati: `git diff base...HEAD` non basta.

Temp e documentation/runs sono esclusi dall'impronta generale, gli input
pertinenti sono vincolati come artefatti espliciti. Inventario
`evidence/supervisor-implementation-review-r005/review-inputs.json` per
navigare, senza obbligo di leggere tutti i log. Include prove/byte ricevuti,
report/CP autore, arbitrato/mandato, review r004 e prove r005. Registri vivi,
nuovi output reviewer, prompt19/20 e checkpoint corrente supervisore sono
esclusi: non alterano il comune. L'esclusione non consente di mutare input
ricevuti o ignorare codice e prove pertinenti.

## Letture e verifiche iniziali

1. AGENTS.md, temp/HANDOVER, parte iniziale PROJECT-CONTEXT, STATE/HANDOVER/
   brief e questo prompt; `handovers/supervisor-implementation-review-r005.md`.
2. Skill manage-implementation-run e verify-conversion-fidelity, protocollo
   run-lifecycle, template review, arbitrato r004 e prompt18 **integrali**.
3. **Entrambi i report r004 sono ammessi**: reviews/review-implementation-r004-chatgpt.md
   e reviews/review-implementation-r004-claude.md, checkpoint e sonde citate;
   sono input precedenti, non report concorrenti r005. Giri anteriori solo
   per requisiti/regressioni pertinenti, senza campagne a001 o nuova pianificazione.
4. Report e CP finale autore r005, final-checks001, PROTOCOL, due guide,
   `documentation/benchmarks/2026-10-09-a002-r005-diagnosi-html-e-domini.md`
   e relativo `-data.json`, receipt di ricezione/transizione del supervisore.
5. Intero delta di codice/test e dipendenze raggiunte: HTMLTree/tokenizer/
   errori/visit/inline/node/offset, audit e safety, raccolta risorse, scorer
   e disponibilità CER/WER, domini/counts/decomposizione, test r005 e regressioni.
   Lettura senza import implicito del prodotto o ML.
6. Piano/errata/ADR0001/0004/roadmap fase1/workflow benchmark/guida uv nei
   requisiti pertinenti. Non riaprire integralmente piano o fix già chiusi.

Verifica branch/HEAD/dev/merge-base/indice/delta, hash snapshot/piano/errata.
Esegui **prima e dopo** la review, registrando argv/cwd/output/exit:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r005
```

È manager stdlib del contesto, non test del prodotto. Non correggere input
o rigenerare il comune. STALE/identità diversa invalidano la review del
precedente oggetto: conserva il fatto e segnala il prossimo passo senza GO.
Vecchi snapshot/stage legano prove storiche e non devono tutti essere MATCH.

## Prove consegnate e limiti

Ricezione statica supervisore: 44067 confronti hash, 42288 input protetti
invariati, 168 input/copie prima-dopo consegna, 42 copie storiche ricevute,
108 input s004→s005 equivalenti e 10 origini installate coincidenti coi sorgenti.
Delta7, receipt e composizione controllati; nessuna nuova build/suite/R/
conversione/rescore/V del supervisore. Non è un GO semantico dei fix.
Alla ricezione 206 input esterni s005 corrispondono, zero differenze.
I tre pyc managed ricevuti restano conservati con nuovi gate; causa storica
non attribuita e nessuna sanatoria dei Chrome NONPASS non consumati.
Dopo ricezione i quattro documenti supervisore cambiano il worktree generale:
non trasferire il preflight integrale s005 a sonde nuove. Identifica input
realmente consumati e registra gate/differenze attuali, senza riscrivere receipt.

Prove sotto `evidence/implementation-r005/` e `work/implementation-r005/`:

- entry-verify/entry-git/received-bytes/protected-descriptors, package-inputs/
  managed-startup e snapshot managed-entry001: input prima del primo managed,
  provenienza/origini reali, copie congelate e controlli dopo le prove.
- Prima: work/before-gpt/probe-summary.json e before-claude-summary.json.
  Finali: work/after-gpt-002/probe-summary.json (16 casi),
  after-claude-summary-002.json (26: 25 con assert, A6 solo diagnostico).
  Nessun unmet; attesi originali preservati e attesi aggiornati motivati dal
  mandato: A3c positivo, A4/A5 nonPASS, B7/E2 INCOMPLETE. Non attribuire ad
  A6 un assert PASS assente. Gold sintetici PROVISIONAL e gate automatici
  distinti da qualità VERIFIED/R; confronto prima-dopo sul percorso reale.
- cli-probe-002/summary.json: nove casi e 18 exit run/evaluate **reali**,
  run.json/final presenti dove previsti, niente traceback. Writer **stub
  sintetico**: non prova di un motore nativo. Gold PROVISIONAL: positivo
  automatico può dare evaluate5, aggiunta certa3, bundle incompleto2.
- R-r005-tests-003: 652 mirati PASS (177 nuovi+475 precedenti),16F04 deselected,
  in16,20s sullo stage s004. R-r005-tests-fast-001:809PASS+310subtest in27,33s
  dopo gate finali s005. Suite prescritta tests/ con ignore API/packaging/
  Docker e smoke radice esclusi; collaudo parziale, nessun download riparatore.
- stage-s005/preparation/S/B/I/benchmark-inputs e wheel canonica reinstallata:
  managed3.12.13/dieci hash e origini, stesso interprete nei figli/R.
  s004-s005-consumed-equivalence.json:108 input helper/test equivalenti.
  budget.py host cambia per contabilità CPU e non è consumato dai workload;
  due report/dati nuovi aggiunti, nuovi gate s005. Verifica dipendenze reali,
  non un trasferimento di tutto il preflight vecchio. input-bytes-s004-before
  copre109file; delivery-before/after168 è consegna, non prima di ogni workload.
- contracts-002:10contratti conformi, H01–H08 R PASS; H12±RFAIL voluti,
  expected0 fissato prima/harness1/wrapper2. final-checks001:19receipt finali
  17PASS+2FAIL, binding prima/dopo/reap/socket/own_pending0. Un contratto che
  attende R FAIL non rende quella receipt R PASS.
- rescore-manifest002/rescore002:64nuovi ID r005b, due set32 ciascuno4PASS/
  28FAIL. engine_runs0 e nuovi giudizi V0; raw/source/gold/V e conversioni
  originali preservati. baseline003 è conversione r002 con misure derivate r005.
  derived-check002:64record, legami/somme/denominatori e distanza globale
  ricalcolati con matrice indipendente, zero differenze numeriche/gate da r004.
  Nuovi reader/ATX/hash non sono nuove conversioni. Prova r004 sui10casi/
  42completamenti riusata per identità metrics.py, non nuova prova runtime r005.
- Fallimenti gate iniziale001 (argv/path errato), diagnostici amministrativi,
  test001630PASS+10FAIL, test002640PASS prima dell'ultimo fix e vecchi R FAIL
  preservati. Intervalli commenti finali coperti da test003, sonde/CLI/contratti/
  rescore002 e suite finale. Non usare prove001 per il reader finale.
  Due copie s002 recuperate dopo SHA congelato MATCH, non ricostruzione
  inventata dei sei helper storici mancanti o sanatoria del gap12output.

## Criteri tecnici identici dei due reviewer

Review mirata F22–F26, intero delta7, dipendenze/regressioni. F19/F20/F05/F15
sono chiusi nel dominio dichiarato dell'arbitrato r004; F16 preservato64.
Residui HTML F13/F14/F09 vanno chiusi dal giro corrente. Non basta contare
test o ripetere soltanto le stringhe dei reviewer; prova i produttori e i
consumatori effettivi della nuova rappresentazione, con positivi discriminanti.

| Criterio | Verifiche sostanziali |
| --- | --- |
| C01 F22 diagnosi | Wrapper div/section/body/article bilanciati, annidati e vuoti positivi; stessi non chiusi e chiusure incoerenti, table/figure e raw fence HTML: diagnostica localizzata sopravvive alla proiezione e raggiunge audit/scorer/exit, INCOMPLETE/2 senza COMPLETE/PASS. Raw/coordinate/tipo corretti, foglie note in ordine, nessuna duplicazione testo/ref/math/unknown né perdita di risorse. Errori parser e copertura del tokenizer hanno responsabilità esplicite. |
| C02 F23 heading | h1–h6 a inizio riga, dentro wrapper e raw fence non passano dal prefisso ATX e non fanno crash. Positive equivalenze/aggiunte/math/ref/asset/Unicode byte-offset effettivi; run.json/final ed exit appropriati. Markdown ATX, note, liste e blockquote conservano la mappa corretta delle coordinate, niente offset locale spacciato per globale. |
| C03 F24 host inline | Heading HTML e code HTML rispettano il profilo inline; table/figure annidate conservano foglie/ref/math/raw/unknown e non si appiattiscono come supportate dando PASS. A4/A5 ora valutabili a runtime dopo F23; positivi inline leciti restano misurabili. Markdown code/fence letterali restano inerti, distinti da code nel tree HTML. |
| C04 F25 token | Commenti, declaration/unknown_decl e PI in testa/mezzo/coda, wrapper, cella, caption e residui hanno politica html-tokens-v1 visibile ai consumatori. Raw/intervalli esatti anche closure --!>/abrupt raggiungibili nel tokenizer locale, diagnostica unsupported coerente e INCOMPLETE; massimo end o raw presente non bastano. Nessuna prosa/token/risorsa silenziosamente persa o doppia; markup dentro commenti resta inerte, niente risorse attive inventate o fetch. Literal < resta positivo. |
| C05 F26 domini | text_lower_bound limita solo la distanza dei completamenti nel dominio globale char/token dichiarato. Non è bound di CER/WER completi/condizionati strutturali: può superarli (DOCX62contro60, EPUB54contro53, D1globale38char10parole contro strutturale0). PROTOCOL/guide/report/dati lo dicono, senza cambiare algoritmo/allineamento/normalizzazione/denominatori o imporre boundglobale≤strutturale. TRIVIAL_ZERO e motivi/versioni non diventano misura completa0/fedeltà. |
| C06 regressioni | Residui/caption/table/figure/unknown, errori di blocco/pagina/ordine/numero/cella/formula/asset/ref e aggiunte certe restano visibili; unknown non wildcard globale. CER/WER completi None quando non misurabili, decomp condizionata localizzabile e somme coerenti, bound nel proprio dominio. Positivi completi0 e negativi noti discriminano un comparatore che rifiuta tutto. Gold/config/matching non adattati alle perdite per fare PASS. |
| C07 misure e V | Identità/hash/raw/source/gold/V, versioni e legami osservazioni→audit→score→final effettivi sui64,4PASS/28FAIL per set, zero nuovi motori/V. TXT0/89,DOCX11/182caption,EPUB8/186,RTFcondizionato19/162 vs globale13/162,TeXcompletoNone/condizionato1/152/globale0. Checker indipendente e attestazioni circoscritte; caption/struttura non attribuite tutte a perdita materiale. V13/A03 sui riferimenti non firma output motori, L04 UNAVAILABLE distinto. |
| C08 prove/toolchain | Delta7 e108equivalenze/copie prima-dopo/protetti, ultimi byte del reader legati a test003/probe002/CLI002/contratti002/rescore002/s005; budget host separato. S/B/I/managed-entry/dieciorigini/installazione reali, stesso interprete figli/R, binding/reap/socket/preflight. 17PASS+2FAIL voluti preservati, vecchi stage stale non riattribuiti. Nessuna suite/API/Docker/ML universale rivendicata; helper fuori wheel hanno input/hash/provenienza. |
| C09 consegna/confini | Quota52h/entry/attese e riserva invariati; CPU lower/RSSsampled/disco conservativo non hard cap o certificazione hardware. Ledger chiusura storico, costi nuovi registrati. Fallimenti, ChromeNONPASS, helper12gap e L01/L02/L04/L05 non sanati. Report/guide/checkpoint coerenti, ruoli e Git manuali preservati; nessuna fase2/scelta motore prematura. |

Una soluzione reversibile non enumerata è ammissibile se chiude il requisito.
Un sintetico può esporre un difetto bloccante anche assente nei64raw. Unsupported
è ammissibile nel subset finito se resta visibile senza nascondere risorse,
foglie o errori e senza PASS spurio. Non richiedere parser/HTML universale,
nuovo schema applicativo o estensione fuori mandato. Per ogni rilievo collega
input ammesso, raggiungibilità, requisito, impatto e attribuzione (nuovo,
ereditato, mandato), fix minimo e prova di chiusura; non inventare rilievi.

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
Ledger autore chiusura132920,871316s (~36,922h), CPU8211,089263s lower,
retained1646699588B/RSSsampled740855808B **storici**, non saldo attuale.
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
Valuta F22–F26 e regressioni pertinenti, compresi i residui F13/F14/F09;
non un voto medio e non GO per numero di test o assenza dell'altro report.

Non modificare STATE/HANDOVER/eventi/indici/changelog/checkpoint supervisore,
snapshot ricevuti o file concorrenti. Nel checkpoint: input/output/esito,
costi/processi propri pendenti e prossimo ruolo **supervisore per ricezione
dei due report r005 e arbitrato**. Concludi in chat con esito e percorsi.
