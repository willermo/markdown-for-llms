# Arbitrato implementazione a002 r008 — NO_GO, consolidamento locale r009

2026-10-10 Europe/Rome. Supervisore Codex/OpenAI; deployment preciso/chatID
non esposti. Applicate manage-implementation-run e verify-conversion-fidelity.
Letti entrambi i report/checkpoint r008 integrali, evidenze pertinenti, tracce
e rami del reader/safety. Nessuna review delegata o simulata, nessuna esecuzione
di prodotto da parte del supervisore; verifiche stdlib/hash/manager/Git readonly.

**NO_GO** sul comune r008. Tre cause bloccanti, accolte in F39–F41;
F42 impone consolidamento, prove delle interazioni e documenti coerenti.
[Prompt30](../prompts/30-implementation-r009-consolidate-reader-boundaries.md)
per una **NUOVA chat implementatrice**, su richiesta/preferenza corrente
utente. Non nuovo piano, nuova run o reset dei contatori. Le due nuove review
arriveranno dopo consegna completa r009. [Nota sul metodo/modelli](../notes/recovery-method-and-models-r009.md).

## Identità e validità

[Comune r008](../../../../../temp/run-a002-fase1-corpus/snapshots/implementation-r008.json):SHA256
`2f6d690a8f3db7af8f192f78a9e6f5c0e3704b9757256f9dfdf72e35ee3b4152`,
impronta`448b854a88dc3dc883f7337f6fe037fd27846513e3a19104fc88529841e07eda`,
325file/76333artefatti,20PDF nella fingerprint generale. Feature
`feature/run-a002-corpus`;HEAD/dev/origin-dev locale/merge-base
`80c31da46e5caab990d3596c79c44e524ec1dc95`,indice vuoto alla ricezione.
MATCH prima/dopo per entrambi i reviewer e alla ricezione del supervisore;
nessuna query remota. [Ricezione](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-implementation-arbitration-r008/reception.json),
[tracce e risultati ricevuti](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-implementation-arbitration-r008/static-premises.json).

- [ChatGPT](../reviews/review-implementation-r008-chatgpt.md):NO_GO,
  GPT-R008-I001,quattro controesempi/due positivi con goldPROVISIONAL,
  R-probes-003 e attesi anteriori; provider OpenAI Codex basato suGPT-6,
  deployment esatto non esposto, nessun report concorrente letto.
- [Claude](../reviews/review-implementation-r008-claude.md):NO_GO,
  CLA-R008-I001/I002, confronto runtime con reader r007 conservato;
  dichiarato Claude Opus5.5/Anthropic in chat nuova indipendente.
- ReceiptR PASS attestano la raccolta e gli esiti registrati, non che tutti
  i casi prodotto siano conformi. Il supervisore legge le prove, non le replica.

## Decisioni su tutti i rilievi

| Rilievo | Decisione e attribuzione | Impatto ed evidenza | Correzione e criterio di chiusura |
| --- | --- | --- | --- |
| CLA-R008-I001 | Accolto, media, **bloccante F39**, regressione r008 della collocazione PassiveHTML | html_nodes/HTMLTree: errore duplicati/depth interrompe feed; aperture successive non validate. D1/D2/D3/D6 runtime diventano UNSUPPORTED invece di UNSAFE, r007 UNSAFE. Nessun file/fetch eseguito; INCOMPLETE/exit2 invariati. È perdita della classificazione richiesta C04 e guardia incompleta, non prova di evasione riuscita. | Ogni apertura reale fuori dai token inerti viene validata anche se la proiezione semantica fallisce. Stessa semantica lessicale, non html(raw) ingenuo che riattiva attributi/commenti di F36. D1/D2/D3/D6 UNSAFE con detail; positivi e D4/D5 preservati; token-body mai risorsa, precedenza sicurezza documentata. |
| CLA-R008-I002 | Accolto, media, **bloccante F40**, regressione r008 incomplete_tag_token | :107–122 attribuisce quote a ogni apice; il tokenizer3.12.13 riconosce invece p title=it's e l’img successivo. D7 perde UNSAFE; variante sicura perde image ref/apertura e testo. Input ammesso nel profilo reale; nessun falsoPASS/rete, ma perdita contenuto e guardia. | Distinguere unquoted/quoted values coerentemente con tokenizer fissato; incompleteness solo reale. D7 UNSAFE, variante p.png conserva ref/testo/asset, mentre <alpha prima di token F36 resta prefisso localeunsupported/opaco. Quote letterali nei valori e token dentro attributi non devono divergere fra scanner/safety/proiezione. |
| GPT-R008-I001 | Accolto, media, **bloccante F41**, residuo F37 (consumer/default ereditato non riallineato) | Scanner usa dominio fisico producer, read_parts strippa quote/list/note e inline ricostruisce inline_scope_end: '- ' nel code soft-line diventa barriera. Quattro controesempi runtime perdono link e producono falsoFAIL/score3 con COMPLETE, contro due positiviPASS/score5. | Consumer riceve dominio/intervallo deciso dal producer e mappa sorgente→logico coerente; niente nuova inferenza dal testo trasformato. Quote/list/note/Unicode e varianti link/image/tick/marker: testo/ref/ordine/offset esatti,COMPLETE/automaticPASS/evaluate5PROVISIONAL. Negativi fuori dominio e token realmente opachi preservati. |

Non demoto I001 solo perché INCOMPLETE impedisce il PASS: C04 richiede tutte
le aperture reali validate, e la regressione cambia le diagnosi. I002 perde
anche un asset reale. Non sono nuove pretese di HTML universale. F41 resta
aperto nonostante i positivi Claude: i due report coprono input diversi,
il controesempio ChatGPT è raggiungibile e confuta la chiusura generale.

## Cosa si chiude e cosa si preserva

F35 chiuso nel perimetro revisionato:progressione scanner verificata daClaude,
T1 e7varianti runtime;prima3TIMEOUT r007 e dopo3PASS autore conservati.
F36 chiuso nei domini verificati, token/prefisso locale e positivo asset dopo
token conformi. F37 caso originario migliorato, **parziale viaF41**.
F38 parziale: documenti dicono tutte le aperture guardate; dopo F39–F41
rettificarli secondo prove reali. F32/F27/F30/F28locale/F26/F16 preservati.

64misure/19campi invarianti, osservazioni salvoIDreader coerenti;4PASS/28FAIL
per set32. Corpus/gold/raw/V13/A03 invariati,L04formuleoutputUNAVAILABLE distinta,
L01/L02/L05 eChromeNONPASS/helper12gap/H15/E9 storico non sanati.
S007 STALE solo quattro documenti conservato, gate reviewer circoscritti reali.
Preflightfail/EPERM/attesi/checker reviewer conservati, non difetti prodotto.
Le due osservazioni statiche Claude rawLaTeX-token e costo quadratico restano
suggerimenti/limiti dichiarati, non nuovi blocchi: non imporre parserTeX o
ottimizzazione generale. Se il consolidamento tocca quei rami, verificarne
le regressioni pertinenti; non cancellare le note precedenti.

## Cambio di metodo entro lo stesso risultato

La sequenza mostra interpretazioni discordanti tra scanner/producer/consumer/
safety; altre condizioni speciali sui payload non danno fiducia sufficiente.
F42 richiede una mappa breve delle decisioni duplicate prima del codice,
invarianti condivise/dati di dominio espliciti e prove finite su interazioni
con oracoli indipendenti. Scelta del mezzo reversibile all’autore, incluso
refactoring locale; niente nuova libreria/download o nuova architettura.
È consolidamento tecnico nei criteri approvati, non cambio del piano/gold/
metrica/perimetro. La nota tecnica fa parte del report, non ritorno di ruolo.

Richiesta utente: contesti freschi per implementazione e supervisione,
valutare modelli/effort per costo. Disposizione di ruolo: nuova chat autrice
r009 e [handover supervisione31](../prompts/31-supervisor-handover-a002-r009.md).
Compattazione come fattore plausibile, causalità non provata. Raccomandazione
Sol/xhigh solo implementazione,Sol/high supervisore/reviewerGPT eClaude
corrente; opzione Astra/high solo autore. Nessuna selezione o aumento
automatico: configurazione scelta dall’utente, mandatovalido ancheSol/high.

## Risorse e prosecuzione

[Disposizione24h](addendum-resources-r008-r001.md) effettiva,entry originale
2026-10-08T07:06:54.369068Z e contatori invariati. Stop nuovi workload
2026-10-11T06:37:31Z (08:37:31Europe/Rome),deadline09:37:31Z (11:37:31),
riserva3h. CPU21600/stop19800s,disco6/stop5GiB,RAM8/stop sampled6GiB,
riserva1800CPU-s/1GiB,R≤900s/1MiBstream/32MiBfile,rete0/€0benchmark.
Accounting della ricezione include entrambi i reviewer senza sommare wall
parallele; CPU propria supervisore non integralmente misurata, nessun upper.
Nessuna nuova quota, tempi non ripartono in nuove chat.

Contesto tecnico fix-r008 dopo soli quattro documenti generali; report/review
storici e byte r008 preservati. Mandato30completo e contesti freschi,poi
nuova consegna/comune/due review indipendenti/arbitrato. Nessun GO futuro,
Git automatico/cleanup/nuova V/motore/ML o fase2.
