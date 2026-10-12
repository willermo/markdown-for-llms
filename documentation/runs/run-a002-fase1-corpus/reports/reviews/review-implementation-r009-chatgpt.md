# Review — implementazione — run-a002-fase1-corpus — r009 — ChatGPT

- Autore: reviewer richiesto ChatGPT; esecutore effettivo OpenAI Codex, basato
  su GPT-6 secondo l'identità esposta. Deployment preciso, effort effettivo e
  chat ID non esposti. Nessuna sostituzione dichiarata, delega o subagente.
- Prompt: [32](../prompts/32-review-implementation-r009-chatgpt.md), SHA256
  `0a3499239702806b9b92f207efddd81ec8dcef43f3c52234cb6c24d8fce0f4fc`.
- Oggetto: r009, fix F39–F42 dopo arbitrato r008 NO_GO; stato ricevuto
  IMPLEMENTATION_REVIEW / FIX_REVIEW_PROMPTS_READY. **Esito: NO_GO**.
- Comune: [implementation-r009](../../../../../temp/run-a002-fase1-corpus/snapshots/implementation-r009.json), SHA256
  `bfc26ebc0db5946f30a0a47b6aa3a558674a167eb1c022dce1caa39fb6a03b9f`;
  worktree `1c5d94c7db9f4a8508143a3bd1e7b72f559dfcaa6bb9909602ac12ebee20eb66`;
  328 file generali / 91038 artefatti espliciti, 20 PDF generali.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto prima/dopo.
  Considerati anche file nuovi e modificati non committati.
- Nuova chat reviewer indipendente; letti entrambi i report/checkpoint r008
  come storia comune. Nessun report, evidenza o arbitrato concorrente r009
  letto; nessuna scrittura a prodotto, input altrui o registri condivisi.
- Delta autore: sette file, reader/PROTOCOL/test r009/due guide/report e dati
  permanenti; quattro documenti generali del supervisore separati. Reader
  `observations-v9-reader-boundaries`, SHA256
  `a8ffa9eb44cf1695ca2e7e611c91ef8c7594605fe38a3bb194811f056cdbc737`.
  Scorer v4, safety, audit, adapter, R, corpus/gold/raw/V13/A03 invariati.
  La collocazione della nuova guardia nel reader è inclusa nel giudizio.

## Ambito e prove

Le prove proprie sono in
[evidence/review-implementation-r009-chatgpt](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r009-chatgpt).
Letti istruzioni e contesto corrente, handover del supervisore, skill
manage-implementation-run e verify-conversion-fidelity, protocollo/template,
arbitrato r008 e mandato30 integrali, disposizione24h e fonte utente, nota
metodo. Letti integralmente entrambi i report r008 e report/checkpoint autore
r009; consultati finding GPT-I001, casi Claude D1–D7/safe e loro ricevute.
Letti report permanente r009 e dati64, delivery-check002, manifest47 e receipt
di consegna, budget/processi e transizione del comune. Consultati ADR0001,
indici/registro ADR/roadmap fase1, piano/errata e mandati05/06 nei contratti
pertinenti, workflow benchmark, guida uv/toolchain.

Reader1040righe letto integralmente e confrontato con il v8 fisico; delta
in `reader-delta.stdout` (diff exit1 atteso). Letti test r009 e regressioni
r003–r008 pertinenti, PROTOCOL e sezioni correnti delle due guide. Seguiti
audit→observe_bundle, PassiveHTML/safety, metrics/allineamento/scoring,
validatore adapter e CLI raggiungibili. Letti collector matrix.py, attesi
manuali, matrice before-corrected001/after003, CLI002 e stdout targeted003/
fast004, contratti002/H12 e rescore/derived-check001 pertinenti. Nessuna
lettura indiscriminata dei payload ambienti o della storia a001.

### Identità, installazione e confini delle prove

Comando proprio prima e dopo, cwd `/home/davide/workarea/markdown-for-llms`:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r009
```

[verify-before.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r009-chatgpt/verify-before.json)
e [verify-after-002.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r009-chatgpt/verify-after-002.json)
conservano argv/cwd/stdout/stderr/exit/costo. **MATCH/exit0** prima e dopo;
anche il controllo intermedio `verify-after.json` è MATCH. Manager host
stdlib, non prova prodotto. Git in `git-identity*.json`, `git-status-after.json`,
`git-index*.json`, `git-mergebase-after.json`; `git-diff-check.json` exit0.

Il S006 generale storico è **STALE/exit2**, conservato in
`verify-s006-stale.json`: soltanto CHANGELOG, indice documentation, roadmap
e next-run del supervisore differiscono. Nessun ripristino o PASS generale
trasferito. Prima del primo managed, `managed-before-first-import.json`
verifica 5193 descrittori, origine locale3.12.13, startup, archivi e dieci
origini sorgente=wheel canonica=installato. Gate propri correnti da5214 a5227
input, `consumed-inputs-00*.json`, target/preflight omonimi; controllo
origine, `-I -B`, prefix/base-prefix, `.pth` e dieci moduli prima delle sonde.
Nessuna build/reinstallazione eseguita dal reviewer.

Snapshot tecnici propri, stessa impronta comune: `review-r009-chatgpt-probes-001`,
`review-r009-chatgpt-oracle-001`, `review-r009-chatgpt-inert-tail-001/002`.
Per la prova decisiva finale, [inert-tail-002](../../../../../temp/run-a002-fase1-corpus/snapshots/review-r009-chatgpt-inert-tail-002.json)
SHA256 `eaad1bb42ab83f312e75a1ebd13200f9d91f286d8bd70a624e4f7389f39406d0`.
Copie fisiche prima/dopo per gli oracle e le sonde decisive, confrontate con
i byte caricati. Per il primo gruppo26, copie fisiche soltanto **dopo**:
gli input anteriori sono fissati da manifest/snapshot/gate e binding R;
non rivendico copie-before retroattive. Dettagli in `copies-after-003.json`.

### Prove autore controllate, senza riesecuzione

[static-audit.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r009-chatgpt/static-audit.json)
PASS: 4102 confronti hash / 2217 path unici. Controllati 2399 descrittori di
copie helper prima/dopo, 561 descrittori runtime finali e47input della consegna.
18receipt finali selezionate:16PASS+2FAILH12, binding registrato/byte invariati/
reap. H12± conservano harness1, expectedR0, wrapper2; nessuna sanatoria.

Matrice autore: identici raw/oracoli308 prima/dopo;178/308v8→308/308v9,
classi89unsafe/12inert-error/56safe-attribute/122domain/5foreign/24token.
35fixture CLI e70exit OS confrontati con manifest e output reali; writer
sintetico dichiarato, nessun motore. Risultati ricevuti1712mirati/16deselected,
1869fast+310subtest,339test r009: **non suite proprie**.

64misure: controllati catena/source/gold/audit/observations/raw/asset/final,
19campi e gate invarianti, aritmetica/denominatori; ricalcolate con routine
stdlib indipendente le distanze Levenshtein dei token e dei flussi globali.
Non ricostruito indipendentemente tutto l'allineamento.4PASS/28FAIL per set32;
56AVAILABLE/EXACT_GLOBAL_DISTANCE,8UNAVAILABLE/TRIVIAL_ZERO. Nel permanente
final=r009, conversion_final/run originali, previous_derived_final/old_*=r008;
summary originale mantiene final storico. Counts condizionati, bound globale
e CER/WER completi distinti; TRIVIAL_ZERO non fullPASS. V13/A03 qualifica il
riferimento, L04outputUNAVAILABLE distinto; L01PDF/OCR e L02compileNOT_MEASURED.

Gli errori autore EPERM/matrice292/CLI001/H01label-parent/deliverychecker-link/
docwrappercollision/H12 sono storia conservata, non prove verdi riscritte.
Nuova308 usa gli stessi raw/oracoli prima/dopo; la revisione delle continuazioni
paragraph corregge la classificazione del trasporto, senza mutare gold canonico
o scorer. Limiti ChromeNONPASS/gap12helper/H15/E9 non chiusi da questa review.

### Sonde proprie

Ogni collector usa gli stessi CPython managed3.12.13 e R/Firejail `--net=none`,
ambiente chiuso e `shell=False`; timeout15s, expected collector0. Argv reali
in `R-*.json/.stdout` e ricevute `R-probes-*/receipt.json`. Attesi manuali
fissati prima, audit→observe→score, goldPROVISIONAL, nessuna V o motore.
**R PASS certifica l'esecuzione del collector, non la conformità del reader**:
i mismatch sono dati conservati e determinano il giudizio.

| Gruppo | Evidenza | Risultato prodotto |
| --- | --- | --- |
| 26sonde guardia/attributi/domìni/mappe | `probe-results-002/summary.json`, `R-probes-002` | 26/26 conformi; recursion/enfasi/link-image, backtick1/2, CRLF/Unicode, seconda reference/code, quote/list/note-tab, token veri e foreign-close |
| 8oracle positivi attributi completi | `oracle-results-001/summary.json`, `R-probes-oracle-001` | 8/8 conformi; tokenizer HTMLParser indipendente sui piccoli input, valori con comment/CDATA/PI/declaration/script-looking bytes, testo/ref/pixelasset/gate/scorer |
| 7sonde HTML inline→code, v9 e v8 fisico | `inert-tail-results-002/summary.json`, `R-probes-inert-tail-002` | 3/7 conformi per reader; 4falsi unsafe per ciascuno, findingI001; controlli code-alone/code-before-HTML e unsafe reale discriminanti |

La prima R delle26 è IMPEDITA per EPERM sul socket sintetico, senza workload;
receipt conservata. Escalation della stessa azione autorizzata approvata,
R/rete0 mantenuti, nessun rifiuto auto-review. Conservati due errori dei miei
checker stdlib: canonical digest con LF aggiunto e schema copia string/dict;
corretti in nuove versioni e rieseguiti, senza effetti sui byte ricevuti.
La prima sonda quote aveva gold.type `blockquote`: il contratto usa paragraph
con block_syntax blockquote. Versione001 preservata; corretto soltanto quel
tipo nel gold002 e ripetuto il gruppo con nuovo snapshot/gate/copie/output.
Raw/testi degli input invariati; i tre casi paragraph erano già validi.

## Rilievi

| ID | Severità e blocco | Posizione | Input ammesso/percorso | Attribuzione e requisito | Problema e impatto | Evidenza | Chiusura richiesta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-R009-I001 | Alta; sì | observations.py:580–590, HTMLTree/validate_html; audit.py:22; benchmark.py:50 | Paragrafo con span HTML bilanciato seguito da inline code; anche quote semplice | Ereditato v8, confermato v9; applicazione della guardia modificata da r009. C01/C05/F39 e contratto codice letterale | L'intera coda Markdown viene data al parser HTML: un tag nei backtick diventa apertura attiva. Falso INCOMPLETE/UNSAFE_REFERENCE, automatic/qualityFAIL, score3; osservazione ripiegata su unknown dell'intero documento | Pipeline propria su entrambi i reader, 4input e3controlli; summary finale, raw/expected/audit/observations/metrics e R corrente | Limitare parser/proiezione/guardia al frammento HTML effettivo, rispettando domini e atomi Markdown inerti; mantenere validazione di tutte le aperture HTML reali dopo errori strutturali. Positivi letterali completi e negativo reale, testo/ref/offset/metriche/exit discriminanti |

### GPT-R009-I001 — HTML nella coda code interpretato fuori dal proprio dominio

Input completo minimo della sonda `span-before-traversal-code`:

````markdown
Known <span>words.</span> `<img src="../outside.png"/>`

# Visible heading

After words.
````

È nel profilo finito: span inline passivo ammesso e codice Markdown letterale
inerte (PROTOCOL270–275/341/383–384). Atteso: tre blocchi, primo testo
`Known words. <img src="../outside.png"/>`, zero reference/asset, auditCOMPLETE,
automaticPASS, qualitàINDETERMINATE e score5 con goldPROVISIONAL.

Osservato v9: auditINCOMPLETE/UNSAFE_REFERENCE, detail
`Path evasivo: '../outside.png'`, automatic/qualityFAIL, score3; observe_bundle
cattura Unsafe e restituisce un solo unknown contenente l'intero raw.
Il raw è conservato, ma heading/paragrafo non rimangono osservazioni tipizzate.
Identico sul v8 fisico: **non una regressione introdotta da r009**. Non è
richiesta di parser universale o estensione fuori profilo: è un difetto
essenziale raggiungibile nella guardia/trasporto esplicitamente revisionati.

HTMLTree riceve `raw[i:]` prima di selezionare il primo sottoalbero bilanciato;
la traversata raggiunge l'img nei backtick e solleva Unsafe. L'except ValueError
porta al fallback `validate_html(raw[i:])`, nuovo in r009, che riattiva gli
stessi byte letterali. Varianti `<script>x</script>` e `<b onclick="x">literal</b>`
dentro code producono lo stesso falso unsafe; trasporto quote confermato nel
gruppo002. Code senza span precedente e code **prima** dello span sono COMPLETE,
testo esatto, automaticPASS/score5. L'img evasivo fuori code resta correttamente
UNSAFE_REFERENCE/score3: correggere il falso unsafe non deve nascondere quello
reale. Nessun path evasivo viene aperto dal codice audit, che termina prima
delle letture asset; nessun fetch. Non rieseguita la CLI su questi input:
run/evaluate2 è conseguenza statica del contratto operativo, non un exit OS
proprio dichiarato.

Correzione minima richiesta per obiettivo: rendere coerente il limite del
frammento e l'inertness lungo selezione del sottoalbero, proiezione e guardia,
senza interrompere la visita delle aperture **reali** per duplicate/depth/cap.
Aggiornare matrice e descrizione della copertura con questa interazione finita;
non serve aggiungere motori, V, scorer o normalizzazioni.

## Motivazione dell'esito e limiti

| Criterio | Valutazione |
| --- | --- |
| C01/F39 | D1/D2/D3/D6 e precedenza unsafe dopo errori risultano corretti nei casi ricevuti e propri. Chiusura generale incompleta: I001 inventa aperture dentro code. |
| C02/F40 | Chiusura supportata nel perimetro: D7/safe, matcher3.12.13 condivisi, quoted/bare e token reale/EOF; otto oracle propri positivi conformi. |
| C03/F41 | Chiusura supportata nei domini ammessi e controllati: GPT-exact, trasporti, ricorsioni, source map char/UTF8 e foreign-close, contenuti/ref/asset effettivi. |
| C04/F42 | Nota tecnica anteriore e consolidamento/matrice/attesi verificabili; manca l'interazione HTML inline→code che I001 rende decisiva. Nessun expected autore derivato dal reader rilevato. |
| C05 | NON soddisfatto: falso FAIL su input ammesso e perdita della vista tipizzata. Nessuna riapertura F28/F29 globali o costo/rawTeX fuori scope. |
| C06 | Storia/errori e limiti conservati; descrizione della guardia da rettificare con I001, senza cancellare i risultati reali308. |
| C07 | Catena64/19campi/aritmetica/distanze e distinzioni V/L04 rispettate nei controlli eseguiti. Counts4/28 osservati, non qualità imposta. |
| C08 | Identità comune e gate correnti dei byte consumati confermati; S006 generale STALE dichiarato, installazione canonica invariata. Copie/limiti propri esplicitati. |
| C09 | Quote e accounting lower/sample/scoped rispettati dalle misure visibili; nessuna nuova acquisizione/servizio/€0. |
| C10 | Review indipendente sul comune valido; nessuna delega, modifica prodotto/Git/stato, nuova fase o approvazione fittizia. |

**NO_GO per GPT-R009-I001**, non per il numero dei test o l'assenza dell'altro
report. F40/F41 hanno prove favorevoli; F39 e la copertura F42 non sono chiusi
nel requisito essenziale di validare solo aperture attive. Nessuna suite,
build, campagnaH/a001, conversione32, engine/render/browser/M/V rieseguita.
I 64output reali non esercitano questo caso sintetico: invarianti numeriche
e corpus stabili non ne dimostrano l'assenza. Nessuna certificazione del
parser universale, della nuova applicazione o del completamento fase1.

Costi, disco e processi in `budget-final-002.json`/`checks-final-002.json`,
con saldo aggiornato in `delivery-budget.json`. Entry originale
2026-10-08T07:06:54.369068Z; disposizione24h ancorata a09:37:31Z del10ottobre,
nessun reset. Stop workload11ottobre06:37:31Z, deadline09:37:31Z, riserva3h;
CPUstop19800/max21600, disco5/6GiB, RSSsampled6/8GiB. CPU ricevuta
12864.569662s lower,45sClaude r007 approssimati separati. Recorder propri
includono figli senza sommare di nuovo R annidate; amministrazione/letture/
assembly non tutta misurata. Disk retained attuale circoscritto, non sommato
al retained storico dello stesso scope né upper cacheOS. RSS dei workload
propri massimo sampled30474240B, storico749760512B, non picco RAM certificato.
R≤900s/stream1MiB/file32MiB, rete/acquisizioni/invii/servizi0/€0.
Tutte le sonde terminate, binding MATCH, figli reaped e socket puliti;
nessun processo workload proprio pendente.

Prossimo ruolo: **supervisore**, per ricezione di entrambi i report e arbitrato
sul comune r009. Nessuna modifica automatica dello stato o indicazione di
commit/merge prima di un eventuale GO successivo. Checkpoint proprio:
[handover reviewer](../handovers/review-implementation-r009-chatgpt.md).
