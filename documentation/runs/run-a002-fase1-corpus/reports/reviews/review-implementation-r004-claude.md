# Review — implementazione — run-a002-fase1-corpus — r004 — Claude

- Autore/provider/modello e chat: reviewer Claude, provider Anthropic, modello Claude Opus 5.5
  (`claude-opus-5-5`, contesto 1M), tramite l'estensione Claude Code per VS Code. ID chat non
  esposto. Chat nuova; reviewer previsto Claude, nessuna sostituzione, nessun agente delegato.
- Prompt di origine: [prompt17](../prompts/17-review-implementation-r004-claude.md),
  SHA256 `64a1517c0469d917ffa03c4a07b1121e4b0b7fa02470bf3a4bcfd3c2be6129c0`.
- Oggetto e revisione esatta: fix r004 dell'implementazione a002 (prompt15, F17–F21), con
  l'intero delta di 8 file, le dipendenze e le regressioni. Piano r001 `3ac23bce…dd4ab8` ed
  errata `a09cbb8b…e0dfd9` verificati per hash. Report autore `reports/implementation-r004.md`
  e checkpoint `handovers/implementation-r004.md` (prevale la sezione finale «Consegna completa»).
- Snapshot, HEAD e impronta: [implementation-r004.json](../../../../../temp/run-a002-fase1-corpus/snapshots/implementation-r004.json),
  SHA256 `7105ec0e91e0783a26e13623ecf9eff37211291323253e39e7dc815e92d768c7`, impronta
  `3976b26937e2f80ee03fb089e7e18a4fd8782edd96c0f821536c367a65064edf`, 313 file e 41997
  artefatti. Verify **prima** MATCH/exit 0, verify **dopo** MATCH/exit 0
  ([entry](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-claude/verify-entry.txt),
  [after](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-claude/verify-after.txt)).
- Indipendenza: non ho letto la review ChatGPT r004, né le sue evidenze o il suo checkpoint.
  Il file `reviews/review-implementation-r004-chatgpt.md` è comparso durante la mia review e
  non l'ho aperto. Nessun arbitrato r004 esiste. Ho letto, come ammesso, i due report r003,
  l'arbitrato r003 e le mie evidenze r003.
- Giro: fix r004, mirato a F17–F21, ai residui F01/F05/F09/F13–F16, al delta e alle regressioni.
- Base e delta: branch `feature/run-a002-corpus`; HEAD, `dev`, `origin/dev` locale e merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto. Confronto fra
  `implementation-fix-context-r003.json` e il comune ([delta-check](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-claude/delta-check.txt)):
  cambiano `observations.py`, `metrics.py`, `PROTOCOL.md`, le due guide; sono nuovi
  `tests/unit/test_conversion_fix_r004.py` e il report/dati permanenti r004 (8 file autore);
  i 4 documenti generali del supervisore sono distinti; 38018 artefatti precedenti invariati.
  `audit.py`, safety, adapter, R, writer, config, lock e corpus invariati.
- Esito: **NO_GO**.

## Ambito e prove

Skill usate: `manage-implementation-run` e `verify-conversion-fidelity`.

**Letture.** AGENTS, `temp/HANDOVER`, parte corrente di PROJECT-CONTEXT, HANDOVER/brief della
run, prompt17 e checkpoint del supervisore r004; le due skill e il template review;
arbitrato r003 e prompt15 integrali; mio checkpoint/evidenze r003; report autore r004;
PROTOCOL e due guide (sezioni r003/r004); report permanente r004 e dati; summary rescore,
completion-check, derived-check, final-checks, contracts-001. Codice letto per intero:
`observations.py`, `metrics.py`, `audit.py`; `safety.html/local_uri/PassiveHTML`,
`benchmark.py` (run/observe/score/evaluate), `test_conversion_fix_r004.py`. Confronto
statico di `html/parser.py` managed 3.12.13 con quello di sistema (differisce solo il buffering).

**Controllo indipendente dei 64 score** ([script](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-claude/own_rescore_check_r004.py),
[risultati](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-claude/own-rescore-check-r004.json)), stdlib,
senza importare reader/scorer/checker autore. Su 64/64: hash dei file legati, raw identici
ai root r002, legami raw→osservazioni→audit, versioni `observations-v4-html-coverage` e
`metrics-v4-conditional-and-bounds`, edit a parole delle entry, somme di categoria e
denominatori, `conditional_text_counts` = totals con `guaranteed_bound=false`, quality/gate/
exit/CER/WER/totals/order/pages/blocks identici a r003. Failures identici a meno del nuovo
campo `policy` sulle caption DOCX/RTF. Il bound globale pubblicato coincide su 64/64 con il mio
ricalcolo (matrice di edit propria, flussi ricostruiti da gold e osservazioni). Per formato:
TXT 0/89, DOCX 11/182 (insertion 11), EPUB 8/186 (omission 8), RTF 19/162 = 11+3+5 con bound
13/162, TeX 1/152 condizionato con bound 0 e WER completo None. Per set 4 PASS e 28 FAIL,
`engine_runs` 0.

**Gate della sonda** ([gate](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-claude/gate)): hash propri
prima/dopo su package-inputs s008 (3813 file, inclusi i tre pyc encodings ricevuti), startup
audit, 206 input esterni, I/B e dieci moduli installati uguali ai sorgenti: tutto PASS.
`check_preflight` S/I s008 → **FAIL STALE**, conservato. Diagnosi: cambiano solo quattro
documenti del supervisore, non consumati. Inventory verify s008 → PASS. Ho quindi legato le
sonde a due **nuovi snapshot tecnici propri** (`review-r004-claude-probe-001/002`), con impronta
uguale al comune, MATCH prima/dopo. Non ho trasferito il PASS s008.

**Sonde R proprie** (interprete managed CPython 3.12.13 della venv s008, firejail `--net=none`,
registry/shell=False, solo `scripts/benchmarks` dal percorso `--modules`). Gli attesi sono
stati fissati prima dell'esecuzione nel file della sonda.
- `R-probe-001`: R **FAIL** exit 2 per traceback della sonda al caso A3; binding MATCH,
  input invariati. Conservato.
- `R-probe-002` (stessi casi, eccezioni registrate per fase): R **PASS**, exit 0
  ([risultati](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-claude/probe-results-002.json)).
- Gold sintetici propri PROVISIONAL. Exit di evaluate derivati dalle regole, non conversioni.

| Caso | Input | Atteso | Osservato |
| --- | --- | --- | --- |
| A1 | heading Markdown con `<table>` inline e testo aggiunto | non PASS | FAIL3 |
| A2 | `<p>` HTML con figure e img assente | INCOMPLETE/2 | INCOMPLETE/2 |
| B0/B5 | table e figure positive | PASS | PASS/0 |
| B1 | `<p>` testo diretto sotto table | FAIL3 html_residual | FAIL3 html_residual |
| B1b | `tfoot` | non PASS | INCOMPLETE (validator) |
| B2 | img assente diretta in tbody | INCOMPLETE/2 | INCOMPLETE/2 |
| B3 | table annidata sotto table | FAIL3 | FAIL3 |
| B4 | figure con due figcaption, img assente | INCOMPLETE/2 | INCOMPLETE/2 |
| B6 | residuo solo whitespace | non PASS | INDETERMINATE/5 |
| C1 | caption pipe aggiunta, gold senza caption | FAIL3 table_caption | FAIL3 |
| C2 | figcaption vuota, gold con caption | FAIL3 | FAIL3 |
| D1 | due paragrafi noti invertiti | ordine FAIL, completo 0 | completo 0, bound 38 car./10 parole, CER bound 0,667 |
| D2 | paragrafo omesso | completo 26 | completo 26, bound 27 |
| D3 | omissione più unknown altrove | bound TRIVIAL_ZERO, FAIL | TRIVIAL_ZERO, FAIL3 |
| E1 | `<=`, `<-`, `3 < 4` | PASS, COMPLETE | PASS |
| E3 | commento top-level | INCOMPLETE | INCOMPLETE/2 |
| A3, A3a, A3b, A3c, A4, A5 | `<h1>`/`<h2>`/`<h3>` HTML, semplici, in `<div>`, in fence `{=html}` | misura o unsupported | **TypeError** in observe_bundle e in audit |

**Non eseguito, con motivo.** Nessuna suite, build, reinstallazione, contratto H, conversione,
V o rescore propri: il mandato li esclude e le receipt autore sono state verificate per hash e
contenuto (`R-r004-tests-005` 475 PASS, `R-r004-tests-fast-001` 632 PASS + 310 subtest,
binding MATCH, input invariati; 20 receipt finali 18 PASS + 2 FAIL corrispondenti agli hash
delle receipt H12; contracts-001 `all_expected` true).

## Rilievi

| ID | Severità e blocco sì/no | Posizione | Raggiungibilità/input ammessi | Attribuzione e requisito | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLA-R004-I001 | Media, **bloccante** | `observations.py:389–390` (h1–h6 HTML → `heading`), `:411` e `:416` (prefisso Markdown `#` obbligatorio per ogni `heading`), chiamate `:460` e `:507` | Qualunque heading HTML `<h1>`–`<h6>` del profilo «HTML passivo»: blocco a inizio riga, sotto `div/section/article/body`, oppure in fence `{=html}`. Input ammesso dal validator e dal visitor | Introdotto in r003 (mappa prefissi F13), ereditato invariato in r004 in un file del delta8 e lungo il percorso HTML producer→node di F17/F18. Viola la consegna di un subset HTML finito «verificabile» e la regola produttore→consumatori: ogni foglia arriva al dominio o diventa unsupported localizzato | `re.match('^ {0,3}#{1,6}[ \t]+', …)` restituisce None sul raw `<h1>…` e solleva TypeError. Né observe_bundle né audit lo intercettano (solo Unsafe/UnicodeError). Il CLI restituisce exit 4: run.json manca, evaluate fallisce con «Run receipt missing». Nessun falso PASS, ma l'output non è misurabile e un difetto del misuratore diventa «internal failure» del tentativo. Nessuno dei 64 raw contiene HTML: cifre invariate | Runtime R-probe-002: A3a `<h1>Titolo</h1>`, A3b `<div><h2>…`, A3c fence `{=html}` `<h3>`, A3/A4/A5 → TypeError in observe_bundle e in audit indipendente. R-probe-001 interrotto allo stesso punto. Nessun test copre `<h1>`–`<h6>` | Calcolare i prefissi Markdown solo per heading di sintassi Markdown (es. `block_syntax='atx'` o raw che inizia con `#`). Nessun prefisso per heading HTML. Test: A3a/A3b/A3c misurati (positivo PASS, testo aggiunto FAIL), offset/`source_position` corretti, audit COMPLETE/INCOMPLETE secondo le risorse. Regressioni: heading/note/list/blockquote Markdown invariati. Chiudere insieme CLA-R004-I002 |
| CLA-R004-I002 | Bassa, non bloccante da solo; **da chiudere con I001** | `observations.py:278` (`child_inline` esclude h1–h6 e `code`), `:309` | Diventa raggiungibile appena I001 è corretto. Esempi: `<h1>Titolo <table><tr><td>99</td></tr></table></h1>` contro gold heading «Titolo 99»; analogo con `<figure>` | Nuovo nel contratto F17 (html-inline-v1). Le guide estendono il vincolo a celle, caption, paragrafi e wrapper inline. Gli heading HTML ammettono solo contenuto phrasing, come `<p>`, ma non sono host | Table/figure dentro un heading HTML vengono appiattite nel testo senza unsupported: se il testo coincide, PASS automatico con struttura inventata. Foglie e risorse sono comunque conservate e auditate. Un'immagine aggiunta in un blocco di testo non è confrontata (limite preesistente, vedi motivazione) | **Statica**: il runtime è bloccato dal crash di I001 (A4/A5) | Aggiungere h1–h6 e `code` al contesto host di html-inline-v1, oppure documentare esplicitamente l'eccezione con motivazione. Prova: A4/A5 non PASS automatico, heading positivi invariati |
| CLA-R004-I003 | Bassa, non bloccante | `metrics.py:112–143`; report permanente r004, tabella «Bound globale garantito»; `text_measure_definitions` nei dati | Corpus reale: 32/64 record (DOCX/EPUB). Sintetici D1/D2 | Nuovo in r004 (F20). Chiarezza del contratto C04: la garanzia è definita sulla distanza globale del flusso. Il nome, i campi `CER/WER` e l'affiancamento alla misura completa suggeriscono un limite inferiore della misura completa o condizionata | Nel caso misurabile il «bound» è una misura diversa, dipendente da ordine e separatori: può superare CER/WER completi sugli stessi denominatori. DOCX: 62 > 60 caratteri, CER bound 0,0512 > completo 0,0496. EPUB: 54 > 53. D1: completo 0, bound CER 0,667. Il bound non è mai positivo quando la misura completa manca, perché le stesse `text_reasons` danno TRIVIAL_ZERO. La garanzia dichiarata è corretta nel suo dominio | Ricalcolo indipendente 64/64; R-probe-002 D1/D2 | Una frase in PROTOCOL/guide/report e nei dati: `text_lower_bound` limita la distanza del flusso globale; non è un limite inferiore di CER/WER completi o condizionati e può superarli (omissione con separatore, inversione); è informativo solo quando la misura completa esiste. In alternativa rinominare il campo o allinearne il dominio. Nessuna nuova misura richiesta |

## Valutazione dei fix e dei residui

- **F17 chiuso.** html-inline-v1 trasporta l'intero sottoalbero, con unknown di profilo,
  formule, risorse e coordinate, senza svuotare il testo né contarlo due volte. I1/I2 non sono
  più PASS. A1/A2 e i test r004 (5 contesti × 2 blocchi × 3 wrapper) confermano; i positivi
  span/em/a/MathML restano validi. Lacuna degli heading HTML come host: CLA-R004-I002.
- **F18 chiuso.** html-table-v1 consuma ogni ramo una sola volta: celle dirette, caption unica,
  thead/tbody, residui interi con raw/posizione; le caption duplicate non scelgono l'ultima.
  html-figure-v1 tratta img + figcaption e i residui. Risorse di celle, caption e residui
  arrivano all'audit; lo scorer consuma `residual_contents` e unknown. B1–B4/B6 e i test
  r004 lo confermano; B0/B5 restano positivi. Il bypass `../outside.png` è respinto prima
  dell'apertura (spy). Nota conservativa: tag fuori validator come `tfoot` danno INCOMPLETE.
- **F19 chiuso.** Confronto caption con `absent-empty-text-equivalent-v1`, documentato: assente
  e vuota sono equivalenti solo come testo; aggiunta, omissione e differenza danno FAIL3;
  componenti tipizzate confrontate a parte; unknown mai PASS (test 3×6, C1/C2).
- **F20 chiuso nel dominio dichiarato.** Misura completa, conteggi condizionati (`guaranteed_bound=false`,
  decomp v2) e `text_lower_bound` sono distinti in codice, dati e guide. Garanzia: zero su
  qualunque testo non misurabile (nonnegatività); distanza globale esatta su flusso misurabile.
  Valore ricalcolato in modo indipendente. Known words / L1 non sono più presentati come bound;
  C1 3/30 resta condizionato/FAIL. Missing, pagine, ordine, numeri e celle sono invariati
  (D1–D3). Chiarezza del rapporto con la misura completa: CLA-R004-I003.
- **F21 chiuso.** `<` seguito da spazio, cifra, `=` o `-` resta testo (E1, test). Veri tag,
  commenti e dichiarazioni top-level e malformed restano INCOMPLETE (E3, test); la coda
  scartata dal tokenizer è rilevata. Osservazione non bloccante: un commento dentro un albero
  HTML è scartato dal tokenizer senza unknown (E2/B7 PASS), come nel rendering HTML, mentre
  un commento top-level dà INCOMPLETE. È una classificazione diversa, non una perdita.
- **Residui.** F01, F13 e F14 sono chiusi per celle, caption e residui. Il ramo HTML di
  F09/F13 è chiuso per link, figure, raw e offset, ma non per tutto il profilo: CLA-R004-I001
  lascia non misurabili gli heading HTML. F05/F15 sono chiusi con la
  separazione delle misure. F16 è preservato: cifre e composizione dei 64 identiche a r003,
  decomp v2 aggiunge solo qualificazioni. Altri fix r002 non riaperti.

## Motivazione dell'esito e limiti

NO_GO per CLA-R004-I001. Il profilo HTML finito al centro di F17/F18 dichiara supportati gli
heading h1–h6, ma il reader e l'audit vanno in crash su qualunque heading HTML (block, wrapper
o fence raw). Non c'è falso PASS: il CLI produce exit 4, internal failure. Il misuratore però
non misura un input ammesso, e il protocollo vuole che un difetto non sia innocuo perché raro o
preesistente. Il fix è locale e piccolo. Se l'arbitrato applicasse il precedente di F21
(falso errore senza falso PASS, non bloccante), il crash va comunque corretto nello stesso
giro locale, con I002. I003 è documentale.

Le parti F17–F21 sono sostanzialmente corrette: i 64 score sono verificati in modo indipendente,
le receipt e il delta coincidono con quanto dichiarato, e corpus, gold, raw e V13 sono
preservati. Toolchain: dieci origini e S/B/I sono coerenti; i pyc managed corrispondono al
manifest ricevuto, senza ripristino o attribuzione di causa. Lo S s008 è STALE solo per i
documenti del supervisore. Limiti: la prova riguarda solo il reader/audit/scorer su sintetici
e sui 64 esistenti, senza motori, L04 o V. Nessun collaudo universale HTML. Il GO non
riguarderebbe altri motori. ChromeNONPASS, gap12 helper, L01/L02/L04/L05, quota 52h e ledger
autore restano come dichiarati e non sono rivalutati. Osservazione fuori delta, senza ID:
le immagini nei blocchi di testo non sono confrontate dallo scorer (preesistente, non nei
requisiti F01–F21).

Costi propri: circa 20 minuti di wall (17:02–17:23 UTC). CPU nota circa 55 s (checker 53,6 s
user, sonde R 0,45 s, gate e verify circa 1 s); RSS massimo campionato 64 MiB (checker).
Disco circa 0,6 MB di evidenze più due snapshot. Rete, acquisizioni e costi esterni zero.
Nessun processo proprio pendente.
