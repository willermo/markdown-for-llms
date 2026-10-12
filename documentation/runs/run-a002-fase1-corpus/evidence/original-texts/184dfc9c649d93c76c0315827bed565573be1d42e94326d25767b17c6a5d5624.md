# Review — implementazione — run-a002-fase1-corpus — r010 — Claude

- **Autore, provider, modello e chat.** Reviewer Claude, Anthropic, Claude Opus 5.5
  (`claude-opus-5-5[1m]`), in Claude Code nell'estensione VS Code. Effort e ID della chat non sono
  esposti. Chat nuova, distinta da autore, supervisore e altro reviewer. Nessuna sostituzione,
  delega o subagente.
- **Prompt di origine.** `prompts/36-review-implementation-r010-claude.md`, SHA256
  `f8407128b3c7a5d017a0251043b1d7857a254bdf3bbcbac58779e6d9f895a250`.
- **Oggetto.** Implementazione r010: fix F43–F45 dopo l'arbitrato r009 NO_GO, mandato prompt34.
  Report e checkpoint dell'autore r010 letti integralmente, nota tecnica inclusa.
- **Snapshot e impronta.** Comune `snapshots/implementation-r010.json`, SHA256
  `cfc1e56f66a2f38c4159b0e4624271c70bb3d8d3092c94149762c0a89b0f3945`, impronta
  `b36a7ec224c2cf1777c0ccda50955eda7f2573702ea2d5a8fc60346ba2ff9bf4`, 331 file e 107203 artefatti.
  - Verify prima (18:54:02Z, 45,41 s): **MATCH**, exit 0.
  - Verify dopo la scrittura degli output: registrato nel checkpoint.
- **Identità Git e input.** Branch `feature/run-a002-corpus`; HEAD, dev, origin/dev locale e
  merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto. Coincidono con gli hash
  ricevuti: piano `3ac23bce…dd4ab8`, errata `a09cbb8b…e0dfd9`, fix-context-r009 `b62f462d…7d6447`,
  author-delivery-001 `3dff2c98…f9fb19`, reader v10 `382d6be9…94b5`, v9 fisico `a8ffa9eb…c737`.
- **Indipendenza.** Non ho letto review, evidenze o checkpoint ChatGPT r010, né un arbitrato r010.
  Nella cartella `snapshots/` compaiono file `review-r010-chatgpt-*`: li ho visti solo nell'elenco e
  non li ho aperti. Ho letto come storia comune entrambe le review e i checkpoint r009, i miei
  `findings-detail.md` e gruppi A/B r009, e `inert-tail-cases-002` e il summary ChatGPT r009.
- **Giro.** Fix r010. Aperti dall'arbitrato r009: F43 (CLA-R009-I001), F44 (GPT-R009-I001), F45
  (prove e documenti, incluso il limite F36 di CLA-R009-I002). F40/F41 chiusi; F39/F42 parziali.
- **Base e delta.** Sette file dell'autore (reader, PROTOCOL, due guide, nuovo test r010, report e
  dati permanenti) più quattro registri generali del supervisore. Diff del reader dal v9 fisico:
  `evidence/review-implementation-r010-claude/reader-v9-v10.diff` (100 righe).
- **Esito: NO_GO**

## Ambito e prove

**Letture.** AGENTS.md, temp/HANDOVER, la prima parte di PROJECT-CONTEXT, STATE/HANDOVER/brief,
il prompt36 e il checkpoint `supervisor-implementation-review-r010`. Le due skill, il protocollo
(review, sandbox, prove) e i template review e handover. Integrali: arbitrato r009, prompt34,
addendum risorse r008-r001 con la nota utente, le due review e i due checkpoint r009, il report e il
checkpoint dell'autore r010. Report permanente r010, sezioni r010 di PROTOCOL, reference e how-to.

**Codice.** `observations.py` v10 per intero e il diff dal v9. `audit.py`, il fallback di
`benchmark.observe_bundle`, `safety.Unsafe/PassiveHTML`, il calcolo degli exit di `metrics.score`
ed `evaluate_exit`, l'exit di `benchmark.run_attempt`. Il test r010, il collector `matrix.py`, il
driver CLI e `fixture-cli.py` dell'autore.

**Prove proprie, host stdlib (nessun import del prodotto).** In `evidence/review-implementation-r010-claude/`:

- `entry.py` → `entry.json`, exit 0. Coincidono:
  - Git e hash del comune e degli input;
  - origine `stage-s009/dev/bin/python` → `.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`;
  - target s009 (0 differenze) e package input (0 differenze);
  - `.pth` di startup;
  - 10/10 origini installate e 10/10 moduli della wheel canonica uguali al sorgente.

  Il verify di `implementation-r010-s009` è **STALE** (exit 2): le sole differenze generali sono i
  4 registri del supervisore. Lo conservo e non lo uso come gate verde.
- `stage-equivalence-002.txt`: s007 (suite veloce) e s009 (finale) hanno gli stessi `.py`,
  gli stessi 206 benchmark input e gli stessi 3813 package input. Differiscono solo S.json e I.json.
- `static_checks.py` → `static-checks.json` e `-002`:
  - matrice before-005/after-002: stessi 303 ID, `document.md` e `expected.json` identici byte per
    byte, 105→303 rispettati, classi 142/78/19/24/12/24/4, nessuna apertura di `../outside.png`;
  - i 14 raw Claude A/B r009 sono identici a quelli della matrice;
  - i 7 raw e gold ChatGPT r009 sono identici;
  - matrice r009 308/308;
  - stdout: «2015 passed, 16 deselected» e «2174 passed, 310 subtests passed».
- `derived_check.py` → `derived-check.json`: 64 record. Catena verificata: `previous_derived_final`
  e `old_*` uguali a final/audit/observations/metrics r009; run, gold e verification invariati. Tutti
  i descrittori sono coerenti. Osservazioni = r009 salvo l'ID del reader. Metrics = r009 salvo
  `observations_sha256` e `audit_sha256`, che derivano dall'ID. Campi numerici e gate identici.
  Per set 4 PASS/28 FAIL; 56 AVAILABLE/`EXACT_GLOBAL_DISTANCE` e 8 UNAVAILABLE/`TRIVIAL_ZERO`.
  Per il set historical, `conversion_final` punta ora al final originale della campagna r001
  (nei dati r009 puntava al final del rescore r002). È coerente con `run` in 64/64 (attempt e hash).

**Prove proprie, runtime.** Interprete managed 3.12.13 dello stage s009 (`-I -B`) dentro
R/Firejail con `--net=none`, `shell=False`, timeout breve e figli raccolti. Gli attesi sono fissati
prima in `expectations-frozen.json` (`9c2d4b8f…`) e `expectations-frozen-002.json` (`15040a26…`).
Il runner `rrun.py` copia i byte consumati prima e dopo (sempre identici) e cattura un manifest
proprio. Crea uno snapshot `review-r010-claude-*` per ogni esecuzione: le versioni del runner sono
conservate (`rrun-001.py`). Preflight: `check_python_origin`, inventario dei 206 input dello stage
s009 e inventario proprio.

| Prova | Esito R | Contenuto |
| --- | --- | --- |
| `v10-001` | PASS | 35 casi su v10: calibrazioni, gruppi H, T, F, R, K |
| `v9-001` | PASS | Gli stessi 35 casi sul v9 fisico, per l'attribuzione |
| `cause-001` | PASS | Eventi del walk inline su 5 valori fissati |
| `cli-001` | PASS | CLI reale run/evaluate su 4 casi, writer sintetico dichiarato dell'autore non modificato |
| `perf-v10-001`, `perf-v9-001` | PASS | Tempo di `observe` su un paragrafo con k commenti, k ≤ 4000 |

Esiti v10 (`probe-check.json`): 18 dei 29 casi di requisito sono conformi; i 4 controlli di profilo
sono conformi; K1 diverge da CommonMark (caso esplorativo).

- **Calibrazione.** A1 UNSAFE, A9 UNSUPPORTED, GPT-after COMPLETE/PASS/5, B1 UNSUPPORTED: coincidono
  con l'autore.
- **F43 nei trasporti assenti dalla matrice.** T1 heading, T2 caption, T3 didascalia di figura,
  T4 enfasi e T5 label di link danno UNSAFE con «Path evasivo». Sul v9 T1, T2, T4 e T5 erano
  COMPLETE: r010 li chiude. T3 sul v9 non è registrato per un mio errore di gold (KeyError nello scorer).
- **F44, varianti.** Restano COMPLETE, con testo literal esatto e nessuna reference:
  F1 span dentro enfasi, F2 dentro label di link, F3 annidato, F4 `<span/>`, F5 `<br>`.
  Sul v9, F3, F4 e F5 erano falsi UNSAFE.
- **F6, prefisso F36 inline seguito da code chiuso.** Dà UNSUPPORTED (sul v9 era falso UNSAFE).
- **Controlli di profilo.** F7 (owner aperto), F8 (owner bilanciato che contiene code) e
  F9 (chiusura estranea) danno UNSAFE, come dichiara il PROTOCOL: l'owner HTML conserva il dominio HTML.
- **Raw inline.** R1 `{=html}` unsafe dà UNSAFE; R2 sicuro dà UNSUPPORTED.
- **Gruppo H.** Vedi il rilievo.

**Prove dell'autore lette, non rieseguite.** Le receipt R dei sette passaggi finali (matrice
before/after final, r009, oracle-binding, targeted, fast-final, CLI) sono PASS con `inputs_unchanged`.
Lo stesso vale per contratti H01–H08, rescore, derived-check e delivery-check-001…003. Sono
conservati: H12± FAIL volute (`R_expected_exit` 0, harness 1, `match` true) e matrix-before-001…003
FAIL (IndexError, `segments`, preflight STALE). CLI dell'autore: 41 fixture e 82 comandi,
`all_expected` true, exit attesi (0, 5, 5) per i literal e (2, 2, 3) per unsafe/unsupported.
Il test r010 contiene 303 + 2 casi, come dichiarato.

**Non eseguito.** Per mandato non ho rieseguito suite, matrice dell'autore, build o reinstallazione,
contratti H, rescore, conversioni, render, Pandoc, M/V, rete e Git di scrittura. Gli argomenti
CommonMark/Pandoc delle osservazioni K1 e `{=html5}` sono letture della specifica, non prove eseguite.

## Rilievi

| ID | Severità e blocco sì/no | Posizione | Raggiungibilità/input ammessi | Attribuzione e requisito | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLA-R010-I001 | **Media, bloccante** | `observations.py:215–217`: regola dei metadata heading attiva anche con `block_context=False`, che alimenta `inline_html_events` (`:509–524`). Il consumer chiude il resto in `:556` (code) e `:576` (math), con `lost_html` falso. Per H2: `except ValueError` (`:622`) assorbe `Unsafe` | Valore inline di list (prima riga o continuazione), quote, note, cella pipe, caption, testo di un heading ATX con blocco attributi finale. Basta una riga logica che inizi con `# ` e termini con `{…}` e contenga un backtick o un `$x` non chiusi seguiti da un tag reale. Esempio: ``- # x {`<img src="../outside.png">}``. Input sintetico ammesso | **Ereditato** per H1/H3–H11: le righe 215–217 sono identiche nel v9, che dà gli stessi COMPLETE. r010 affida la guardia a questo walk e non lo chiude. H2 è una **regressione r010** di classificazione (v9 UNSAFE → v10 UNSUPPORTED), fail-closed. Requisiti C01/F43 (aperture reali in trasporti e ricorsioni; A9 mai COMPLETE con risorsa persa), C03/F45 (walk, producer e consumer concordi), C05. Il PROTOCOL r010 dice: «un opener senza chiusura accettata non rende inerte il tag seguente», «Non si riclassificano i marker rimossi dai prefissi quote/list/note». Smentiti anche la riga della tabella nella reference e il passo 1 della how-to | Il walk inline salta `{…}` come se fosse un metadata heading: il tag non diventa evento e la guardia non lo vede. Il consumer non conosce la regola e lo chiude in una foglia unknown con `resource_syntax: false`. Risultato: audit **COMPLETE**, CLI **run 0 / evaluate 5** (gate INDETERMINATE, score 5 quando il gold corrisponde) per `<img src="../outside.png">` e `<script>`, che il tokenizer fissato riconosce. Con `p.png` la risorsa sicura si perde con COMPLETE (H3). Nessun file aperto né fetch. La matrice 303 non ha valori che iniziano con `# ` né host heading o caption | `out-v10-001`: H1, H3–H7, H9–H11 COMPLETE con score 5; H8 COMPLETE; H2 UNSUPPORTED. `out-v9-001`: H1/H3–H11 uguali, H2 UNSAFE. `out-cause-001`: 0 eventi con `# ` iniziale, 1 senza; 5/5 previsioni. `out-cli-001`: exit OS 0/5 per H1, H6 e H3, 2/2 per il controllo HC1. HC2 (heading reale) resta COMPLETE: consumer concorde. Dettaglio in `findings-detail.md` | Il walk inline non riconosce metadata heading, che valgono solo per il blocco (per esempio: limitare `:215–217` a `block_context`), oppure si allinea al consumer in modo equivalente. H1 e H4–H11 danno `UNSAFE_REFERENCE` con detail. H3 dà INCOMPLETE/UNSUPPORTED, oppure immagine proiettata e asset auditato. H2 torna UNSAFE. Restano invariati HC2, matrice 303, regressioni r009 e F36. La matrice va estesa con righe logiche iniziali `# ` e con i trasporti heading e caption; i documenti vanno rettificati |

Ho scritto un solo rilievo perché i casi H, incluso H2, hanno una sola causa e una sola correzione.

### Chiusure F43–F45 e criteri comuni

| Criterio | Esito nel perimetro controllato |
| --- | --- |
| C01 F43 | **Parziale.** Chiuso per i sei host della matrice (A1–A10, 108 fallback, foreign e first-unsafe) e per i miei T1–T5 (heading, caption, didascalia di figura, enfasi, label di link). A9 usa l'alternativa INCOMPLETE/UNSUPPORTED autorizzata. Visita continuata dopo la prima violazione: test r010 letti. Non chiuso per le righe logiche `# … {…}` dei valori inline (CLA-R010-I001), dove il fallback nasconde ancora l'apertura con COMPLETE |
| C02 F44 | **Chiuso** nei casi controllati. I quattro falsi unsafe GPT sono COMPLETE/PASS/5 con tre blocchi e nessuna reference. F1–F5 (enfasi, link, annidato, self-closing, void) sono literal e COMPLETE. F6 con prefisso F36 non genera più un falso unsafe. L'unsafe reale fuori code resta UNSAFE (GPT-real, A8). Owner aperto, bilanciato o con chiusura estranea danno UNSAFE secondo il profilo dichiarato |
| C03 F45 | **Parziale.** La nota tecnica precede il codice e la matrice finita ha attesi manuali identici prima e dopo (105→303). Il binding di 21 discriminanti è verificato da me. L'invariante «walk, producer e consumer concordi» non vale in modalità inline per la regola dei metadata heading. Il walk termina; a k=4000 commenti in un paragrafo v10 impiega 7,3 s contro 1,4 s di v9 (osservazione) |
| C04 F36 | **Conforme.** B1–B4 INCOMPLETE/UNSUPPORTED, nessuna apertura; i dati registrano le aperture del parser puro. La divergenza è documentata e non è estesa ai fallback F43 |
| C05 regressioni | Matrice r009 308/308, suite veloce e mirati dell'autore. F40/F41 e il mapping char/byte non cambiano nel delta (le righe di `html_tag_boundary` e della mappa sono invariate). F32, F27, F30, F28 locale e F26 non sono toccati. Regressione H2 (UNSAFE → UNSUPPORTED) dentro CLA-R010-I001 |
| C06 documenti | Storia r009, errori conservati (IndexError, segments, output relativo, STALE dopo test, argv del budget, collisione del doc wrapper) e H12 corretti. Il limite F36 e i 12 math chiusi come negativi di scoring sono descritti con precisione. PROTOCOL, reference, how-to e report r010 affermano una copertura F43 che CLA-R010-I001 smentisce |
| C07 64 misure e V | **Conforme** (`derived-check.json`). Nessun motore, nessuna V nuova. V13/A03 riguarda i riferimenti; L04 output resta UNAVAILABLE, distinta. L01/L02 NOT_MEASURED, L05 lower/sampled. TRIVIAL_ZERO non è un PASS completo |
| C08 toolchain | Gate host di origine, target, startup, package, archivi e wheel confermati sullo stage s009. Suite su s007 con Python identico a s009 (`stage-equivalence-002.txt`). Il generale s009 è STALE per i soli 4 registri ed è conservato. Gate propri circoscritti; nessun PASS trasferito |
| C09 risorse | Dentro la disposizione 24h (sotto). Ogni workload proprio con timeout breve e reap; nessun download o invio |
| C10 ciclo | Il report dell'autore è completo, senza GO. Ruoli separati, comune MATCH. Fix locale, non nuovo piano. Nessun Git, cleanup, a001, fase 2, motore o V |

## Osservazioni non classificate come rilievi

Dettaglio in `findings-detail.md`; nessuna è bloccante.

- **K1, lunghezza dei run di backtick.** `code_end` (`:75–79`) chiude un run anche dentro un run più
  lungo: ``See ``a``` <img src="../outside.png"> ` tail.`` dà COMPLETE (anche sul v9). Per
  CommonMark l'img è raw HTML attivo. Lexer e consumer sono concordi, il codice è fuori dal delta e
  il PROTOCOL r010 non promette la semantica CommonMark completa. Proposta: backlog.
- **Formati raw `{=html5}` e `{=html }`.** Il consumer valida solo `html` esatto (`:562–565`): gli
  altri formati sono `raw inline format unsupported` senza `resource_syntax`, quindi COMPLETE.
  Lettura statica; Pandoc non è stato eseguito.
- **Costo della guardia inline.** Per carattere si aggiunge una scansione degli intervalli inerti: v10
  è circa 5 volte più lento di v9 a k=4000 commenti in un paragrafo. Calcolo terminante, nessun requisito.
- **ReaderID del fallback unsafe.** `observe_bundle` (`benchmark.py:55`, fuori dal delta) scrive
  `observations-v1`: i 142 record unsafe della matrice non portano l'ID del reader reale.

## Motivazione dell'esito e limiti

**NO_GO** sul comune `implementation-r010`. F44 è chiuso nelle mie varianti indipendenti. F36 è
documentato come chiesto. F43 è chiuso per i sei host della matrice e per heading, caption,
didascalia di figura, enfasi e label di link (sul v9 heading, caption, enfasi e label erano ancora
COMPLETE). Le 64 misure sono
invariate e coerenti.

La guardia inline però dipende dal walk `markdown_lexemes` in modalità inline, e questo walk
applica ancora una regola di blocco: i metadata di heading. Un'apertura reale che segue un code o
un math non chiusi, su una riga logica `# … {…}` di un valore list, quote, note, cella, caption o
heading, non diventa evento. Il fallback la nasconde e l'audit dà COMPLETE: run 0 / evaluate 5
misurati sulla CLI reale. È la stessa classe di CLA-R009-I001 (media, bloccante) e lo stesso
requisito F43, in un trasporto non coperto dalla matrice. Contraddice anche le nuove dichiarazioni
del PROTOCOL r010. La correzione è locale. Non chiedo un parser universale: chiedo che producer e
consumer del dominio finito siano concordi.

Il GO non certificherebbe comunque HTML o Markdown universali, L04, altri motori o la fase 1.

**Limiti.** Le mie prove runtime coprono 35 casi sintetici su due reader (70 esecuzioni), 5 valori
di causalità, 4 casi CLI (8 comandi) e 10 misure di tempo; non tutte le combinazioni di trasporti.
Gli argomenti CommonMark/Pandoc delle osservazioni non sono stati verificati con un renderer. Le
prove dell'autore le ho lette e ricontrollate negli hash e nei byte, non rieseguite. Gold sintetici
PROVISIONAL, nessuna V.

**Errori propri conservati.**

- `static_checks.py`: nome della cartella B4 sbagliato, corretto in `static-checks-002.json`.
- `derived_check.py`: non escludeva `audit_sha256`; differenze diagnosticate nel report.
- Prima equivalenza degli stage con percorsi collassati: ripetuta in `-002`.
- KeyError dello scorer sul mio gold `figure` in T3, solo sul v9.

Nessun input ricevuto è stato modificato.

**Costi propri** (`costs.json`). Review dalle 18:54:02Z. Comandi misurati con `/usr/bin/time`,
CPU user+sys:

| Comando | CPU (s) | Wall (s) |
| --- | --- | --- |
| verify prima | 45,39 | 45,41 |
| entry | 2,05 | 2,05 |
| v10-001 | 5,49 | 5,20 |
| v9-001 | 8,98 | 8,13 |
| cause-001 | 6,30 | 5,83 |
| cli-001 | 17,84 | 14,83 |
| perf-v10-001 | 17,12 | 15,22 |
| perf-v9-001 | 8,52 | 7,84 |
| derived_check | 0,54 | 0,53 |

- **CPU.** Totale **112,23 CPU-s**, lower bound: checker host minori e verify finale non sono inclusi
  (il verify dopo è nel checkpoint).
- **Memoria.** maxrss esterno 324756 KiB, non un picco certificato.
- **Disco.** Evidenze 10612962 B apparenti (13201408 B allocati) alla misura delle 19:14Z, più
  6 snapshot propri per 444308 B.
- **Rete e servizi.** Rete, acquisizioni, invii e servizi: 0 e €0.

**Wall e contabilità CPU.**

- Wall cumulativo dall'entry 2026-10-08T07:06:54.369068Z alle 19:14:09Z: 216435,3 s.
- Margine allo stop cumulativo di 257436,63 s: circa 41001 s.
- Stop nuovi workload 2026-10-11T06:37:31Z, deadline 09:37:31Z.
- CPU in contabilità: 14002,663555 s visibili alla consegna del supervisore più 112,23 s miei,
  circa 14114,89 s lower bound, sotto lo stop di 19800 s.
- Esclusi: la CPU del reviewer ChatGPT r010, che non conosco; i 45 s Claude r007 approssimati
  restano separati.
- Nessun processo pendente.

Prossimo ruolo: **supervisore**, per la ricezione di entrambi i report r010 reali e l'arbitrato.
Git resta manuale e solo dopo un GO finale.
