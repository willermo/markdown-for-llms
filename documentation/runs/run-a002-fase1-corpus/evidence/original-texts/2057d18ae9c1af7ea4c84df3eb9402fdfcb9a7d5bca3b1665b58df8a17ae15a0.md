# Review — implementazione — run-a002-fase1-corpus — r009 — Claude

- Autore/provider/modello e chat: reviewer Claude, Anthropic, Claude Opus 5.5 (`claude-opus-5-5[1m]`),
  in Claude Code nell'estensione VS Code. Effort e ID chat non sono esposti. Chat nuova, distinta da
  autore, supervisore e altro reviewer. Nessuna sostituzione, delega o subagente.
- Prompt di origine: `prompts/33-review-implementation-r009-claude.md`, SHA256
  `43840121d830818d71a732aa251a91ebeb8c3e9dee290dfdabac4d90c7de5ae4`.
- Oggetto e revisione esatta: implementazione r009, giro di fix F39–F42 dopo l'arbitrato r008 NO_GO.
  Mandato: prompt30 e addendum risorse r008 r001 (DISPOSED24h). Report e checkpoint autore r009
  integrali, con la nota tecnica iniziale.
- Snapshot, HEAD e impronta verificati prima/dopo: `snapshots/implementation-r009.json`, SHA256
  `bfc26ebc0db5946f30a0a47b6aa3a558674a167eb1c022dce1caa39fb6a03b9f`, impronta
  `1c5d94c7db9f4a8508143a3bd1e7b72f559dfcaa6bb9909602ac12ebee20eb66`, 328 file e 91038 artefatti.
  Verify prima (15:35:44Z, 41,15 s): **MATCH**, exit 0. Il verify dopo la scrittura degli output è
  nel checkpoint. Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto. Piano `3ac23bce…dd4ab8`, errata
  `a09cbb8b…e0dfd9`, contesto fix-r008 `f6cd0f93…c6cc99aa` e snapshot autore delivery-001
  `1e54271b…c11892` coincidono con gli hash ricevuti.
- Indipendenza: non ho letto la review ChatGPT r009, le sue evidenze o il suo checkpoint, né un
  arbitrato r009. Nella cartella snapshot esiste un file `review-r009-chatgpt-probes-001.json`:
  l'ho visto solo nell'elenco, non l'ho aperto. Ho letto come storia comune le due review r008,
  la traccia GPT-R008-I001 e l'arbitrato r008.
- Giro: fix r009. Aperti dall'arbitrato r008: F39 (CLA-R008-I001), F40 (CLA-R008-I002), F41
  (GPT-R008-I001), F42 consolidamento, prove e documenti.
- Base iniziale e delta del giro: 7 file autore (reader, PROTOCOL, nuovo test r009, due guide,
  report e dati permanenti) più 4 registri generali del supervisore. Diff del reader dai byte v8
  fisici `work/implementation-r009/r008-observations.py` (`817852a8…`) al v9 (`a8ffa9eb…`) in
  `evidence/review-implementation-r009-claude/reader-v8-v9.diff`.
- Esito: **NO_GO**

## Ambito e prove

**Letture.** AGENTS.md, temp/HANDOVER, STATE/HANDOVER/brief della run, il checkpoint
`supervisor-implementation-review-r009`, le due skill, il protocollo (sezioni review, sandbox,
prove) e il template. Poi arbitrato r008, prompt30, addendum risorse e nota utente; le due review
r008 con i checkpoint e la traccia `finding-I001.md`; F36 nell'arbitrato e nella review Claude r007.
Report e checkpoint autore r009 integrali; PROTOCOL (sezioni r004, r008, r009), report e dati
permanenti r009.

Codice: `observations.py` v9 per intero e il diff dal v8; `audit.py`, `benchmark.observe_bundle`,
`safety.PassiveHTML/html`, il percorso exit di `metrics.py` dal PROTOCOL; il test r009 e il collector
`work/implementation-r009/matrix.py`. Ho letto anche il tokenizer fissato
`.venv-python/cpython-3.12.13-…/lib/python3.12/html/parser.py` (`952f1d7d…`): `tagfind_tolerant`,
`attrfind_tolerant`, `locatetagend`, `parse_starttag`, `parse_endtag` e `goahead`.

**Prove proprie, statiche (host stdlib, nessun import del prodotto).** Tutte in
`evidence/review-implementation-r009-claude/`:

- `entry.py` → `entry.json`, exit 0. Coincidono Git, hash del comune e degli input, origine
  `stage-s006/dev/bin/python` → `.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`,
  1170 descrittori del target S006 (0 differenze), package input (0 differenze), `.pth` di startup
  e 10/10 origini installate uguali al sorgente. Il verify di `implementation-r009-s006` è
  **STALE** (exit 2); le sole differenze generali sono i 4 registri del supervisore.
- `wheel_check.py` → `wheel-check.json`: 10/10 moduli uguali tra sorgente, wheel canonica S006
  (`6a602f60…`) e installato.
- `static_checks.py` → `static-checks.json`, exit 0. Dati permanenti r009: 64 record, 0 errori di
  descrittore o catena, `final` distinto da `conversion_final` e da `previous_derived_final`. Per set
  4 PASS/28 FAIL (historical-r001 e baseline-r002). Osservazioni r009 = r008 salvo l'ID del reader
  su 64/64; metriche identiche su tutte le chiavi. 56 `EXACT_GLOBAL_DISTANCE`/AVAILABLE e
  8 `TRIVIAL_ZERO`/UNAVAILABLE. Per formato: TXT WER 0 (0/89); DOCX 0,0604 (11/182, globale 11);
  EPUB 0,0430 (8/186, 8); RTF 0,1173 (19/162 condizionato, 13 globale); TeX WER None (1/152
  condizionato, 0 globale `TRIVIAL_ZERO`). La prima versione del checker è fallita per un mio errore
  (chiave `conversion_run` assente nei dati r009); la conservo come `static_checks-001-failed.py`.

**Prove proprie, runtime.** Interprete managed 3.12.13 dello stage S006 (`-I -B`), dentro
R/Firejail con `--net=none`, `shell=False`, timeout 60 s e figli raccolti. Gli attesi (criterio
indipendente dal reader e previsione statica) sono stati fissati prima in `expectations-frozen.json`
(`e68accf2…`). Il runner `rrun.py` copia i byte consumati prima e dopo (sempre identici), cattura un
manifest proprio e crea uno snapshot tecnico `review-r009-claude-*` per ogni esecuzione. I preflight
sono: `check_python_origin`, l'inventario dei 206 input runtime dello stage S006 e l'inventario
proprio. S006 integrale non è usato come gate verde.

| Prova | Esito R | Risultato |
| --- | --- | --- |
| `s006-conserve-001` | **FAIL** conservato | Gate S006 ricevuto: `Stage STALE` prima di ogni workload |
| `v9-001` | **FAIL** conservato | Errore mio di preflight: `check_preflight.py` vuole lo snapshot S006 generale MATCH. Nessun workload avviato. Sostituito con gate circoscritti |
| `v9-002` | PASS | 42 casi su v9 (gruppi A–E) |
| `v8-001` | PASS | Gli stessi 42 casi sul v8 conservato, per l'attribuzione |
| `r007-001` | PASS | Gli stessi 42 casi sul reader r007 `5d443387…` |

Risultati v9 (dettaglio in `findings-detail.md`, confronto in `probe-check.json`):

- **F39, gruppo C.** D1, D3 raw, D6, D7 e un tag sconosciuto dopo duplicati danno
  `UNSAFE_REFERENCE` con detail (`Path evasivo`, `onerror`, `HTML attivo/sconosciuto: alpha`);
  v8 dà UNSUPPORTED, r007 UNSAFE. D7 sicuro conserva immagine `p.png` aperta e testo «Don't».
- **F40, gruppo D.** 14 forme di attributo non presenti negli ATTRS dell'autore (slash, apice in
  bare value con `/>` e `\n<br>`, `"` e `'` misti, `=` iniziale, backtick, opener dentro valori).
  La fine del tag del reader coincide con il tokenizer fissato in **14/14**, senza prefisso
  incompleto. v8 diverge su D08 e D12–D14. Su 9 casi il mio atteso «immagine conservata» era
  sbagliato: avevo usato nomi di attributo fuori dal subset passivo e PassiveHTML dà
  correttamente UNSAFE.
- **F41, gruppo E.** Link con code su soft-line dentro un'enfasi in quote; list con marker `> `
  nel code; due backtick; reference sulla terza riga di una quote; reference dentro un'enfasi sulla
  seconda riga; note con tab; immagine HTML inline sulla seconda riga. Testo e reference corretti,
  COMPLETE, automatic PASS, qualità INDETERMINATE e score 5 sui gold scritti a mano. Ogni posizione
  char/byte coincide con l'indice nel raw UTF-8. v8 fallisce E1, E2 ed E6.
- **Gruppi A e B**: vedi i rilievi.

**Prove dell'autore lette, non rieseguite.** Matrice `before-corrected-001` 178/308 (19 unsafe,
14 safe-attribute, 97 domain non rispettati) e `after-003` 308/308, stessi ID, nessuna apertura di
`../outside.png`. Le receipt R di after-matrix-003, before-corrected-matrix-001, targeted-003,
fast-004, cli-002, derived-check-001 e delivery-check-002 sono PASS con `inputs_unchanged`.
Lo stdout dà «1712 passed, 16 deselected» e «1869 passed, 310 subtests passed». H01–H08 PASS,
H12 positivo e negativo FAIL voluti. CLI 35 casi e 70 comandi con exit OS attesi. Il conteggio delle
classi (89/12/56/122/5/24 = 308) e i 31 test aggiuntivi tornano con il sorgente del test.

**Non eseguito.** Per mandato non ho rieseguito: suite, matrice dell'autore, build o
reinstallazione, contratti H, CLI con writer, rescore, conversioni, render, Pandoc, M/V, rete e Git
di scrittura. Per gli argomenti CommonMark (gruppo A) mi baso sulla specifica e sul comportamento
dello scanner v9, non su un renderer eseguito.

## Rilievi

| ID | Severità e blocco sì/no | Posizione | Raggiungibilità/input ammessi | Attribuzione e requisito | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLA-R009-I001 | **Media, bloccante** | `observations.py:520` (code non chiuso), `:540` (math non chiusa), `:557` (link fuori profilo): foglia unknown fino a `leaf_end`. `:888` valida solo nodi unknown senza lista `unknown`. `audit.py:40–43` | Paragrafo, quote, cella o list con un backtick o un `$x` non chiusi, oppure un link fuori profilo, seguiti nello stesso producer da un tag reale: `<img src="../…">`, `<script>`, `onerror`. Lo scanner v9 avanza del solo opener (`:233–235`) e riconosce il tag (`:270–278`); per CommonMark è raw HTML attivo. Input sintetico ammesso | **Ereditato** (identico su r007 e v8), non regressione r009. Ricade nel requisito esplicito di questo giro: C01 e F39 («tutte le aperture reali riconosciute dalla semantica lessicale scelta passano PassiveHTML»; «prima diagnosi unsafe precede unsupported»), F42 («ogni tag attivo validato»). PROTOCOL r009: «Un unsafe reale ha precedenza sulle diagnosi unsupported strutturali o lessicali». PROTOCOL r004: «Ogni src/href … validato anche in fallback; sintassi risorsa non rappresentabile rende INCOMPLETE» | Il tag viene inglobato nella foglia unknown senza `validate_html` e senza `resource_syntax`. L'audit dà **COMPLETE** con reason `None`: script, `onerror` e path evasivi non danno alcuna diagnosi di sicurezza o di risorsa. Per la regola di uscita del PROTOCOL, bundle completo significa run exit 0 (dedotto, CLI non eseguita). Con link fuori profilo l'esito è INCOMPLETE/UNSUPPORTED invece di UNSAFE. Con `p.png` si perde il riferimento immagine reale senza INCOMPLETE. Nessun file aperto né fetch; il nodo unknown fa fallire lo score. La documentazione r009 e r004 è smentita. La matrice 308 combina solo errori strutturali HTML con tag unsafe | Runtime proprio `v9-002`: A1/A2/A4/A5/A6/A10 COMPLETE; A3 UNSUPPORTED; controlli A7 (code chiuso) e A8 (tag prima) UNSAFE. Il tokenizer fissato registra lo start tag reale in ogni caso. `v8-001` e `r007-001` danno gli stessi esiti | **Fix minimo**: quando una foglia unknown di code/math/link ingloba sintassi di apertura riconosciuta dallo scanner, validarla con `validate_html` (unsafe prevale) e marcarla `resource_syntax`, come nel ramo `inline HTML unsupported` (`:588–593`). In alternativa la guardia può seguire le aperture di `markdown_lexemes` fuori dagli span inerti. **Chiusura**: A1/A2/A4/A5/A6/A10 e A3 danno `UNSAFE_REFERENCE` con detail. A9 è almeno INCOMPLETE con diagnosi. Restano invariati i code chiusi letterali (nessun unsafe inventato), i controlli A7/A8, la matrice 308 e F36. Matrice e documenti estesi a questa interazione |
| CLA-R009-I002 | Bassa, **non bloccante** (decisione all'arbitro) | `observations.py:93–124` (`html_tag_boundary`): opener inerte in posizione di nome tag/attributo = prefisso incompleto «prima della fine del tag». Il guard `FiniteHTML` eredita la scelta | Blocco HTML di tipo 6 (`<div …`) o fence `{=html}` con `<!--` in posizione di attributo e un `>` reale prima del commento chiuso o di EOF: `<div <!-- -- ><img src="../outside.png">`, `<div <!-- -- ><script>`. Sintetico ammesso | Estensione F36 introdotta in r008 e mantenuta in r009 **per mandato**: prompt30 «F36 `<alpha` prima del token resta locale unsupported»; arbitrato r008 «non ripristinare semplicemente un html(raw)… prefissi F36 come attributi». C02 «prefisso veramente incompleto» | Il tokenizer fissato (e HTML5) chiude il tag al `>` e riconosce `img`/`script` reali. Il reader li mette nel corpo di un commento fino a `-->` o EOF, quindi non li valida: INCOMPLETE/UNSUPPORTED invece di UNSAFE. Nessun PASS né fetch, exit 2. PROTOCOL e report non dicono che le aperture complete per il tokenizer dentro un corpo F36 restano senza validazione | Runtime: B1/B2/B3/B4 su v9 e v8 INCOMPLETE/UNSUPPORTED, su r007 UNSAFE. Il tokenizer fissato riporta `<div <!-- -- >` poi `<img src="../outside.png">` (B1) e `img` con `src="../outside.png"` (B2) | Arbitro: accettare come limite documentato (PROTOCOL, guide, report) oppure richiedere, nei soli producer blocco HTML e raw, una guardia fail-closed anche sul tokenizer puro, senza cambiare token e proiezione F36. Nel secondo caso va rivista l'attesa della classe `prefix` |

Ho scritto CLA-R009-I001 come un unico rilievo: code, math e link fuori profilo sono la stessa
causa (foglia del consumer che ingloba un'apertura), con un'unica correzione.

### Chiusure F39–F42 e criteri comuni

| Criterio | Esito nel perimetro controllato |
| --- | --- |
| C01 F39 | **Chiuso per i frammenti HTML, non in generale.** `html_nodes` chiama `validate_html` prima dell'albero (`:688`). `HTMLGuard` conserva la prima violazione e prosegue; duplicati, depth>16 e il cap di 20000 nodi restano nella proiezione. D1/D3/D6/D7 e il tag sconosciuto dopo duplicati sono UNSAFE con detail; D4/D5 e gli inert-error dell'autore sono invariati. Resta aperto CLA-R009-I001: un'apertura reale dentro una foglia Markdown non supportata non arriva mai alla guardia |
| C02 F40 | **Chiuso.** Scanner (`tag_end`), `FiniteHTML`, guardia e albero condividono `html_tag_boundary`, basata su `tagfind_tolerant`/`attrfind_tolerant`/`locatetagend` del 3.12.13. Le quote valgono solo per i valori quotati. 14/14 forme nuove e i 20 confronti dell'autore coincidono col tokenizer. D7 è UNSAFE; la variante `p.png` conserva testo, ref e asset aperto. Il prefisso F36 resta locale come da mandato (osservazione CLA-R009-I002). Il `parse_endtag` del tree non applica F36 agli end tag, mentre lo scanner sì: nelle mie letture non produce aperture diverse, solo un token non segnalato |
| C03 F41 | **Chiuso.** I producer passano `producer_limit` e `_source_map`; il consumer e le ricorsioni (label, span, enfasi, `inline_delimiter`) non usano più `inline_scope_end` con il limite del producer. `node` usa la mappa per char/byte. La quote considera i code segment. GPT-exact (autore) e i miei E1–E7 (enfasi, list `> `, due backtick, terza riga, tab, HTML inline) sono esatti, COMPLETE, PASS/5 INDETERMINATE. I `foreign` dell'autore non prendono chiusure estranee |
| C04 F42 | **Parziale.** Nota tecnica prima del codice, decisione lessicale unica, matrice finita con attesi scritti a mano (uguali prima e dopo, 178→308) e assert lungo audit→observe→score. Manca l'interazione «letterale Markdown non chiuso × apertura unsafe» (CLA-R009-I001). La progressione terminante non è toccata dal delta |
| C05 regressioni | Nessuna regressione trovata nel delta. Le righe di autolink (F32), `</>` (F27), reason (F30), CDATA locale (F28) e scorer (F26) sono invariate o equivalenti; i 64 dati sono invariati (F16). Le regressioni r002–r008 dell'autore sono PASS (1712). I controlli A7/A8 e i code chiusi restano corretti. CLA-R009-I001 è ereditato, non una regressione |
| C06 documenti | PROTOCOL, guide e report descrivono correttamente F40/F41, conservano la storia r008 e i fallimenti (EPERM, matrice 292, CLI001, H01 label/parent, link nel delivery checker, collisione del doc wrapper, H12). Due affermazioni sono smentite dal mio CLA-R009-I001: «Un unsafe reale ha precedenza sulle diagnosi unsupported … lessicali» e «HTMLGuard valida tutte le aperture reali». Il limite di CLA-R009-I002 non è dichiarato |
| C07 64 misure e V | **Conforme** (`static-checks.json`). Nessun motore, nessuna V nuova. V13/A03 riguarda i riferimenti; L04 formule output resta UNAVAILABLE, distinta. L01/L02 NOT_MEASURED, L05 lower/sampled. TRIVIAL_ZERO non è contato come PASS completo |
| C08 toolchain | Prove dell'autore coerenti: stage S006 con S/B/I, wheel canonica e 10 hash nello stesso interprete dei figli. Il mio gate host conferma origine, target, startup, package e wheel. S006 integrale è STALE per i soli 4 registri ed è conservato; i miei gate sono circoscritti e non trasferisco PASS generali |
| C09 risorse | Dentro la disposizione24h, lontano dallo stop (sotto). Ogni workload proprio con timeout breve e reap; nessun download |
| C10 ciclo | Report autore completo, senza GO; ruoli separati; comune valido MATCH. Fix locale, non nuovo piano. Nessun Git, cleanup, a001, fase 2, motore o V |

## Motivazione dell'esito e limiti

**NO_GO** sul comune `implementation-r009`. Le chiusure richieste dall'arbitrato r008 sono reali
nelle mie prove indipendenti: D1/D3/D6/D7 tornano UNSAFE, la fine del tag coincide col tokenizer
fissato anche su forme non coperte dall'autore, e i domini e le mappe dei producer reggono in
sottocontesti non presenti nella matrice. Le 64 misure sono invariate e coerenti.

Il criterio F39/C01 però chiede che ogni apertura reale fuori dai corpi inerti passi dalla guardia
e che un unsafe reale prevalga sulle diagnosi unsupported lessicali, e il PROTOCOL r009 lo afferma.
Un tag reale dopo un backtick o un `$` non chiusi viene invece inglobato in una foglia unknown
senza validazione e senza `resource_syntax`, e l'audit risulta COMPLETE. È un difetto ereditato,
non introdotto da r009, ma riguarda un requisito esplicito di questo giro e la stessa classe dei
due rilievi r008 (guardia che dipende dalla proiezione). La correzione è locale.
CLA-R009-I002 è una tensione con la semantica F36 decisa dall'arbitro; la lascio non bloccante.

Il GO non certificherebbe comunque HTML o Markdown universali, L04, altri motori o la fase 1.

Limiti: le mie prove runtime sono 42 casi sintetici su tre reader (126 esecuzioni), non coprono
tutte le combinazioni di trasporti. L'argomento CommonMark del gruppo A non è stato verificato con
un renderer. Le prove dell'autore le ho lette e ricontrollate negli hash, non rieseguite. Gold
sintetici PROVISIONAL, nessuna V.

**Costi propri** (`costs.json`). Review dalle 15:35:44Z. Comandi misurati con `/usr/bin/time`,
CPU user+sys:

| Comando | CPU (s) | Wall (s) |
| --- | --- | --- |
| verify prima | 41,11 | 41,15 |
| entry | 1,07 | 1,05 |
| s006-conserve-001 | 1,00 | 1,01 |
| v9-001 (FAIL preflight) | 4,04 | 3,88 |
| v9-002 | 5,30 | 5,04 |
| v8-001 | 4,94 | 4,70 |
| r007-001 | 4,94 | 4,70 |
| static 001 (FAIL) + 002 | 0,76 | 0,74 |

Totale **63,16 CPU-s**, lower bound: wheel/probe check, letture e verify finale non sono misurati
(il verify dopo è nel checkpoint). maxrss esterno 316236 KiB, non picco certificato. Disco nuovo
circa 6,2 MB: evidenze 5860986 B alla misura, più 4 snapshot propri per 291000 B. Rete, acquisizioni,
invii e servizi: 0 e €0.

Wall cumulativo dall'entry 2026-10-08T07:06:54.369068Z al 15:54Z: 204437,4 s (circa 56,79 h).
Margine allo stop cumulativo 257436,63 s: circa 52999 s. Stop nuovi workload
2026-10-11T06:37:31Z, deadline 09:37:31Z. CPU in contabilità: 12835,064462 s della ricezione del
supervisore più 63,16 s miei, circa 12898,22 s lower bound, sotto lo stop di 19800 s; 45 s Claude
r007 approssimati restano separati. Non è inclusa la CPU del reviewer ChatGPT r009, che non conosco.
Nessun processo pendente.

Prossimo ruolo: **supervisore**, per la ricezione di entrambi i report r009 e l'arbitrato.
Git resta manuale e solo dopo un GO finale.
