# Review — implementazione — run-a002-fase1-corpus — r007 — Claude

- Autore/provider/modello e chat: reviewer Claude, Anthropic, Claude Opus 5.5 (`claude-opus-5-5[1m]`),
  in Claude Code nell'estensione VS Code. L'ID chat non è esposto. Chat nuova, distinta da autore,
  supervisore e reviewer dei giri precedenti. Nessuna sostituzione o delega, nessun agente avviato.
- Prompt di origine: `prompts/26-review-implementation-r007-claude.md`, SHA256
  `71fcfd0b0badc04bccf286517563c7f1c49aa7ad40e57e6c070bb17b22ae5e24`.
- Oggetto e revisione esatta: review dei fix dell'implementazione r007 (F31–F34). Mandato:
  arbitrato r006 con prompt24. Report e checkpoint autore r007, con le sezioni finali
  «Consegna completa» e «Chiusura amministrativa finale» autorevoli. Non è una review del piano.
- Snapshot, HEAD e impronta verificati prima e dopo: `snapshots/implementation-r007.json`, SHA256
  `6a5235582225134b364818c96d0e3029597d505d39db936b16e0bec039de7ab4`, impronta
  `296a3cc7b90044cb6abb2f38fcf2c5d030820e34f03d361a45d078c5c329b535`, 322 file e 65669 artefatti.
  Verify prima (08:14:54Z) e dopo (08:26:38Z) MATCH, exit 0.
  Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto. Piano `3ac23bce…dd4ab8` ed errata
  `a09cbb8b…e0dfd9` coincidono.
- Indipendenza: non ho letto la review ChatGPT r007, le sue evidenze o checkpoint, né un arbitrato
  r007. Ho letto come input ammessi le due review r006 (ChatGPT e la mia), l'arbitrato r006 e prompt24.
- Giro: fix r007, quarto giro sui confini lessicali. Bloccanti F31–F33, F34 documentale.
  Da valutare F28 globale, F29 e i residui F13/F14/F09 tramite questi fix.
- Base iniziale e delta del giro: da `implementation-fix-context-r006` al comune r007 cambiano
  7 file autore e 4 registri del supervisore (CHANGELOG, documentation/README, next-run, roadmap).
  I file autore sono `observations.py`, `PROTOCOL.md`, le due guide e tre file nuovi non committati:
  test r007, report e dati r007. I 58941 artefatti del contesto sono invariati.
  `metrics.py`, `audit.py` e `safety.py` sono byte-identici ai ricevuti.
- Esito: **NO_GO**

## Ambito e prove

**Vincolo temporale.** Ho letto il prompt alle 08:14:48Z, cioè dopo lo stop per nuovi workload
(08:06:54Z). Per mandato non ho eseguito sonde R, test, CLI, conversioni o rescore. La review è
**statica**: letture, hash, aritmetica sui JSON e il manager stdlib richiesto. Non ho importato il
reader, audit, scorer o altri moduli del prodotto e non ho creato snapshot tecnici propri.

**Letture.** AGENTS.md, temp/HANDOVER, HANDOVER della run, checkpoint
`supervisor-implementation-review-r007`, arbitrato r006 e prompt24 integrali, le due review r006
(la mia per intero, quella ChatGPT tramite l'arbitrato e le sonde ricevute), report e checkpoint
autore r007 nelle sezioni finali. Poi PROTOCOL (r006/r007), le due guide (sezione r007), report e
dati permanenti r007, `final-checks-002`, il riepilogo delle sonde `after-004-*` e `domains-after-004`.

Sul codice: `observations.py` integrale e diff r006 → r007 contro i byte ricevuti (`ffd83938…`):
`html_lexeme`, `inline_scope_end`, `source_scope_end`, `metadata_at`, `command_at`, `code_end`,
`math_end`, `tag_end`, `markdown_lexemes`, `markdown_units`, `pipe_cuts`/`pipe_row`,
`markdown_label`/`markdown_link`, `inline` con `leaf_end`/`sub_inert`, `HTMLTree` e i producer di
`observe` con le barriere. Inoltre `audit.py`, `benchmark.observe_bundle`, le parti di `metrics.py`
su unknown/gate/exit e il test r007 integrale.

**Prove proprie (statiche).** Tutte in `evidence/review-implementation-r007-claude/`:

- `verify-before.*` e `verify-after.*`: argv `python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus
  --label implementation-r007`, cwd radice, entrambi exit 0, «MATCH: oggetti invariati».
- `static_checks.py` e `static-checks.json`: script stdlib proprio, `python3 -I -B`, exit 0.
  Il delta del contesto è di 11 file (7 autore e 4 supervisore); 58941 artefatti con 0 cambiati e
  0 rimossi. Il reader corrente ha hash `5d443387…`. Sui 64 record: 4 PASS/28 FAIL per set,
  `engine_runs` 0, nuove V 0. Ho trovato **zero differenze** su 14 campi rispetto ai dati r006.
  I 256 descrittori (observations/audit/metrics/final) hanno hash e byte corretti, il reader è v7
  su 64/64, quality ed exit coincidono con i file. Ci sono 0 fence `{=latex}` nei 64 raw e 0 blocchi
  fence `{=latex}` nei test. Il reader finale è vincolato negli input-bytes per ciascun workload.
- `static-traces.md`: tracce a mano, riga per riga, dei tre rilievi.

**Prove dell'autore lette, non rieseguite.** Ho letto `final-checks-002`: 24 receipt (22 PASS e
2 FAIL H12 deliberati), 58941 protetti, 146 byte consumati, `own_pending` vuoto. Le sonde
`after-004` danno GPT 21/21, trasporti 14/14, metadata 15/15, Claude Y/K 15/15 e 12/12 domini. Ho
letto anche suite veloce, test mirati e CLI. Sono coerenti con il report, ma sono prove dell'autore.

**Non eseguito.** Per lo stop e per mandato non ho eseguito suite, test, build/reinstall, contratti,
CLI, sonde R, rescore, conversioni, render, M/V, rete o Git di scrittura. I rilievi sono quindi
**solo statici**: non hanno una riproduzione runtime propria.

## Rilievi

| ID | Severità e blocco sì/no | Posizione | Raggiungibilità/input ammessi | Attribuzione e requisito | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLA-R007-I001 | **Alta, bloccante** | `observations.py:163` imposta `limit=inline_scope_end(...)` per il fence `{=latex}`. `:147` e `:151` non aggiornano `limit` mentre `raw_fence` è attivo. `:186–188` e `:189–191` fanno `i=min(i+2,bound)` o `i=bound` con `bound<i`. Anche `:744` (`observe`), audit.py:18 e benchmark.py:53 | Blocco raw LaTeX `` ```{=latex} `` con una riga vuota (o una riga simile a un marker di blocco) seguita da `\` o da un backtick prima della chiusura. È la forma con cui il writer Markdown di Pandoc emette un RawBlock latex con `raw_attribute`. Ambienti con paragrafi interni sono comuni nel TeX. Accade anche in output di altri convertitori. Non presente nei 64 raw e in nessun test | **Regressione r007**: il ramo `raw_mode=='latex'` è nuovo. Nel reader r006 un fence `{=latex}` era saltato come letterale fino alla chiusura e terminava. È la parte F33 «raw TeX» e trasporti (C03/C04). Il requisito «profili/limiti 32 MiB/profondità/safety invariati» presuppone un reader finito | `markdown_lexemes` **non termina**: l'indice torna all'inizio della riga vuota e ripete la stessa sequenza a stato identico. `observe`, `audit` e `observe_bundle` restano bloccati, senza bundle, misura o diagnosi. In R il comando finisce solo per timeout di 900 s; senza R resta appeso. Un documento LaTeX ordinario rende impossibile la misura | Statica: traccia T1 su `\begin{center}` / riga vuota / `\end{center}`. Python: `Pattern.match` con `endpos<pos` non trova corrispondenze, `min(i+2,bound)<i`. Il caso autore `after-004-transport/raw-latex` non esercita il ramo, perché il commento ricalcola `limit` a `:221–222`. Nessuna riproduzione runtime propria (stop) | **Fix minimo**: dentro un raw fence latex `bound` non deve mai essere minore di `i`. Si può ricalcolare il dominio a ogni riga vuota o marker del corpo, oppure usare come dominio l'intero corpo fino alla chiusura, che è il valore del consumer a `:819`. In alternativa basta garantire che ogni ramo avanzi strettamente. **Chiusura**: test e sonda R con fence `{=latex}` che contengono righe vuote e `\`, backtick o `$` dopo la riga vuota: terminazione, chiusura del fence, blocchi seguenti separati, nessuno span spurio. Vanno ripetute le varianti autore raw-latex/raw-html e verificata l'invarianza dei 64 |
| CLA-R007-I002 | Media, **bloccante** | `observations.py:189–191` (backtick non chiuso: `i=bound`), `:195–198` (`$x`/`$$`: `i=bound`), `:224–227` (`tag_end` None: `i=limit`). Consumer `:452`, `:469`, `:489`: il letterale non chiuso termina a `leaf_end`, cioè prima della barriera inerte | Paragrafo con backtick, `$x`, `$$` o `<alpha` non chiusi, seguiti **nello stesso blocco** da `<!--` o `<![CDATA[` che proseguono oltre una riga vuota. Una variante è `` See `x `` seguita a capo da `<!--` (inizio riga, blocco HTML in CommonMark). Delimitatori spaiati e commenti multi-paragrafo sono contenuti ordinari; ammessi sintetici | Residuo di F31: sono stati chiusi solo i casi in un blocco diverso. Criterio C01: «Code/math/tag incompleti … non saltano commenti/CDATA successivi». Arbitrato r006: commento effettivamente aperto intero e inerte; corpo senza prosa/ref/asset/aperture; q/pagina 2 missing autonomi | Lo scanner salta il token, quindi nessuno span e nessuna barriera. Il corpo del commento diventa Markdown attivo: figura con `p.png` aperta e hashata dall'audit, `UNSAFE_REFERENCE` con `../outside.png`, heading dal corpo, `q` soddisfatto dal testo commentato (missing e pagina 2 mascherati). L'unknown `unclosed inline code`/`math` non ha `resource_syntax`, quindi l'audit atteso è COMPLETE se l'asset esiste. Non c'è PASS automatico (reader UNAVAILABLE), ma foglie, reference e misure sono falsate. Con `$5 <!--` il token è invece riconosciuto: il risultato dipende da un delimitatore che resta testo letterale | Statica: traccia T2. Copertura autore: i 6 prefissi di `test_F31_literal_domain_does_not_suspend_next_real_token`, i casi GPT `comment-open-*`, i Claude Y e i 12 domini hanno tutti il delimitatore in un blocco **separato** da una riga vuota. Nel consumer `leaf_end` prevede una barriera che lo scanner non può produrre nello stesso dominio | **Fix minimo**: il letterale non chiuso non deve nascondere un token reale successivo nel proprio dominio. Ad esempio, senza chiusura avanzare oltre il solo delimitatore, come per testo letterale, e continuare a cercare token. Per coerenza col consumer si può anche fermare il salto al primo `<!--`/`<![CDATA[`/PI/declaration riconosciuto da `html_lexeme`. **Chiusura**: varianti backtick/`$x`/`$$`/`<alpha` × stessa riga/soft line × commento/CDATA chiuso/EOF, con immagine esistente e URI unsafe. Token intero, nessuna apertura dal corpo, `q`/pagina 2 missing. Restano positivi i code/math bilanciati sulla riga e i casi autore |
| CLA-R007-I003 | Media, **bloccante** | `observations.py:183–185` restringe `bound` alla fine della riga dentro una label. `markdown_label :355` cerca la chiusura del code fino a `inline_scope_end`, cioè oltre la riga. Seguono `:189–191` (`i=bound`) e `:214–220` (span del token fino a EOF) | Link o immagine valida la cui label contiene un inline code spezzato su soft line con dentro `<!--`, `<?`, `<![CDATA[` o `<!x`. Ad esempio `` [a `b `` a capo `` c <!-- ` d](https://…) ``. È CommonMark valido, perché il code ha precedenza sulle parentesi, e il reader lo supporta nel consumer. Ammesso come sintetico; non presente nei 64 | Stessa classe di CLA-R006-I001/GPT-R006-I002 (F32/F33): un falso opener in una foglia letterale estende il blocco. Criteri C04 (label, opener a cavallo di soft line, «nessun raw/text/offset/line … occultato») e C02/C03 «letterali non assorbiti» | Lo scanner vede `<!--` fuori dal code e crea uno span fino a `-->` o EOF. `markdown_units` unisce tutto il seguito nel primo paragrafo. Il consumer ritrova lo stesso link con il code chiuso e non emette diagnosi. Ne risulta un audit COMPLETE con 0 token, mentre heading e paragrafi successivi vengono assorbiti e lo scorer li conta come perdite del convertitore. Le guide e il report r007 affermano invece che scanner e consumer condividono gli atomi | Statica: traccia T3. Il test della label usa `$5`/commento senza code; il caso quote soft-line non riguarda le label | **Fix minimo**: lo scanner usi per il code nella label lo stesso confine del walk `markdown_label`. In alternativa il walk usi la fine riga, ma lo stesso confine deve valere in tutti e due. **Chiusura**: link e immagine con code su soft line contenente comment/PI/CDATA/declaration, chiuso in altro blocco o mai: stessi blocchi del caso senza opener, COMPLETE, 0 token. Restano positivi la label EOF senza destinazione e i casi `test_F31_source_barrier…` |

### Fix e requisiti valutati

- **F31 (C01).** Sono chiusi i casi con il delimitatore in un **blocco diverso**: Y3–Y9, le otto
  varianti GPT, K1–K5, lo stop a blank/marker con `source_scope_end`/`inline_scope_end` e la non
  ripresa dentro i fence. Lo vedo dal codice e dalle sonde dell'autore che ho letto. **Non chiuso
  come classe**: CLA-R007-I002 (stesso blocco).
- **F32 (C02).** `AUTOLINK` è unico in scanner (`:216`), inline (`:518`), `source_scope_end` e
  producer HTML (`:861`); non entra più in `html_stack`. I test r007 coprono http/https/mailto ×
  comment/PI/CDATA/declaration/`</>` × code e la chiusura in altro blocco. Staticamente non ho
  trovato regressioni: **chiuso** nei domini verificati.
- **F33 (C03).** `metadata_at`, `command_at`, la metadata dell'heading e `pipe_cuts` sono condivisi
  e ref/eqref/label/foo restano locali. Per paragrafi, heading, celle, link, span e code è chiuso. Il
  nuovo ramo raw LaTeX **non termina** (CLA-R007-I001) e la label con code su soft line non è
  coerente (CLA-R007-I003).
- **F34 (C06).** PROTOCOL, le due guide e il report r007 rettificano r006 senza riscriverlo. E9,
  i FAIL S003/S004, launcher e whitelist sono conservati e il gold PROVISIONAL è distinto dalla V
  reale. Due affermazioni sono smentite dai rilievi: «raw … latex gli atomi finiti» (I001) e
  «scanner e consumer condividono atomi finiti» (I003). Vanno rettificate dopo i fix.
- **F27/F30/F28 locale (C05).** Sono preservati: `parse_endtag` intercetta `</>`; CDATA e commenti
  usano `html_lexeme`; la reason neutra è invariata. Le sonde autore basic-14/15/16 e
  diagnostic-* sono rispettate. F28 globale e F29 restano **parziali** per I001–I003. I residui
  F13/F14/F09 sono unsupported localizzati, salvo i casi dei rilievi.
- **C05/C07 misure.** Lo scorer è byte-identico e F26 è distinto: condizionato contro globale, con
  TRIVIAL_ZERO non completo. Ho verificato dai file le cifre: TXT 0/89, DOCX 11/182, EPUB 8/186,
  RTF 19/162 condizionato contro 13/162 globale, TeX WER None, 1/152 condizionato, globale 0
  TRIVIAL_ZERO. 4 PASS/28 FAIL per set, 0 differenze da r006, 0 motori e 0 nuove V. I nuovi ID
  r007b sono legati ai file. I 64 raw non contengono gli input dei rilievi, quindi le misure non
  cambierebbero; questo non chiude i rilievi.
- **C08.** Il reader `5d443387…` è vincolato negli input-bytes per workload di test004, delle sonde
  004, dei domini, delle CLI 001/E9 002, del rescore 002, del checker e della suite veloce. I 22
  PASS e i 2 FAIL H12 sono conservati. Non ho rieseguito gate S/B/I né preflight; non trasferisco
  alcun PASS.
- **C09.** Il wall attuale è circa 49,3 h dalla entry (2026-10-08T07:06:54Z), lo stop è già
  trascorso e il ledger autore di 40,625 h è storico. Dopo lo stop non ho avviato workload né
  nuove sonde. L01/L02/L04/L05, Chrome NONPASS e il gap helper12 non sono sanati.

## Motivazione dell'esito e limiti

NO_GO. CLA-R007-I001 è una regressione introdotta da r007: un blocco raw LaTeX ordinario con una
riga vuota impedisce al reader di terminare. Così si perde la misura e non si ottiene alcuna
diagnosi. CLA-R007-I002 e CLA-R007-I003 mostrano che il contratto scanner → confine → consumer non è
ancora uniforme. Nel primo caso il letterale non chiuso nasconde un token reale nello stesso
blocco. Nel secondo, nella label scanner e walk usano due confini diversi per lo stesso code. Sono
gli stessi effetti già giudicati bloccanti negli arbitrati r005/r006: corpo del commento attivo,
asset aperti, missing mascherato, blocchi validi assorbiti con COMPLETE e perdite inventate.

I fix sono locali e finiti e non serve un parser universale. Bisogna garantire un avanzamento
stretto nel raw latex, non saltare i token dopo un letterale non chiuso nel proprio dominio e usare
lo stesso confine per il code della label. Sono chiusi F32, F27/F30, F28 locale, F26 e la misura
dei 64; F31 è chiuso tra blocchi diversi.

Limiti. Per lo stop dei workload i tre rilievi sono **statici**: ho tracciato il codice riga per
riga, ma non ho una riproduzione runtime propria. L'impatto su audit e scorer è dedotto dal codice
(audit.py, metrics.py), non osservato. La raggiungibilità da Pandoc per I001 è argomentata, non
misurata con una conversione. Le prove dell'autore sono lette, non rieseguite. Questo esito non
certifica altri motori né L04.

Costi propri: la review va dalle 08:14:48Z alle 08:28Z circa (wall circa 14 minuti). Due verify
del comune: 25,7 s di wall e 25,7 s di CPU (user+sys) la prima, 19,3 s e 19,3 s la seconda. Lo
script statico dura circa 1 s ed è CPU amministrativa non misurata separatamente. Le evidenze
occupano circa 64 KB. Nessuno snapshot proprio. Rete, acquisizioni e servizi esterni: €0.
Nessun processo pendente.
