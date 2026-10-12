# Review — implementazione — run-a002-fase1-corpus — r013 — Claude

- **Ruolo e modello.** Reviewer Claude, Anthropic, Claude Opus 5.5 (`claude-opus-5-5[1m]`), Claude
  Code nell'estensione VS Code. Effort e ID della chat non esposti. Chat nuova, distinta da autore,
  supervisore e altro reviewer; nessuna delega, subagente o impersonificazione.
- **Prompt.** `prompts/45-review-implementation-r013-claude.md`; mandato43 F54–F57 dopo NO_GO r012.
- **Oggetto.** Comune `snapshots/implementation-r013.json`, SHA256
  `ac928bb37a34e637b352013241652132e03daa86412c32e2705655e92549bccc`, impronta
  `062e6fd310234f0e3953f0c48b3c9478f94b5ac2223a8d2f430534ec4d141685`. Reader
  `observations-v13-ordered-guards-fences`, 79328 B, `27d1b580…3c20c43ad`; V12 fisico
  `work/implementation-r013/r012-observations.py` `5cabe677…bc22`; piano `3ac23bce…4ab8`, errata
  `a09cbb8b…dfd9`. Hash ricalcolati da me.
- **Git.** `feature/run-a002-corpus`; HEAD/dev `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice
  vuoto; `git diff --check` exit 0. Nessun comando Git mutativo o remoto.
- **Verify del comune.** Ingresso 16:30:44Z: **MATCH**, exit 0, 47,02 CPU-s. Verify finale nel
  checkpoint.
- **Indipendenza.** Non ho aperto review, evidenze o checkpoint ChatGPT r013 né arbitrati r013;
  nessuno compariva nei listing consultati.
- **Esito: NO_GO.** F54 (ordine e visite) e F57 sono chiusi nelle mie sonde. Restano due requisiti
  bloccanti, riprodotti sotto R con AST Pandoc: la classe F55 non è chiusa fuori dal contesto interno
  del container (I001), e l'HTML dentro gruppi `{…}` o code di opener div che Pandoc non accetta come
  attributi non raggiunge mai la guardia (I002, ereditato). Esito in entrambi: **COMPLETE** con script
  attivo non guardato, path evasivo non segnalato o asset sicuro perso.

## Letture

- **Contesto.** AGENTS; temp/HANDOVER; prima parte di PROJECT-CONTEXT; STATE (intestazione),
  HANDOVER e brief della run; checkpoint supervisore r013; skill manage-implementation-run e
  verify-conversion-fidelity; ADR0001; addendum risorse r012-r001; nota utente no deadline;
  `budget-final.json` del supervisore.
- **Integrali.** Arbitrato r012, mandato43, recovery-method-r013; review e checkpoint r012 Claude e
  ChatGPT; il mio predecessore r012 (cases/probe/rrun/check) come base degli strumenti; report e
  checkpoint autore r013; censimento `implementation-r013-guard-inventory.md`; report permanente r013.
- **Codice.** Reader V13 **integrale** (1237 righe) e diff V12→V13 (`reader-v12-v13.diff`);
  `audit.py`; `observe_bundle`; sezione r013 del PROTOCOL; sezioni r013 delle due guide; elenco
  estensioni pinned `pandoc-markdown-extensions-3.1.3.txt`.
- **Prove autore lette nei campi decisivi.** cases-004/005, check-baseline-003/004/005,
  check-after-005/006, freeze-004/005, overlay legacy, diff test r006 (F29), due diff di mutazione e
  check-mutation-002, delivery-check-003, budget-consegna, process-and-resources-final-001,
  derived-check-002, observation-delta-002, final-binding-003, delivery-verify-001.

Non letti integralmente: dati permanenti (12,8 MB), 64 catene, receipt S/B/I, CLI, H01–H12, test r013
riga per riga. Per questi mi affido alle receipt ricevute, senza attribuirmene la verifica.

## Metodo e sonde

- **Statico prima.** Dalla lettura del reader ho isolato quattro contesti assenti dalla matrice
  autore. Scansione dei 285+16 casi, della fixture350 e dei test r002–r013: zero casi per ciascuno
  (`previous` della matrice fence è solo interno all'host; `preceding_block` solo heading/code).
- **Casi.** 33 microcasi in `cases.py`, max 199 B. Attesi contrattuali, classe AST prevista e
  previsioni statiche V13/V12 congelati prima di qualunque esecuzione (`freeze-receipt.json`,
  16:41:21Z, SHA256 `48e02b68…02ced7`). Per i casi con premessa di dialetto l'atteso è condizionato:
  la classe AST dell'apertura marcata (`data-c13="…"`) sceglie tra `if_active` e `if_inert`, che
  restano manuali.
- **Esecuzione.** `rrun.py probe-001 300` → `run_offline.py` Firejail `--net=none`, interprete
  stage-s010 (managed 3.12.13, `check_python_origin` PASS dentro R). Preflight: origine, inventory
  runtime s010, inventory propria. Il `check_preflight` S/I generale s010 non è usato: è STALE per
  costruzione dopo i quattro registri supervisore, nessun PASS trasferito.
- **Esito R.** PASS. Snapshot tecnico proprio `review-r013-claude-probe-001` MATCH prima e dopo;
  `inputs_unchanged` true; socket pulito; workload exit 0; 8,86 s wall.
- **Collector.** `probe.py`: V13 e V12 fisico sugli stessi byte, `observe_bundle` + `audit`. Una spia
  su `guard_event`/`drain_guards` registra per ogni singola `observe` gli eventi drenati e la
  molteplicità grezza. AST `/usr/bin/pandoc -f markdown -t json` da stdin: 3.1.3, SHA256
  `3dd27364…d701b`.
- **Checker host.** `check.py`, stdlib, nessun import prodotto. Posizioni attese ricavate dal raw con
  `str.find`, indipendenti dal reader → `check-002.json`.
- **Errore mio conservato.** `check-001-failed-syntax.py`: SyntaxError nel mio checker, corretto in
  002.
- **Premessa mia smentita.** K2c (lista lazy): prevedevo HTML attivo, l'AST dà CodeBlock e V13 è
  coerente. Non è un finding.
- **Previsioni.** Le mie 33 previsioni statiche V13 e le 33 V12 risultano tutte confermate.

## Rilievi

### CLA-R013-I001 — ownership del fence decisa senza il contesto di paragrafo esterno al producer (media, bloccante)

**Input decisivi** (tutti seguiti da `\n\n# Visible heading\n\nAfter words.\n`; `S` è
`<script data-c13="…">x</script>`; `I` è `<img src="p.png" data-c13="…">`):

| Caso | Raw | AST Pandoc 3.1.3 dell'apertura | V13 | V12 |
| --- | --- | --- | --- | --- |
| K1a quote | `Intro text\n> ~~~\n> S\n> ~~~` | `Plain[…'>','~~~']`, **RawBlock html** | COMPLETE | COMPLETE |
| K1b quote img | `Intro text\n> ~~~\n> I\n> ~~~` | **RawInline html** | COMPLETE, assets [] | COMPLETE |
| K1c list | `Intro text\n- ~~~\n    S\n    ~~~` | **RawBlock html** | COMPLETE | COMPLETE |
| K1d note | `Body[^n].\nIntro text\n[^n]: ~~~\n    S\n    ~~~` | **RawBlock html** | COMPLETE | COMPLETE |
| K2a lazy quote | `> Intro text\n~~~\nS\n~~~` | `BlockQuote[Plain[…'~~~'], RawBlock html, …]` | COMPLETE | COMPLETE |
| K2b lazy quote img | `> Intro text\n~~~\nI\n~~~` | **RawInline html** | COMPLETE, assets [] | COMPLETE |
| K2d lazy note | `Body[^n].\n\n[^n]: Intro text\n~~~\nS\n~~~` | `Note[Plain[…'~~~'], RawBlock html, …]` | COMPLETE | COMPLETE |
| K2e figura | `![a](../../../../../temp/run-a002-fase1-corpus/reviews/p.png)\n~~~\nS\n~~~` | `Plain[Image,'~~~']`, **RawBlock html** | COMPLETE | COMPLETE |

**Controlli corretti in V13.**

- K1e backtick: Code inline in Pandoc, inerte nel reader.
- K1f `~~~{=html}`: UNSAFE script.
- K2c lista lazy: Pandoc dà CodeBlock.
- K2f backtick lazy: CodeBlock.

**Percorso** (traccia statica, confermata dalla sonda):

1. **Container non separato.** `PARAGRAPH_BOUNDARY` (`observations.py:41`) lascia che `>`, i marker
   di lista e `[^n]:` interrompano un paragrafo radice. Le estensioni pinned del progetto dicono il
   contrario: `+blank_before_blockquote`, `-lists_without_preceding_blankline`, `+blank_before_header`.
   Il producer di container (`read_parts`, 1081–1095) parte con `paragraph_open=False`
   (`markdown_lexemes`, 308). Così `fence_decision` (49–77) assegna il code al tilde iniziale; il corpo
   viene saltato (341) e `inline` (710–716) aggiunge un segmento `code` senza unknown.
2. **Continuazione lazy o riga di figura.** Il producer radice chiama `fence_decision(raw,starts[i])`
   senza `paragraph_open` (1152). Lo scanner radice sulla stessa riga ha invece `paragraph_open=True`
   (347) e non riconosce il fence: è un disaccordo scanner/producer sulla condizione che il mandato
   chiedeva condivisa.

**Requisiti violati.**

- Mandato43 F55: «stato precedente paragrafo/inizio/separazione … sono parte della decisione»; «lo
  script reale riceve guardia UNSAFE/detail; img sicura conservata … mai COMPLETE vuoto»; «B7 radice
  stessa condizione ereditata inclusa».
- Criteri C03 e C04.
- Il PROTOCOL r013 promette: «Un marker tilde dopo testo non nasconde il corpo HTML reale».

**Impatto.** È lo stesso esito di CLA-R012-I002, arbitrato media/bloccante: falso COMPLETE con script
attivo non guardato e asset sicuro perso senza diagnosi. Non ho osservato fetch né esecuzione HTML.

**Attribuzione.**

- (1) Deriva dalla proprietà dei fence nei container introdotta in r012 (in r012 Claude B6, V11 dava
  UNSAFE per il tilde iniziale del quote) e non è chiusa da r013. V12 e V13 eseguiti da me sugli
  stessi byte; V11 non eseguito in questo giro.
- (2) È il percorso radice ereditato, lo stesso confine di B7 che l'arbitrato ha incluso in F55.

**Perché le prove r013 non lo vedono.** La dimensione `previous` del generatore (begin/text/blank)
vive dentro l'host, come lo stato `paragraph_open` dell'implementazione. Generatore e codice
condividono la premessa che il contesto rilevante cominci all'inizio del container: è il caso
descritto nella recovery-method-r013. La mutazione fence-everyline altera solo l'interno di
`fence_decision`, non l'inizializzazione del contesto nei chiamanti.

**Correzione attesa.** La decisione di blocco deve rispettare le regole pinned anche ai confini tra
producer:

- inizio container non separato da un paragrafo;
- continuazione lazy di quote e note;
- riga di figura seguita da tilde.

In alternativa, dove l'ownership non è dimostrata, il corpo va guardato o dichiarato non
rappresentato. Le prove devono includere il contesto **esterno** al container. Non serve un parser
universale.

### CLA-R013-I002 — HTML dentro gruppi `{…}` e code di opener div consumati come metadata senza guardia (media, bloccante, ereditato)

| Caso | Raw | AST Pandoc | V13 | V12 |
| --- | --- | --- | --- | --- |
| K4a span | `See [a]{<img src="../x.png" …>} end.` | `'[a]{'`, **RawInline html**, `'}'` | COMPLETE (atteso UNSAFE `../x.png`) | COMPLETE |
| K4b heading | `# Title {S}` | `Plain['#','Title','{']`, **RawBlock html** | COMPLETE | COMPLETE |
| K4c link | `See [a](https://example.invalid){S} end.` | `Link`, `'{'`, **RawBlock html** | COMPLETE | COMPLETE |
| K4d code | ``See `x`{<img src="../x.png" …>} end.`` | `Code "x"`, `'{'`, **RawInline html** | COMPLETE | COMPLETE |
| K4e span img sicura | `See [a]{I} end.` | **RawInline html** | COMPLETE, assets [] | COMPLETE |
| K3a div | `::: <img src="../x.png" …>\nBody.\n:::` | `Para[':::', RawInline html, …]` | COMPLETE | COMPLETE |
| K3d div | `::: S\nBody.\n:::` | `Plain[':::']`, **RawBlock html** | COMPLETE | COMPLETE |

**Controllo.** K4f `[a]{.cls data-c13="k4f"}` è un attributo vero in Pandoc (Span) e dà COMPLETE in
V13: corretto.

**Percorso** (statico, confermato):

1. `ATTRIBUTE`/`metadata_at` (43, 101), la regex span (399) e `peel_attrs` (505–508) accettano
   qualunque contenuto `[^{}]*` come metadata.
2. Lo scanner salta l'intervallo (`destinations`/`metadata`, 355–356, 390, 401), quindi nessun evento
   HTML viene raccolto.
3. `attrs` (486–502) parcheggia i token non riconosciuti in `values['unknown']`.
4. Il ramo div (1141–1150) mette l'intera coda nei metadata del container.
5. `audit` non legge gli attrs (`audit.py:40–43`).

Resta quindi una fonte Unsafe del reader che non diventa mai evento.

**Requisiti violati.**

- F54/C01 chiede il censimento completo delle fonti Unsafe; il PROTOCOL r013 afferma «Tutte le fonti
  Unsafe del reader confluiscono in eventi».
- C04 sull'ownership.
- AGENTS: HTML non fidato da gestire esplicitamente.
- Principio di non dichiarare completo ciò che non lo è.

**Impatto.** Come I001: falso COMPLETE con script attivo, path evasivo accettato in silenzio, asset
sicuro perso. Si aggiunge la perdita del testo letterale `{…}` che Pandoc conserva.

**Attribuzione.** Ereditato, non regressione r013: V12 è identico sugli stessi byte, e regex e `attrs`
sono invariati nel delta. Il censimento r013 non elenca la proprietà «il gruppo `{…}` è davvero
attributo», e nessun caso della matrice, della fixture350 o dei test contiene HTML dentro le graffe o
nella coda di un opener div. Secondo i criteri comuni, un difetto ereditato raggiungibile può violare
un requisito essenziale. Lascio all'arbitro il perimetro; non abbasso la severità.

**Correzione attesa.** Riconoscere come metadata solo la sintassi di attributo pinned (`#id`, `.class`,
`key=value`/quoted, eventuali speciali ammessi). Altrimenti il gruppo resta testo e il suo HTML passa
dalla guardia. Lo stesso vale per la coda dell'opener div. In alternativa, guardare comunque il
contenuto del gruppo. Una regex più stretta da sola non basta: serve un caso per producer.

## Osservazioni non bloccanti

- **CLA-R013-O001 — sovra-segnalazione** (K1h, stessa radice di I001-1). `Intro text\n# Heading
  {onclick=…}`: Pandoc lo legge come testo (`+blank_before_header`, AST Str); V13 e V12 danno UNSAFE
  `Active markdown attribute`. È un falso positivo di fedeltà, non un buco di sicurezza; da correggere
  insieme a I001.
- **CLA-R013-O002 — baseline 132/133 chiarita, senza riscrivere dati.**
  - `check-baseline-004.json` dà 133/285: confronta l'output V12 di `baseline-004` con gli attesi di
    cases-004.
  - `freeze-005` (14:58:26Z) corregge tre premesse dopo l'AST: `context-list-code` e le due
    `context-note-*-heading`, con raw identici.
  - `check-baseline-005.json` ricontrolla gli stessi output contro cases-005 e dà **132/285**.
    Differisce solo `context-list-code`: V12 ora fallisce su detail/visite.
  - Il permanente `before_checks` vale 132 e coincide con il check 005. Il report cita «su
    baseline-004», cioè la directory degli output, non la receipt 005. È un difetto di citazione: non
    c'è conferma falsa del checker né un errore del prodotto.
- **CLA-R013-O003 — collisione di namespace** (statico, non eseguito). `public()` (1055–1058) rimuove
  la chiave `_guard_coverage` a qualunque profondità, compresi i valori di `attrs` forniti dall'utente
  (`{_guard_coverage=x}`). Impatto trascurabile; si può limitare la pulizia ai nodi prodotti dal
  reader.

## Chiusure verificate dalle mie sonde

- **F54 ordine e visite.** 11 casi con sequenza esatta di eventi per singola `observe`; gli eventi
  della `observe` di `audit` sono identici:
  - A1–A4 (detail `../first.png`; in V12 era `Active markdown attribute`);
  - A5;
  - A7, ora ATTR per primo nello stesso dominio inline (in V12 era `../second.png`);
  - A8 come limite documentato;
  - A9 (GPT26: script, attr, img last);
  - A10: UTF8/CRLF, sei cause tra paragrafo, cella, caption di figura e attrs. Posizioni char
    8/21/69/96/122/151, byte coerenti; la coda viene sempre visitata;
  - A11: errore d'albero (attributo duplicato), poi img, attr e img del blocco successivo;
  - A12: reset, con COMPLETE dopo una sessione unsafe.

  Le visite duplicate sono solo le doppie chiamate attese (ledger e proiezione) dello stesso token
  (A7, A10). Il dedupe le collassa senza nascondere aperture distinte.
- **F57.** P1 con tabella pipe e caption, figura e tabella HTML con caption: V13 non ha chiavi `_…` a
  nessuna profondità, V12 ne ha 6 `_guard_coverage` annidate. Le 640 rimozioni su 56 osservazioni e i
  19 campi invariati sono ricevuti (observation-delta-002, derived-check-002), non ricalcolati.
- **Mutazioni.** Due copie private sui byte finali, con diff mirati: `raise` in `attrs` e rimozione
  dei due `return None` in `fence_decision`. Il checker le respinge 0/3 ciascuna su detail, visite e
  asset. Sono prove reali delle due assunzioni richieste, non della copertura dei chiamanti (vedi
  I001).
- **Overlay EOF ed F29.** I 20 attesi r012 cambiati e F29 r006 sono coerenti con F55/B9 e con l'AST:
  l'EOF ordinario non dà ownership code; HTML guardato o img conservata/residuo; il commento opaco
  resta tale. La fixture350 originale è preservata.

## Criteri comuni

| ID | Esito | Motivo e prova |
| --- | --- | --- |
| C01 F54 fonti | **FAIL** | Ordine assoluto e prima detail corretti per attrs e HTML raccolti (A1–A11, sonda), GPT26 e A1–A7 chiusi; A8 resta limite. Il censimento però non comprende HTML in `{…}` e code div (I002). |
| C02 F54 visite | **PASS** (sonda) | Identità, molteplicità, sequenza e posizioni char/byte esatte; errore d'albero non troncante; reset; nessun accesso URI. |
| C03 F55 fence | **FAIL** | B1–B4/B7/B9 originali nella matrice autore e mutazione; K1e/K1f/K2c/K2f corretti. La classe resta aperta fuori dal container (I001). |
| C04 ownership | **FAIL** | Il metodo AST sul corpo discriminante è corretto e ha corretto tre premesse dell'autore. Ownership non dimostrata assegnata a code (I001) e a metadata (I002). |
| C05 F56 prove | **Parziale** | Tabella censimento → proprietà → casi pubblicata, freeze prima del runtime, AST obbligatorio, due mutazioni reali. Dimensioni del generatore allineate all'assunzione del codice; nessun caso per I001/I002. |
| C06 regressioni | **PASS** nei rami sondati | F44/owner, raw `{=html}` container (K1f), F29/EOF overlay giustificati; 350/153/303/308 e suite 2997+310 ricevuti, non rieseguiti. |
| C07 F57 coverage | **PASS** (sonda) | Nessun privato nel JSON pubblico, anche annidato in cell/caption/figure/HTML table; O003 minore. |
| C08 attribuzione | **PASS** nei limiti | S008/S010, 10 moduli e 37 helper, 25 receipt (23 PASS + 2 H12 FAIL) e binding003 coerenti nei campi letti. 132/133 chiarito (O002). Nulla rieseguito di suite/CLI/rescore. |
| C09 documenti | **FAIL** (minore) | PROTOCOL e guide promettono «tilde dopo testo non nasconde» e «tutte le fonti»: smentiti da I001/I002. Gli altri limiti (F36, quattro r010, A8, L01/L02/L04/L05, NONPASS) sono dichiarati. Nessuna deadline né GO autore. |

## Esito e limiti

**NO_GO.** I001 e I002 sono bloccanti: input raggiungibili nel profilo, AST Pandoc pinned eseguito,
V13/V12 sugli stessi byte, atteso congelato prima. F54 e F57 sono chiusi nei miei casi. Non dichiaro
la chiusura universale di nessuna classe.

Non eseguiti:

- suite, build, reinstallazione, rescore, CLI, H01–H12, S/B/I e catene;
- V11 (attribuzione storica tramite la review r012);
- ricerca esaustiva di altri contesti di blocco.

Pandoc è usato solo come oracolo sintattico, non come prova di sicurezza o fedeltà. L01/L02/L04/L05,
F36 e gli altri NONPASS storici restano invariati.

## Costi e passaggio

- **CPU misurata finora.** Verify d'ingresso 47,02 s; R probe-001 10,08 s (wrapper con figli); checker
  0,20 s (001 fallito) e 0,14 s (002). Letture, scansioni e hash host non misurati, piccoli.
- **Base.** Ledger supervisore `budget-final.json`: CPU visibile lower 19168,993788 s; 631,006 s allo
  stop prima delle review. Costi concorrenti ChatGPT r013 non noti a me.
- **Disco e memoria.** Evidenze circa 4,9 MB allocati più uno snapshot tecnico; picco RSS del wrapper
  316232 KiB.
- **Rete, acquisizioni, invii.** 0; €0.
- **File fuori dalla cartella evidence.** `snapshots/review-r013-claude-probe-001.json` (snapshot
  tecnico, non un'approvazione).

Totale, verify finale e budget sono nel checkpoint. Prossimo ruolo: il supervisore, dopo entrambi i
report reali, per l'arbitrato. Nessun Git, cleanup, fase 2, motore, ML o nuova V.
