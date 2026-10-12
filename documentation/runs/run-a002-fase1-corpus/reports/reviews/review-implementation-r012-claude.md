# Review — implementazione — run-a002-fase1-corpus — r012 — Claude

- **Ruolo e modello.** Reviewer Claude, Anthropic, Claude Opus 5.5 (`claude-opus-5-5[1m]`),
  Claude Code nell'estensione VS Code. Effort e ID della chat non esposti. Chat nuova, distinta da
  autore, supervisore e altro reviewer; nessuna delega, subagente o impersonificazione.
- **Prompt.** `prompts/42-review-implementation-r012-claude.md`; mandato40 F50–F53 dopo NO_GO r011.
- **Oggetto.** Comune `snapshots/implementation-r012.json`, SHA256
  `a8ad87d33d3d5ea6fecf40557cb0e470b1282bd043c312b005fd7ad17b048625`, impronta
  `c5c7eb56c0933f8653a6a71354b224d0c885cfe6105a4ffb99dc982be9108378`. Reader
  `observations-v12-shared-atoms` `5cabe677…bc22`; v11 fisico `af5299bb…8482`; piano `3ac23bce…4ab8`,
  errata `a09cbb8b…dfd9`; snapshot autore delivery001 `fd15440f…5dbc5` (non GO).
- **Identità Git.** `feature/run-a002-corpus`; HEAD, dev, origin/dev locale e merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto. Receipt in `evidence/review-implementation-r012-claude/entry.json`.
- **Verify del comune.** Ingresso 13:07:31Z: **MATCH**, exit 0, 43,23 s user + 3,19 s sys. Il
  verify finale dopo gli output è registrato nel checkpoint.
- **Indipendenza.** Non ho letto review, evidenze o checkpoint ChatGPT r012, né un arbitrato r012. La
  cartella `evidence/review-implementation-r012-chatgpt/` è comparsa solo come nome in un elenco e non
  è stata aperta.
- **Esito: NO_GO.** Due requisiti bloccanti, entrambi verificati con sonde proprie eseguite sotto R.
  Il primo è una regressione r012 sull'ordine della prima unsafe, il secondo una regressione r012 sui
  fence ordinari nei container, confermata con l'AST di Pandoc.

## Letture

- **Contesto.** AGENTS; temp/HANDOVER; STATE, HANDOVER e brief della run; checkpoint del supervisore
  r012; skill manage-implementation-run e verify-conversion-fidelity; addendum risorse r012 e nota
  utente no deadline.
- **Integrali.** Arbitrato r011; mandato40; nota recovery-method-r012; review e checkpoint Claude e
  GPT r011 con le tracce T1–T4 e la traccia GPT; report e checkpoint dell'autore r012; nota
  inventario; report permanente r012.
- **Codice.** Reader v12 integrale e diff dal v11 fisico, salvato in `reader-v11-v12.diff`. Inoltre
  `audit.py`, `benchmark.observe_bundle`, la sezione r012 del PROTOCOL, le sezioni r012 delle due
  guide, il test e la fixture r012 (struttura e scansione dei 350 raw) e lo schema del generatore.
- **Prove autore**, lette nei campi riassuntivi: delivery-check-001 (PASS_SCOPED_BYTES, 10 moduli,
  37 Python, 350 oracoli, 67 CLI), final-delivery-inputs-001, budget-delivery-final-001,
  process-and-resources-final-001, cost-001, derived-check-001 (PASS 64, 0 delta) e
  observation-delta del supervisore.

Non letti integralmente: dati permanenti (2,5 MB), 64 catene una per una, receipt S/B/I individuali,
CLI, H01–H08/H12, arbitrati r010/r009/r008 e mandati precedenti. Per questi mi affido alle receipt
dell'autore e alla ricezione del supervisore, senza attribuirmene la verifica.

## Sonde proprie

Le sonde sono 31 casi sintetici in `cases.py`. Gli attesi contrattuali e le previsioni statiche
v12/v11 sono stati scritti e hashati prima di qualunque esecuzione del prodotto (`freeze-receipt.json`,
13:24:34Z, `cases.py` SHA256 `9a715a2d…d284`).

Ogni caso passa sullo stesso raw per `observe_bundle` e `audit`, sia con il reader v12 sia con il v11
fisico caricato come modulo separato. Una spia registra gli eventi validati da ogni
`inline_html_events` v12. Solo per i casi con premessa di dialetto, cioè B e D1, ho eseguito il
microparsing AST con `/usr/bin/pandoc -f markdown -t native` 3.1.3 già installato, solo stdin.

- **Esecuzione.** probe-005, `rrun.py` con interprete stage-s005 (managed 3.12.13,
  check_python_origin PASS prima di ogni import). Preflight dentro R: origine PASS, inventory s005
  `PASS_EXTERNAL_INPUTS dbe2048e…`, inventory propria `06190233…`.
- **Isolamento.** Snapshot tecnico proprio `review-r012-claude-probe-005` MATCH prima e dopo;
  Firejail `--net=none` (netns interno distinto dall'host); comando exit 0 in 0,56 s; figli reaped,
  socket pulito; `inputs_unchanged` true; byte di input identici prima e dopo.
- **Fallimenti conservati.** probe-001: `--bundle` non ammette comandi. probe-002: inventory s004
  STALE su PROTOCOL.md per costruzione, quindi uso s005 senza sanare il vecchio gate. probe-003: mio
  errore di `sed`. probe-004: mio `mkdir` senza `parents`.
- **Errori dei miei oracoli**, in `check-005-corrections.json`:
  - A13 atteso sbagliato: `<b>` è fuori dal profilo passivo.
  - A12: il checker sommava le visite di due `observe`.
  - Premessa Pandoc: attendevo RawInline e l'AST dà RawBlock per lo script, con sostanza invariata.
- **Checker.** `check.py`, host stdlib → `check-005.json`.

## Rilievi

### CLA-R012-I001 — la prima unsafe non è più la prima in ordine sorgente (media, bloccante)

- **Input.** `Text <img src="../first.png"> end.` seguito da `# Heading {onclick="x"}`, più la coda.
  Varianti con `::: {onclick="x"}`, fence ```` ```{onclick="x"} ```` e `![cap](../../../../../temp/run-a002-fase1-corpus/reviews/p.png){onclick="x"}`
  (A1–A4).
- **Atteso.** Detail `Path evasivo: '../first.png'`, secondo mandato40 F52, C03 e PROTOCOL r012: «la
  sessione conserva la prima Unsafe … `observe` solleva la prima diagnosi».
- **Osservato v12.** 4/4 `Active markdown attribute`.
- **Osservato v11.** 4/4 `../first.png`.
- **Controlli.** A5 (attributo per primo) e A6 (attributo sicuro) sono corretti in entrambi.
- **Percorso.** `report_unsafe` (`observations.py:151–154`) differisce la prima violazione nella
  sessione di `observe` (983–989). `attrs()` invece solleva `Unsafe` direttamente (442). Lo
  raggiungono i producer di blocco successivi: div 1077, info del fence 1100, heading via
  `peel_attrs` 1120/452, figure 1145. L'eccezione esce da `_observe` prima di `raise session['first']`
  e sostituisce la diagnosi già registrata.
- **Impatto.** Diagnosi globale errata su metadata già nel profilo (F46). Il documento resta
  UNSAFE: nessuna risorsa nascosta.
- **Attribuzione.** Regressione r012: il v11 sollevava subito alla prima violazione.
- **Stessa classe, ereditato.** A7, `See [a]{onclick="x"} and <img src="../second.png">`: gli
  attributi Markdown si decidono in proiezione, dopo la guardia HTML del dominio, quindi v12 e v11
  riportano `../second.png`.
- **Correzione attesa.** Le violazioni degli attributi devono passare per la sessione in ordine
  sorgente; almeno, un `Unsafe` diretto non deve scavalcare `session['first']`.

### CLA-R012-I002 — fence ordinari nei container riconosciuti dove Pandoc non li apre (media, bloccante)

- **Input.** `> Intro text`, `> ~~~`, `> <script>alert(1)</script>`, `> ~~~`, più la coda (B1).
  Stessa forma in list (B2) e note (B3).
- **Pandoc 3.1.3.** `BlockQuote [Plain [… Str "~~~"], RawBlock (Format "html") "<script>alert(1)</script>", Para [Str "~~~"]]`:
  nessun CodeBlock, script attivo. Il controllo B5 conferma la differenza: dopo testo un fence a
  backtick produce CodeBlock. B6 lo conferma per il tilde a inizio blocco.
- **Osservato v12.** B1–B3 **COMPLETE**, senza unknown, con lo script dentro un segmento `code`.
- **Osservato v11.** B1–B3 UNSAFE `script`.
- **Asset sicuro (B4).** `RawInline … <img src="p.png">`: v12 dà COMPLETE con `assets=[]`, cioè
  risorsa persa senza diagnosi; v11 proiettava `p.png`.
- **Percorso.**
  1. `read_parts` → `InlineDecision(blocks=True)`.
  2. Il walk `_container_blocks` (269–292) accetta un fence su ogni inizio riga, senza considerare la
     riga di paragrafo precedente per il tilde né la presenza della chiusura.
  3. L'atomo `code_block` salta il corpo, quindi nessun evento e nessuna guardia.
  4. `inline` (650–656) aggiunge un `code` senza unknown, e `complex_quote` ignora `~~~`.
  5. `_guard_coverage` (0, len) dichiara il dominio guardato e `audit` restituisce COMPLETE.
- **Requisiti violati.** C02 (ownership effettiva prima di guardia/fallback, mai COMPLETE
  silenzioso) e C05 (la copertura dichiarata non corrisponde a una guardia completa). Smentisce anche
  PROTOCOL r012 e la guida reference («Fence normale chiuso … nessuna apertura»).
- **Attribuzione.** Regressione r012 nei container.
- **Ereditato.** Al livello superiore B7 dà lo stesso COMPLETE in v11 e v12, per
  `PARAGRAPH_BOUNDARY` (41) e il ramo fence (1079–1100).
- **Aspetto minore.** Il fence non chiuso (B9), per Pandoc `Plain "```"` + RawBlock script, è
  UNSUPPORTED in v12 e UNSAFE in v11. Non è COMPLETE, ma la diagnosi è più debole.
- **Correzione attesa.** Allineare la decisione di ownership del fence alle condizioni del dialetto
  (il tilde non interrompe un paragrafo, serve la chiusura). Dove l'ownership non è certa, guardare il
  corpo o dichiararlo non rappresentato. Non serve un parser universale.

## Osservazioni non bloccanti

- **CLA-R012-O001 — `_guard_coverage` serializzato.** `located()` (582–588) rimuove `_source_map` ma
  non `_guard_coverage`. Le 640 occorrenze in `cell_contents`/`caption_content` su 56 osservazioni
  vengono da qui: celle pipe e caption Markdown, non HTML.
  - Nessun consumatore le legge: `node` legge solo il campo di primo livello. I 19 campi e i gate
    restano a delta 0.
  - Il campo però è dichiarato privato, cambia i byte e l'hash delle osservazioni e non è documentato
    come schema.
  - Va rimosso dalla serializzazione o dichiarato. Per le celle la copertura è comunque reale, perché
    ogni cella ha la propria decisione guardata.
- **CLA-R012-O002 — ordine tra reader e audit (ereditato, A8).** Con `![a](../../../../../temp/run-a002-fase1-corpus/first.png)` seguito da
  HTML evasivo, il detail è `../second.png` in v11 e v12. Il path Markdown è controllato da `audit`
  dopo `observe`. PROTOCOL limita la promessa a `observe`, quindi la segnalo come limite da
  dichiarare, non come finding.

## Criteri comuni

| ID | Esito | Motivo e prova |
| --- | --- | --- |
| C01 F50 | **PASS** (sonda) | C1: formula `x <img …> \⏎y`, link, tre blocchi, char 13/byte 14 verificati a mano. C3, quote con code+LF ed escape+LF: formula intera, char 15/byte 16. C2, non chiuso: UNSAFE `../outside.png`. C4, token con LF: UNSUPPORTED senza formula prestata. v11: C1/C3 UNSAFE, chiusura attribuita a v12. |
| C02 F51 | **FAIL** | Raw `{=html}` esatto chiuso: C5 nested quote, C6 list tilde, C7 note tab (UNSUPPORTED con `container raw block not projected`), D1 raw dopo testo (Pandoc RawBlock), C8 letterale, C9 latex non UNSAFE. Però CLA-R012-I002: fence ordinario dopo testo dà COMPLETE con script attivo. |
| C03 F52 | **FAIL** | A9/A10 GPT originali corretti; A12: visite una volta ciascuna, ordine sorgente, nessuna duplicazione dopo unsafe per osservazione. Però CLA-R012-I001 (A1–A4) e A7 ereditato. |
| C04 F53 | **FAIL** (parziale) | Ledger/slice/source-map coerenti nei rami label e LF. Restano due decisioni discordanti: ownership del fence nel container rispetto al dialetto, e prima unsafe differita rispetto a `raise` diretto. |
| C05 coverage | **FAIL** | `_guard_coverage` viene emesso per tutto il dominio anche quando un corpo classificato come fence non è stato guardato (B1–B4). O001 per la serializzazione. |
| C06 regressioni | Parziale | Regressioni r012 rispetto a v11 su A1–A4 e B1–B4. F31/F36/F40/F41/F46/F48 non contraddetti nei rami sondati. Le regressioni 153/303/308 e la suite 2688+310 sono ricevute, non rieseguite. |
| C07 oracoli | Parziale | Generatore separato e 350 attesi presenti; la fixture non contiene attributi Markdown attivi né fence tilde dopo testo nei container (scansione: 0 casi). La spia verifica la sola membership; la mia A12 copre un caso di ordine e molteplicità. |
| C08 evidenze | PASS nei limiti | Receipt finali coerenti nei campi letti; S004 runtime e S005 documentale distinti (la mia probe-002 conferma lo STALE di s004 sul PROTOCOL). Le 56 osservazioni private sono attribuite alla causa O001. Non ho ricalcolato 64 catene, CLI, H01–H12 né S/B/I. |
| C09 documenti | **FAIL** (minore) | PROTOCOL e guide promettono la prima Unsafe della sessione e un «fence ordinario» letterale: smentiti da I001 e I002. F36, le quattro osservazioni r010, O001 microcosto e i limiti L01/L02/L04/L05 sono dichiarati correttamente. |

## Esito

**NO_GO.** CLA-R012-I001 e CLA-R012-I002 sono bloccanti, con input raggiungibili nel profilo, prove
proprie eseguite, attribuzione v11/v12 sugli stessi byte e, per I002, AST Pandoc. Le chiusure
originali F50 e F52, e F51 per il raw esatto, sono confermate nei casi sondati.

Restano limiti: nessuna suite, build, rescore o CLI rieseguiti; Pandoc solo come oracolo sintattico,
non prova di fedeltà o sicurezza; L01/L02/L04/L05 e i NONPASS storici invariati.

## Costi e passaggio

- **CPU misurata.** Verify d'ingresso 46,42 CPU-s. Gate e probe 001–005 19,5 CPU-s figli
  (`check-005-corrections.json`). Checker 0,15 s.
- **Non misurato.** Letture, hash e diff host.
- **Memoria.** Picco RSS campionato 389615616 B nell'albero R.
- **Rete, acquisizioni, invii.** 0; €0.
- **Processi.** Nessun processo pendente.
- **File fuori dalla cartella evidence.** Snapshot tecnici propri `snapshots/review-r012-claude-probe-00{2..5}.json`.

Il budget finale è nel checkpoint.

Prossimo ruolo: il supervisore, dopo entrambi i report reali, per l'arbitrato. Nessun Git, cleanup,
fase 2, motore o nuova V.
