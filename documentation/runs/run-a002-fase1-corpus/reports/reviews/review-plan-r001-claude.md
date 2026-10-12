# Review — piano — run-a002-fase1-corpus — r001 — Claude

- Autore/provider/modello e chat: Claude, provider Anthropic, eseguito in Claude Code
  (estensione VS Code). Modello esposto dalla sessione: Opus 5.5, ID
  `claude-opus-5-5[1m]`. ID della chat non esposto. Ruolo esclusivo: reviewer Claude;
  nessuna sostituzione di provider, nessun agente delegato.
- Prompt di origine: `temp/run-a002-fase1-corpus/prompts/03-review-plan-r001-claude.md`
  (testo ricevuto in chat, coincidente con il file).
- Oggetto e revisione esatta: piano r001, `temp/run-a002-fase1-corpus/plans/plan-r001.md`,
  SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8` (MATCH),
  letto integralmente (righe 1–899).
- Snapshot, HEAD e impronta verificati prima/dopo: `snapshots/plan-r001.json`, SHA256
  `db1dfa3a54fb7d7235e96030a196ba546409c7cd4ea18ceda3753b80edb16026` (MATCH), impronta
  worktree `2aef47fc0f089f82c9d6904cef67d5aa04da3de3232eebb055642021c0ea79d8`, 132 file e
  9 artefatti. `run_context.py verify … --label plan-r001`: MATCH/exit 0 prima
  (2026-10-08T07:19:03+02:00) e dopo la review (vedi `evidence/review-plan-r001-claude/verify-after.json`).
- Indipendenza: non ho letto report, evidenze o checkpoint del reviewer ChatGPT né
  arbitrati correnti. La directory `evidence/review-plan-r001-chatgpt/` è comparsa
  soltanto in un elenco `ls`; nessun suo file è stato aperto.
- Giro: iniziale; nessun rilievo precedente, nessuna parte invalidata.
- Base iniziale e delta del giro: branch `feature/run-a002-corpus`, HEAD = `dev` =
  merge-base = `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto; 7 file
  documentali modificati e 8 nuovi non ignorati ereditati (documentazione/archivio a001),
  identici all'elenco della ricezione del supervisore. Nessun codice prodotto nel delta.
- Esito: **NO_GO** sul piano r001. Quattro rilievi bloccanti, tutti correggibili con
  errata locali verificabili; non serve un nuovo piano né un cambio di metodo o perimetro.

## Ambito e prove

**Letture**, nell'ordine del prompt: AGENTS.md, `temp/HANDOVER.md`, sezione corrente di
`temp/PROJECT-CONTEXT.md`, STATE/HANDOVER/brief della run; `run-lifecycle.md`, skill
`manage-implementation-run` e `verify-conversion-fidelity`, template review, checkpoint
`supervisor-plan-review-r001.md`; piano integrale, prompt01, `handovers/planning-r001.md`;
`inspection.json`, `tool-help.json`, `document-checks-r001/r002.json`, `reception.json`,
`snapshot-receipt.json`; ADR 0001/0004, registro ADR, indice `documentation/README.md`
(parte iniziale), roadmap fase 1, `next-run-a002-corpus.md`, `open-questions.md`, workflow
`converter-benchmark.md`; guida uv, reference toolchain, `pyproject.toml`, `MANIFEST.in`,
`tests/conftest.py`; codice pertinente letto senza importarlo né eseguirlo:
`unified_converter.py` (54–148, 402–500), `logging_config.py` (60–130), `config.py`
(250–268), `run_offline.py` (95–124, 160–311), `run_tests.py`, `check_preflight.py`,
`run_trusted_script.py`, `run_baseline.py` (359–375, 605 tramite grep); manifest a001
(righe 1–40) solo per il limite delle quote temporali.

**Comandi** (argv/cwd/output/exit in `evidence/review-plan-r001-claude/`):

- Git di sola lettura: branch, `rev-parse HEAD dev`, `merge-base`, `diff --cached`,
  `status --porcelain`, `diff --check` (exit 0, nessun problema). Hash di piano e snapshot.
- `python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label plan-r001`
  prima e dopo: MATCH. Si tratta di un uso stdlib del gestore di contesto (shim pyenv
  3.12.3), non di una prova del prodotto.
- Controllo aritmetico ricalcolato (`arithmetic-check.json`): base 600 min, 2.100 CPU-s,
  2.200 MiB; margine 180 min = 30 %; inviluppo 900 min = 54.000 s; CPU attesa 4.500 s;
  disco 3,95 ≤ 4 GiB; stop a 13 h e 3,2 GiB; 9 pagine PDF, 12 pagine di controllo,
  32 tentativi. **PASS**: la r002 corregge il FAIL storico del pianificatore.
- Sonde di sola presenza/help (`probes.md`): renderer HTML offline presenti (firefox,
  google-chrome, wkhtmltopdf, calibre); `pandoc --list-extensions=markdown` (exit 0).
  Nessun documento passato agli strumenti.

**Non eseguito, per mandato:** conversioni, build, installazioni, prove R, import
applicativi o ML, benchmark, rete. In particolare non ho convertito documenti con Pandoc.
Quanto scrivo sul comportamento del writer (estensione `smart`) e di texmath (OMML/MathML→TeX)
deriva dalla documentazione nota di Pandoc e dalle estensioni elencate localmente; non
l'ho osservato su 3.1.3. I rilievi CLA-P001 e CLA-P005 restano validi a prescindere,
perché riguardano il contratto di misura e non un esito runtime.

### Criteri controllati

| Criterio | Esito della verifica | Rilievi |
| --- | --- | --- |
| C01 obiettivo e confini | Copre corpus, gold, protocollo/runner, positivi/negativi e baseline reale; contratto del benchmark distinto dalla fase 2; nessun ML, servizio remoto o Git implicito. L'apparato è ampio ma pertinente; il costo è trattato in P004 | P004 |
| C02 corpus e provenienza | 10 ID e 9 pagine PDF verificati; coprono formati, IT/EN, una e due colonne, scansioni, pagina mista (C10 p3) e tutti i fenomeni T/N/O/F/C/A/R. Licenza MIT per i sintetici e notice DejaVu separata. BYTE_IDENTICAL distinto da CONTENT_REPRODUCIBLE. Scansioni pulite e sintetiche, dichiarate come tali | — |
| C03 gold e dipendenza umana | Gold separato dai motori, nessuna firma umana impersonata, PROVISIONAL se manca il controllo. Mancano lo strato meccanico per le proprietà non visive, la definizione del perimetro e del formato del controllo umano e un tempo credibile | **P002** |
| C04 reader e allineamento | Osservazioni gold-blind con prova del digest; raw/offset conservati; nodi `unknown` resi UNAVAILABLE; abbinamento uno-a-uno con margine; AMBIGUOUS senza tie-break; secondo allineamento per le inversioni. Le soglie 0,75/0,15 sono dichiarate euristiche di allineamento. Adeguato; l'efficacia dipende dal profilo del writer (P005) | P005 |
| C05 metriche e matematica | Unità, denominatori e stati (N/A, UNAVAILABLE, NOT_MEASURED) ben definiti; nessuna media composita; tipografia, sintassi e semantica separate. La metrica sintattica delle formule e i numeri in matematica dipendono dalla grafia | **P001**, P005 |
| C06 positivi e negativi | La matrice copre tutte le mutazioni del brief; il positivo fallisce con un rifiuto universale; mutazioni solo su copie, con hash before/after. I positivi sono scritti nella grafia del gold (P001); la riga sui path non distingue il lato output (P006) | P001, P006 |
| C07 identità e stati | experiment/attempt/config/hash, omonimia, rerun, parziali, cinque livelli di stato, precedenze di exit. Exit 2/3/5 incompatibili con il criterio PASS di R | **P003**, P006 |
| C08 baseline reale e lacune | 5+3 percorsi praticabili sul codice letto; nessuna riparazione dal gold; FAIL conservati; L01/L02 delimitate. Il percorso TXT soddisfa il requisito «≥1 completo» solo in modo vacuo; identità del legacy da precisare | P005, P007, P008 |
| C09 contenuti non fidati | Limiti ZIP/XML/SVG/path/include attuabili con stdlib; `shell=False`, registry esplicito, nessun filtro o template preso dai dati; R non presentato come sandbox del filesystem | P006 |
| C10 toolchain e prove | Nuovi S/B/I con i dieci hash, wheel reinstallata per ultima, R nella stessa invocazione, manifest esterno, test di tampering, snapshot autonomi, output vivi esclusi. Verificato nel codice: R lega lo snapshot prima e dopo (`binding_before/after`); le receipt E di `run_tests.py` non hashano `tests/fixtures/conversion-corpus/` né `scripts/benchmarks/`, quindi il vincolo passa per snapshot e manifest esterno, come il piano prevede | P003 |
| C11 risorse e recupero | Aritmetica PASS; ledger, stop con riserva, contatori senza reset, cap del tool (≤900 s, 1 MiB per stream, FSIZE 32 MiB) verificati in `run_offline.py`. Stima di base non credibile rispetto al perimetro | **P004** |
| C12 consegna e ciclo | A01–A12 con prova, esito e costo; L01–L05 motivate; ruoli, doppia review, Git manuale, EPERM/escalation e rifiuto rispettati | P007 |

## Rilievi

| ID | Severità e blocco sì/no | Posizione | Raggiungibilità/input ammessi | Attribuzione e requisito | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLA-P001 | Media-alta; **sì** (errata) | §4 tabella metriche: «Formule sintattiche» r. 298, «Numeri e segni» r. 296, «Testo/Unicode» r. 294; annotazioni r. 200–201; positivi r. 413–428 | Percorso previsto: C02 (OMML) e C03 (MathML) via Pandoc diretto e legacy, C05; riuso per Docling/Marker | Introdotto dal piano. Brief: normalizzazioni dichiarate e motivate per metrica; riuso delle metriche; ADR 0001 | La metrica confronta la grafia LaTeX con graffe preservate, non la struttura; il meno in matematica è trattato come in prosa. Ne derivano FAIL sistematici non dovuti a perdite. I positivi, scritti nella grafia del gold, non lo rilevano | Dettaglio sotto | Forma strutturale canonica dichiarata prima del workload, con positivi in grafia alternativa e negativi invariati |
| CLA-P002 | Media-alta; **sì** (errata + disposizione) | §3 r. 205–238; controlli r. 417; formule r. 299–300; tempi r. 744, 772; L03 r. 864 | Inevitabile: A03 e quality PASS richiedono VERIFIED | Introdotto dal piano / dipendenza da disporre. Brief: gold controllabile sul sorgente; C03 del prompt | Il controllo visivo umano è chiamato a verificare anche proprietà non visibili. Perimetro (gold, controlli, giudizi sugli output) e formato non definiti; 60–90 min non credibili | Dettaglio sotto | Strati M (meccanico) / V (visivo), pacchetto leggibile, budget contato e disposto |
| CLA-P003 | Media; **sì** (errata) | §5 exit r. 393–403; §8 r. 602–606, 623–630, 640 | Alta: il piano prevede FAIL e bundle incompleti dei motori (r. 507–508, 825) | Introdotto dal piano nell'interazione con R ereditato | Con `run` exit 2 o `score` exit 3/5, R segna FAIL anche se l'isolamento è corretto; l'alternativa è scegliere `--expected-exit` dopo aver visto l'esito | `run_offline.py` r. 197–201, 270–272; precedente `run_baseline.py` r. 359–375 | Harness con exit 0 a esecuzione registrata; exit del benchmark nei JSON; test dedicati |
| CLA-P004 | Media; **sì** per la disposizione della quota | Stima r. 764–777; A02 r. 743; stop r. 801–806 | Certa: è la quota del mandato | Introdotto dal piano. C11 del prompt; brief «quota unica adeguata» | Le stime di base (90 min corpus+gold, 240 min runner, 45 min negativi) sono incompatibili con il perimetro elencato; prevedibile stop a 13 h o estensione | Dettaglio sotto; manifest a001 r. 18–21 | Ristima dal basso con conteggi, poi quota maggiore oppure priorità che protegga i criteri essenziali |
| CLA-P005 | Media; no | Argv r. 469–476, 483–485; reader r. 253–260 | Percorso previsto (5 Pandoc e 3 legacy) | Introdotto dal piano. Brief: normalizzazioni dichiarate; integrità del confronto | Writer con `+smart`, `+multiline_tables`, `+raw_tex`, `+fenced_divs`, `+bracketed_spans`, `+link_attributes`; il reader legge solo pipe table e non gestisce attributi, div e span. Rischio di INDETERMINATE diffusi o di configurazione del motore regolata sul gold | `pandoc-list-extensions-markdown.txt` | Profilo fissato e motivato a priori; cambi successivi come nuova config |
| CLA-P006 | Bassa-media; no | Mutazioni r. 454; audit r. 352–357; exit r. 395–396; stati r. 380 | Un motore può produrre `../`, path assoluti o symlink nell'output | Introdotto dal piano | Exit 4 «prima di conversione» è impossibile per un difetto dell'output e lo attribuisce al runner; lo stato del bundle non è definito | Testo del piano | Separare il lato input (exit 4) dal lato output (raw conservato, INCOMPLETE o UNSAFE, exit 2) |
| CLA-P007 | Bassa; no | r. 122, 330, 492–499, 526–528; A10 r. 751 | Certa nella formulazione | Introdotto dal piano. Brief, criterio 2 | Il TXT (adapter del benchmark) soddisfa in modo vacuo «tutti i media referenziati esistenti»; incoerenza con A10; due ID diversi | Testo del piano | «≥1 completo» = convertitore esistente su un documento con almeno un asset; un solo engine_id |
| CLA-P008 | Suggerimento; no | §7 r. 510–521, 330–333 | Percorso legacy previsto | Ereditato dal legacy (comportamento da caratterizzare) | Pandoc risolto via PATH, `is_available()` aggiuntivo, `.temp.md` accanto alla sorgente, log nel cwd, default non misurato | `unified_converter.py:62-69,75,87,90`; `logging_config.py:79-99` | PATH chiuso e hash del binario risolto; tempi non confrontati col diretto; facoltativo: default reale come config separata |

### CLA-P001 — Formule e numeri in matematica confrontati sulla grafia LaTeX

La riga «Formule sintattiche» (r. 298) prescrive un tokenizer che preserva graffe,
operatori e ordine, con soglia 100 % e senza «alias non dichiarati». Il gold dichiara
per ogni formula un LaTeX (r. 200–201). Per i documenti OMML (C02) e MathML (C03) non
esiste però una grafia LaTeX canonica: il convertitore ne produce una legittima tra molte.
Le varianti sono equivalenti per TeX: `x_i` e `x_{i}`; `\begin{bmatrix}` e
`\left[\begin{matrix}`; spazi attorno agli operatori. Con le graffe preservate queste
differenze diventano FAIL. Inoltre, in modalità matematica il `-` di TeX è il segno meno:
MathML `<mo>−</mo>` ed equivalenti OMML vengono resi tipicamente come `-`. La regola
sui numeri (r. 296: «nessuna … meno Unicode … normalizzati») e la regola «`−`/`-` mai
equivalente» (r. 294), applicate alle occorrenze dentro le formule, producono FAIL anche
quando il segno è conservato.

Impatti:

- Il gate delle formule tende al FAIL per ragioni di grafia, contro il principio del brief
  di misurare perdite reali.
- Il prossimo confronto Docling/Marker classificherebbe i motori per stile TeX.
- I positivi (r. 413–428) sono scritti a mano nella grafia del gold, quindi non possono
  rivelare il problema.
- Correggere dopo aver visto gli output sarebbe una normalizzazione a posteriori, che il
  piano stesso vieta (r. 316–321).

Requisito: brief, «normalizzazioni solo dichiarate e motivate per metrica» e riuso delle
metriche; ADR 0001.

**Risoluzione (errata):**

1. Definire nel PROTOCOL, prima del workload ufficiale, una forma strutturale canonica
   per un sottoinsieme TeX dichiarato: token, gruppi, apici/pedici con argomento a token
   singolo normalizzato, argomenti di `\frac`, righe di aligned, celle di matrice con il
   tipo di delimitatore come attributo, spazi TeX.
2. Equiparare `-` di TeX al segno meno U+2212 **solo** nei nodi matematici; prosa, celle
   e codice restano invariati.
3. Il gold conserva il LaTeX di riferimento più la struttura attesa; i costrutti fuori
   sottoinsieme sono UNAVAILABLE.
4. Ogni regola ha un positivo in grafia alternativa (diversa dal gold), che deve dare
   PASS, e negativi che restano FAIL: `+`→`−`, `i`→`j`, riga o cella rimossa, segno
   eliminato.

**Chiusura:** errata verificata con l'elenco delle regole e la matrice dei test.

### CLA-P002 — Verifica umana: proprietà non visive, perimetro, formato, tempo

**(a) Proprietà non verificabili a vista.** La checklist (r. 221–226) chiede al
verificatore umano di controllare «tutti i numeri/segni» e Unicode. Alcune proprietà
richieste dal corpus non sono però verificabili a vista:

- NFC/NFD intenzionali di C01;
- U+2212 contro U+002D (C01, tabella C06), affidabile solo con font e ingrandimento
  favorevoli;
- spazi non separabili;
- `0/O` e `1/l` in C08, dove la distinzione dipende dal font.

Un VERIFIED umano su queste proprietà attesterebbe qualcosa di non controllato. Il
piano non prevede uno strato meccanico indipendente da reader e comparatore che
confronti i code point dell'input canonico con il gold. Per TXT/XML/XHTML/RTF/TeX basta
una lettura stdlib; per i PDF testuali serve un'estrazione Poppler/MuPDF dichiarata come
controllo incrociato (oggi pdftotext è limitato alla diagnosi dell'immagine, r. 180–182).

**(b) Perimetro non definito.** Oltre al gold di 10 documenti, il piano attribuisce
lavoro umano anche:

- ai controlli, «controllate separatamente come per il gold» (r. 417);
- ai giudizi semantici per ogni formula di ogni output (r. 300: C02/C03 diretto e
  legacy, C05 diretto);
- alle checklist tipografiche (r. 299).

Tutto rientra negli stessi 85 minuti, condivisi con la verifica degli output da parte
dell'agente (r. 772). Rileggere con attenzione tecnica circa 2.000–3.500 parole con ogni
numero, segno, cella e formula su 12 pagine, 2 capitoli, TXT e TeX richiede, secondo la
mia stima di reviewer (non una misura), più delle 1–1,5 h indicate già per il solo gold.

**(c) Formato non definito.** `annotations.json` (bbox, offset) non è verificabile da una
persona. Non è definito un pacchetto leggibile, derivato in modo deterministico dal gold
e non dai motori, né come il verificatore registra le proprie risposte senza che
l'agente compili gli esiti al suo posto (r. 227–232).

**Risoluzione (errata):**

1. **Strato M** (meccanico, indipendente da reader e comparatore): code point e byte
   dell'input canonico confrontati con il gold. Le discordanze vanno a V.
2. **Strato V** (umano, visivo): layout, overflow, glifi presenti, ordine e colonne,
   figura/didascalia, appartenenza alla pagina, C10 p3, aspetto delle formule.
3. Pacchetto per pagina: render, trascrizione impaginata con i token critici evidenziati,
   checklist compilata dal verificatore; l'agente calcola soltanto gli hash.
4. Budget derivato dal conteggio degli elementi annotati; budget separati per il gold
   (A03) e per i giudizi sugli output (L04). Questi ultimi possono restare
   PENDING/UNAVAILABLE se non disposti.
5. Chiarire se i controlli sono verificati per costruzione (meccanicamente rispetto alla
   loro specifica) oppure da una persona.
6. Regola PROVISIONAL per documento, non globale.

**Da disporre prima del mandato:** verificatore designato, finestra e durata realistica,
browser offline per EPUB (presenza osservata, non provata sotto R).

**Chiusura:** errata con la ripartizione M/V e il budget contato, più la disposizione
nell'arbitrato.

### CLA-P003 — PASS di R e codici di uscita del benchmark

`run_offline.py` assegna PASS solo se `exit_code == expected_exit` (r. 197–201) e
propaga lo stato interno (r. 270–272). Il piano esegue il workload ufficiale e lo score
sotto R (r. 602–606, 623–630) con exit 2 per PARTIAL/EMPTY/INCOMPLETE e 3/5 per qualità
FAIL o INDETERMINATE (r. 393–403). Prevede inoltre esplicitamente FAIL dei motori
(r. 507–508, 825). Una baseline corretta produrrebbe quindi receipt R FAIL. L'unica via
per ottenere un PASS sarebbe fissare `--expected-exit` dopo aver visto l'esito, cioè un
gate adattato al risultato; in alternativa si classificherebbe come problema di
isolamento (IMPEDITA, r. 640) un esito valido.

Il precedente a001 risolve con un harness: `run_baseline.py` registra `expected_exit` e
`exit_matches` per ogni sotto-comando (r. 359–375, 605) e torna 0 come comando R.

**Risoluzione (errata):**

- Il comando R ufficiale è un harness (modalità del benchmark o wrapper) con exit 0 se e
  solo se ha completato e registrato tutti i tentativi e l'audit.
- Gli exit e gli stati del benchmark restano nei JSON e nel report.
- `--expected-exit` è fissato a priori sul contratto dell'harness.
- Test: un motore fallisce → R PASS con batch PARTIAL ed exit 2 registrati; l'harness
  fallisce → R FAIL.

### CLA-P004 — Credibilità della quota

Il perimetro operativo comprende:

- generatori stdlib per cinque formati: OOXML con OMML, note e relazioni; EPUB3 con nav e
  MathML; writer RTF; progetto TeX; PNG via zlib;
- 9 pagine SVG con matematica impaginata a mano e bbox;
- 10 gold con annotazioni tipizzate (bbox, offset, celle, link);
- 12 render di controllo e schermate EPUB, due generazioni con confronto, manifest e NOTICE;
- un reader con offset e HTMLParser e un tokenizer/canonicalizzatore TeX;
- due allineamenti con ambiguità e nove famiglie di metriche;
- una CLI a cinque modalità con receipt e ledger;
- campionatore `/proc` a 10 ms, wait4 e gestione dei process group;
- expander TeX, launcher legacy, validatori di confinamento e audit;
- circa 17 famiglie di mutazioni più positivi e negativi di sicurezza;
- S/B/I/R, tre documenti e il report.

Per questo il piano stima 90 min (corpus, gold e render), 240 min (runner) e 45 min
(positivi, negativi e regressioni). La mia stima di reviewer, non misurata, è 4–6 h per
generazione e gold e 8–12 h per runner e test. Le quote temporali a001 risultano NON PASS
(manifest a001 r. 18–21): è un segnale storico di stime strette, non una prova sul nuovo
lavoro.

Impatto: arresto sostanziale prevedibile a 13 h con criteri essenziali incompleti, oppure
pressione a ridurre le verifiche.

**Risoluzione:**

1. Ristima dal basso per componente, con conteggi (blocchi, pagine, metriche, mutazioni).
2. Poi una delle due strade:
   - quota maggiore disposta dal supervisore (costo €0, risorse locali);
   - ordine di priorità che protegga entro la riserva i criteri essenziali: corpus, gold
     con strato M, reader/scorer, negativi e baseline nativa prima delle rifiniture
     documentali. Le riduzioni ammesse non devono indebolire requisiti; per esempio, bbox
     solo a livello pagina/regione nei PDF non convertiti in questa run, se le metriche
     previste non ne dipendono.

### CLA-P005 — Profilo del writer Markdown e del reader

La sonda locale conferma che nel profilo `markdown` di Pandoc 3.1.3 sono attive di
default `smart`, `multiline_tables`, `raw_tex`, `raw_html`, `fenced_divs`,
`bracketed_spans` e `link_attributes`. L'argv proposto (r. 469–476) disattiva soltanto le
tabelle simple e grid. Il reader dichiarato (r. 253–260) gestisce pipe table e HTML
passivo, non attributi, div o span.

Conseguenze attese, documentate da Pandoc ma non osservate qui:

- le tabelle DOCX con larghezze di colonna escono come multiline;
- `smart` rende in ASCII virgolette tipografiche, trattini ed ellissi;
- immagini e ancore portano `{…}` o `[]{#id}`.

Il risultato sarebbe una serie di nodi `unknown` e di INDETERMINATE, oppure FAIL dovuti
alla configurazione. Il piano consente di modificare l'argv sugli input effettivi
(r. 467, 484), ma non vieta di farlo dopo il confronto con il gold.

**Raccomandazione:** fissare nel PROTOCOL, prima del workload ufficiale, il profilo
Markdown del benchmark (per esempio `-smart-multiline_tables`, con una decisione
esplicita su raw_html/raw_tex/attributi/div/span: letti dal reader oppure disattivati),
motivato dal dialetto obiettivo e non dai risultati. Servono test del reader per ogni
costrutto ammesso. Ogni cambio successivo diventa una nuova config/hash, conservando i
risultati precedenti; il legacy resta misurato con la propria configurazione.

### CLA-P006 — Riferimenti non confinati prodotti dal motore

Nella riga di r. 454 sono unificati due casi diversi:

- input malevolo (membro ZIP evasivo, include fuori root): exit 4 prima della
  conversione, corretto;
- output del motore con `../`, path assoluto o symlink: rilevabile solo dopo la
  conversione e attribuibile al motore, non al runner.

Lo stato di bundle per un riferimento non confinato non è definito (r. 352–357, 380).

**Raccomandazione:** per il lato output conservare il raw, non seguire mai il
riferimento, assegnare bundle INCOMPLETE (o uno stato dedicato come
UNSAFE_REFERENCE), exit 2 e asset FAIL. Dichiarare la politica sui path assoluti
interni alla root e aggiungere un test per ciascun lato.

### CLA-P007 — Percorso TXT e criterio «≥1 nativo completo»

Il TXT è un adapter del benchmark: incapsula il testo in una fence che il reader poi
estrae. Non ha media, quindi il criterio di r. 526–528 («tutti i media referenziati
esistenti») risulta soddisfatto in modo vacuo, mentre A10 (r. 751) richiede «media
realmente presenti». Inoltre l'engine_id compare in due forme: `utf8-literal-v1`
(r. 122) e `txt-literal-v1` (r. 330).

**Raccomandazione:** «≥1 completo» = Pandoc diretto o legacy, su un documento con almeno
un asset atteso, referenziato e presente. Nel report il TXT va presentato come controllo
della pipeline, non come prova di un motore. Usare un solo ID.

### CLA-P008 — Identità del percorso legacy (suggerimento)

Comportamento osservato nel codice:

- `PandocConverter` costruisce `['pandoc', file_path] + options.split()`
  (`unified_converter.py:90`): Pandoc è risolto via PATH, non con un path assoluto;
- `convert()` invoca `is_available()` (r. 62–69, 75), cioè un `pandoc --version` in più
  a ogni tentativo;
- scrive `.temp.md` accanto alla sorgente (r. 87);
- all'istanza crea `pipeline_logs/` nel cwd e scrive i log su stdout
  (`logging_config.py:79-99`);
- non passa `--from` e usa `--strip-comments` di default.

**Suggerimento:**

- PATH chiuso e hash del binario effettivamente risolto;
- sorgente copiata in una directory dedicata, così il `.temp.md` non altera gli audit
  di identità;
- tempi legacy (avvio interprete e `is_available`) non confrontati con quelli diretti;
- facoltativo e poco costoso (12 tentativi): misurare anche il default reale del prodotto
  senza `extract-media`, come config separata, per caratterizzarne il comportamento
  effettivo.

## Motivazione dell'esito e limiti

Il piano è solido nell'impianto: copertura del corpus, separazione tra gold, osservazione
e scoring, stati espliciti, assenza di medie composite, mutazioni sostanziali,
identità/omonimia, contenuti non fidati, catena S/B/I/R e lacune L01–L05 delimitate senza
indebolire i requisiti. L'aritmetica è corretta e i cap del tool R citati corrispondono
al codice.

L'esito è **NO_GO** per quattro blocchi:

- **P001** (validità della metrica delle formule, riusata dai confronti futuri);
- **P002** (il gold umano attesterebbe proprietà non verificabili; perimetro e tempo
  della dipendenza da disporre non definiti);
- **P003** (contraddizione reale fra il contratto di exit e il criterio PASS di R,
  raggiungibile nel percorso atteso);
- **P004** (quota non credibile per il perimetro dichiarato).

Tutti si chiudono con errata locali nell'arbitrato, senza un nuovo piano. P005–P008
sono raccomandazioni non bloccanti: P005 è consigliato come errata, perché tutela
l'integrità del confronto.

Limiti della review:

- nessuna conversione o prova runtime; le affermazioni su `smart` e texmath sono di
  natura documentale;
- le stime di tempo umano e di sviluppo in P002 e P004 sono giudizi del reviewer;
- non ho riletto l'intera storia a001 né svolto un sondaggio sulle versioni dei motori
  candidati;
- la presenza dei renderer HTML non prova il loro funzionamento sotto R;
- questo esito non è prova di alcun runtime futuro.
