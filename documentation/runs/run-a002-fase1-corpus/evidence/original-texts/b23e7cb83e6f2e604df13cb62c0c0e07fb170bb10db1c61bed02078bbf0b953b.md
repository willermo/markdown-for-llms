# Review — implementazione — run-a002-fase1-corpus — r011 — Claude

- **Autore, provider, modello e chat.** Reviewer Claude, Anthropic, Claude Opus 5.5
  (`claude-opus-5-5[1m]`), in Claude Code nell'estensione VS Code. Effort e ID della chat non sono
  esposti. Chat nuova, distinta da autore, supervisore e altro reviewer. Nessuna sostituzione,
  delega o subagente.
- **Prompt di origine.** `prompts/39-review-implementation-r011-claude.md`, SHA256
  `84204ca00849f593e923b05dcf115c750637bc6abd74bfb7ba987a15de829ef1`.
- **Oggetto.** Implementazione r011, cioè F46–F49 del mandato37 dopo il NO_GO r010.
  Report e checkpoint dell'autore letti integralmente, nota tecnica inclusa.
- **Snapshot e impronta.** Comune `snapshots/implementation-r011.json`, SHA256
  `29efea9167ad4809fc08e1ec9a1923998c6434d2f518a92c9c663811f53c16f5`, impronta
  `525b853f32493abe8c3637bcb455219f557442cc032ac69d6b4358dcb370b680`.
  - Verify all'ingresso (07:09:17Z, 43,6 s wall, 43,57 CPU-s): **MATCH**, exit 0.
  - Verify dopo gli output: registrato nel checkpoint.
- **Identità Git e input.** Branch `feature/run-a002-corpus`. HEAD e dev coincidono con
  `80c31da46e5caab990d3596c79c44e524ec1dc95`. Hash verificati:
  - reader v11 `af5299bb…8482`;
  - v10 fisico `382d6be9…94b5`;
  - v9 storico `a8ffa9eb…c737`;
  - summary del rescore `f7a08e22…b57d`;
  - checker derivato `3f29bc11…8d19`.
- **Indipendenza.** Non ho letto review, evidenze o checkpoint ChatGPT r011, né un arbitrato r011.
  La cartella `evidence/review-implementation-r011-chatgpt/` compare solo in un elenco e non è
  stata aperta.
- **Modalità.** Review **statica**: lo stop dei nuovi workload (06:37:31Z) era già passato
  all'ingresso. Ho usato solo letture, hash, JSON, `diff` e il manager host stdlib. Nessun import
  del prodotto, R, CLI, test, suite, build, rescore o Pandoc.
- **Esito: NO_GO**

## Ambito e prove

**Letture.**

- AGENTS, temp/HANDOVER, STATE e il checkpoint `supervisor-implementation-review-r011`.
- Integrali: arbitrato r010 e prompt37.
- Integrali: report e checkpoint dell'autore r011.
- Della review Claude r010: intestazione e formato.
- Sezione r011 di PROTOCOL e righe 225–240 (contratto del fallback).
- Dati permanenti r011: casi, before/after, limiti, CLI e prove.
- Receipt `delivery-check-001`, `derived-check-001`, `final-tests-001` e `final-binding-001`.
- Il caso `after-final-002/metadata-closed-quote-1`.

Restano non letti: findings-detail e probe r010 integrali, guide how-to/reference, receipt S/B/I
e le singole receipt R. Per questi mi affido alla ricezione del supervisore, senza attribuirmi
quella verifica.

**Codice.**

- `observations.py` v11: scanner `markdown_lexemes`, `markdown_label`/`markdown_link`,
  `inline_html_events`, `inline`, `node` e producer di `observe`.
- Diff dal v10 fisico, in `evidence/review-implementation-r011-claude/reader-v10-v11.diff`:
  quattro hunk.
- Diff v9→v10 dei rami toccati.
- `audit.py`, `safety.PassiveHTML` e la riga del fallback di `benchmark.observe_bundle`.
- Il test `test_conversion_fix_r011.py`: casi, generatori e check.

**Prove proprie.** Tutte statiche, in `evidence/review-implementation-r011-claude/`:

- `verify-entry.json`;
- `reader-v10-v11.diff`;
- `static-traces.md`, con le tracce T1–T4, gli attesi indipendenti, le varianti finite e
  l'attribuzione v9/v10/v11.

Nessuna traccia è stata eseguita: sono analisi del codice, non test. Le prove dell'autore sono
ricevute e non rieseguite.

## Criteri

| ID | Esito | Motivo |
| --- | --- | --- |
| C01 F46 metadata | **PASS** (statico) | Riga 217: metadata solo con `block_context`. Nel walk inline la riga `# … {…}` torna contenuto e ogni tag diventa evento. HC2 resta coperto da `peel_attrs` sul vero heading, H7 separa testo e `{.c}`. H3 esce con `resource_syntax` → UNSUPPORTED. Trasporti list/quote/note/cell/caption/heading coperti dai 42 casi generati. |
| C02 F47 bound | **FAIL** | La LF posseduta dal code non taglia più e il token resta barriera: corretto. Ma `markdown_label:434` accetta anche label che attraversano LF con `\`+LF, e la guardia taglia ancora lì mentre il consumer no: CLA-R011-I001. |
| C03 F48 owner | **PASS** (statico) | Riga 226: il salto di `label_codes` è subordinato a `html_stack`. Gli eventi sono raccolti in ordine di sorgente e la guardia continua dopo la prima violazione. Il code dopo la chiusura dell'owner resta inerte. Le voci stantie del ledger estendono solo il bound e coincidono con il consumer, che non taglia. |
| C04 fallback quote | **FAIL** | `'segments' not in result` (942) disattiva il fallback per **tutte** le quote complesse: la proiezione inline non vede un fence raw `{=html}` dentro la quote. Esito: COMPLETE con script o risorsa nascosti (CLA-R011-I002). |
| C05 F49 invarianti | **FAIL** (parziale) | Oracoli e trasformazioni finiti e indipendenti (153). La matrice però non varia il vettore di accettazione della label (code / token / `\`+LF) né la struttura di blocco dentro quote complesse: I001 e I002 restano fuori. |
| C06 regressioni | Parziale | Nessuna regressione statica rilevata su F43/F44/F40/F41/F36/F31 token nei rami toccati, e mi baso sulle matrici 303/308 ricevute. I002 è una regressione rispetto a v9 e v10. |
| C07 limiti | PASS | F36 B1–B4 e le quattro osservazioni Claude sono conservati in PROTOCOL e in `limits`, senza ampliamenti. |
| C08 prove | PASS nei limiti | Receipt finali coerenti (PASS_SCOPED_BYTES, derived PASS, exit 0) e hash ricontrollati a campione. s006 dichiarato solo documentale. Non ho rieseguito nulla. |
| C09 documenti | **FAIL** (minore, legato a I002) | PROTOCOL r011 («Il fallback di un blocco già proiettato/validato inline non rilegge il raw») presuppone una proiezione completa che per le quote complesse non esiste, e contraddice la riga 233. Risorse, clock e stop sono rispettati, senza sanatorie. |

## Rilievi

| ID | Severità | Rilievo e prova | Attribuzione |
| --- | --- | --- | --- |
| CLA-R011-I001 | Alta, **bloccante** | F47 è chiuso solo per un vettore. Esempio: `È See [Known $x <img src="../outside.png"> \`⏎`y$ words](https://example.invalid) end.` più la coda a tre blocchi. Il walk della label (434) consuma `\`+LF e accetta la label. Alle righe 240–242 la LF non è in `label_codes`, quindi `bound=at+1`; la math non si chiude e l'img diventa evento → UNSAFE. Il consumer (595, `leaf_end=len(label)`) chiude invece `y$` come formula. L'impatto è identico a F47: unknown di documento, con perdita di formula, link e blocchi (T1). | Regressione r010 non chiusa da r011: v9 senza guardia preventiva (dedotto dal codice); v10 e v11 producono un falso UNSAFE. Il commento alle righe 233–237 prevede solo code e token. Nessuno dei 24 casi math-label usa `\`+LF. |
| CLA-R011-I002 | Media, **bloccante** | Esempio: `> ```{=html}`⏎`> <script>alert(1)</script>`⏎`> ```` → nodo `unknown` (quote complessa) con `segments`, guardia senza eventi (fence letto come code inline chiuso) e fallback 942 saltato → `unknown=[]` → audit **COMPLETE**. Con img `../outside.png` l'esito è lo stesso; con `p.png` la risorsa va persa e l'esito è COMPLETE. In Pandoc (`raw_attribute`) il blocco è HTML attivo. Il percorso è confermato dall'artefatto autore `metadata-closed-quote-1`: unknown, `unknown=[]`, COMPLETE (T2). | Regressione r011: v9 e v10 eseguono `validate_html(raw)` → UNSAFE oppure UNSUPPORTED. Viola C04, PROTOCOL 233/541 e «mai COMPLETE con risorsa persa». |

Il fix serve a far concordare guardia e consumer, non a definire un nuovo contratto:

- **I001.** Trattare `\`+LF accettata dal walk come LF posseduta, oppure rifiutare la label.
  Aggiungere casi `\`+LF × trasporti × img/script, con il controllo non chiuso (T1z).
- **I002.** Restringere l'esenzione ai blocchi la cui struttura è interamente proiettata. Per le
  quote complesse va mantenuta la validazione del raw, o un equivalente per i fence `{=html}`
  interni. Il caso `metadata-closed-quote` deve restare COMPLETE solo se nessuna struttura di
  blocco attiva è nascosta. Aggiungere T2a–c e il controllo fence letterale T2z.

## Osservazioni non bloccanti

- **CLA-R011-O001 — costo cubico del nuovo loop LF** (240–243). Con N code multilinea in una
  sola label il costo è circa N³/6 passi Python per ogni pass del walk; il v10 era lineare.
  Stima, non misurata: con circa 20 KB si può superare R 900 s, e l'esito sarebbe un timeout
  esplicito, non una riuscita falsa. Debito come il timing dei commenti r010; si risolve con una
  ricerca dell'intervallo ordinata.
- **CLA-R011-O002 — fence raw in list e note** (ereditato, fuori delta). `- ```{=html}` con
  continuazione `    <img …>` produce un paragraph con code inline, quindi COMPLETE in v9, v10 e
  v11. Stessa radice di I002; l'interpretazione Pandoc non è verificata con esecuzione. Va al
  supervisore se includerla nello stesso fix.
- Chiusure accolte: F46 (C01) e F48 (C03), più F47 per il vettore code (otto originali e 16
  varianti dichiarati PASS dall'autore e coerenti con la traccia). Restano conservati F36, le
  quattro osservazioni r010, `conversion_final` r001 distinto, i 64 rescore senza delta,
  L01/L02/L04/L05 e i NONPASS storici.

## Esito

**NO_GO.** Rimangono due requisiti bloccanti:

- I001: l'allineamento dei bound F47 è incompleto, con falso UNSAFE e perdita di
  formula/link/blocchi;
- I002: regressione r011 su una quota di HTML attivo nascosto, con COMPLETE senza diagnosi.

Gli input sono attribuiti al comune r011, che risulta MATCH. Il NO_GO non nasce dall'assenza di
sonde: deriva dalle tracce statiche, e per I002 anche da un artefatto autore ricevuto che conferma
il ramo. Le prove eseguibili T1/T2 richiedono una disposizione temporale esplicita.

## Costi e passaggio

- Verify d'ingresso 43,57 CPU-s, RSS max 126628 KiB.
- Letture, hash e diff host non misurati singolarmente: pochi secondi in totale.
- Rete, acquisizioni e invii 0; €0.
- Nessun processo pendente.
- Il verify finale è nel checkpoint.

Prossimo ruolo: il supervisore, dopo entrambi i report reali, per l'arbitrato. Nessun Git,
cleanup, fase 2, motore o nuova V.
