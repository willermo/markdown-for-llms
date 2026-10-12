# Review — implementazione — run-a002-fase1-corpus — r008 — Claude

- Autore/provider/modello e chat: reviewer Claude, Anthropic, Claude Opus 5.5 (`claude-opus-5-5[1m]`),
  in Claude Code nell'estensione VS Code. L'ID chat non è esposto. Chat nuova, distinta da autore,
  supervisore e reviewer dei giri precedenti. Nessuna sostituzione, delega o subagente.
- Prompt di origine: `prompts/29-review-implementation-r008-claude.md`, SHA256
  `b3f8d7ba457861a69379a7525cc14a04fae43b3a27a2ca245ff25728244907e8`.
- Oggetto e revisione esatta: implementazione r008, giro di fix F35–F38. Mandato: prompt27, arbitrato
  r007 (NO_GO) e addendum risorse r008 r001 (DISPOSED24h). Report e checkpoint autore r008, con le
  sezioni finali «Consegna completa» e «Chiusura amministrativa finale» autorevoli.
- Snapshot, HEAD e impronta verificati prima e dopo: `snapshots/implementation-r008.json`, SHA256
  `2f6d690a8f3db7af8f192f78a9e6f5c0e3704b9757256f9dfdf72e35ee3b4152`, impronta
  `448b854a88dc3dc883f7337f6fe037fd27846513e3a19104fc88529841e07eda`, 325 file e 76333 artefatti.
  Verify prima (13:22:36Z) e dopo le prove (13:44:46Z): MATCH, exit 0. Un verify finale dopo la
  scrittura degli output è nel checkpoint. Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev
  locale/merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto. Piano `3ac23bce…dd4ab8`
  ed errata `a09cbb8b…e0dfd9` coincidono.
- Indipendenza: non ho letto la review ChatGPT r008, le sue evidenze o il suo checkpoint, né un
  arbitrato r008 (nessuno esisteva all'ingresso). Ho letto come input storici le due review r007 con
  le tracce statiche e l'arbitrato r007.
- Giro: fix r008. Bloccanti r007: F35 (CLA-R007-I001), F36 (CLA-R007-I002), F37 (CLA-R007-I003 e
  GPT-R007-I001); F38 documenti e prove. Da valutare anche la nuova collocazione di PassiveHTML.
- Base iniziale e delta del giro: dal contesto `implementation-fix-context-r007-resources-r001`
  cambiano 7 file autore e 4 registri del supervisore. I file autore sono `observations.py`,
  `PROTOCOL.md`, le due guide e tre file nuovi non committati (test r008, report e dati r008).
  Ho confrontato il reader con i byte r007 ricevuti (`5d443387…`): diff in
  `evidence/review-implementation-r008-claude/reader-r007-r008.diff`. Scorer, audit e safety invariati.
- Esito: **NO_GO**

## Ambito e prove

**Letture.** AGENTS.md, temp/HANDOVER, la parte corrente di PROJECT-CONTEXT, STATE/HANDOVER/brief
della run, il checkpoint `supervisor-implementation-review-r008`, il protocollo, le due skill, il
template review. Poi arbitrato r007, prompt27 e addendum risorse con la nota utente; le due review e
i checkpoint r007 con le tracce `static-traces.md` e `static-label-trace.md`; report e checkpoint
r008 integrali; PROTOCOL r008, le due guide, report e dati permanenti r008.

Sul codice ho letto `observations.py` per intero e il diff r007 → r008. Ho seguito
`markdown_lexemes`, `raw_fence_end`, `tag_end`, `incomplete_tag_token`, `markdown_label` con
`label_codes`, `markdown_link`, `markdown_units`, `strip_inline_end`, `inline`, `HTMLTree` e
`html_nodes`, poi i producer di `observe`. Ho letto anche `audit.py`, `benchmark.observe_bundle`,
`safety.PassiveHTML/html`, il percorso exit/reason di `metrics.py` e il test r008 per intero.
ADR, roadmap e guida uv solo dove servono, attraverso report e protocollo.

**Prove proprie, statiche.** Tutte in `evidence/review-implementation-r008-claude/`:

- `entry.py` → `entry.json` (host stdlib, exit 0). Registra Git, hash del comune, piano/errata e
  reader. Origine del managed: `stage-s007/dev/bin/python` risolve a
  `.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12`. Coincidono 1127 descrittori del
  target S007, 3813 input package/startup, i `.pth` di startup, le 10 origini installate e gli
  archivi wheel `43a2bb92…` e sdist `905005ce…`. Il verify di `implementation-r008-s007` è
  **STALE**: le sole differenze generali sono i 4 registri del supervisore.
- `static_checks.py` → `static-checks.json` (host stdlib, exit 0, nessun import del prodotto).
  Verifica i 64 record dei dati permanenti r008 e trova 0 errori di descrittore. Altri risultati:
  - catena raw/osservazioni/audit coerente; reader v8 su 64/64;
  - per set: historical-r001 4 PASS/28 FAIL e baseline-r002 4 PASS/28 FAIL;
  - **osservazioni r008 identiche a r007 salvo l'identificativo del reader su 64/64**, metriche
    r008 = r007 su tutte le chiavi;
  - per formato (WER, parole condizionate/riferimento, parole globali, stato del bound): TXT
    0/89 PASS; DOCX 0,0604 con 11/182 e 11; EPUB 0,0430 con 8/186 e 8; RTF 0,1173 con 19/162
    condizionato e 13 globale; TeX WER None con 1/152 condizionato e 0 globale `TRIVIAL_ZERO`.
- Diff del reader e lettura delle receipt dell'autore (sotto).

**Prove proprie, runtime.** Tutte con lo stesso interprete managed 3.12.13 (`-I -B`), dentro R/firejail
con `--net=none`, `shell=False`, figli raccolti. Gli attesi sono stati fissati prima di ogni
esecuzione (`expectations-frozen*.json`, con hash). Il runner proprio `rrun.py`:

- copia i byte consumati in un archivio per contenuto, prima e dopo, e li trova identici in ogni prova;
- cattura un manifest `benchmark-inputs` (scripts/benchmarks, corpus, le mie sonde, `html/parser.py`
  del managed) e usa due preflight: `check_python_origin` e `launch.py inventory verify`;
- prende il target S007 con i descrittori aggiunti delle mie sonde e crea uno snapshot proprio
  `review-r008-claude-*`.

S007 integrale non è usato come gate verde. Le sonde importano solo stdlib e `scripts/benchmarks`;
dal site-packages viene caricato solo `_virtualenv.py`, non il package installato.

| Prova | Esito R | Risultato |
| --- | --- | --- |
| `s007-conserve-001` | **FAIL** conservato | Gate S007 ricevuto: `Stage STALE` (binding) prima di ogni workload |
| `f35-001` | **FAIL** conservato | Errore mio di preflight: `check_python_origin` vuole la radice `.venv-python`. Nessun workload avviato. Corretto in `rrun.py` |
| `f35-002` | PASS | F35, 8/8 conformi in 0,61 s complessivi (timeout R 20 s). A1 è il T1 esatto. Gli altri casi: backtick dopo blank, `$$` attraverso blank, fence `~~~` con ```` ``` ```` interno, non chiusa con EOF, chiusura più lunga, righe-marker nel corpo, CRLF. Tipi, raw/offset char/byte, stato, formula e blocchi seguenti corretti |
| `main-001` | PASS | 14/19 conformi. Tutti F36 (B1 è il T2 esatto, poi PI, declaration, doppio backtick, math chiusa che nasconde `<!--` letterale), il positivo B6 (immagine attiva dopo un token chiuso, aperta e hashata), F37 (C1 è il T3 esatto, image-in-link con code soft-line, Unicode con posizione della reference), C3/C4 (veri token nella label EOF o dopo il link) ed E1. **Non conformi i 5 casi C04 D1/D2/D3/D6/D7**, come previsto; controlli D4/D5 conformi |
| `r007-html-001` | PASS | Gli stessi casi C04 sui byte r007 `5d443387…`: tutti `UNSAFE_REFERENCE` (detail: Path evasivo, script, onerror). Il tokenizer fissato riporta `img`/`script` come start tag reali, in D7 anche `p title="it's"` |
| `extra-r007-001` / `extra-r008-001` | PASS / PASS | Variante sicura di D7 (`p.png`). r007: riferimento immagine `p.png` proiettato e aperto. r008: nessun riferimento, nessuna apertura, leaf `incomplete-start` su `<p title=it's><img src="p.png"> Don't ` |

**Prove dell'autore lette, non rieseguite.**

- `final-checks-002`: 25 receipt finali, 23 PASS e 2 FAIL H12 deliberati, hash e stato ricontrollati
  da me. Storia di 47 receipt (38 PASS e 9 FAIL). Restano conservati tre TIMEOUT a 3 s con reap sul
  reader r007, after-matrix001, targeted002, cli-image002/003 e H12.
- Il prima della matrice e del loop è legato ai byte r007 `5d443387…`, il dopo/CLI/rescore/suite ai
  byte v8 `817852a8…`. matrix-before-001 ha 128/128 casi non conformi, matrix-after-005 ne ha 0/128.
- `semantic-equivalence-002`: 205 file runtime, 3813 package, 10 moduli, 124 helper/test; stage e
  binding nuovi.
- Ho letto anche mirati 1373/16 deselected, suite veloce 1530 + 310 subtest e 57 casi CLI con 114
  exit OS. Sono coerenti con il report, ma restano prove dell'autore.

**Non eseguito.** Per mandato non ho rieseguito: suite, test campaign, build/reinstallazione,
contratti H, CLI con writer, rescore, conversioni, render, M/V, rete e Git di scrittura. F32/F27/
F30/F28/F26/F16 li ho controllati staticamente sul diff (righe non toccate) e con le regressioni
dell'autore, senza sonde dedicate.

## Rilievi

| ID | Severità e blocco sì/no | Posizione | Raggiungibilità/input ammessi | Attribuzione e requisito | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLA-R008-I001 | **Media, bloccante** | `observations.py:645–650` (`html_nodes`): rimossa la chiamata `html(raw)` che in r007 precedeva il tree. `:586–591`: PassiveHTML ora gira solo in `HTMLTree.handle_starttag`. `:591` (attributi duplicati) e `:595` (profondità >16) sollevano `ValueError` e fermano il feed. Raggiunto da `observe` `:924–931` (blocco HTML), `:869–870` e `:880–881` (fence ```` ```{=html} ````), poi `audit.py:18–20` | Output Markdown di un motore (dato non fidato) con un blocco HTML o una fence raw HTML. Nello stesso blocco un tag con attributo duplicato, o un annidamento >16, precede uno `<script>`, un attributo evento o un `src="../…"`. Input sintetico ammesso | **Regressione r008** dovuta alla nuova collocazione della guardia. C04: «aperture HTML reali passano tutte dalla guardia PassiveHTML», «niente attenuazione safety». PROTOCOL r008, le due guide e il report permanente dichiarano che «ogni opening tag effettivamente riconosciuto passa il medesimo PassiveHTML» e che le URI unsafe «conservano le guardie». `final-checks-002` dice «actual HTML openings invoke unchanged PassiveHTML» | Dopo il primo `ValueError` non-Unsafe il parser si ferma e i tag successivi non passano da PassiveHTML. L'audit dà `INCOMPLETE/UNSUPPORTED_RESOURCE_SYNTAX` invece di `UNSAFE_REFERENCE`: script, `onerror` e path evasivi diventano «sintassi non supportata». Nessuna apertura né fetch; status ed exit 2 invariati. Cambia la classificazione di sicurezza in audit e nel motivo `bundle` delle metriche. Le affermazioni documentali sono smentite | Runtime proprio. `main-001`: D1/D2/D3/D6 danno `UNSUPPORTED_RESOURCE_SYNTAX`, con unknown «Duplicate HTML attributes outside finite profile» e «HTML depth>16»; i controlli D4 (stesso img senza duplicato) e D5 (percorso inline) restano UNSAFE. `r007-html-001`: sui byte r007 tutti UNSAFE, con detail «Path evasivo», «script», «onerror» | **Fix minimo**: validare con PassiveHTML ogni start tag che lo stesso tokenizer del tree riconosce, anche quando la proiezione si interrompe. Si può fare con un passaggio di sola validazione che usa la stessa `parse_starttag`, oppure registrando duplicati e profondità come errori senza fermare il feed. **Chiusura**: D1/D2/D3/D6 danno `UNSAFE_REFERENCE` con detail e D4/D5 restano invariati. Le varianti F36 HTML bare/wrapper/cella/caption/raw-html restano `INCOMPLETE/UNSUPPORTED`, senza attributi inventati dai corpi dei token. Rettificare PROTOCOL, guide e report |
| CLA-R008-I002 | **Media, bloccante** | `observations.py:107–122` (`incomplete_tag_token`, nuova): ogni `'` o `"` apre uno stato quotato. La usa `HTMLTree.parse_starttag` (`:600–606`) prima del parser fissato | Un blocco HTML o una fence raw HTML con un valore d'attributo non quotato che contiene un apice (es. `title=it's`). Dopo la chiusura dell'apice, e prima di un `>` non quotato, compare un html_lexeme (commento, PI, declaration, CDATA) nello stesso blocco. Sintetico ammesso | **Regressione r008** (nuova funzione). Contratto r008: `incomplete-start` vale solo per un prefisso realmente incompleto, e `html_lexeme` dichiara «matching the pinned 3.12.13 tokenizer». Valgono anche C04 (aperture reali alla guardia) e C05 (nessuna perdita inventata) | Il tokenizer fissato chiude `<p title=it's>` al primo `>`. La funzione invece salta quel `>` e i tag reali successivi, e crea un leaf `incomplete-start` fino al token. Il tag `img` interno non è né validato né proiettato. Con `../` l'esito è UNSUPPORTED invece di UNSAFE. Con `p.png` spariscono nell'unknown il riferimento immagine reale e il testo «Don't», e l'asset non viene verificato. Il documento è comunque INCOMPLETE per il token; nessun falso PASS | Runtime proprio. `main-001` D7 dà `UNSUPPORTED_RESOURCE_SYNTAX`. `r007-html-001` D7 dà UNSAFE, e il tokenizer fissato riporta `[p title="it's"]`, `[img src="../outside.png"]`. `extra-r008-001`: image_refs `[]`, opened `[]`. `extra-r007-001`: image_refs `['p.png']`, opened `['p.png']` | **Fix minimo**: riconoscere le virgolette come fa il tokenizer fissato, cioè solo all'inizio di un valore dopo `=`. In alternativa trattare un tag come incompleto solo se il tokenizer non ne trova la fine. **Chiusura**: D7 dà `UNSAFE_REFERENCE`; la variante `p.png` conserva il riferimento immagine e il testo. Restano invariati i casi `<alpha <!--…` di F36 e i test r008 sui producer HTML |

I due rilievi riguardano lo stesso requisito C04, ma hanno cause e correzioni distinte; si possono
chiudere in un unico giro. Non li ho uniti a F38: la rettifica documentale fa parte della chiusura.

### Chiusure e regressioni F35–F38 e criteri comuni

| Criterio | Esito nel perimetro controllato |
| --- | --- |
| C01 F35 | **Chiuso.** Ho controllato a mano tutte le transizioni di `markdown_lexemes`: ognuna aumenta `i` strettamente. Righe e fence vanno all'end fisico. Le chiavi di destinazioni, metadata e `label_codes` puntano a end > chiave. Comandi, code e math avanzano di almeno l'opener. Nel raw fence `limit` è la vera chiusura o EOF, sempre ≥ cursore; dopo un token che la scavalca viene ricalcolato dal suo end. Il token non può finire su un inizio riga se non a EOF. Il T1 e sette varianti terminano (`f35-002`). Il prima/dopo dell'autore è legato ai byte r007/v8, con 3 TIMEOUT a 3 s e reap e poi 3 PASS. Tipi, raw, offset e diagnosi della fence non chiusa sono conservati |
| C02 F36 | **Chiuso nei domini verificati.** Un code, un `$x`, un `$$` o un `<alpha` incompleto ora avanza del solo opener (`:227`, `:234`, `:265`), e il consumer termina l'unknown alla barriera. Ho verificato T2 esatto, PI, declaration, doppio backtick e math chiusa che contiene `<!--` letterale. Il token è intero e alla posizione char/byte esatta, il corpo non viene aperto, i path `../` non danno esito, q e pagina 2 risultano missing e i blocchi seguenti sono preservati. Il positivo B6 discrimina un reader che rende tutto inerte |
| C03 F37 | **Chiuso.** `markdown_label` riceve il bound del producer e restituisce `codes`; lo scanner li consuma prima del limite a newline (`:212–213`, `:239`, `:244`). T3 esatto, image-in-link con code soft-line e la label Unicode danno tre (o due) blocchi in ordine, testo esatto, reference e posizione char/byte corrette, zero token e COMPLETE. Un vero token dopo il code (C3, C4) resta intero e inerte |
| C04 guardie HTML | **Non chiuso**: CLA-R008-I001 e CLA-R008-I002. Il percorso inline mantiene `html()` in fallback (D5 UNSAFE), e i tag validati dal tree conservano URI, attributi ed eventi (D4, E1). La relocazione invece lascia senza guardia i tag dopo un abort del tree o dentro un `incomplete-start` inventato |
| C05 regressioni | Sono intatte nel diff le righe di autolink (F32), `</>` (F27), reason neutra (F30) e CDATA locale (F28). Lo scorer è byte-identico (F26). Le regressioni mirate r002–r007 dell'autore risultano PASS. I miei positivi (A1–A8, C1/C2/C7, E1, B6) escludono un reader che rifiuta tutto. Fuori da I002 non ho trovato falsi FAIL su input validi |
| C06 F38 | **Parziale.** PROTOCOL, guide e report descrivono correttamente F35–F37, conservano storia, fallimenti, H12 e TIMEOUT e distinguono gold PROVISIONAL e writer sintetico. È falsa l'affermazione sulla validazione PassiveHTML di ogni apertura (I001/I002), che va rettificata col fix |
| C07 64 misure e V | **Conforme.** Verificato con il mio checker statico (vedi sopra). Nessun motore e nessuna V nuova. V13 riguarda i riferimenti; L04 formule output resta UNAVAILABLE, distinta. L01/L02/L05 sono invariati |
| C08 toolchain | Prove dell'autore coerenti: S007 con S/B/I nuovi, equivalenza S006→S007 limitata, 23 PASS + 2 FAIL finali, binding prima/dopo. Le mie prove usano gate propri sugli input realmente consumati, dopo aver conservato il FAIL di S007, senza trasferire PASS generali né ripristinare input |
| C09 risorse | Disposizione24h rispettata. Stop e deadline sono lontani (sotto). Ogni workload proprio ha avuto timeout breve e reap; nessun download |
| C10 ciclo | Report dell'autore completo, senza GO; ruoli separati; comune valido MATCH prima/dopo. Nessun Git, cleanup, a001 o fase 2 |

**Osservazioni non bloccanti, statiche.** (a) Nel raw LaTeX il corpo è letto dal consumer inline
Markdown, come in r007. Un `<!--` non chiuso diventa quindi un token che scavalca la vera chiusura
della fence e assorbe i blocchi seguenti. È documentato come profilo («token inerti attraversano
false chiusure») e dà sempre INCOMPLETE; non l'ho sondato. (b) Già in r007, `local=raw[:bound]` a
ogni `!`/`[` (`:236`) rende lo scanner quadratico su input grandi. Termina, ma non con costo lineare;
è un suggerimento fuori dal perimetro F35.

## Motivazione dell'esito e limiti

NO_GO sul comune `implementation-r008`. F35, F36 e F37 risultano chiusi nelle prove dell'autore e
in varianti indipendenti eseguite da me (T1, T2 e T3 esatti inclusi). Le 64 misure sono invariate e
coerenti. Il delta r008 però ha spostato la guardia PassiveHTML di `html_nodes` dentro il tree, e
questo crea due regressioni raggiungibili e misurate:

- tag attivi o evasivi dopo un abort del tree (CLA-R008-I001);
- tag reali dentro un `incomplete-start` inventato da un'euristica sulle virgolette diversa dal
  tokenizer fissato (CLA-R008-I002).

In entrambi i casi l'esito non è più `UNSAFE_REFERENCE`, che r007 dava. Nel secondo si perde anche un
riferimento immagine reale. C04 chiede esplicitamente di verificare la raggiungibilità di questa
collocazione e di escludere attenuazioni della safety, e la documentazione r008 afferma il contrario.
Non ci sono aperture di file, fetch o falsi PASS, e status ed exit restano INCOMPLETE/2. L'impatto
reale è la classificazione di sicurezza e, per I002, la fedeltà di un documento già INCOMPLETE.
L'arbitro può pesarlo; per me resta bloccante, perché è una regressione di una guardia richiesta con
una correzione locale piccola. Il GO non certificherebbe comunque altri motori, L04 o la fase 1.

Limiti: le prove runtime sono mie e piccole (28 casi sintetici distinti, 36 esecuzioni sui due
reader) e non coprono tutte le combinazioni di trasporti. Le prove dell'autore le ho lette e ricontrollate negli hash, non rieseguite. Le
osservazioni (a) e (b) sono solo statiche. Il gold delle sonde è sintetico, senza V.

**Costi propri.** Wall della review circa 25 minuti (13:22:31Z → circa 13:47Z). Comandi misurati
con `/usr/bin/time`, CPU user+sys:

| Comando | CPU (s) | Wall (s) |
| --- | --- | --- |
| verify prima | 30,07 | 30,14 |
| verify dopo | 22,49 | 22,49 |
| entry | 0,98 | 0,96 |
| s007-conserve | 0,70 | 0,71 |
| f35-001 | 3,29 | 3,19 |
| f35-002 | 7,98 | 7,18 |
| main-001 | 4,09 | 3,93 |
| r007-html-001 | 3,93 | 3,79 |
| static | 0,45 | 0,44 |
| extra-r007 | 3,83 | 3,68 |
| extra-r008 | 3,86 | 3,74 |

Totale **81,67 CPU-s**, lower bound (i comandi di lettura non sono misurati); il verify finale è
nel checkpoint. RSS massimo campionato 291812 KiB. Disco nuovo circa 7,7 MB: evidenze finali 7443208 B
più 6 snapshot propri per 285870 B complessivi. Rete, acquisizioni, invii e servizi: 0 e €0.

Wall cumulativo dall'entry al 13:44:31Z: 196657,75 s (circa 54,63 h); margine allo stop 257436,63 s
di circa 60779 s. Lo stop resta 2026-10-11T06:37:31Z. CPU in contabilità: 11655,304610 s dell'autore
più 81,67 s miei, circa 11737 s lower bound, sotto lo stop di 19800 s. Non sono inclusi la CPU di
ricezione del supervisore e il reviewer ChatGPT r008, che non conosco. Non ci sono processi pendenti.

Prossimo ruolo: **supervisore**, per la ricezione di entrambi i report r008 e l'arbitrato reale.
Git resta manuale e solo dopo un GO finale.
