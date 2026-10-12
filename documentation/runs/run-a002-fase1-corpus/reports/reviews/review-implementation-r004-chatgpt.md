# Review — implementazione — run-a002-fase1-corpus — r004 — ChatGPT

- **Esito: NO_GO**, per GPT-R004-I001. GPT-R004-I002 è un completamento non bloccante. I casi originali F17–F20 sono risolti nel perimetro controllato; resta una perdita della diagnosi di markup fuori profilo. Oggetto valido: verify prima/dopo **MATCH**, exit0.
- Reviewer ChatGPT, agente Codex, provider OpenAI, famiglia GPT-6 dichiarata dall'ambiente. Deployment esatto e chat ID non esposti; nessuna sostituzione, delega o impersonazione di Claude.
- Prompt origine: [16-review-implementation-r004-chatgpt.md](../prompts/16-review-implementation-r004-chatgpt.md). Review mirata dei fix r004, non nuova review del piano.
- Indipendenza: chat nuova, distinta da autore/supervisore e dalle review precedenti. Letti entrambi i report/checkpoint **r003**, ammessi. Non letti report, evidenze o checkpoint del concorrente **r004**, né arbitrato r004.
- Snapshot: [implementation-r004.json](../../../../../temp/run-a002-fase1-corpus/snapshots/implementation-r004.json), SHA256 `7105ec0e91e0783a26e13623ecf9eff37211291323253e39e7dc815e92d768c7`; impronta `3976b26937e2f80ee03fb089e7e18a4fd8782edd96c0f821536c367a65064edf`, 313 file e 41997 artefatti espliciti. Non modificato o rigenerato.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`. Indice vuoto; modifiche e file nuovi locali considerati, oltre al diff tra commit.
- Piano r001 SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`; errata SHA256 `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`, verificati. Autorità: piano/errata/GO e mandati vigenti, [arbitrato r003](../arbitrations/arbitration-implementation-r003.md) e [prompt15](../prompts/15-implementation-r004-fix-html-coverage-and-lower-bounds.md).
- Skill applicate: [manage-implementation-run](../../../../../.agents/skills/manage-implementation-run/SKILL.md) e [verify-conversion-fidelity](../../../../../.agents/skills/verify-conversion-fidelity/SKILL.md). Struttura del [template review](../../../../development/templates/review.md).

## Ambito e prove

### Letture e delta

Letti AGENTS, handover globale, parte corrente PROJECT-CONTEXT, STATE/brief e passaggi correnti HANDOVER/checkpoint supervisore; protocollo run-lifecycle, skill e template; arbitrato r003 e prompt15 integrali; report/checkpoint r003 e sonde pertinenti; report autore r004 e sezione finale autorevole del checkpoint, final-checks001, PROTOCOL, entrambe le guide benchmark e report/dati permanenti r004; receipt/transizione di ricezione. Consultati requisiti pertinenti piano/errata/GO, ADR0001/0004, indici architetturali, fase1 roadmap, workflow benchmark e guida uv. Non ripetuta la campagna a001 o la review integrale del piano.

Letti integralmente observations/metrics, il loro delta r003→r004, audit/safety, nuovo test r004 e test r003; consultati regressioni r002/benchmark pertinenti, math_structure, data, observe_bundle/score_bundle/evaluate, adapter/registry, launcher/inventario e guardie S/B/I/R. Le letture statiche non hanno importato il prodotto. Ispezionati gli oracle autore completion/check-derived: matrice indipendente dalle funzioni del comparatore, ma completamenti finiti e alignment registrato non costituiscono una garanzia universale.

Il [checker proprio](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-chatgpt/artifact_audit.py) confronta i due snapshot e la transizione: **8 file autore**, distinti da quattro registri documentali del supervisore:

- `scripts/benchmarks/observations.py`, `metrics.py`, `PROTOCOL.md`;
- `tests/unit/test_conversion_fix_r004.py`;
- `docs/how-to/benchmark-fedelta.md`, `docs/reference/benchmark-fedelta.md`;
- `documentation/benchmarks/2026-10-09-a002-r004-profili-html-e-bound.md` e relativo `-data.json`.

Audit/safety/adapter/R/resources/writer/config/PATH/lock/pin e sorgenti corpus restano invariati. I rilievi sotto raggiungono il produttore HTML modificato e i contratti F13/F14/F21; non riaprono indiscriminatamente fix r002 estranei.

### Verifiche ed esecuzioni proprie

Cwd: `/home/davide/workarea/markdown-for-llms`. Comando prima e dopo:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r004
```

[verify-before.txt](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-chatgpt/verify-before.txt) e [final-checks.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-chatgpt/final-checks.json): exit0/MATCH. Il primo ha stdout/stderr uniti; la sua ora di completamento è ricavata dall'mtime del file, senza inventare uno start. Il secondo registra argv/cwd/start/end/output/exit. Manager stdlib, non test del prodotto. Git read-only finale conferma identità, indice vuoto e diff-check exit0.

Prima del primo managed, [startup-input-audit.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-chatgpt/startup-input-audit.json) confronta package-input, prerequisiti del target e startup `.pth`, senza differenze, e i dieci moduli origine/installato. Managed reale `.venv-python/cpython-3.12.13-linux-x86_64-gnu`, venv finale s008 ricevuta; archivi s008 poi verificati nel nuovo B, nessuna build o reinstallazione reviewer.

Nuove receipt **S/B/I**, origine e preflight propri tutti exit0: chiamate e stdout/stderr nei file `*-command.json`. Package manifest, helper e inventario sono vincolati al nuovo [snapshot tecnico inputs](../../../../../temp/run-a002-fase1-corpus/snapshots/review-r004-chatgpt-inputs-001.json); target/preflight/receipt e helper al nuovo [snapshot tecnico workload](../../../../../temp/run-a002-fase1-corpus/snapshots/review-r004-chatgpt-workload-001.json). Label conformi `review-r004-chatgpt-*`; nessuno snapshot ricevuto mutato. L'impronta generale coincide col comune r004. R usa lo stesso Python dei figli, Firejail `--net=none`, ambiente chiuso, timeout120s, expected-exit0 fissato prima, nessun motore/render/browser.

| Invocazione propria | Esito effettivo | Attribuzione |
| --- | --- | --- |
| [R-probes-001](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-chatgpt/R-probes-001/receipt.json) | IMPEDITA, wrapper2, EPERM sul socket sintetico prima del workload | Conservata. Diagnosi ed escalation della stessa operazione autorizzata; nessun rifiuto automatico di approvazione. |
| [R-probes-002](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-chatgpt/R-probes-002/receipt.json) | **R PASS**, wrapper/workload0 | Escalazione approvata mantenendo R. Preflight S/I e inventario correnti PASS, binding prima/dopo MATCH, figli reaped, socket pulito. 16 casi effettivi. |

[probes.py](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-chatgpt/probes.py) fissa input e attesi prima del workload; [probe-summary.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-chatgpt/probe-summary.json) e `probe-output/` conservano raw/gold/observazioni/audit/score. Gold sintetici **PROVISIONAL**, nessuna V. Gate PASS dei casi sotto non è quality globale PASS: questa resta INDETERMINATE. Exit evaluate sono **derivati dalla regola** su audit/score reali, non conversioni o evaluate end-to-end. R PASS indica esecuzione isolata, non conformità dei payload.

| Caso proprio | Bundle / automatic_gates / unknown / WER | Conclusione |
| --- | --- | --- |
| `text-positive`, `wrapper-positive` | COMPLETE / PASS /0 /0 | Positivi discriminanti |
| `wrapper-unclosed` | **COMPLETE / PASS /0 /0** | I001 |
| `wrappers-unclosed` | **COMPLETE / PASS /0 /0** | I001, due wrapper |
| `wrapper-unclosed-table` | **COMPLETE / PASS /0 /None** | I001 anche su tabella; celle2/2, dominio prosa vuoto |
| `wrapper-unclosed-inline` | INCOMPLETE / FAIL /1 /None | Controllo: il ramo inline conserva l'errore |
| `comment-tail` | INCOMPLETE / FAIL /1 /None | Controllo della nuova guardia tail |
| `comment-middle`, `declaration-middle` | **COMPLETE / PASS /0 /0** | I002: trattamento dipendente dalla posizione |
| `table-positive` | COMPLETE / PASS /0 /None | Positivo F18 |
| `table-residue-missing` | INCOMPLETE / FAIL /1 /None | Risorsa assente raggiunge audit |
| `table-caption-added` | COMPLETE / FAIL /0 /None | F19 discriminante |
| `inline-table`, `unknown-text` | COMPLETE / FAIL /1 /None | F17/F20: niente completo0 spurio |
| `text-insertion` | COMPLETE / FAIL /0 /1.5 | Conteggio e bound3/2, denominatore2 |
| `literal-less-than` | COMPLETE / PASS /0 /0 | F21 letterale fedele |

### Prove consegnate e controlli autonomi

[artifact-audit.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r004-chatgpt/artifact-audit.json): **43085 controlli PASS**, stdlib, nessun import reader/scorer/generator/checker autore. Conferma 38018 input protetti,42 copie ricevute,160 input con copie prima/dopo e hash effettivi,48 helper/test consumati s007→s008 equivalenti, delta8+4,20 receipt finali **18PASS+2FAIL** con binding e socket. Dopo il workload, ulteriori **4167 righe** package/target/inventario confrontate senza differenze; non trasferito il preflight s008 ai documenti generali modificati dal supervisore.

Le64 misure sono controllate nei file effettivi: raw copiati byte-identici; legami raw→observations→audit→score, gold/V e identità conversione; versioni reader/scorer; pubblicazione dei campi numerici e diagnostici confrontata con metrics reali. Ricalcolate distanze token delle entry KNOWN, somme/categorie, offset, role_candidates, denominatori pieni e distanze globali char/token. **Zero differenze nei campi storici numerici/gate** confrontati; due set32, ciascuno **4PASS/28FAIL**. Conteggi ottenuti dai risultati, non imposti per far superare il checker. Non è un nuovo rescore del comparatore né una prova universale dell'alignment.

Confermati: TXT0/89; DOCX11/182 (caption2+9); EPUB8/186; RTF19/162 condizionato =11aligned+3omissione+5inserzioni, contro **bound13/162**; TeX completoNone, condizionato1/152 contro **bound0/152**. Decompv2 qualifica tutte le entry come conteggi strutturali, `guaranteed_bound=false`. Gate non sommati come perdite indipendenti, caption divenute prosa non dichiarate materialmente perse. Baseline003 resta conversione r002 del2026-10-08UTC, misura r004 derivata del2026-10-09UTC, engine_runs0. Corpus/gold/V13 preservati; nessuna nuova firma umana, L04 separata.

Ricalcolate anche le distanze dei **42 completamenti/10 casi** ricevuti e le disuguaglianze rispetto ai bound. Gli attesi dell'oracle non chiamano `metrics.levenshtein`; la garanzia matematica dipende comunque dalla politica esplicita, non dal campione finito.

Letti receipt e stdout autore: mirati475PASS/16F04deselected; suite veloce632PASS+310subtest in27.76s, comando prescritto con ignore API/packaging/Docker e smoke radice esclusi. Contratti H01–H08 R PASS; H12± R **FAIL** deliberati, harness1/wrapper2, expected0 prefissato. Non riclassificati come R PASS. Verificata equivalenza dei48 input dipendenti s007→s008 e receipt finali, non assunto che il nome stage certifichi l'esecuzione. Build/reinstallazione/preparation sono prove autore ricevute, non mie esecuzioni.

Fallimenti prepare relativo, collection SyntaxError,446PASS29FAIL e474PASS1FAIL malformed `<img` restano conservati. Il mio primo checker statico è fallito confrontando i JSON prima/dopo integralmente: il secondo aggiunge legittimamente `after_sha256`. Script/stdout/stderr001 conservati, confronto corretto sui descrittori e hash effettivi nella versione002 PASS. Nessuna modifica degli input autore o trasferimento di risultati dal tentativo fallito.

## Rilievi

| ID | Severità e blocco | Posizione | Input/percorso/requisito e attribuzione | Problema e impatto | Evidenza | Fix minimo e prova di chiusura |
| --- | --- | --- | --- | --- | --- | --- |
| **GPT-R004-I001** | Media, **bloccante** | `observations.py:307`, `:316`, `:323`; audit resource_unknown | `<div><p>Known words.</p>` e wrapper section/div non chiusi, anche intorno a table finita. Input passivo raggiunge observe→html_nodes→visit→audit→score. Difetto **ereditato dal ramo r003**, raggiunto dalla nuova politica r004 e dai requisiti F13/F14 markup non bilanciato, C06, PROTOCOL/guide «markup non bilanciato … INCOMPLETE». | `parsed(wrapper)` produce correttamente unknown/resource_syntax, ma `visit` scarta `base` del wrapper e visita solo i figli. Nessun parser.errors raccoglie i container non chiusi. Risultato senza unknown: **COMPLETE/PASS automatico**, WER/CER0 nel caso testo; stessa violazione nel caso table. Il raw generale resta conservato e il testo noto non è perso: viene persa la diagnosi di sintassi fuori profilo, con un falso completo rispetto al contratto dichiarato. Non dimostrato bypass URI o fetch. | R-probes-002 PASS; casi `wrapper-unclosed`, `wrappers-unclosed`, `wrapper-unclosed-table`, contro `wrapper-positive` e `wrapper-unclosed-inline`; raw e JSON propri. Attribuzione statica sul received-bytes r003, non runtime vecchio inventato. | Rendere gli errori strutturali del parser/wrapper indipendenti dalla proiezione dei nodi (oppure trasportarli esplicitamente ai consumatori), mantenendo foglie/raw/offset una volta. Test su div/section/body/article e annidamenti, prosa/table/figure, raw HTML fence; negativi INCOMPLETE/2 e nessun PASS, positivi completi invariati. Test/regressioni dipendenti e rescore/invarianza64 dopo il fix. |
| **GPT-R004-I002** | Bassa-media, **non bloccante**, completamento F21 | `observations.py:227` classe HTMLTree; `:257` covered_end | `<p>Known <!-- comment -->words.</p>` e `<p>Known <!bad>words.</p>`, contro commento finale. Callback comment/declaration non rappresentate. **Ereditato** per l'omissione dei callback; nuova guardia tail r004 copre solo la coda. F21/C06 richiede che veri commenti/dichiarazioni non spariscano e mantengano i controlli; test r004 ha solo casi standalone. | Un massimo degli end non prova copertura degli intervalli interni. Questi byte vengono ignorati dal tokenizer e, se seguiti da contenuto/nodo, risultano **COMPLETE/PASS**, unknown0; in coda lo stesso commento è INCOMPLETE. Raw generale conservato, nessuna perdita di prosa visibile o asset eseguito dimostrata; resta omissione della diagnosi e incoerenza della classificazione del profilo. | R-probes-002: `comment-middle`, `declaration-middle`, `comment-tail`; ispezione statica callback/covered_end. Non rivendicato HTML universale o un asset dentro un commento come risorsa attiva. | Conservare i token comment/declaration con raw/offset e politica esplicita del profilo, o unknown localizzato consumato da audit/scorer; non usare solo l'estremo massimo come prova di copertura. Negativi a inizio/interno/fine e dentro cella/caption/residui, positivi literal `<` e code invariati. |

### Valutazione esplicita dei fix e criteri

| Fix/criterio | Esito e limite |
| --- | --- |
| F17 / C01 | **Chiuso nel profilo inline controllato**: trasporto del sottoalbero completo, block inline unsupported con testo noto/ref/math/unknown; I1 nonPASS, I2 completoNone; test su cinque ospiti e wrapper, positivi span/a/enfasi/MathML/code. Rami di blocco richiedono ancora I001. |
| F18 / C02 | **Casi originali chiusi**: walk dei residui non selettivo, caption duplicate consumate una volta, audit delle risorse; H1 INCOMPLETE, H2/F1FAIL, H0/F0PASS; formule/residui/URI/offset nei test. Non certificata la copertura universale del produttore a causa di I001/I002. |
| F19 / C03 | **Chiuso**: confronto assente/vuoto/testo, componenti tipizzate separate, unknown nonPASS. Equivalenza assente-vuoto limitata al testo e documentata; caption aggiunta propria FAIL. |
| F20 / C04 | **Chiuso**: conteggi condizionati espliciti, bound globale distinto. Per unknown qualsiasi stringa è ammessa e0 è garantito dalla nonnegatività; flusso misurabile fissato con LF/token concatenati, distanza minima libera. Limite di calcolo→zero con ragione/versione. Denominatori interi originali. Nessun requisito arbitrario di bound stretto su C1. |
| C05 / F15 | **Preservato nei controlli**: unknown non wildcard del matching, missing/page/order/numero/cella/math/ref/asset restano discriminanti; disponibili completiNone dove richiesto; inserzione certa3/2, C1condizionato3/30FAIL. L1condizionato20/30 distinto dal bound0; decomp e domini coerenti. |
| F21 / C06 | Letterale `<` **chiuso**, inclusa sonda propria; veri tag/duplicati/malformed inline/tail hanno negativi. **Copertura incompleta** per I001/I002, con falsi COMPLETE/PASS del profilo. |
| F16 / C07 | **Preservato sulle64 misure**, checker autonomo e pubblicazione coerente,4/28 per set,engine_runs0,V13 invariata,L04 distinta. Nessuna nuova conversione attribuita al rescore. |
| C08 | **Soddisfatto nel perimetro**: byte protetti/copiei/input, dieci origini, nuovi gate propri, equivalenza48,receipt18+2, S/B/I/R pertinenti. Suite/contratti autore letti, non rieseguiti; nessun PASS da stage stale trasferito. |
| C09 | **Soddisfatto**: quota/entry/stop invariati, ledger closing storico, CPU lower/RSS sampled/retained con limiti, fallimenti/helpergap12/ChromeNONPASS/L01/L02/L04/L05 conservati. Guide usabili, claim generale sul markup va completato, ruoli/Git manuali, nessuna fase2/scelta motore. |

Residui: **F01** chiuso sulle foglie/risorse e caption dei casi r003, ma il contratto generale unsupported resta aperto per i rilievi di copertura; **F05** chiuso su inserzioni/disponibilità/garanzia separata; **F09** Markdown preservato e risorse HTML residue corrette, ma la completezza del profilo HTML resta limitata da I001. **F13/F14** aperti soltanto per il residuo di propagazione/copertura descritto; **F15** chiuso nel dominio esplicito; **F16** preservato. Gli altri fix r002 non sono riaperti senza dipendenza reale.

## Motivazione dell'esito e limiti

**NO_GO sul comune r004 identificato**, per il requisito esplicito di non dichiarare completo il markup non bilanciato/outside-profile. I001 è riprodotto sul percorso reale, ha un positivo e un controllo inline e non dipende dal concorrente o dal numero dei test. Non contesto la fedeltà del testo visibile dei payload: contesto la cancellazione di una diagnosi che il produttore ha già generato e il falso stato COMPLETE/PASS. I002 è un completamento locale non bloccante coerente con l'azione F21 precedente; può essere corretto insieme, senza nuova pianificazione o parser universale.

Le64 misure ricevute e F20 non sono invalidati aritmeticamente da questi casi: nei raw benchmark l'HTML interessato non è presente. Il bound zero su unknown è una scelta garantita e ammissibile. La diagnosi dei wrapper e la copertura lessicale vanno completate senza adattare gold/alignment, ridurre i domini o rendere unknown una wildcard globale.

Non rieseguiti suite generale/mirati/contratti, build/reinstallazione, motori, renderer/browser, M/V o rescore prodotto; vietati dal mandato reviewer. Per queste attività sono state valutate le prove autore, non rivendicate come proprie. Sonda sui moduli helper del repository, wheel legacy solo verificata; nessun nuovo supporto universale HTML, ABI, OCR, GPU/altri OS o hard cap risorse. Sicurezza URI invariata: nessun bypass dimostrato. Nessuna fonte remota consultata o nuova versione raccomandata.

Costi misurati propri: R-probes-002 workload0.240573485s, wait4CPU0.22262s, picco albero **campionato**25571328B; figli reaped. Gate/snapshot/wrapper registrati: wall12.943524226s, CPU attesa figli12.904726s; checker statico finale3.426869659s wall/3.42561234s CPU processo. Questi sono sottoinsiemi misurati, non un upper-bound della review e non vanno sommati al CPU R già incluso nei wrapper. Primo checker fallito e letture/Git/tools hanno costi parzialmente non misurati. Intervallo tra completamento prima verify17:03:20.107149UTC e final-check17:12:27.340523UTC:547.233374s; non è un tempo attivo umano.

Alla final-check il wall cumulativo dalla entry originale è122732.971455s (~34.0925h), distinto dal ledger closing autore120881.053504s. Quota52h/CPU21600s/disco6GiB/RAMoperativa8GiB/stop sampled6GiB, stop nuovi49h/19800CPU-s/5GiB e riserva3h/1800CPU-s/1GiB invariati; stop2026-10-10T08:06:54.369068Z, deadline11:06:54.369068Z. Evidenze proprie circa1.53MB prima di report/checkpoint/check finali, non un inventario dell'intero disco. Nessun saldo CPU/disco universale inventato; autore CPU7524.451826s lower/retained1479827278B/RSS735604736B restano storici. Rete/acquisizioni/invio/costo esterno0, nessun privilegio persistente, Git di scrittura, cleanup o modifica prodotto.

Nessun processo proprio persistente pendente: chiamate concluse e receipt R reaped/socket clean. Prossimo ruolo: **supervisore per ricezione dei due report r004 e arbitrato reale**, con [checkpoint proprio](../handovers/review-implementation-r004-chatgpt.md). Stato condiviso, input ricevuti e file del concorrente non modificati.
