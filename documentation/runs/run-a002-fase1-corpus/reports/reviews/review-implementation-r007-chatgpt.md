# Review — implementazione — run-a002-fase1-corpus — r007 — ChatGPT

- **Esito: NO_GO**, un rilievo medio bloccante, `GPT-R007-I001`.
- Autore/provider/modello: reviewer ChatGPT; OpenAI, Codex basato su GPT-6 come esposto dalla sessione. Identificatore esatto del deployment e chatID non esposti. Nessuna sostituzione o delega.
- Prompt: [25-review-implementation-r007-chatgpt.md](../prompts/25-review-implementation-r007-chatgpt.md), SHA256 `d0177847fa74d7eb6be3466ef500818d67e30f3d878a1c73f81f8f35076bbecf`.
- Oggetto: fix F31–F34 r007, consegna completa dell'autore; stato ricevuto IMPLEMENTATION_REVIEW / FIX_REVIEW_PROMPTS_READY. Questa è una review dell'implementazione.
- Indipendenza: nuova chat di review, distinta dalle attività precedenti; letti entrambi i report **r006**, come autorizzato. Non letti report, evidenze, checkpoint o arbitrato concorrenti **r007**. Nessun agente avviato.
- Base: branch `feature/run-a002-corpus`; HEAD, `dev`, `origin/dev` locale e merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`. Indice vuoto. Nessuna operazione Git di scrittura.
- Snapshot comune: [implementation-r007.json](../../../../../temp/run-a002-fase1-corpus/snapshots/implementation-r007.json), SHA256 `6a5235582225134b364818c96d0e3029597d505d39db936b16e0bec039de7ab4`; impronta `296a3cc7b90044cb6abb2f38fcf2c5d030820e34f03d361a45d078c5c329b535`, 322 file e 65669 artefatti espliciti, inclusi i 20 PDF versionabili.
- Verify prima/dopo: **MATCH / exit 0**. [Prima](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/verify-before.json), [dopo letture e audit](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/verify-after.json), [dopo scrittura degli output](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/verify-after-final.json). Oggetto valido per questa review; MATCH non equivale a GO.
- Piano r001 SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`; errata SHA256 `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`, verificati.

## Ambito e prove

Il delta autore rispetto al contesto r006 comprende **sette file**, anche nuovi non committati: `scripts/benchmarks/observations.py`, `scripts/benchmarks/PROTOCOL.md`, `tests/unit/test_conversion_fix_r007.py`, le due guide `docs/how-to/benchmark-fedelta.md` e `docs/reference/benchmark-fedelta.md`, il report permanente `documentation/benchmarks/2026-10-10-a002-r007-contesti-markdown.md` e il relativo `-data.json`. Il confronto fisico è in [reader-delta.diff](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/reader-delta.diff). Quattro documenti amministrativi del supervisore sono separati nella transizione: CHANGELOG, indice architetturale, next-run e roadmap. Non sono fix del prodotto.

Letti AGENTS, HANDOVER globale, contesto iniziale, stato/HANDOVER/brief della run e checkpoint supervisore r007; skill manage-implementation-run e verify-conversion-fidelity, protocollo del ciclo e template review; prompt24 e arbitrato r006 integrali; entrambi i report/checkpoint r006 e sonde pertinenti; report e checkpoint finale autore r007, final-checks002, inventario e transizione di ricezione. Le sezioni finali **Consegna completa** e **Chiusura amministrativa finale** prevalgono sui passaggi WIP.

Letti l'intero reader corrente e delta, test r007, dipendenze audit/safety/scorer/risorse/adapter/profili/launcher raggiunte e regressioni pertinenti r002–r006. Controllati scanner, walk delle label, inline, HTMLTree, producer, barriere, pipe/trim, offset e primitive code/math/command/metadata/autolink. Consultati piano/errata, ADR0001/0004, fase1 della roadmap, workflow benchmark, addendum V e guida uv nei requisiti pertinenti. Nessuna riapertura della pianificazione o campagna a001.

La prima ora effettiva rilevata è **2026-10-10T08:14:41Z (10:14:41 Europe/Rome)**: lo stop nuovi workload delle 08:06:54.369068Z era già trascorso. Sono state eseguite esclusivamente letture, manager stdlib, hash/archivi/JSON/aritmetica e scrittura degli output propri. **Nessun import del prodotto, interprete managed, R, test, CLI, conversione, rescore del prodotto, build/reinstallazione, render/browser, M/V o rete** eseguito dal reviewer. Nessuno snapshot tecnico proprio necessario. Il controesempio seguente è una traccia statica, non una sonda runtime.

I comandi propri decisionali sono registrati con argv/cwd/output/exit:

| Comando, cwd repository | Risultato e attribuzione |
| --- | --- |
| `python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r007` | Prima e dopo MATCH, exit 0; manager stdlib, senza rigenerazione del comune. |
| `python3 -I -B temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/static_audit.py` | Prima versione exit 1: errore proprio di assunzione dello schema stage-only, preservato con script e log. Versione corretta exit 0, 195297 confronti/assert statici, zero errori. |
| `python3 -I -B temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/supplemental_audit.py` | Exit 0, 8583 confronti/assert statici, zero errori. |

Gli script e i log sono output propri: [audit statico](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/static-audit.json), [log corretto](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/static-audit-command-002.json), [fallimento conservato](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/static-audit-command-001-failure.json), [audit supplementare](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/supplemental-audit.json), [log supplementare](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/supplemental-audit-command-001.json). Non costituiscono una nuova esecuzione delle prove dell'autore. Il numero di assert non decide il GO.

### Controlli sostanziali e chiusure

| Criterio | Risultato nel perimetro controllato |
| --- | --- |
| C01 F31 domini/confini | Il nuovo bound prima del merge e le barriere correggono i payload ricevuti; scanner e consumer **divergono ancora nella label con code soft-line**: I001. Non si può chiudere integralmente F31 e il residuo globale F28/F29/F09 su questa base. |
| C02 F32 autolink | AUTOLINK condiviso e discriminante rispetto al vero HTML; ispezionati scanner, inline e producer. Le sonde Y/K e varianti ricevute chiudono i casi r006 pertinenti, senza fetch. Nessun ulteriore rilievo fondato nell'ambito letto. |
| C03 F33 metadata/raw TeX | Intervalli finiti command/metadata/attrs, destinazione/titolo e foglie separati; semantic ref/eqref/label preservati, unknown locale. Le prove attributi finali sono coerenti coi byte correnti. Nessuna richiesta di supporto TeX universale; resta il limite trasversale I001. |
| C04 trasporti/barriere | Controllati unità fisiche, prefix stripping, nesting, celle/pipe, caption, heading, quote/list/note, fence e label senza destinazione. I001 dimostra una regressione raggiungibile del nuovo ramo labels; le chiusure di code accettate dal walk non sono le stesse dello scanner. |
| C05 chiusure/misure | F27 `</>` diagnosticato fuori dal codice, F28 locale CDATA e F30 reason neutra conservati nelle regressioni pertinenti. `metrics.py` byte-identico a r006. Unknown non diventa wildcard per pagine, ordine, numeri, celle, asset o reference. Counts condizionati/decomposizione distinti dal bound globale; quest'ultimo non è un bound delle CER/WER strutturali. TRIVIAL_ZERO non certifica errore completo zero. |
| C06 F34 documenti | PROTOCOL, guide e report/dati permanenti distinguono prove finali, gold PROVISIONAL, writer sintetico e V reale; r006 storico non riscritto. Corretta distinzione fra veri EOF-label e assert/launcher propri. La pretesa generale di scanner/consumer coerenti resta smentita da I001; nessun secondo rilievo duplicato. |
| C07 64 misure/V | Verificati da file tutti i 64 ID r007b, catena corrente observe/audit/score/final, rawtree, versioni, riferimenti conversion_final/run e old_* r006; 4 PASS + 28 FAIL per ciascuno dei due set. Nessun motore o V nuovo. Dettagli sotto. |
| C08 prove/toolchain | Hash/copie/receipt e binding controllati staticamente; nuovi S/B/I e gate S006 distinti dai vecchi. Prove funzionali S005 attribuibili ai 205 byte runtime identici S006 nei limiti dichiarati. Nessun preflight corrente o PASS storico trasferito dal reviewer. |
| C09 risorse/ciclo | Stop rispettato; nessun nuovo workload. Ledger autore storico distinto da wall corrente, attese incluse, quota non resettata. L01/L02/L04/L05, Chrome NONPASS e gap helper12 restano; nessuna fase2, scelta motore, commit/merge/push/deploy o firma umana. |

I risultati ricevuti tests004 **1112 PASS / 16 F04 deselected**, con 176 nuovi, e suite S006 **1269 PASS + 310 subtest in 34,41 s** sono stati letti e legati ai byte finali, non rieseguiti. La suite selezionata resta parziale, con ignore API/packaging/Docker e smoke radice esclusi. Test rossi intermedi e cause proprie sono conservati; il reviewer non li riclassifica come PASS.

Le sonde prima/dopo verificate per hash e JSON danno GPT 21/14/15: unmet 8/9/8 prima, zero dopo004; Claude Y/K 15 conformi, domains004 12/12. Claude B16 conserva E9 come unico unmet storico: l'atteso evaluate0 è sbagliato con gold PROVISIONAL, evaluate5 è coerente. CLI001/E9-002 dichiarano **38 casi / 76 exit OS**, writer sintetico, zero motori e V: dieci positivi principali più E9 run0/evaluate5, automatic gate PASS e qualità INDETERMINATE; 27 negativi INCOMPLETE/run2/evaluate2/score3 bundle. Lettura di helper/manifest e risultati ricevuti, senza CLI propria. Non si tratta di V reale o baseline dei motori.

Controllati **58941 descrittori protetti**, **73 copie ricevute**, copie fisiche/input-store per-workload e **146 delivery002**; le delivery non sono dichiarate tutte anteriori a ogni workload. Verificati 205 runtime identici S005→S006; su 206 input benchmark, solo PROTOCOL cambia. Controllati i 3813 input package/startup e dieci origini/hash installati, confrontando source, installazione e wheel canonica dei due stage. Binding del reader v7 (`5d4433877b1ab2723c6ee2adec6cd2ec927cebea52ed3c411c33aae8079ffeca`) riscontrato nelle copie fisiche di dodici workload finali. Per i 141 helper/test sono state controllate le copie presenti nel manifest S006 e la coerenza del resoconto di equivalenza, senza attribuire l'intera consegna a un singolo workload.

Le 24 receipt final-checks002 hanno **22 R PASS e 2 R FAIL H12 deliberati**: expected0, harness1/wrapper2 conservati. Preflight/interprete padre-figlio, snapshot, binding input prima/dopo, namespace/socket e reap delle receipt letti separatamente dagli assert. L'origine locale managed 3.12.13 è documentata e vincolata nelle prove ricevute; non è stata avviata ora. I nuovi S006 non cancellano lo STALE del vecchio S integrale dopo i quattro documenti amministrativi del supervisore. Tre pyc encodings correnti ricevuti preservati; nessun restore o causa storica inventata.

Il checker proprio ha ricontrollato i 19 campi numerici/gate/somme/denominatori rispetto a r006 e calcolato con matrice di edit indipendente le distanze dei flussi registrati condizionati e globali. **L'allineamento dei blocchi non è stato ricostruito indipendentemente**. Dati verificati: TXT 0/89; DOCX 11/182, inclusa caption; EPUB 8/186; RTF condizionato 19/162, globale 13/162; TeX WER completo None, condizionato 1/152, globale 0 TRIVIAL_ZERO. Non sono somme di gate come perdite indipendenti. Nessuna differenza nei 19 campi r006; l'attuale final deriva dal nuovo rescore, mentre conversion_final/run resta storico. V13/A03 sono riferimenti a V precedenti, senza nuova firma; L04 formule output UNAVAILABLE resta distinta.

## Rilievi

| ID | Severità e blocco sì/no | Posizione | Raggiungibilità/input ammessi | Attribuzione e requisito | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-R007-I001 | **Media, blocco sì** | `scripts/benchmarks/observations.py:183–191`, in relazione a `markdown_label:348–366`, `markdown_units:236–247`, `inline:492–502`, `observe:870–877` | Link Markdown valido; code span su soft-line nella label, contenente `<!--`, seguito da heading e paragrafo autonomi. Nessun asset o servizio richiesto. | Regressione del nuovo bound labels r007 rispetto al reader r006 ricevuto; F31/C01 e C04, preservazione F29 e residui F13/F14/F09 pertinenti a struttura/ordine. | Scanner limita il code al primo newline, walk/inline lo accettano oltre il newline. Falso commento fino a EOF fonde i blocchi; il consumer assorbe l'opener nel code e non produce unknown. Heading presente nel raw diventa missing artificiale, compatibile con COMPLETE senza diagnosi. | [Traccia statica completa](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/static-label-trace.md), delta fisico e input/attesi sotto. Nessuna esecuzione propria dopo stop. | Un solo intervallo code/limite producer condiviso fra scanner, walk e inline. Tre blocchi attesi, reference/raw/offset corretti, COMPLETE e zero token spurio; positivi/varianti e veri token inerti discriminanti, senza allargare confini né rifiuto generico del documento valido. |

Input minimo, fissato nella traccia statica:

````markdown
See [Known `first
<!--` words.](https://example.invalid)

# Visible heading

After words.
````

Atteso: paragraph `See Known first <!-- words.`, heading `Visible heading`, paragraph `After words.`, in questo ordine; una reference `https://example.invalid`, nessun token HTML spurio, COMPLETE.

Il walk della label salta il code fino al backtick della seconda riga con `inline_scope_end`, e `markdown_link` accetta il link prima del clipping. Il ramo scanner `if labels` riduce invece bound a newline+1: non trova il backtick, riprende sulla seconda riga e registra `<!--` sino a EOF. `markdown_units` sopprime i confini interni al falso span; `observe` aggiunge l'unità fusa al primo paragrafo. Nell'inline ricorsivo il falso span non è interamente dentro la label e non viene trasportato; il code span è letto correttamente e non emette unknown. Il seguito resta testo del paragrafo. Questa catena basta alla decisione statica; **COMPLETE e struttura attuale sono dedotti, non exit OS misurati**.

Nel reader r006 SHA256 `ffd83938197fa2d203484d77a75d32e55d8cab9f55fdf5f2c2a6909182cc00af`, lo scanner trova il backtick della seconda riga e non crea il falso span. È quindi attribuibile al delta r007, non a un nuovo requisito. I test r007 quote soft-code e label con token reale non coprono questa interazione. Il fix minimo può riutilizzare gli intervalli atomici accettati dal walk, mantenendo il limite del producer; non serve HTML/Markdown universale. Varianti pertinenti: CDATA, due backtick, prefix quote/list/note, positivo code su una riga e soft-code fuori label, veri commenti che devono restare inerti sino a chiusura/EOF.

## Motivazione dell'esito e limiti

**NO_GO sul comune identificato.** Le prove finali ricevute chiudono i payload r006, le 64 misure conservano i limiti dichiarati e l'attribuzione S005/S006 è sostenuta dai byte. Tuttavia I001 espone un errore del misuratore su input già ammesso: il raw contiene un heading che il reader trasforma in testo di un altro blocco senza diagnosi. Non è chiudibile con il numero di test, la rarità del caso o l'assenza di una review concorrente.

F31/F34 non possono essere dichiarati integralmente chiusi finché la divergenza di label resta. F32/F33 e chiusure precedenti risultano sostenuti nei domini e casi effettivamente letti; nessuna certificazione di tutti i trasporti arbitrari o altri motori. Nessun difetto ulteriore fondato viene inventato. Il caso non ha conferma runtime propria perché lo stop è precedente all'ingresso; l'impedimento riguarda nuove sonde, non l'identità dell'oggetto né la conclusione statica.

La quota unica rimane 52h dall'entry `2026-10-08T07:06:54.369068Z`, con deadline `2026-10-10T11:06:54.369068Z`. Il ledger autore 146250,572159 s (~40,625h), CPU lower 10400,457453 s, disco retained 2086483026 B e RSS sampled 740855808 B è **storico**, non saldo corrente. Costi propri e fine amministrativa sono in [costs.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/costs.json): misure parziali senza upper o certificazioni hardware. Errori del checker proprio conservati, nessun FAIL/E9/STALE convertito in PASS.

I controlli documentali propri sono registrati in [document-checks.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/document-checks.json): link locali degli output e documenti del delta presenti, `git diff --check` exit 0; hash prompt/snapshot nell'audit statico. Gli identificatori degli anchor non sono validati. La prima passata documentale ha segnalato il link al proprio output JSON prima della sua creazione: [FAIL preservato](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r007-chatgpt/document-checks-001-forward-reference-failure.json), poi ripetizione PASS sui file presenti, senza fix prodotto. Non sono collaudo del prodotto. Tutti gli output sono esclusivamente nel perimetro del reviewer; nessun input ricevuto/common o registro condiviso modificato.

Prossimo ruolo: **supervisore, per ricezione dei due report r007 e arbitrato**. Ricevere I001, decidere il fix e l'eventuale prova necessaria entro i vincoli temporali reali; nessun nuovo workload è autorizzato automaticamente da questa review, nessun reset della quota. Commit/merge/push e promozione restano manuali dell'utente dopo l'eventuale GO futuro. [Checkpoint reviewer](../handovers/review-implementation-r007-chatgpt.md).
