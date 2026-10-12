# Review indipendente fix implementazione a002 r002 — Claude

Agisci esclusivamente come **reviewer Claude** in una NUOVA chat, separata
da autore/supervisore e dalla precedente review. Registra provider effettivo
ed eventuale sostituzione, senza impersonare quello previsto. Non leggere
la review concorrente **r002** o il suo eventuale arbitrato corrente.

Output esclusivamente tuoi, sotto `temp/run-a002-fase1-corpus/`:

- Report: `reviews/review-implementation-r002-claude.md`.
- Evidenze: `evidence/review-implementation-r002-claude/`.
- Checkpoint: `handovers/review-implementation-r002-claude.md`.
- Nuovi ID rilievi: `CLA-R002-I001`, progressivi, distinti dai rilievi r001.
- Prompt origine: `prompts/11-review-implementation-r002-claude.md`.

## Oggetto comune e identità

Run `run-a002-fase1-corpus`, review dei fix dell’implementazione **r002**.
Stato IMPLEMENTATION_REVIEW / FIX_REVIEW_PROMPTS_READY. Consegna completa
ricevuta; F01–F12 sono dichiarati chiusi dall’autore e attendono valutazione.
Nessun GO o arbitrato r002. Le due review r001 NO_GO e il relativo arbitrato
sono input ammessi per verificare le correzioni. Questa review copre i fix,
l’intero loro delta e le regressioni nelle dipendenze raggiunte.

- Snapshot: `temp/run-a002-fase1-corpus/snapshots/implementation-r002.json`.
- SHA256 snapshot: `da3ecd3c352afe15bc07a0dba3ed27e52831e779fa1ab19981ad7fe92621291a`.
- Impronta worktree: `9dae2da3d81663f51189fab604effc659ca02fc85bc38bfb2c899e72e52418ca`.
- 307 file di worktree e 25706 artefatti espliciti. I 20 PDF del corpus sono
  ora inclusi nell’impronta generale e ammessi in Git.
- Branch `feature/run-a002-corpus`; HEAD, dev, origin/dev locale e merge-base:
  `80c31da46e5caab990d3596c79c44e524ec1dc95`. Indice vuoto alla ricezione.
- Piano r001 SHA256: `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`.
- Errata SHA256: `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.

I percorsi relativi della run sono sotto `temp/run-a002-fase1-corpus/`.
Il report è `reports/implementation-r002.md`. Nel checkpoint dell’autore,
`handovers/implementation-r002.md`, prevale la **sezione finale “Consegna
completa r002”**: le sezioni iniziali “in corso” descrivono passaggi storici.

Il delta dell’autore è registrato in
`evidence/implementation-r002/implementation-delta-003.json`: 76 file, di cui
20 PDF ammessi in Git senza cambiamento di byte. Comprende reader, audit,
metriche, monitor delle risorse, R, validatore, adapter/PATH/figli, campaign,
profile/inventory, M, generatori, confronto e derivazione V, quattro nuovi
helper e test. Converter legacy, pin, lock, registry e opzioni writer sono
dichiarati invariati: controlla le dipendenze effettive. Il lavoro è locale;
`git diff base...HEAD` da solo non mostra nuovi file e modifiche non committate.

`temp/` e `documentation/runs/` sono esclusi dall’impronta generale, ma gli
input pertinenti sono vincolati esplicitamente. Usa
`evidence/supervisor-implementation-review-r002/review-inputs.json` per navigare;
non occorre leggere integralmente tutti i log. Sono inclusi input e byte
preservati r001/r002, canonici rev4, report delle review r001, arbitrato e
origini installate attribuite. Ambienti e cache terzi sono esclusi, salvo
origini esplicite. Nuovi output dei reviewer, registri vivi, checkpoint del
supervisore e prompt10/11 sono esclusi; il checkpoint finale dell’autore è
input fisso. Le esclusioni non permettono di nascondere codice o prove pertinenti.

## Letture e verifica prima/dopo

Usa e dichiara le skill `.agents/skills/manage-implementation-run/SKILL.md` e
`.agents/skills/verify-conversion-fidelity/SKILL.md`. Leggi nell’ordine:

1. `AGENTS.md`, `temp/HANDOVER.md`, sezione corrente iniziale di
   `temp/PROJECT-CONTEXT.md`, STATE/HANDOVER/brief della run, questo prompt e
   `handovers/supervisor-implementation-review-r002.md`.
2. `documentation/development/run-lifecycle.md`, template review, le skill,
   `arbitrations/arbitration-implementation-r001.md` e
   `prompts/09-implementation-r002-fix-benchmark-fidelity.md` integrali.
   Sono ammessi entrambi i report **r001** e le prove pertinenti:
   `reviews/review-implementation-r001-chatgpt.md` e
   `reviews/review-implementation-r001-claude.md`. Non sono review concorrenti r002.
3. Report e checkpoint finali dell’autore r002;
   `documentation/benchmarks/2026-10-09-a002-r002-rettifica-e-baseline.md` e
   `documentation/benchmarks/2026-10-09-a002-r002-data.json`;
   delta003, `final-checks-001.json`, `delivery-data-001.json` e
   `delivery-budget-final-001.json` nelle evidenze dell’implementatore.
4. Requisiti pertinenti di piano, errata, arbitrato GO del piano, prompt05/06 e
   `arbitrations/addendum-v-human-r001.md`; ADR0001/0004, roadmap fase1,
   open-questions, workflow benchmark, guide benchmark e PROTOCOL corrente
   quanto necessario a delimitare i requisiti. Non riaprire tutto il piano.
5. Codice modificato e dipendenze raggiunte, incluso monitor condiviso
   `scripts/benchmarks/resources.py` e R `run_offline.py`, helper
   derive_verification/invariance/metadata_revision/pandoc_guard, nuovo
   `tests/unit/test_conversion_fix_r002.py`, diagnostici e test modificati,
   corpus/spec/header/transcript. Leggi il codice senza importarlo implicitamente.
6. Prove indicate di seguito. Evita scansioni indiscriminate della storia a001:
   integrazione e pulizia sono già ricevute e non vanno ripetute.

Verifica branch, HEAD, dev, merge-base, indice, delta e hash di snapshot,
piano ed errata. Esegui prima e dopo la review:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r002
```

È un controllo stdlib del contesto, non una prova del prodotto Python.
Registra argv/cwd/output/exit nelle tue evidenze. Non rigenerare lo snapshot
né correggere input comuni. STALE o identità diversa invalidano la review
sull’oggetto ricevuto: conserva il risultato e segnala fatto e prossimo passo;
non dare GO sullo stato precedente. Gli snapshot storici r001, del contesto
fix e degli stage preliminari attribuiscono prove precedenti e non devono
risultare tutti MATCH dopo modifiche autorizzate.

## Ricezione e differenza nell’ambiente

Il supervisore ha eseguito letture, controlli statici, hash e identità Git:
nessuna suite, build, R, conversione o nuova V. La `reception.json` del ruolo
registra 11369 confronti hash, 9189 input storici invariati, 208 input di stage
preservati e 230 prima/dopo baseline, 20 PDF non ignorati, 13 receipt derivate
V e conteggi della baseline. Non certifica la semantica delle correzioni.

**Tre binari esterni sono diversi rispetto all’inventario stage-s013:**
`/opt/google/chrome/chrome`, `/opt/google/chrome/chrome-sandbox` e
`/opt/google/chrome/chrome_crashpad_handler`. Dei 206 input dell’inventario,
203 coincidono. Hash e dimensioni precedenti/attuali sono nella reception;
la causa non è accertata. Lo snapshot comune vincola workspace e artefatti,
non tutti i binari esterni. Le receipt storiche rimangono attribuite ai loro
input registrati. Non trasferire il vecchio preflight completo a nuovi probe,
non sanare il precedente Chrome NONPASS e non ripristinare/scaricare binari.
Per eventuali prove proprie aggiorna target, inventario e preflight sulle
**dipendenze realmente consumate**, senza modificare le receipt altrui.

## Prove finali da valutare

Tutte le prove seguenti sono sotto `evidence/implementation-r002/`, salvo
indicazione diversa. I nomi abbreviati rimandano ai percorsi citati nel report.

- `baseline-bytes.json` e `received-helper-delta-001.patch`: byte/diff della
  consegna r001 effettivamente ricevuta. Non ricostruiscono i due helper
  storici mancanti della campagna002: quella lacuna resta visibile.
- `metadata-revision-005.json`, `transcript-invariance-001.json`,
  `ignored-pdf-entry-check.json`, `git-pdf-and-complete-corpus-001.json`:
  26 file di metadati/derivati, 11 didascalie in 9 transcript, fonti e asset
  invariati. Corpus completo: 129 file.
- `M-gold-007/M.json`, `M-controls-007/M.json`: 415+93 controlli dichiarati PASS,
  inclusi 15 controlli tecnici sugli header. `comparison-006/packet.json` e
  `verification-derived-002/summary.json` con 13 receipt e view-check schema3.
  La V reale originale è conservata; `new_human_confirmations=0`. Verifica
  vecchio/nuovo gold, packet/pair, scope, delta ammesso, fonti e viste effettive.
  `technical_header_qualified_by` attribuisce il controllo a M, non alla persona.
  Non richiedere una nuova V generale per modifiche di soli metadati o cambio chat.
- Stage-s013: S/B/I, dist, origini installate, target/preflight/inventory/audit;
  `input-bytes-s013.json` (208) e `baseline-input-bytes-before-003.json` /
  `baseline-input-bytes-after-003.json` (230), con copie effettive. Managed
  CPython 3.12.13, wheel canonica reinstallata, dieci hash e figli nello stesso
  interprete. I documenti finali post-workload non rendono corrente il vecchio
  preflight su tutto il worktree: confronta gli input realmente consumati.
- `R-r002-tests-008`: 247 test mirati; `R-r002-tests-fast-004`: 388 test e
  310 subtest dichiarati PASS. Suite selezionata con API/packaging/Docker e
  smoke radice esclusi, non collaudo completo. Leggi stdout e testitems;
  i fallimenti intermedi sono conservati. Correzioni degli attesi non devono
  indebolire i requisiti.
- `guards-current-004`: 18 casi; `R-child-guards-summary-004.json`: otto R
  esterne, **sei FAIL attesi e due PASS**, non otto PASS. Stream di 1056768
  byte dopo 100–200 ms, setsid, timeout oltre il leader, positivi 200/400 ms,
  esclusione di orfani estranei. `contracts-007`: 20 esiti attesi; H15 corrente
  è R FAIL STREAM_LIMIT anche con leader exit0. Il monitor usa resources.py,
  supervisore dedicato e pidfd via libc. Fallimenti storici, false receipt,
  Chrome NONPASS, EPERM e rifiuti effettivi di approvazione restano conservati.
- Ultimo validatore XML/resource e regressioni EPUB pre-tool: src in NCX,
  href OPF con case diverso, xml:base, HTML attivo travestito da XML. Verifica
  che i workload finali consumino l’ultima versione. Baseline001/002 su
  s011/s012 sono precedenti storici dopo le ulteriori modifiche.
- `rescore-input-002.json` e `rescore-historical-002/`: 32 raw storici,
  12 campaign002+20 campaign003, 606 input protetti invariati, **engine_runs=0**.
  Nuova osservazione/audit/scoring: celle DOCX 9/9, RTF 6/6, TeX 9/9;
  numeri 12/13, 5/8, 8/9; WER DOCX 11/182 e RTF 19/162; TeX completo
  UNAVAILABLE con lower-bound 1/152. Totale dichiarato 4 PASS/28 FAIL.
- `baseline-003/`: esperimento `a002-r002-baseline-003`, otto percorsi per
  quattro nuovi tentativi, 32 SUCCESS/COMPLETE e qualità 4 PASS/28 FAIL.
  A10 asset referenziati e byte verificati è distinto dalla fedeltà caption.
  28 prove Pandoc e 12 parent/child legacy concordanti per PATH/identità;
  source/helper prima/dopo. Tempi di conversione, child API, audit, score ed
  evaluate distinti; primo avvio+tre ripetizioni, cache OS non controllata,
  RSS campionato: non prova universale cold-start o hard-cap aggregato.
- `final-checks-001.json`: 19 receipt R positive finali, separate dai sei
  negativi R attesi. Nessun processo proprio pendente dichiarato. Bilancio
  finale autore: wall 55151,43 s (~15,32 h) **alla consegna**, CPU nota
  lower-bound 6195,769 s, 918503388 byte conservati più 1890137 byte repo
  prudenziali, picco RSS campionato 735604736 byte; costi/acquisizioni/invio €0.
  Non è il wall corrente. Quota 52 h e ingresso 2026-10-08T07:06:54.369068Z
  invariati: stop nuovi workload 2026-10-10T08:06:54.369068Z, deadline
  2026-10-10T11:06:54.369068Z. Nessun reset dei contatori.

## Criteri comuni di chiusura e regressione

| Fix / rilievo r001 | Verifica sostanziale |
| --- | --- |
| F01 / GPT-I001 | Celle e didascalie pipe/HTML conservano inline tipizzati, raw, offset, coordinate, formule, riferimenti e unknown. Audit di tutte le occorrenze e risorse; scorer vede aggiunte e perdite senza doppio conteggio. Cella realmente vuota PASS; formula supportata aggiunta FAIL; asset mancante INCOMPLETE/exit2 e qualità exit3; immagine presente con alt vuoto non rende valida una caption extra. Positivi con formule/asset in cella e caption; ordine delle caption HTML. Unknown non occulta contenuto noto. |
| F02 / GPT-I002 | FP01–FP13 end-to-end observe→align→score, frazioni con/senza braces e AST equivalenti, gate automatico PASS distinto da L04 UNAVAILABLE. Raw conservato, subset finito; one-to-one, duplicati, ripetuti ambigui, missing/unexpected, FN/FU discriminanti. Nessuna equivalenza algebrica generale implicita. |
| F03 / GPT-I003 | Custodia fino a quiescenza/reap/chiusura writer, oltre leader e setsid; limiti stream/tempo/namespace/RSS e controllo finale. Supervisore stdlib dedicato e pidfd via libc non segnalano o reap processi estranei, PID riusati o orfani stranieri. I 18 casi, 8 R e 20 contratti esercitano il codice consegnato; negativi senza falso verde, positivi entro cap compreso 400 ms, nessun processo proprio pendente. PASS del probe distinto dall’esito del comando interno. |
| F04 / GPT-I004 | Subset passivo validato prima di tool/estrazione: srcset, escape CSS, SVG/ZIP/script, URI XML/NCX/OPF, case e namespace, xml:base, HTML travestito da XML. EPUB negativi non aprono target/motore (marker), exit4; positivi canonici con asset locali e link inerti. No regex globale come validatore universale, fetch o fallback. Rete0 non equivale a sandbox filesystem. |
| F05 / GPT-I005 | WER include inserzioni dei blocchi non allineati, denominatore atteso stabile: esempio 3/2=1,5, positivo0, omissioni/sostituzioni/misti. Scope testo/math/code dichiarato; matematica equivalente non penalizzata come grafia raw. Unknown/unresolved rendono la metrica completa UNAVAILABLE con lower-bound attribuito, non zero/PASS. Rettifiche DOCX/RTF/TeX riproducibili. |
| F06 / CLA-I001 | 20 PDF non ignorati, versionabili e inclusi nel general fingerprint; stessi hash di canonici/render/derivati ricevuti. Eccezione limitata al corpus sintetico, senza staging forzato o PDF privati. Tutti i 129 file necessari a un clone pulito, manifest offline, fixture/test, licenze e font notice disponibili senza dipendenza accidentale da ignored file. |
| F07 / CLA-I002 | Header semantico in spec/gold/M derivato da DOCX/RTF/TeX senza header e EPUB con th; policy PDF/raster manuale esplicita. Reader gold-blind distingue header/body e placeholder pipe realmente vuoto; coordinate e numeri corretti. Celle 9/9,6/6,9/9 su output reali; promozione indebita di body a header FAIL; header/celle/righe vuoti distinti da assenti. Nessun drop ingenuo o riallineamento sul gold. Canonici invariati. |
| F08 / CLA-I003 | Perdita di blocchi/pagine FAIL prevale su unknown estraneo. `exact-visible-unique-v1` locale e conservativo, con duplicati/context/aggiunte/inversioni; K02 pagina2 mancante+unknown estraneo produce exit3 e missing page. FAIL noto e UNAVAILABLE coesistono. Unknown associato non copre altri mancanti né produce PASS/recall spurio; denominatori e provenance coerenti. |
| F09 / CLA-I004 | Target e titolo separati per link/figure, titoli doppi/singoli/parentetici, escape e path bilanciati; colon nel titolo non rende URI unsafe, URI vietata sì. Raw inline con attributi è unsupported tipizzato e localizzato, distinto da codice reale. Riferimenti in celle/caption non supportati restano espliciti e auditati. Rescore TeX e WER UNAVAILABLE motivati; reader limitato dichiarato. |
| F10 / CLA-I005 | Nove transcript sono esattamente i vecchi byte con le undici caption già nel gold, una volta e nell’ordine. Checker indipendente dal generatore/scorer; niente fonti rigenerate o caption inventate. Completezza, hash/diff conservati e riproducibilità dei derivati verificabili. |
| F11 / CLA-I006 | PATH effettivo con allowlist chiusa; token pandoc risolto e realpath/hash/versione prima/dopo, concordi parent/child legacy. Fake/user/cwd/PATH vuoto o relativo/mutazione rifiutati con exit4/H10 prima del tool. 28 prove Pandoc e 12 legacy; registry/writer/input invariati; nessun segreto o fallback implicito. |
| F12 / CLA-I007 | 208 input e 230 prima/dopo preservati, hash e byte degli helper correnti attribuibili; baseline nuova 8×4 e 20 contratti provano la versione attuale. Diff della consegna r001 distinto dagli helper storici irrecuperati: vecchi 12 raw mantengono la lacuna. Origine interprete/wheel, dieci hash, S/B/I, figli e R/--no-sync effettivi. |
| V derivata | Schema3 lega V reale, risposta originale, packet/pair/scope, vecchio gold/receipt, nuovo M e delta ammesso. Invarianza dei contenuti visibili nelle vecchie/nuove viste dei 13 documenti, fonti/render/asset/include/numeri/math/caption/riferimenti/ordine; controllo indipendente, non sola autoconsistenza dei nuovi dati. Tamper, rejected, stale, scope/set/ripetuti e header non visibile non diventano VERIFIED. Zero nuove conferme umane dichiarato; L04 sugli output resta UNAVAILABLE. |
| Misure, stati e documentazione | Rescore storico senza motori distinto dalle 32 nuove conversioni; 606 input, raw/metriche/report storici preservati. Rettifica per fenomeno; 4 PASS/28 FAIL non prefissati. SUCCESS/COMPLETE distinto da qualità, batch, coverage ed exit previsto; A10 asset distinto da caption. Guide/report/dati e ledger coerenti, limiti di tempo/RSS/cache dichiarati, nessun vincitore/scelta app/fase2/costo/fetch/modello/hardware implicito. |

Valuta il fix e tutto il suo delta, oltre ai test nominati. Non ignorare
requisiti essenziali o nuovi difetti raggiungibili nelle dipendenze colpite.
Un FAIL reale di Pandoc è un risultato valido; misura viziata, falso COMPLETE /
VERIFIED / PASS, perdita occultata o confine violato possono bloccare anche
su input sintetico o raro. L01 PDF/OCR, L02 compilazione TeX, L04 giudizio
umano sulle formule output e L05 costi/RSS/cache/hardware restano lacune precise.
Nessuna media o quota di PASS sostituisce i contenuti e i denominatori attesi.

## Autonomia, report ed esito

Sono consentiti letture/ispezione statica, hash, link, aritmetica, snapshot,
diagnostica stdlib e propri output. Eventuali riproduzioni mirate richiedono
managed locale CPython 3.12.13, origine/installazione/S/B/I/dieci hash verificati,
nuovi target/inventari/preflight e R nello stesso interprete per i dati consumati.
Puoi preparare manifest/preflight/snapshot propri nella tua directory, senza
modificare input comuni. Niente build, reinstallazioni, download, nuove
conversioni/render o campagne/suite intere senza rischio o delta pertinente.
Se manca un prerequisito, registra il limite esatto e la prova necessaria.
Timeout ≤900 s, cap R, isolamento e rete0 restano invariati. EPERM/EACCES
richiedono diagnosi ed eventuale escalation tool per l’azione autorizzata;
un effettivo rifiuto di approvazione richiede stop, senza aggirare il confine.

Non implementare fix né modificare sorgenti, gold, packet, receipt, report
altrui, STATE/HANDOVER/eventi/indici/changelog o checkpoint del supervisore.
Non rigenerare snapshot/corpus, inventare firme V o eseguire Git di scrittura,
pulizia, fase2, ML o remoto. Non avviare agenti/reviewer delegati: questa è
una delle due nuove chat indipendenti. Fonti primarie aggiornate con URL/data
solo se valuti nuove versioni/licenze/capacità; puoi restare sulle prove locali.

Usa il template review. Dichiara autore/provider/modello/chat ID effettivi
(senza inventare identificativi non esposti), prompt, giro r002, oggetto/hash,
impronta/base/delta/indice, verifica prima/dopo, indipendenza, file letti,
comandi e limiti. Esito **GO o NO_GO** motivato; invalidità fattuale se STALE.
Per ogni F01–F12 indica chiuso/aperto/non valutato, prova, limite e regressioni.
Per nuovi rilievi: ID propri, severità/blocco, posizione/riga/input ammesso,
raggiungibilità, impatto, requisito e attribuzione (nuovo/legacy/mandato),
correzione minima e prova di chiusura. Non inventare rilievi, gate opzionali,
prove mai eseguite o certificazioni universali.

Nel checkpoint riporta input/output/esito/processi propri pendenti e prossimo
ruolo **supervisore per ricezione di entrambi i report e arbitrato reale**.
Concludi in chat con esito reale e percorsi. Non avviare fix o operazioni Git.
