# Review indipendente implementazione a002 r001 — Claude

Agisci esclusivamente come **reviewer Claude**, separato dall’autore e dal
supervisore, in una NUOVA chat. Registra provider effettivo ed eventuale
sostituzione senza impersonare il reviewer previsto. Non leggere la review
concorrente o un eventuale arbitrato corrente dell’implementazione.

Output esclusivamente tuoi, sotto `temp/run-a002-fase1-corpus/`:

- Report: `reviews/review-implementation-r001-claude.md`.
- Evidenze: `evidence/review-implementation-r001-claude/`.
- Checkpoint: `handovers/review-implementation-r001-claude.md`.
- ID rilievi: `CLA-I001`, progressivi.
- Prompt origine: `prompts/08-review-implementation-r001-claude.md`.

## Oggetto comune e stato

Prima review **completa dell’implementazione r001**, run `run-a002-fase1-corpus`.
Stato IMPLEMENTATION_REVIEW, consegna completa ricevuta; non esiste ancora
alcuna review implementativa o GO codice. Il GO piano r002 approva requisiti,
non il loro adempimento futuro. Non ripetere la review integrale del piano.

- Snapshot comune: `temp/run-a002-fase1-corpus/snapshots/implementation-r001.json`.
- SHA256 snapshot: `6d843889bd8de41b2b6c0842fe6377ba14b9145cc30bbf722662ec77888aa911`.
- Impronta worktree: `457568ae5fef44d597c9dbf66d402f56c918ed358da46355dfbd9e8e7169fb70`.
- Snapshot: 280 file worktree e 9190 artefatti espliciti.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base:
  `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto alla ricezione.
- Piano: `plans/plan-r001.md`, SHA256
  `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`.
- Errata: `plans/errata-plan-r001-r001.md`, SHA256
  `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.
- Disposizione completa: arbitrato piano r002 + prompt05 + addendum-v-human-r001
  e prompt06. Questi ultimi prevalgono sulla modalità umana E02/prompt05.

Percorsi abbreviati dei registri sotto `temp/run-a002-fase1-corpus/`.
Il delta è nel worktree e nei file nuovi non committati: `git diff base...HEAD`
da solo è vuoto e non identifica l’implementazione. Nuovi scripts/benchmarks,
fixture, test, guide/report; modifica al runner R `run_offline.py` ereditato da
a001. Nessuna modifica converter/pin/lock dichiarata: verificarla.

`temp/` e `documentation/runs/` sono esclusi dall’impronta generale; gli input
pertinenti della run sono vincolati direttamente come artefatti espliciti, con
inventario `evidence/supervisor-implementation-review-r001/review-inputs.json`.
L’inventario serve per navigare, non impone di leggere integralmente 9190 log.
Ambienti/cache e dipendenze terze sono esclusi, salvo i venti moduli installati
attribuiti nei due stage finali. I checkpoint vivi del supervisore, registri
comuni, prompt07/08 e vostri output sono esclusi; il checkpoint dell’autore
consegnato è invece un input fisso. Non nascondere codice nei percorsi esclusi.

## Letture e verifica dell’identità

Usa `.agents/skills/manage-implementation-run/SKILL.md` e
`.agents/skills/verify-conversion-fidelity/SKILL.md`, dichiarando l’uso in chat.
Leggi nell’ordine, senza acquisire tutta la storia a001:

1. AGENTS.md, temp/HANDOVER.md, sezione iniziale temp/PROJECT-CONTEXT.md,
   STATE/HANDOVER/brief della run e questo prompt.
2. Protocollo documentation/development/run-lifecycle.md, le due skill,
   template documentation/development/templates/review.md e checkpoint
   handovers/supervisor-implementation-review-r001.md.
3. Piano r001 ed errata E01–E08 integrali, arbitrations/arbitration-plan-r002.md,
   addendum-v-human-r001.md, prompt05 e06. L’arbitrato del **piano** è un input
   autorizzato; non esiste un arbitrato implementativo corrente da leggere.
4. Report autore correnti reports/implementation-r001.md e
   implementation-r001-v2-fix.md, parte corrente handovers/implementation-r001.md,
   documentation/benchmarks/2026-10-08-a002-verified-reference.md e supplemento
   2026-10-08-a002-human-comparison.md. Il report 2026-10-08-a002-native.md è
   storico; usa le parti necessarie per campagna002/contratti/R e attribuzione.
5. documentation/README.md, registro ADR e ADR0001/0004, roadmap fase1,
   next-run-a002-corpus.md, open-questions e workflow converter-benchmark.
   Guide docs/how-to/benchmark-fedelta.md e docs/reference/benchmark-fedelta.md,
   scripts/benchmarks/PROTOCOL.md (verifica i nomi reali nei rispettivi indici).
6. **Tutto il delta prodotto pertinente**: moduli scripts/benchmarks, corpus/
   manifest/manual-spec/gold/controlli/provenance e generatori, nuovi test unitari
   benchmark/comparison e test integration writer; diff di run_offline.py,
   test/diagnostici effettivi del runner. Codice legacy raggiunto dagli adapter,
   unified_converter.py, packaging e tests/conftest.py quanto serve per contratti.
7. Prove effettive elencate sotto, usando report e inventario per i percorsi.
   Receipt a001/S/B/I sono pertinenti solo per riuso attribuito; non caricare
   tutta la campagna a001 o ripeterla per il cambio di chat. Pulizia già ricevuta.

Verifica branch/HEAD/dev/merge-base, indice, modificati e nuovi non ignorati,
hash di piano/errata/snapshot. Esegui prima e dopo la review (stdlib):

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r001
```

Registra argv/cwd/output/exit nelle tue evidenze. Non rigenerare lo snapshot.
Se STALE/identità errata, conserva il risultato e dichiara la review invalida
per l’oggetto ricevuto; nessun GO ricavato dall’assenza di difetti sullo stato
precedente. Gli snapshot **storici dell’autore** identificano prove precedenti:
non pretendere MATCH attuale da tutti dopo modifiche autorizzate. Verifica le
dipendenze consumate dal workload e i cambiamenti successivi rilevanti.

## Prove da esaminare e distinzioni vincolanti

Ricezione supervisore: reception.json e snapshot-receipt.json nella cartella
supervisor-implementation-review-r001. Sono controlli statici di identità,
556hash MATCH,201input tecnici,33oggetti supervisore invariati prima dei nuovi
registri e10coppie origine installata, senza suite o conversioni rieseguite.
Non sostituiscono la tua verifica dei contenuti e della raggiungibilità.

V-comparison-001/packet.json, viste originali/Markdown, reference-map e file
effettivi; V-received-chat-001.txt e V-received-chat-metadata-001.json;
V-response-document-001/response.json, verification-document-001/summary.json
e tredici receipt/view-check. Tutti sotto evidence/implementation-r001/.
Digest canonico packet `ac0921defac79ca063ef89fe270ed09e8d991e7dc17d9a0aeae3c2c4c06a5c6f`,
SHA file packet `4c206f02efbc1bce1030d86b281ceb825e263eb3becc0995b0001215909ff942`.
Risposta originale SHA `2b34e76549d084988bb09ba56d121e66b0b20d81443e2922f01ef1a3cda87f9e`:
«Tutti  idocumenti corrispondono. Ad un esame visivo, valido le conversioni verificate.»

L’utente ha effettivamente confermato l’insieme dei confronti presentati. Il
supervisore riceve tale conferma **documentale collettiva** come V nel contesto
del mandato umano semplificato: una risposta originale, proiezione esplicita
su13documenti, senza215firme atomiche o tredici risposte individuali inventate.
Valuta fedeltà della registrazione, ambito e parser, non imporre all’utente il
vecchio workflow respinto. Una nuova verifica umana non è dovuta per il solo
cambio di chat. V certifica la corrispondenza visibile del riferimento, non gli
output dei motori. Inizio/durata attiva e timestamp piattaforma indisponibili;
260min storici e stima60–90min non sono misure reali. PROVISIONAL nei vecchi
gold/packet PENDING è stato conservato: stato qualificato nella receipt separata.

M-gold-006/M.json (403required PASS), M-controls-004/M.json (89PASS), checker
comparison-check-002 e suoi esiti (1906PASS), test caption-source-contract e
due generazioni013/014: esamina indipendenza e completezza, non solo contatori.
Vecchi canonici rev3/fallimenti sono preservati; caption DOCX/RTF/TeX corretta
contro specifica, con controllo discriminante. La lacuna precedente M sulle
caption non è attribuibile al motore. Raster e compilazioneTeX hanno limiti.

Stage-s040: preparazione, S/B/I, audit wheel, package-inputs, benchmark-inputs,
preflight, target e comandi effettivi. R-tests-comparison-006, R-tests-fast-004,
R-document-response-001, R-document-verification-001,
R-rescore-verified-reference-002: receipt/inside/runner/stdout/stderr.
163test mirati e304test+310subtest suite dichiarati PASS; suite selezionata,
esclusioni API/packaging/Docker e smoke radice, non collaudo completo.
implementation-delivery-consistency-002.json controlla201input finali identici;
documenti scritti dopo il workload non rendono current il vecchio preflight
intero worktree. I dieci moduli installati039/040 sono identici; helper esterni
e nuovi test cambiano e hanno inventario proprio. Non confondere le due cose.

Campaign002 (32conversioni iniziali), campaign003 (20rerun dipendenti), i relativi
S/B/I/R,20contratti, test negativi/risorse e preservazione di errori/parziali.
rescore-verified-reference-input-001.json e
rescore-verified-reference-001/summary.json: ultimo R002 è scoring, zero engine_runs.
32raw validi =12TXT/EPUB della002 +20DOCX/RTF/TeX della003,8gruppi×4.
356file protetti e misure strutturali dichiarati invariati;4PASS/28FAIL qualità.
Sei delta helper storici (safety/adapters) espliciti sui vecchi TXT/EPUB:
non conversioni con codice corrente. Valuta riuso in base alla dipendenza
reale, senza richiedere indiscriminatamente tutti i workload per soli documenti.

Runner R modificato: code/diff e prove guard/post-exit/reap/processi/stderr;
errore precedente falso PASS, H15 guard ucciso senza receipt finale e Chrome
R NONPASS per namespace annidati sono conservati. Verifica che la consegna non
li sani attribuendo loro altri PASS. Firejail dopo errori unshare/permessi:
escalation strumenti autorizzata, isolamento da verificare, non sandbox FS
universale. Le immagini Chrome diagnostiche non provano R PASS del rendering.

## Criteri comuni delle due review

| Criterio | Verifica sostanziale sul codice, sui contenuti e sulle prove |
| --- | --- |
| I01 obiettivo/contratti (A01–A12/E01–E08) | Risultato intero entro incremento, gold/runner/negativi/baseline reali. L01–L05 espliciti, nessun schema applicazione o vincitore/fase2 anticipati. Requisiti del piano corretti realizzati, non sola presenza di file. |
| I02 corpus/reproducibilità (A02) |10documenti/9PDFpagine/IT-EN/colonne/scansioni/misto stesso foglio e3controlli, numeri/formule/matrici/celle/figure/note/bibliografia. Specifiche indipendenti, asset/font/licenze e manifest offline. Byte-identical distinto da content-reproducible, generazioni reali e overflow/glifi/caption verificabili, source/include/hash coerenti. |
| I03 M e gold (A03/E02) | Controlli indipendenti da motore, comparator e gold derivato; testo/codepoint/numeri/formule/XML/asset e raster coperti con limiti dichiarati. Test prima FAIL/poi PASS della caption, non solo gold auto-consistente. Identità source/gold/render/packet/M uguali per tutti13. |
| I04 V e presentazione (A03/prompt06) | Originale effettivo e Markdown di riferimento completo/celle vuote/asset/caption/MathML/note/riferimenti/ordine leggibili. Nessun gold omesso e nessun output motore passato per gold. Una dichiarazione reale, attribuzione/granularità/ambito documentali trasparenti. Parser finito rifiuta negazioni, vaghi, eccezioni, parziali, stale e tampering; verifica riclassifica raw invece di fidarsi solo di ACCEPTED. Ricevute per documento, storia e distinzione Vgold/L04output. |
| I05 observe e alignment (A04/A06) | Gold-blind, typed/raw/offset e unknown; test con gold alternativo. Matching one-to-one, duplicati/aggiunte/mancanti/ambigui e denominatori attesi, inversioni non omissioni. Euristiche0.75/0.15 non generano PASS falso né tuning su output ufficiali. |
| I06 metriche (A05/A06/E01) | Testo Unicode/NFC/NFD, segni e numeri, formule strutturali/AST limitato con grafiche alternative x_i/x_{i}/OMML/MathML, meno in matematica distinto dalla prosa, unknown visibili. Celle vuote/distinte da assenti, ordine/pagine/blocchi/asset/caption/note/link/refs. Denominatori e localizzazione corretti, CER/WER non occultano perdite; unavailable/manuale non diventano PASS. |
| I07 positivi/negativi (A06/A07/A11) | Positivi reali discriminano reject-all e equivalenze lecite; mutazioni discriminano perdite pagina/blocco/numero/segno/simbolo/ordine/cella/asset/note/rawvuoto e batcherrato. Assert contenuto/denominatore/localizzazione/stato/exit, casi ambigui e ripetuti, nessun baseline/gold corretto usando output da valutare. |
| I08 identità/stati/batch (A07/E03/E06) | Document/experiment/attempt/config/hash/omonimia/retry. SUCCESS, COMPLETE, qualità, coverage e batch separati; raw/parziali/errori conservati. Harness sottoR exit0 conserva exit effettivi attesi2/3/5 senza scegliere dopo risultato; fault imprevisto visibile. Inputunsafe4 distinto da outputunsafe2, non seguito.20contratti pertinenti anche dopo delta. |
| I09 baseline reale (A10/E05/E07/E08) |5diretti e3legacy praticabili/effettivi con4attempt ciascuno, writer/profile/media/resource/cwd/PATH/scratch/TeXincludes reali. Requisito nativo conasset soddisfatto da Pandoc/legacy con contenuti/hash/riferimenti, non vacuamenteTXT del benchmark.4PASS/28FAIL non sanati dalla V. Scoring32raw e riuso12storici identificati,20dipendenti ripetuti; conseguenze dei6helperdelta raggiungibili e prove invalidate non nascoste. |
| I10 confini di sicurezza (A08/A09) | Path/symlink/hardlink/ZIP/XML/HTML/SVG/font/includes/resource, nessun fetch/comando da dati/fallbackremoto; shell=False, registry esplicito. Negativi prima di scritture o esecuzioni pericolose. Parser reference/math/Markdown sicuri anche dati invalidi; R non sandboxfilesystem totale. Nessun aggiramento di approvazioni negate. |
| I11 toolchain/R/provenance (A09/A11) | Managed3.12.13, wheel canonica/diecihash/S/B/I e figli nello stesso interprete, preflight prima import/no-sync. Helper esterni conmanifest/audit/tampering, snapshot esecuzione e consumi attribuiti. R stesso interprete/rete/socket/pre-post/guard per tutta esecuzione compreso dopoexit, caps/timeout/reap/figli/SIGKILL/receipt assente o errore senza falsoPASS. Prove correnti e storiche differenziate, errori conservati; suite effettiva non help/versione. |
| I12 risorse/documenti/consegna (A01/A12/E04) |52h quota unica/entry07:06:54.369068UTC/riserva senza reset. Ledger008wall41737.61s,CPUnota1465.67lower-bound,465182720B allocati conservati, RAMsampled10ms/PID/cacheOS/peakdisco limiti; nessuna falsa certificazione dei non misurati.€0/acquisizioni/invio0. Guide/report/logs collegati su prove reali, archiviazione futura/Git manuale/doppia review,nessun resto roadmap avviato. |

Esamina il delta completo anche oltre il fix V. Un FAIL qualità Pandoc è
risultato ammissibile se misurato e pubblicato correttamente: non richiedere
che il motore diventi perfetto o che vengano eseguiti PDF/OCR/TeXcompile fuori
mandato. Viceversa una misura viziata, perdita nascosta o falso VERIFIED/PASS
è un difetto del benchmark e può bloccare. Disponibilità L04output resta una
lacuna autorizzata, non deroga alla misura strutturale delle formule prevista.

Per ogni rilievo: posizione precisa, input ammesso e raggiungibilità, impatto,
requisito e attribuzione (nuovo codice, legacy raggiunto, mandato o dipendenza),
evidenza, correzione minima/prova di chiusura. Difetto raro o sintetico non è
innocuo; difetto legacy può bloccare se impedisce requisito esplicito raggiunto.
Non inventare rilievi, non trasformare preferenze reversibili in nuovi gate.

## Autonomia e limiti del reviewer

Consentiti letture/ispezioni del codice e dei contenuti, hash/link/snapshot/
aritmetica, diagnostica statica/stdlib e scrittura dei tuoi output. Non eseguire
implicitamente il codice importandolo per ispezionarlo. Se serve una riproduzione
mirata del prodotto, usa solo runtime locale managed verificato e gate effettivi
installazione/S/B/I/diecihash/inventario/preflight/R dello stesso interprete,
input fissi e output nella tua directory. Un vecchio preflight whole-worktree
non è automaticamente valido dopo i documenti finali. Puoi preparare i tuoi
snapshot tecnici/manifest/preflight senza cambiare input comuni; non riattribuire
una receipt storica al nuovo target. Rispetta timeout≤900s/caps R, rete0 e isolamento.
Se la prova richiede build/reinstallazioni o prerequisiti assenti, registra il
limite preciso e la prova necessaria per chiudere il rilievo; non fingere PASS.
Non ripetere intere campagne o suite già attribuite senza rischio/delta pertinente.

Niente implementazione/fix, mutazioni di sorgenti/fixture/gold/packet/receipt,
rigenerazioni corpus, build/install/download, import ML, servizi/remoto, benchmark
motori nuovi, pulizia o Git di scrittura. Output nuovi solo nei percorsi propri.
Non modificare STATE/HANDOVER/eventi/indici/changelog, checkpoint supervisore,
file dell’altro reviewer o snapshot comune. Non avviare agenti/delegati: questa
è una delle due nuove chat indipendenti richieste. Consultazioni di versioni/
licenze/capacità nuove, se realmente necessarie, solo fonti primarie aggiornate
con URL/data; puoi restare sulle versioni locali effettive senza nuove ricerche.

## Report e consegna

Usa il template review. Dichiara autore/provider/modello effettivo e chat ID
se disponibile, senza inventare informazioni non esposte; prompt origine,
giro iniziale completo, snapshot/hash/impronta e verify prima/dopo, branch/
HEAD/base/indice/delta, indipendenza e input effettivamente letti.
Esito **GO oppure NO_GO** sull’implementazione identificata, motivato dai
criteri controllati, prove e limiti; stato invalido fattuale se oggetto stale.
Non approvare prove mai eseguite e non chiamare la review una ricertificazione
universale. Distingui bloccanti, suggerimenti e lacune già autorizzate.

Checkpoint proprio con input/output/esito, processi propri pendenti e prossimo
ruolo **supervisore per ricezione di entrambi i report e arbitrato reale**.
Concludi in chat con esito reale e percorsi. Non avviare fix o chiusura Git.
