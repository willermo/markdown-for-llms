# Review indipendente fix implementazione a002 r003 — Claude

Agisci esclusivamente come **reviewer Claude** in una **NUOVA chat**,
separata da autore/supervisore e dalle review precedenti. Registra provider
reale ed eventuali sostituzioni senza impersonare quello previsto.
Non leggere la review concorrente **r003** né l’eventuale arbitrato r003.

Output esclusivamente tuoi, sotto `temp/run-a002-fase1-corpus/`:

- Report: `reviews/review-implementation-r003-claude.md`.
- Evidenze: `evidence/review-implementation-r003-claude/`.
- Checkpoint: `handovers/review-implementation-r003-claude.md`.
- Nuovi ID rilievi: `CLA-R003-I001`, progressivi e distinti dai giri precedenti.
- Prompt origine: `prompts/14-review-implementation-r003-claude.md`.

## Oggetto comune e identità

Run `run-a002-fase1-corpus`, **review dei fix dell’implementazione r003**.
Stato IMPLEMENTATION_REVIEW / FIX_REVIEW_PROMPTS_READY. Consegna completa
ricevuta; F13–F16 dichiarati corretti dall’autore, chiusure e regressioni da
valutare. Nessun GO o arbitrato r003. È review del codice e delle prove,
non una nuova review del piano.

- Snapshot: `temp/run-a002-fase1-corpus/snapshots/implementation-r003.json`.
- SHA256 snapshot: `6dc951642d3108b6b5ed31cc36677ed6c1187b849812474fe74e1d7644f6d6e8`.
- Impronta worktree: `9eb2033e07e62ba509be8a795308f1dc0c2d362caaf79745eff9d4249251ddd1`.
- 310 file di worktree e 37588 artefatti espliciti;
  tutti i 20 PDF del corpus sono nell’impronta generale.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base:
  `80c31da46e5caab990d3596c79c44e524ec1dc95`. Indice vuoto alla ricezione.
- Piano r001 SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`;
  errata SHA256 `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.
- Mandato piano/errata/GO del piano e prompt05/06/09, poi arbitrato
  `arbitrations/arbitration-implementation-r002.md` e prompt12.
- Report `reports/implementation-r003.md`; nel checkpoint
  `handovers/implementation-r003.md` prevale la **sezione finale “Consegna
  completa r003”**. Le sezioni iniziali “in corso” sono storia.

Percorsi relativi della run sotto `temp/run-a002-fase1-corpus/`.
Il delta effettivo dell’autore è di **nove file**: observations/audit/metrics/
PROTOCOL, `tests/unit/test_conversion_fix_r003.py`, due guide benchmark e
nuovi report/dati permanenti. È registrato in
`evidence/implementation-r003/final-checks-002.json`; la ricezione indipendente
statica del supervisore conferma il delta. Sono aggiunti poi quattro registri
documentali del supervisore, distinti nella transition.json del ruolo.
Le modifiche sono locali e includono file nuovi: `git diff base...HEAD` non basta.

Temp e documentation/runs sono esclusi dall’impronta generale; input pertinenti
sono vincolati direttamente come artefatti espliciti. Inventario
`evidence/supervisor-implementation-review-r003/review-inputs.json` per navigare,
non obbligo di leggere integralmente tutti i log. Include prove/byte preservati,
input delle review r002, arbitrato, report/CP finale autore, prove r003 e dieci
origini installate stage010. Ambienti/cache terzi esclusi salvo origini esplicite.
Nuovi output reviewer r003, registri vivi, checkpoint supervisore e prompt13/14
esclusi. Il CP autore finale è fisso. Non nascondere codice o prove nelle esclusioni.

## Letture e verifica prima/dopo

Usa e dichiara le skill `.agents/skills/manage-implementation-run/SKILL.md` e
`.agents/skills/verify-conversion-fidelity/SKILL.md`. Leggi nell’ordine:

1. AGENTS.md, temp/HANDOVER e sezione iniziale corrente PROJECT-CONTEXT,
   STATE/HANDOVER/brief run, questo prompt e
   `handovers/supervisor-implementation-review-r003.md`.
2. Protocollo run-lifecycle, template review e skill. Arbitrato implementazione
   r002 e `prompts/12-implementation-r003-fix-html-and-text-metrics.md` integrali.
3. **Entrambi i report r002 sono ammessi**: reviews/review-implementation-r002-chatgpt.md e reviews/review-implementation-r002-claude.md, relativi checkpoint,
   sonde/evidenze necessarie alle chiusure. Non sono report concorrenti r003.
   R001/arbitrato r001 solo per requisiti o regressioni effettivamente pertinenti.
4. Report/CP finale autore r003, report e dati
   `documentation/benchmarks/2026-10-09-a002-r003-html-e-metriche.md` e
   `documentation/benchmarks/2026-10-09-a002-r003-html-e-metriche-data.json`,
   delta/final-checks002, PROTOCOL e due guide benchmark correnti.
5. Intero delta di codice/test e dipendenze raggiunte: produttori HTML/inline,
   trasporto/offset/fence/prefissi, fallback e audit, scorer/domain/availability/
   lower-bound/decomposizione, test r003 e regressioni precedenti. Leggi il
   codice necessario senza importarlo implicitamente per una mera ispezione.
6. Prove effettive sotto e requisiti piano/errata/ADR0001/0004/roadmapfase1/
   workflow benchmark/guide uv pertinenti. Non ripetere analisi integrale del
   piano o storia/campagne a001. A001 e pulizia sono già ricevute.

Verifica branch/HEAD/dev/merge-base/indice/delta, hash snapshot/piano/errata.
Esegui **prima e dopo**:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r003
```

Registra argv/cwd/output/exit nelle tue evidenze. È stdlib del contesto,
non prova del prodotto Python. Non rigenerare snapshot né correggere input.
STALE/identità diversa invalidano la review sull’oggetto ricevuto: conserva
il risultato, segnala il fatto/prossimo passo, non dare GO sullo stato precedente.
Snapshot r002/fixcontext e stage preliminari attribuiscono prove precedenti,
non devono tutti essere MATCH dopo le modifiche autorizzate.

## Ricezione statica e prove da valutare

Il supervisore ha controllato 27532 confronti hash, 26141 input protetti
invariati, 136 input consumati con copie prima/dopo, 39 copie ricevute r002,
identità/delta, risultati di sonde e conteggi nelle receipt. Nessuna nuova
build/test/R/conversione/V del supervisore. Non è GO semantico sui fix.

Alla ricezione i **206 input esterni stage-s010 coincidono**. Lo stage usa le
versioni esterne correnti: i tre Chrome mutati dopo s013 sono storia nota,
non sono stati ripristinati né eseguiti. Chrome renderer NONPASS resta tale.
Dopo la ricezione quattro registri documentali del supervisore cambiano
l’impronta generale: non trasferire automaticamente il preflight integrale
s010 a prove nuove. Usa target/inventario/preflight sulle dipendenze realmente
consumate e registra le eventuali differenze esterne osservate adesso.

Prove sotto `evidence/implementation-r003/`:

- `entry-verify.json`, `received-bytes.json` (39), `protected-inputs.json`
  (26141), manifest/input/hash di confronto r002 preservati. Le sonde originali
  dei reviewer sono protette e non adattate per eliminare il fenomeno.
- Prima: `before-gpt/`, `before-claude-summary.json`; dopo finale:
  `final-gpt/probe-summary-003.json`, `final-claude-summary.json` e relativi
  raw/observations/audit/metriche/receipt. R PASS è esecuzione della sonda,
  non qualità dei payload. GPT usa gold PROVISIONAL, Claude V derivata reale;
  L04 resta UNAVAILABLE, nessuna nuova firma umana.
- `duplicate-before-001/audit.json`: src duplicato dava COMPLETE spurio se
  primo target assente e ultimo presente. Tre regressioni finali; rifiuto
  esplicito degli attributi duplicati nel parser, non modifica del validator input.
- `R-r003-tests-007`:373 mirati dichiarati PASS (247 precedenti+126 nuovi);
  `R-r003-tests-fast-002`:514 test e310 subtest PASS, suite prescritta dopo
  gate, API/packaging/Docker e smoke radice esclusi; collaudo parziale.
  Confronta attesi autonomi e contenuti, non solo contatori. I fallimenti
  309PASS/29FAIL e versioni338/360/370, fast511/310 storici sono conservati.
- `contracts-002/summary.json`:10 contratti H01–H08 e H12 negativo/positivo.
  H01–H08 R PASS; H12 entrambi R FAIL deliberati con harness1/wrapper2,
  expected-exit0 fissato prima. Non dieci R PASS. H06 UNSAFE/run2/score3/
  evaluate2. Non rieseguire F03/H15/F04 senza dipendenza modificata.
- `rescore-manifest-002.json`, `rescore-002/summary.json`,
  `derived-check-002.json` e checker autonomo:64 raw, nuovi ID r003b,
  **engine_runs=0**, zero nuovi giudizi umani, due set32 ciascuno4PASS/28FAIL.
  Baseline003 rimane conversione r002 con misure derivate r003, non nuova
  campagna motori. Tutti64 raw senza HTML; otto TeX con unknown o0013
  attribuito. Lo scan accompagna il rescore, non lo sostituisce. Vecchi raw/
  osservazioni/audit/metriche/final/receipt/report/corpus/gold/packet/V immutati.
  Verifica confronto completo e differenze dichiarate di versione/schema,
  source_position, disponibilità/provenance/text_complete e decomposizione.
- WER per tentativo: TXT0/89; DOCX11/182=11 inserzioni caption e0 aligned;
  EPUB8/186=8 omissioni; RTF19/162=11 aligned+3 omissione titolo+5 inserzioni
  (titolo3+caption2); TeX completo UNAVAILABLE, noto1/152 aligned. Diagnostiche
  di ruolo sono candidate testuali esatte e non cambiano il matching.
- `stage-s010/preparation.json`, S/B/I/dist/installation/target/preflight/
  inventory: managed CPython3.12.13, wheel canonica reinstallata, dieci hash,
  origine/startup e stesso interprete dei figli. Final fast su s010; test/
  sonde/rescore/contratti s009: equivalenza dei59 input realmente consumati
  in `stage-s009-to-s010-consumed-equivalence.json`. Stage010 incorpora i
  documenti finali; valuta il riuso per dipendenza, non per nome dello stage.
- `input-bytes-delivery-before.json` / after:136 input con copie fisiche,
  helper/test/config/scripts propri/SBI/target prima/dopo; input fuori wheel
  ispezionabili. `final-checks-002.json`:16 receipt finali,14PASS+2FAIL H12
  attesi, binding/reap/own_pending0. S004 gate FAIL STALE e stage preliminari
  conservati; nessuna receipt verde trasferita a byte nuovi.
- `delivery-budget-final-001.json`:wall94489,40259s (~26,247h) **alla consegna**,
  CPU nota lower-bound7104,999667s, retained con supplementi1204121202B;
  RSS massimo noto campionato735604736B. Non wall/saldo corrente, non upper o
  hard-cap universali. Costi/rete/acquisizioni/invio0; quota52h dalla entry
  2026-10-08T07:06:54.369068Z invariata, attese incluse. Stop nuovi workload
  2026-10-10T08:06:54.369068Z, deadline11:06:54.369068Z; nessun reset.

## Criteri tecnici comuni

| Criterio / azione | Verifica sostanziale |
| --- | --- |
| C01/F13 trasporto tipizzato | Il ponte HTML mantiene tutte le foglie text/math/code/ref/anchor/unknown, raw/AST/segmenti/ordine e localizzazione, in celle/caption/paragrafi/list/quote/fence del subset. Span/a annidati, mixed/duplicati/limiti e fallback non cancellano contenuti. K01 vuota PASS; formula supportata aggiunta FAIL3; MathML senza annotation unknown e gate non PASS; formula attesa equivalente PASS automatico distinto da L04. Contratto end-to-end, non solo html_nodes. |
| C02/F14 risorse e completezza | Ogni risorsa HTML supportata o in contenitore/fallback raggiunge audit oppure rende INCOMPLETE. B1–B4 asset assente run/evaluate2; B5 URI evasiva UNSAFE prima dell’apertura; spy/sentinella effettivi. Positivi locali con asset/hash/caption indipendenti, unsafe distinto da semplice unknown. Duplicati src/href/case/annidamenti/malformed/depth gestiti o rifiutati senza perdita del primo target. Code letterale è inerte; nessun fetch/browser o regex globale universale, confini path/symlink/hardlink conservati. |
| C03/F15 disponibilità e lower-bound | CER/WER completi None/UNAVAILABLE quando il dominio testuale non è interamente misurabile, incluse osservazioni unmatched. ID/offset/scope/ragione attribuiti; denominatori completi, numeratori noti conservativi effettivamente garantiti, non solo etichettati lower-bound. Positivo0 e aggiunta nota3/2; foreign unknown completo indisponibile; C1 noto3/30 e completoNone. Unknown associato/estraneo/mixed/list/inline, code literal e domini locali math/celle/caption separati senza invalidare metriche estranee. FAIL noto/missing/page loss prevalgono; confronto oltre limite non diventa AVAILABLE0. |
| C04/F16 composizione WER | Per64 tentativi categorie/entry/text/type/ID/offset/denominatori sommano i numeratori certi; AVAILABLE completo vs lower-bound coerenti. DOCX caption e RTF titolo declassato distinguibili dagli edit aligned. Candidate esatte diagnostiche non mutano matching/normalizzazioni per ottenere verde. Report/dati spiegano sovrapposizioni dei gate e non attribuiscono automaticamente tutti gli edit a parole materialmente perse/inventate. Checker indipendente legge file reali e ricalcola, non importa scorer/generator o copia totali. |
| C05 intero delta e regressioni | Nove file e dipendenze reali, oltre ai casi originali: foglie tipizzate, prefissi/fence/char/byte/source_position, raw/ref/asset/formule/ordine/header/unknown, alignment one-to-one/duplicati/ambiguità, limiti/exit/batch. Attesi autonomi discriminano rifiuto di tutto e falso verde. Nessuna gold-leak nel reader/audit o compensazione di output sul gold. F01/F05/F09 chiusura effettiva e nessuna riapertura indiscriminata dei fix r002 già chiusi. |
| C06 identità delle prove | Byte nuovi consumati da test/probe/rescore/contratti/suite attribuiti, origine/interprete/wheel/diecihash/SBI/R/preflight/input fuori wheel veri. R009→010 equivalenza59 sufficiente per ogni prova riusata; ultimo fix duplicati esercitato nei workload finali. Errori vecchi preservati, esiti R/qualità/contratti distinti. 14PASS+2FAIL non sanati come16PASS; nessun processo proprio pendente o prova non eseguita dichiarata PASS. |
| C07 misure derivate e input protetti |64 raw e vecchie prove/corpus/V invariati, rescore senza motori e nuove identità/hash. Due set32 separati,4PASS28FAIL risultanti e non imposti. I dati legano raw→observe→audit→score/final e versione effettiva; differenze note e campi quantitativi/stati verificati dai veri file. V13 sui riferimenti distinta da L04 output; nessuna nuova V umana generica. Lacuna helper12/ChromeNONPASS/L01/L02/L05 preservata. |
| C08 risorse e documentazione | Guide/PROTOCOL/report/dati/ledger coerenti con comportamenti e misure, costi e quota originale senza reset. CPU lower-bound/RSS sampled/cache/hardware non certificati oltre prova; no winner/schema app/fase2. Archeologia limitata ai requisiti raggiunti, archivio futuro conserva byte/provenienza necessari senza richiedere ora Git/cleanup. |

Valuta chiusure e regressioni del fix **e tutto il suo delta**. Sintetici/rari
sono validi quando esercitano input ammessi. Un FAIL reale Pandoc rimane
risultato valido; false COMPLETE/VERIFIED/PASS, perdite o misure complete
spurie possono bloccare anche se i64 raw non esercitano il ramo.
Non trasformare ogni scelta reversibile in un nuovo requisito o richiedere
parser universale/nuova V/nuove campagne. Non ignorare un requisito essenziale
scoperto nelle dipendenze modificate. Non inventare rilievi per riempire tabelle.

## Autonomia, report ed esito

Consentiti letture/ispezione statica, hash/link/aritmetica/snapshot, diagnostica
stdlib e propri output. Eventuali riproduzioni mirate richiedono managed locale
3.12.13, origini/installazione/S/B/I/diecihash verificati, nuovi target/inventari/
preflight sulle dipendenze consumate e R dello stesso interprete. Puoi creare
manifest/snapshot propri nella tua directory senza mutare input comuni.
Niente build/reinstall/download, nuovi motori/render o suite/campagne intere
senza delta/rischio pertinente. Prerequisito mancante: registra limite esatto
e prova necessaria, non inventare PASS. Timeout≤900/cap R/isolamento/rete0
conservati. EPERM/EACCES: diagnosi ed escalation tool per azione autorizzata
mantenendo confini; effettivo rifiuto di approvazione richiede stop.

Niente fix o mutazioni sorgente/gold/packet/V/receipt/report altrui, STATE/
HANDOVER/eventi/indici/changelog/CP supervisore o file dell’altro reviewer.
Non rigenerare snapshot/corpus, fare Git di scrittura/cleanup, avviare fase2/
ML/remoto. Non delegare agenti o simulare reviewer: questa è una delle due
NUOVE chat indipendenti. Fonti primarie aggiornate URL/data solo se valuti
nuove versioni/licenze/capacità; puoi restare sulle prove locali.

Usa template review; autore/provider/modello/chat ID effettivi (senza inventare
ID non esposti), prompt/giro r003, oggetto/hash/impronta/base/delta/indice,
indipendenza, verify prima/dopo, letture/comandi/prove/limiti. Esito **GO o
NO_GO** motivato, invalidità fattuale se STALE. Per F13–F16: chiuso/aperto/non
valutato, evidenza/limiti/regressioni; lega anche i residui F01/F05/F09 originali.
Nuovi rilievi con ID propri, severità/blocco, posizione/riga/input ammesso,
raggiungibilità, requisito/attribuzione/impatto, fix minimo e prova di chiusura.
Nessun GO per assenza di report o per prove future non eseguite.

Checkpoint proprio con input/output/esito/processi propri pendenti e prossimo
ruolo **supervisore per ricezione dei due report r003 e arbitrato reale**.
Concludi in chat con esito reale e percorsi. Non avviare implementazione o Git.
