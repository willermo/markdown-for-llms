# Review indipendente implementazione a002 r004 — Claude

Agisci esclusivamente come **reviewer Claude** in una **NUOVA chat**,
distinta da autore/supervisore e dalle review precedenti. Registra il provider
effettivo ed eventuali sostituzioni senza impersonare quello previsto.
Non leggere la review concorrente **r004**, le sue evidenze/checkpoint o
l'eventuale arbitrato r004. Non avviare agenti o review delegate.

Output esclusivamente tuoi, sotto `temp/run-a002-fase1-corpus/`:

- Report: `reviews/review-implementation-r004-claude.md`.
- Evidenze: `evidence/review-implementation-r004-claude/`.
- Checkpoint: `handovers/review-implementation-r004-claude.md`.
- Nuovi ID: `CLA-R004-I001`, progressivi, distinti dai giri precedenti.
- Prompt origine: `prompts/17-review-implementation-r004-claude.md`.
- Per eventuali prove proprie, nuovi snapshot tecnici **solo** con label
  `review-r004-claude-*`, sotto `snapshots/`, creati con il manager stdlib.
  Sono output del reviewer; non modificare o rigenerare alcuno snapshot ricevuto.

## Oggetto comune e identità

Run `run-a002-fase1-corpus`, **review dei fix dell'implementazione r004**,
stato IMPLEMENTATION_REVIEW / FIX_REVIEW_PROMPTS_READY. Ricevuta la consegna
completa, F17–F21 completati secondo autore: chiusure e regressioni da valutare.
Nessun GO/review/arbitrato r004 preesistente. Non è una nuova review del piano.

- Snapshot: `temp/run-a002-fase1-corpus/snapshots/implementation-r004.json`.
- SHA256 snapshot: `7105ec0e91e0783a26e13623ecf9eff37211291323253e39e7dc815e92d768c7`.
- Impronta worktree: `3976b26937e2f80ee03fb089e7e18a4fd8782edd96c0f821536c367a65064edf`.
- 313 file di worktree e 41997 artefatti espliciti;20PDF del corpus
  inclusi nell'impronta generale, corpus/gold/raw e V13 preservati.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`, indice vuoto alla ricezione.
- Piano r001 SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`;
  errata SHA256 `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.
- Mandato: piano/errata/GO piano, prompt05/06/09/12 e addendum V, poi arbitrato
  `arbitrations/arbitration-implementation-r003.md` e prompt15/F17–F21.
- Report autore: `reports/implementation-r004.md`; nel checkpoint
  `handovers/implementation-r004.md` prevale la **sezione finale Consegna
  completa r004**, parti iniziali conservate come storia.

Percorsi relativi della run sotto `temp/run-a002-fase1-corpus/`. Delta autore
**8 file**: observations.py/metrics.py/PROTOCOL, nuovo test r004, due guide,
nuovi report/dati permanenti. Audit/safety/adapter/R/resources/PATH/writer/
config/lock/pin e sorgenti corpus invariati. Quattro documenti generali
aggiunti dal supervisore sono distinti in transition.json; nessun fix qui.
Il lavoro è locale, con file nuovi: `git diff base...HEAD` non basta.

Temp e documentation/runs esclusi dall'impronta generale; input pertinenti
vincolati come artefatti espliciti. Inventario
`evidence/supervisor-implementation-review-r004/review-inputs.json` per
navigare, non obbligo di leggere tutti i log. Include byte/prove ricevuti,
report/CP autore, arbitrato/mandato, precedenti review r003 e nuove prove.
Registri vivi/output nuovi dei reviewer, prompt16/17 e checkpoint corrente
supervisore sono esclusi; non alterano il comune. L'esclusione non permette
di ignorare codice o prove pertinenti né di mutare gli input ricevuti.

## Letture e verifiche iniziali

1. AGENTS.md, temp/HANDOVER, parte iniziale PROJECT-CONTEXT, STATE/HANDOVER/
   brief e questo prompt; checkpoint `handovers/supervisor-implementation-review-r004.md`.
2. Skill manage-implementation-run e verify-conversion-fidelity, protocollo
   run-lifecycle, template review, arbitrato r003 e prompt15 **integrali**.
3. **Entrambi i report r003 sono ammessi**: reviews/review-implementation-r003-chatgpt.md
   e reviews/review-implementation-r003-claude.md, checkpoint e sonde citate;
   sono input precedenti, non report concorrenti r004. R002/r001 solo se pertinenti.
4. Report e CP finale autore r004, final-checks001, PROTOCOL, due guide benchmark;
   `documentation/benchmarks/2026-10-09-a002-r004-profili-html-e-bound.md`
   e relativo `-data.json`, receipt di ricezione e transizione del supervisore.
5. Intero delta di codice/test e dipendenze raggiunte: HTML tree/inline/raw
   coverage/projection/offset, audit e safety, residual scorer/caption,
   alignment/domain/availability/conditional counts/global bounds/decomp,
   test r004 e regressioni pertinenti. Lettura senza import implicito prodotto.
6. Requisiti pertinenti piano/errata/ADR0001/0004/roadmap fase1/workflow
   benchmark/guida uv; non ripetere analisi integrale del piano o campagne a001.

Verifica branch/HEAD/dev/merge-base/indice/delta, hash snapshot/piano/errata.
Esegui **prima e dopo** la review, registrando argv/cwd/output/exit:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r004
```

È manager stdlib del contesto, non test prodotto. Non correggere input o
rigenerare il comune. STALE o identità diversa invalidano la review del
precedente oggetto: conserva il fatto, segnala il prossimo passo senza GO.
Vecchi contesti/stage legano prove storiche e non devono tutti risultare MATCH.

## Prove consegnate e limiti

Supervisore: 39648 confronti hash,38018 input protetti invariati,
160 input consumati con copie prima/dopo,42 copie ricevute r003,48 helper/test
s007→s008 equivalenti e10origini installate corrispondenti ai sorgenti.
Delta8 e receipt/composizione verificati staticamente; nessuna nuova build/
suite/R/conversione/rescore/V del supervisore. Non è GO semantico dei fix.

Alla ricezione 206 input esterni stage-s008 coincidono,0differenze.
I tre pyc managed encodings cambiati nella review precedente sono conservati
come byte ricevuti e vincolati a nuovi manifest/gate, nessun ripristino; causa
storica non attribuita. Sono distinti dai tre Chrome storici non consumati.
Diagnosi statica managed non equivale a collaudo universale dell'interprete.
Dopo ricezione quattro documenti supervisore cambiano il worktree generale:
non trasferire il preflight integrale s008 a prove nuove. Inventaria input
realmente consumati e registra nuove differenze, senza riscrivere vecchie receipt.

Prove sotto `evidence/implementation-r004/` e `work/implementation-r004/`:

- entry-verify/entry-git/received-bytes/protected-inputs, managed-static-input-audit,
  entry-package-inputs, managed-startup-guard-host e snapshot managed-entry001:
  byte prima del primo managed, nuovi gate e provenienza reale.
- Prima GPT before-gpt/probe-summary-v2.json e before-claude-summary.json;
  dopo GPT work/implementation-r004/after-gpt/probe-summary-v2.json e
  after-claude-summary.json, raw/osservazioni/audit/score/receipt effettivi.
  GPT gold PROVISIONAL, Claude K01/V reale (LT sintetico); gate automatici,
  qualità e R distinti. Exit run/evaluate delle sonde derivati dalle regole,
  non nuove conversioni end-to-end; contratti H01–H08 hanno exit operativi reali.
- R-r004-tests-005:475mirati dichiarati PASS (118nuovi+357pertinenti),16F04
  deselected; R-r004-tests-fast-001:632PASS+310subtest in27,76s. Suite
  prescritta tests/ dopo preflight/gate/R, ignore API/packaging/Docker,
  smoke radice esclusi. Non collaudo completo, nessuna collection riparata scaricando.
- contracts-001/summary.json:10contratti conformi; H01–H08 R PASS,
  H12± R FAIL deliberati/harness1/wrapper2, expected0 fissato prima.
  final-checks001:20receipt finali18PASS+2FAIL, binding prima/dopo/reap/socket,
  own_pending0. Non trasformare FAIL atteso nel contratto in R PASS.
- stage-s008/preparation/S/B/I/benchmark-inputs e canonical reinstall:
  dieci hash/origini, managed3.12.13, helper esterni alla wheel. Equivalenza
  s007→s008 dei48input consumati giustifica i workload s007; s008 cambia
  due report/dati, nuovi gate e suite finale. Verifica le dipendenze, non il nome stage.
- completion-check001:10casi42completamenti char/token con matrice autonoma;
  derived-check001:64record, legami/somme/denominatori/global distance,
  zero differenze nei campi storici numerici/gate comparati. Controlla
  indipendenza/attesi/limiti del checker; le prove finite non sono la garanzia matematica.
- rescore-manifest001/rescore001:64nuovi ID r004a, due set32 ciascuno4PASS/
  28FAIL, engine_runs0, raw/source/gold/V e identità originali preservati.
  Conversione baseline003 è r002 (2026-10-08UTC), misura r004 è derivata.
- Fallimenti prepare-s001 relativo, collectionSyntaxError,446PASS29FAILrootHTML,
  474PASS1FAIL malformed img, precedenti R/gate FAIL preservati; nessuna
  sanatoria per il numero finale di test. Lega prove passate ai byte effettivi.

## Criteri tecnici identici dei due reviewer

Review mirata di F17–F21, intero delta8, dipendenze e regressioni. F16 è
chiuso sulle64misure precedenti, ora da preservare con decomp v2; altri fix
r002 chiusi non si riaprono senza difetto raggiungibile nel delta/percorso.
Non basta contare test o ripetere solo le stringhe degli ultimi report.

| Criterio | Verifiche sostanziali |
| --- | --- |
| C01 F17 inline/blocco | html-inline-v1 distingue strutture inline e blocchi; table/figure in cella/caption/paragrafo/wrapper conserva foglie, raw, coordinate, math/ref/unknown oppure unsupported visibile. I1/I2 non PASS spurio; nessun testo svuotato o WER completo0. Positivi span/a/enfasi/MathML/code e normali blocchi validi. Copertura condivisa dei produttori/trasporto, senza doppio conteggio. |
| C02 F18 residui | html-table-v1/figure-v1 rende conto di ogni ramo, inclusi testo/math/asset fuori td/th/caption, caption duplicate, figure con altro contenuto, annidamenti, spanning e malformed. Residui effettivamente consumati da audit/scorer, non solo raw conservato. Asset mancante INCOMPLETE/2; URI evasiva respinta prima; aggiunta nota FAIL oppure unsupported conservativo mai PASS. Positivi H0/F0 e risorse presenti discriminano il rifiuto di tutto. |
| C03 F19 caption | Atteso None/vuoto/testo contro assente/vuoto/equivalente/aggiunto/differente/math/unknown. Equivalenza absent-empty-text documentata e motivata; aggiunta nota/omissione/differenza rilevate. Formula/ref/unknown tipizzati e scope separati dalla prosa, senza truthiness che salta il dominio. |
| C04 F20 garanzia | Misura completa, conditional_text_counts/decompv2 e text_lower_bound sono distinti in codice/stati/guide/dati. Known words e L1 non presentano omissione+inserzione come bound certo. Politica globale: dominio e normalizzazione per tipi, flusso char con LF, token concatenati, denominatori interi, corrispondenze libere; su flusso misurabile distanza minima corretta, su unknown/limite0 conservativo con motivo/versione. Nessun bound supera la distanza dei completamenti ammessi. Zero garantito è scelta esplicita consentita, non completezza0 o obbligo di bound stretto; C1 conserva3/30 **condizionato**/FAIL, non va richiesto3come bound globale su flusso incerto. |
| C05 regressioni misura | Unknown non wildcard globale: page/block loss e FAIL noti, ordine, aggiunte certe, numeri/celle/math/asset/ref e denominatori preservati. CER/WER completi None dove non misurabili; bound/conditional scopes coerenti con typed leaves e literal/code. Positivi completi0 e aggiunta interamente nota3/2 restano discriminanti. Matching/gold non adattati al risultato. Decomp categorie/ID/offset e somme verificabili, diagnosi strutturale distinta da perdita materiale. |
| C06 F21 e confini | `<` letterale seguito da spazio/cifra fedele con audit COMPLETE; veri tag/commenti/dichiarazioni/duplicati/malformed/raw tail non spariscono. Risorse in fallback continuano a essere rilevate, code inerte. Sicurezza prima degli import/scritture, niente fetch/comandi da dati; non rivendicare nuovo bypass URI se il caso è già respinto. |
| C07 64 prove/pubblicazione | Identità/raw/hash/source/gold/V, osservazioni nuove→audit→score→final e versioni corrette, engine_runs0;4PASS/28FAIL per set verificati da file effettivi. DOCX11/182caption, EPUB8/186, RTF19/162condizionato vs13/162bound, TXT0/89, TeXNone/condizionato1/152/bound0. Indipendenza dell'oracle, composizione F16, gate non sommati, report r004 rettifica claim storici senza sovrascriverli. V13 non firma output motori; L04 UNAVAILABLE distinta. |
| C08 toolchain/prove | Input before/after/copie protette, delta8, nuovi S/B/I/origini/diecihash e managed-entry attuali, interprete R uguale ai figli, equivalenza48s007→s008 reale. Receipt18PASS+2FAIL voluti, preflight/binding/reap/socket. Nessun PASS da stage stale o byte diversi; helper/test fuori wheel vincolati. Suite/test/contratti selezionati pertinenti, limiti legacy/API/Docker/ML preservati. |
| C09 confini/consegna | Piano/mandato e quota52h/entry/attese invariati; ledger closing storico non saldo attuale, CPU nota lower/RSS sampled/retained conservativo non hard cap universale. Fallimenti, helpergap12, ChromeNONPASS e L01/L02/L04/L05 non sanati. Guide/report/checkpoint usabili e ruoli/Git manuali preservati, nessuna fase2 o motore vincitore scelto. |

Un dettaglio reversibile non enumerato è ammissibile se chiude il requisito.
Un sintetico esprime un difetto raggiungibile anche se manca nei64raw: rarità
non è deroga. Unsupported dichiarato è ammissibile nel subset finito se non
cancella foglie, nasconde risorse o passa i gate per assenza di contenuto.
Non inventare rilievi né richiedere collaudo universale; collega requisito,
input ammesso, percorso, impatto e attribuzione (nuovo/ereditato/mandato).

## Autonomia e limiti delle prove proprie

Consentiti letture statiche, hash/link/aritmetica, manager stdlib, propri
report/evidenze/checkpoint. Piccole sonde sintetiche sul percorso reale
reader/audit/scorer sono consentite, se servono a decidere un rilievo, sotto
R corrente e gate validi, su copie proprie, con attesi fissati prima. Nessuna
nuova V/firma umana agente, motore/render/browser, suite generale, build,
reinstallazione/download/modello/remoto, benchmark a pagamento o Git di scrittura.
Non correggere prodotto, test/corpus/gold canonici o input delle altre chat.

Per nuove sonde registra startup/package input e origine managed3.12.13,
dieci hash e archivi/installazione effettiva; nuovi controlli S/B/I/preflight
sulle dipendenze consumate. Puoi creare manifest/launcher/target propri e
**nuovi snapshot tecnici** con label reviewer sopra, vincolando manifest e
helper prima del workload e output vivi esclusi. È autorizzato in questo
mandato anche se l'helper risolve snapshot nella directory comune: non
modificare i vecchi/common e non chiedere freeze intermedio al supervisore.
Nessun trasferimento dei PASS s008 o esecuzione con gate rossi. In presenza
di differenze conserva il FAIL, diagnostica e prepara nuovi input reali;
non ripristinare pyc per mascherare il delta o inventarne la causa.

R/no-network/same interpreter, shell=False/registry/confini preservati;
R ≤900s per comando/1MiB per stream/32MiB per file, sonde piccole e motivate.
Per EPERM/EACCES diagnosi ed escalation tool della stessa operazione
autorizzata se necessaria, mantenendo R; fermati al rifiuto effettivo, senza
aggirarlo. Statico e runtime distinti: se la prova fallisce prima del workload,
non attribuirle il risultato dedotto. Prove author lette non diventano proprie.

Quota unica implementativa52h dall'entry2026-10-08T07:06:54.369068Z,
attese incluse, CPU21600s/6GiB/RAM8GiB/stop sampled6GiB; stop nuovi
49h/19800CPU-s/5GiB, riserva3h/1800CPU-s/1GiB. Stop nuovi workload
2026-10-10T08:06:54.369068Z, deadline11:06:54.369068Z. Nessun reset/incremento.
Ledger autore chiusura120881,053504s wall (~33,578h), CPU7524,451826s lower,
retained1479827278B/RSSsampled735604736B sono storici. Registra costi propri
e tempi reali, non inventare upper/certificazioni hardware. Nessun accesso
di rete/acquisizione/document-send/costo esterno implicito; consulta fonti
primarie attuali solo se necessario per valutare nuove versioni/capacità.

## Report ed esito

Usa il template review. Dichiara provider/modello/chatID effettivi se esposti,
prompt/revisione/snapshot/hash/impronta/base/delta/indipendenza, verify prima/
dopo, criteri effettivamente verificati, prove/argv/input e limiti. **GO**
oppure **NO_GO** reale sul comune identificato; GO non certifica altri motori
o L04. Se l'oggetto è invalido, dichiara impedimento fattuale e prossimo passo.
Per ciascun rilievo: ID, severità/blocco, posizione, input/percorso/requisito/
attribuzione, impatto, evidenza statica/runtime, fix minimo e prova di chiusura.
Valuta esplicitamente F17–F21 e residui F01/F05/F09/F13–F16, non un voto medio.
Nessun GO per assenza del concorrente o numero di test, nessun fix reviewer.

Non modificare STATE/HANDOVER/eventi/indici/changelog/checkpoint supervisore,
snapshot ricevuti o file del concorrente. Nel checkpoint: input/output/esito,
costi/processi propri pendenti e prossimo ruolo **supervisore per ricezione
dei due report r004 e arbitrato**. Concludi in chat con esito e percorsi.
