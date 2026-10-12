# Archivio — run-a002-fase1-corpus

- Obiettivo/fase: corpus e strumenti di misura, primo incremento fase1.
- Esito: **CLOSED / ACCEPTED_REDUCED_SCOPE_WITH_DEBT**, accettazione utente12ott2026.
- Review finale: ClaudeNO_GO statico,OpenAIINCOMPLETA; nessun GOtecnico retroattivo.
- Branch/base: feature/run-a002-corpus da dev80c31da46e5caab990d3596c79c44e524ec1dc95.
- Commit applicativo e integrazione manuale in `dev`: **verificati** a `7caff336119b6d7d78978204142cfdc16d620fff`; feature e dev identiche alla ricezione.
- Push/deploy/main: non eseguiti; origin/dev locale ancora a `80c31da46e5caab990d3596c79c44e524ec1dc95`, nessuna query remota.
- Registri di integrazione e24log storici: predisposti per un successivo commit documentale, non ancora committati.
- Archiviazione:2026-10-12,supervisore Codex/OpenAI, hoststdlib/hash solamente.

[Sintesi della chiusura](closure.md),
[decisione originale conservata](reports/arbitrations/decision-close-reduced-scope-r001.md),
[disposizione utente](reports/notes/user-accepted-reduced-scope-2026-10-12.md),
[debito obbligatorio](../../benchmarks/a002-residual-debt.md),
[ADR0008](../../decisions/0008-maintained-parser-and-provisional-benchmark.md),
[prossimo incremento](../../development/next-run-a003-engine-comparison.md).

## Materiale conservato e istruzioni di lettura

- Piani,prompt,review,arbitrati,report,checkpoint e note storiche: `reports/`.
- Originali Markdown byte-identici:`evidence/original-texts/`; le copie di
  lettura inreports hanno link adattati. I report originali non sono riscritti.
- Snapshots originali byte-identici:`evidence/snapshots/`; restano impronte
  storiche riferite al worktree originale, non snapshot attuali né backup codice.
- Evidenze decisive r014/baseline/AST/mutazioni/receipt/runtime/delta e costi:
  `evidence/`; nessuno script archiviato eseguito, codice attuale resta nel repo.
- 64bundle raw/asset e misure ricevute:`evidence/derived-r014/`; nessuna nuova
  conversione, misura oV. [Indice di valutazione umana](human-evaluation.md).
- Provenienza/origine/destinazione/SHA/motivo:[inventario JSON](evidence/archive-inventory.json).

L’inventario contiene ogni originale selezionato e relativa copia; hash ebyte
confrontati. JSON originali conservano percorsi assoluti storici: usare la mappa
origine→archivio, non eseguire automaticamente argv storici. Copie di lettura
con link adattati possono ancora puntare alla temp per evidenze non selezionate:
dipendenze elencate nell’inventario, non rappresentate come archiviate.

## Verifica, limiti e pulizia

Verifiche:byte/hash selezionati e coppie raw/asset preservate, link dei documenti
correnti e indice umano, whitespace eGitdiffcheck; ledger e receipt chiusura
registrano esiti/costi reali. Test3156+310 e64ricalcoli sono ricevuti dall’autore,
non eseguiti qui. H12FAIL/NO_GO/reviewincompleta/STALE/NONPASS conservati.
Vecchio comune revisionato diventa STALE amministrativo per chiusura/ADR/guide:
non restaurato né rigenerato. Snapshot di chiusura attribuisce byte,nonGO.

**Nessuna pulizia eseguita.** Tutta la temp a002, cache/venv/input-store e prove
non selezionate restano disponibili perché hanno dipendenze storiche. Non
eliminarli finché eventuale esportazione restante è verificata. Archivio
selettivo reale, non promessa di portabilità completa di ogni runtime storico.
Corpus/gold restano nel repo tests/fixtures/conversion-corpus;versioni/byte
runtime sono identificati, non migliorati dalla chiusura.

Controllo aggiuntivo: 2886 hash originali/copie di lettura PASS; 11 collegamenti storici già indisponibili, elencati in [receipt](evidence/archive-integrity-check.json). Non sanati o trasformati in prove.

Contesto vivo alla chiusura: [STATE](evidence/context-at-closure/STATE.md) e [HANDOVER](evidence/context-at-closure/HANDOVER.md). Le copie in reports/context sono la fotografia precedente alla finalizzazione; gli originali sono conservati.

## Ricezione Git2026-10-12

[Ricevuta](evidence/integration-reception-2026-10-12.json):352filegenerali e2artefatti
identici alla consegna,soloHEAD/branch diversi. Workingtree pulito alla ricezione;
24logarchiviati erano ignorati e assenti nel commit. RegolaGitignore corretta
per il solo archivio,byte immutati; completamento con commitdocumentale.
Nessun parser cambiato/test nuovo/GOtecnico. Commit amministrativi successivi
possono avanzaredev senza cambiare il commit applicativo identificato qui.
