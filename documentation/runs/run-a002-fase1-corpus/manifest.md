# Archivio — run-a002-fase1-corpus

Nota17ott2026: riferimenti operativi eliminati o già indisponibili sono
resi testo; esiti storici invariati. [Disponibilità delle prove](../../development/cleanup-2026-10-17.md).

## Disponibilità aggiornata — 2026-10-17

La pulizia autorizzata rimuove archivi operativi, duplicati, build e cache
superati; conserva sintesi/esiti, casi decisivi del debito e dipendenze del
prodotto. Le descrizioni di conservazione nelle sezioni storiche sotto si
riferiscono alla data originaria, non alla disponibilità attuale.

[Inventario selettivo e limiti](../../development/cleanup-2026-10-17.md).

**Aggiornamento amministrativo2026-10-12:** commit registri/24log completato
e osservato localmente a `9f702a2d368cb8f4a3a6bfb7b9146c6757ecd9df`, padre7caff33;
delta soltanto documentazione/Gitignore, workingtree pulito all'ingresso a003.
Dev/HEAD/origin-dev locali coincidono,senza query remota/prova di push.
[Completamento manuale](manual-integration.md),
[nuova preparazione a003](../../development/a003-engine-comparison-brief.md).
Stato CLOSED e debito invariati,nessuna review tecnica riaperta. Le diciture
«registri da committare» e i ref precedenti sotto sono la fotografia storica
della ricezione precedente al commit amministrativo.

- Obiettivo/fase: corpus e strumenti di misura, primo incremento fase1.
- Esito: **CLOSED / ACCEPTED_REDUCED_SCOPE_WITH_DEBT**, accettazione utente12ott2026.
- Review finale: ClaudeNO_GO statico,OpenAIINCOMPLETA; nessun GOtecnico retroattivo.
- Branch/base: feature/run-a002-corpus da dev80c31da46e5caab990d3596c79c44e524ec1dc95.
- Commit applicativo e integrazione manuale in `dev`: **verificati** a `7caff336119b6d7d78978204142cfdc16d620fff`; feature e dev identiche alla ricezione.
- Push/deploy/main: non eseguiti; origin/dev locale ancora a `80c31da46e5caab990d3596c79c44e524ec1dc95`, nessuna query remota.
- Registri di integrazione e24log storici: predisposti per un successivo commit documentale, non ancora committati.
- Archiviazione:2026-10-12,supervisore Codex/OpenAI, hoststdlib/hash solamente.

[Sintesi della chiusura](closure.md),
decisione originale conservata (riferimento storico non disponibile),
disposizione utente (riferimento storico non disponibile),
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
- Provenienza/origine/destinazione/SHA/motivo:inventario JSON (riferimento storico non disponibile).

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

Controllo aggiuntivo: 2886 hash originali/copie di lettura PASS; 11 collegamenti storici già indisponibili, elencati in receipt (riferimento storico non disponibile). Non sanati o trasformati in prove.

Contesto vivo alla chiusura: STATE (riferimento storico non disponibile) e HANDOVER (riferimento storico non disponibile). Le copie in reports/context sono la fotografia precedente alla finalizzazione; gli originali sono conservati.

## Ricezione Git2026-10-12

Ricevuta (riferimento storico non disponibile):352filegenerali e2artefatti
identici alla consegna,soloHEAD/branch diversi. Workingtree pulito alla ricezione;
24logarchiviati erano ignorati e assenti nel commit. RegolaGitignore corretta
per il solo archivio,byte immutati; completamento con commitdocumentale.
Nessun parser cambiato/test nuovo/GOtecnico. Commit amministrativi successivi
possono avanzaredev senza cambiare il commit applicativo identificato qui.
