# Review implementazione r003 — criteri comuni ChatGPT e Claude

Agisci solo come reviewer assegnato dal wrapper, in **una nuova chat indipendente**.
Repository `/home/davide/workarea/markdown-for-llms`, branch feature/run-a001-uv,
HEAD/dev/base `66ba82200e5def5a4db76f9bafccb0731b506091`, indice vuoto alla
ricezione. Il worktree autorizzato comprende file nuovi: la sola diff fra commit
non identifica il risultato.

## Oggetto

Snapshot comune **impl-r001-stage-final-s003**:
`temp/run-a001-fase0-uv/snapshots/impl-r001-stage-final-s003.json`.
Digest/bytes/worktree in
`evidence/supervisor-implementation-r003/reception-r001/freeze-identity.json`.
I percorsi sotto sono relativi a temp/run-a001-fase0-uv, salvo sorgenti permanenti.

Confronta l'identità e verifica MATCH **prima e dopo** con il managed ricevuto:

```bash
/home/davide/workarea/markdown-for-llms/.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -I -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-final-s003
```

Se STALE, identifica il delta e non dare GO per un oggetto diverso; niente
snapshot sostitutivi o correzioni. Checkpoint implementatore vivo e futuri
output r003 sono esclusi dal freeze; la copia ricevuta del checkpoint è inclusa.
Non leggere report/checkpoint/evidenze r003 concorrenti, non delegare o avviare
agenti, non aggiornare prodotto, registri condivisi, prove o arbitrati.
Le review **r002 di entrambi** sono storia condivisa legittima.

## Input essenziali e perimetro mirato

Leggi AGENTS, skill manage-implementation-run, protocollo/template review e
stato/indici pertinenti. Recupera:

- Arbitrato implementazione-r002, review r002, prompt59, R021/R022; piano/
  arbitrato piano r003 per D2/R/documentazione, senza concatenare i prompt storici.
- Report-r020, copia checkpoint ricevuto e reception/delta-s002-to-received/
  supervisor-documentation-delta della ricezione corrente.
- `evidence/implementation-r001/fix-review-r003/`: calls, execute.py,
  R-a001/b002 con S/I/target/preflight/wrapper/runner/inside/process-collection,
  monitor e invocation; clone b002/snapshot reale e inventario; static/reuse/
  matrix/delivery, manifest e manifest-closing, entry/closing-accounting,
  finalize.py/finalize-failure/close_cost.py. Non confondere i manifest prima
  delle riscritture con il riepilogo di chiusura successivo.
- Cinque path del fix: `scripts/diagnostics/run-a001-fase0-uv/run_offline.py`,
  `tests/unit/test_core_diagnostics.py`, `docs/how-to/ambiente-uv.md`,
  `docs/how-to/marker-legacy-docker.md`, `docs/reference/toolchain-legacy.md`.
  Dipendenze effettive del launcher/preflight/manifest/R e prove riusate quando
  necessarie. Delta documentation del supervisore separato e non applicativo.

Rapporto e hash di ricezione non sostituiscono il tuo giudizio. Il source a001
s076 era MATCH prima dei delta documentali del supervisore; ora è storico.
Non attribuire prove di s076 al nuovo snapshot finale senza spiegare quel delta.

## Chiusure e regressioni da decidere

| ID | Criterio comune |
| --- | --- |
| GPT-I002 / residuo CLA-I001 | Launcher ufficiale non vincolato ad a001: schema int1, run_id/label validi, path assoluto confinato, symlink file/antenati respinti, verify reale, binding prima/dopo conservato. Due R reali a001/b002 usano il launcher permanente corretto, target installato/preflight effettivo e stesso managed nei figli, isolamento Firejail/D4, exit/postcheck e raccolta propri. Il solo S/I o un mock non dimostra il positivo R. Controlla la copia del tool in b002 e il confinamento del clone/Git alternates readonly, senza confondere Git sintetico con scritture al repository. |
| GPT-I001, regressione | Tre cleanup negativi ripetuti e validazione stantio prima del socket/workload con receipt/errore originale; cleanup corretto nei due R reali. Nessuna rimozione di guardie o falso PASS. |
| GPT-I003 / CLA-I007 / residuo CLA-I004 | Guida senza flag verify_distribution inesistente; scope I derivato da S. RUN_IMAGE_CONTEXT_RECEIPT nella reference, variabili e opzioni coerenti col codice, link/whitespace validi. Non riaprire le74+9 disposizioni già approvate per una frase se non emerge una regressione concreta. |
| CLA-I008 | Alternativa **documentale** scelta dall'arbitrato: C-docker richiede RUN_TEST_MODE nell'ambiente, il solo flag pytest non seleziona il driver; scelte discordanti rifiutate. Tre guide coerenti. Non richiedere supporto CLI aggiuntivo né nuovo Docker per questa nota; il limite va dichiarato senza promettere equivalenza universale del flag. |
| CLA-I006 e nuova chiusura | Ledger storico corretto almeno8321,418589/overrun41,418589, upper ignoto/NON PASS, originali immutati. Nuovo ingresso prudenziale8380 +60s letture, cap8880; carica8947,997655 =osservato8946,997655 +1s margine. Formula, script/riscritture, mtime e reale confine dei workload: ultimi R8671,266940. Quota prospettica NON PASS, finalize FAIL conservato; cerca attività ulteriori/escluse e motiva impatto sul giudizio. Nessuna sanatoria o prova per cancellare un costo passato. |

Dodici test ricevuti sono stdlib/mock (5core,3cleanup,4nuovi con sottocasi),
non una suite applicativa sotto R. La vecchia C-fast137+300 resta al suo harness;
i nuovi test non sono dichiarati eseguiti dentro quella suite. Valuta se queste
prove mirate coprono il delta effettivo autorizzato, senza chiedere tutte le
campagne per il solo cambiamento del file dei test.

Riusi: B56 conserva gli input build/payload/installazione; Docker r007 e
C-docker conservano contesto/producer/driver/probe/test/conftest esatti.
Host/V1/standalone/CLI/fedeltà ai propri stage, vecchi R storici; nuove R
attestano il launcher. Controlla identità/equivalenze delle dipendenze toccate,
non assegnare nuovi PASS ai vecchi stage. Baseline62/5/perdite e FAIL restano
caratterizzazione storica; nuova architettura/inferenza/GPU/Windows/V10/V11
fuori scope. A7/arbitrato ancora pendente non è autonomamente un difetto.

## Costi propri e consegna

R022: letture/diff/hash/AST, per dubbio concreto sonde stdlib/mock proprie
**<=30s cumulativi**, timeout finiti, output **<=1MiB** nel pool352MiB vigente,
radici/esclusioni/riserva invariate. Rimisura spazio per i tuoi output; la
ricezione/freeze non sono nuove quote. Nessun prodotto/suite/R reale,
installer/build/Docker/rete/ML o costo trasferito dal core esaurito. Un mock
non è nuovo PASS R/V1. Se una nuova prova prodotto è essenziale, registra
motivo/limite e continua le letture disponibili, senza inventare un risultato.

Report e checkpoint secondo template nei percorsi del wrapper: provider/modello/
chat effettivi e indipendenza, snapshot/digest/MATCH prima-dopo, letture, prove
eseguite/non eseguite, costi e limiti. Tabella delle chiusure sopra con risolto/
ancora aperto, motivo/prova e blocco. Nuovi rilievi: ID progressivi, posizione,
input/raggiungibilità, impatto/requisito, attribuzione base/run, evidenza e
criterio di risoluzione. Suggerimenti estranei in backlog, blocchi validi non
ignorati. **GO o NO_GO** motivato per questo oggetto; nessuna integrazione o
GO dell'altra chat inventati. Consegna al supervisore per l'arbitrato r003.
