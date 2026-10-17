# ADR 0005 — Documentazione e governance locali

> **Partially superseded by [ADR 0010](0010-multi-model-agent-governance.md) (2026-10-17):** the agent governance part (`.agents/workflows/`, instructions for a single tool) is replaced. The split between `documentation/`, `docs/` and `temp/` remains valid until the documentation tranche revises it.

- Data: 2026-10-02
- Stato decisione: accettata
- Stato implementazione: struttura iniziale presente; guide del nuovo prodotto da scrivere
- Origine: richiesta esplicita dell'utente

## Decisione

Conservare proposte, ADR, roadmap, sintesi e report selezionati in `documentation/`;
le evidenze operative restano in `temp/` secondo la precisazione del 2026-10-15.
Organizzare in
`docs/` la documentazione operativa e di sviluppo secondo i quattro bisogni di
Diátaxis: apprendere con tutorial, svolgere un compito con guide pratiche, consultare
una reference, comprendere con spiegazioni. Il pubblico utente/sviluppatore è un
attributo della pagina, non un quinto tipo di documento.

Usare `AGENTS.md` alla radice per le istruzioni comuni, `.agents/skills/` per procedure
riutilizzabili scoperte da Codex e `.agents/workflows/` per sequenze di lavoro locali
richiamate esplicitamente. Quest'ultima cartella è una convenzione del repository,
non un motore di automazione. La futura CI avrà workflow eseguibili separati.

Mantenere il README come punto d'ingresso, migrandolo con le funzionalità effettive.
Una documentazione pianificata deve dichiararlo e non esporre comandi inesistenti.
Gli MCP sono strumenti di sviluppo facoltativi, valutati separatamente dal runtime.

## Conseguenze e riferimenti

Si preserva la storia delle motivazioni senza mescolarla ai percorsi d'uso. La
manutenzione richiede collegamenti e aggiornamenti contestuali alle modifiche.

### Precisazione sulla conservazione — 2026-10-15

Disposizione esplicita dell'utente dopo l'accumulo dei bundle a003: la formula
iniziale «evidenze in documentation» non autorizza a versionare tutte le prove.
La memoria permanente deve spiegare progressi, decisioni, risultati e limiti;
stdout/stderr, receipt, matrici generate, snapshot, copie raw/asset e strumenti
temporanei rimangono nelle rispettive run in `temp/`, esclusa da Git. Le prove
ancora necessarie si conservano localmente; dopo validazione e verifica delle
dipendenze possono essere eliminate senza copiarle tutte in Git.

Implementazione della regola: protocollo, template, AGENTS e guardie `.gitignore`
aggiornati nel working tree. Dopo l’arbitrato r007, spostati in temp gli
8973 file operativi a003 non tracciati (721.430.863 byte), con hash identici e
mappa dei percorsi storici. In documentation/runs/a003 resta la sintesi.
Nessuna prova cancellata, rimozione degli archivi già tracciati o riscrittura
Git; lo spostamento non riduce lo spazio complessivo del workspace. Il [protocollo](../development/run-lifecycle.md#archivio-e-pulizia)
definisce selezione, disponibilità delle prove e transizione.

- [Diátaxis](https://diataxis.fr/).
- [AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
- [Skill locali Codex](https://learn.chatgpt.com/docs/build-skills).
- [Indice documentazione](../README.md), [struttura operativa](../../docs/README.md).
