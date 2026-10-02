# ADR 0005 — Documentazione e governance locali

- Data: 2026-10-02
- Stato decisione: accettata
- Stato implementazione: struttura iniziale presente; guide del nuovo prodotto da scrivere
- Origine: richiesta esplicita dell'utente

## Decisione

Conservare proposte, ADR, roadmap ed evidenze in `documentation/`. Organizzare in
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

- [Diátaxis](https://diataxis.fr/).
- [AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
- [Skill locali Codex](https://learn.chatgpt.com/docs/build-skills).
- [Indice documentazione](../README.md), [struttura operativa](../../docs/README.md).
