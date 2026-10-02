# Progettazione del convertitore di documenti

Questa cartella è la memoria progettuale del repository: conserva motivazioni,
decisioni, proposte e avanzamento. La documentazione d'uso e sviluppo vive invece in
[`docs/`](../docs/README.md), organizzata secondo Diátaxis.

**Situazione al 2026-10-02:** impostazione documentale della nuova versione; il codice
eseguibile è ancora quello legacy. Branch di lavoro: `feature/document-converter-v2`,
creato dal `dev` locale al commit `8fa5be9`. Nessun merge è stato eseguito.

## Percorso di lettura

1. [Draft preliminare della proposta discussa](architecture/preliminary-draft.md).
2. [Registro delle decisioni architetturali](decisions/README.md).
3. [Web UI: obiettivi e percorso utente](architecture/web-ui.md).
4. [Roadmap a fasi e criteri di completamento](roadmap.md).
5. [Questioni aperte](open-questions.md) e [debito tecnico legacy](legacy-findings.md).
6. [Strumenti di sviluppo e valutazione MCP](tooling/mcp.md).

## Come mantenere questa base

Il draft conserva la proposta iniziale, riorganizzata per la consultazione. Gli ADR
(Architecture Decision Records) sono la fonte delle scelte correnti: una proposta
non equivale a una decisione accettata, e una decisione accettata non equivale a una
funzionalità implementata. Le scelte iniziali accettate riflettono l'impostazione
concordata e la richiesta esplicita di FastAPI, governance e Diátaxis.

Per un cambiamento architetturale usare il [template ADR](decisions/template.md),
aggiornare il registro e collegare la fase interessata. Conservare gli ADR superati,
indicando quale nuovo ADR li sostituisce. Le evidenze dei futuri benchmark saranno
salvate qui con versioni, corpus, hardware e limiti del confronto.

Le [istruzioni del repository](../AGENTS.md), le
[skill locali](../.agents/README.md) e i
[workflow](../.agents/workflows/README.md) indicano come lavorare su questi materiali.
