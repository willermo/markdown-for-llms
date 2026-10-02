# Strumenti locali per lo sviluppo assistito

Questa cartella è versionata nel repository. Non installa skill globali, non modifica
la configurazione personale e non avvia servizi.

## Skill

| Skill | Quando usarla |
| --- | --- |
| [record-architecture-decision](skills/record-architecture-decision/SKILL.md) | Formalizzare o aggiornare una decisione, mantenendo ADR e roadmap coerenti |
| [verify-conversion-fidelity](skills/verify-conversion-fidelity/SKILL.md) | Valutare modifiche a parser, OCR, cleaning, formule, asset o serializer |
| [write-diataxis-docs](skills/write-diataxis-docs/SKILL.md) | Scrivere documentazione utente/sviluppatore nella categoria Diátaxis appropriata |

Le skill hanno il formato `SKILL.md` con metadati e istruzioni; Codex supporta
la scoperta da `.agents/skills/` nel repository, come descritto nella
[documentazione ufficiale](https://learn.chatgpt.com/docs/build-skills).
Possono essere richiamate esplicitamente per nome o selezionate dal client quando
pertinenti. Non sono necessari script o dipendenze MCP per queste tre procedure.

## Workflow

I [workflow](workflows/README.md) sono procedure Markdown del progetto richiamate
dalle skill o dalle istruzioni di lavoro. Non sono hook, job GitHub Actions o comandi
speciali di Codex. La CI eseguibile sarà introdotta con il package nella fase 2.

## Manutenzione

Mantenere nelle skill soltanto le procedure specifiche e rimandare ad ADR e workflow
per i dettagli condivisi. Quando cambiano, verificare frontmatter, collegamenti e
istruzioni su una richiesta concreta; la validazione del formato non dimostra da
sola che il client le abbia caricate o che la procedura sia efficace.

Esempi di impiego: «Registra la scelta del frontend» attiva la skill ADR;
«Verifica che il nuovo OCR preservi gli indici matematici» attiva la verifica di
fedeltà; «Scrivi la guida per ripristinare un backup» attiva la skill Diátaxis.

Riferimenti comuni: [AGENTS.md](../AGENTS.md),
[base progettuale](../documentation/README.md).
