# Registro delle decisioni architetturali

Stati della decisione: **proposta**, **accettata**, **superata**, **ritirata**.
Ogni ADR riporta separatamente lo stato dell'implementazione e l'origine della
decisione. Data iniziale del registro: 2026-10-02.

| ADR | Decisione | Stato | Implementazione |
| --- | --- | --- | --- |
| [0001](0001-document-fidelity.md) | Documento completo, fedeltà e derivati opzionali | Accettata | Da realizzare nella nuova pipeline |
| [0002](0002-fastapi-web-ui.md) | Backend FastAPI, worker e Web UI | Accettata | Da realizzare; frontend da scegliere |
| [0003](0003-persistence.md) | SQLite e storage su filesystem | Accettata per la prima versione locale | Da realizzare |
| [0004](0004-execution-and-configuration.md) | Locale/remoto, profili hardware, configurazione `.env` | Accettata come indirizzo | Motori e profili da verificare |
| [0005](0005-documentation-and-governance.md) | Governance locale e documentazione Diátaxis | Accettata | Struttura iniziale presente |
| [0006](0006-python-toolchain-uv.md) | Toolchain Python uv prima dei benchmark | Accettata | Autorizzata, da pianificare nella run a001 |
| [0007](0007-supervised-development-runs.md) | Run supervisionate, doppie review e handover | Accettata | Protocollo e bootstrap presenti |

Usare [template.md](template.md) per il prossimo numero libero. Collegare alternative,
conseguenze e prove utili; evitare un ADR per ogni dettaglio di implementazione.
Una nuova richiesta esplicita può cambiare una decisione: documentare il cambiamento
senza introdurre un ulteriore passaggio di autorizzazione non necessario.
