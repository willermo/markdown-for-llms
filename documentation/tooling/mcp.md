# Strumenti di sviluppo e valutazione MCP

Valutazione iniziale: 2026-10-02. **Nessun nuovo MCP installato o attivato da questo
intervento.** Gli strumenti descritti aiutano lo sviluppo; non sono dipendenze
necessarie al funzionamento del convertitore.

## Raccomandazioni

| Strumento | Utilità concreta | Decisione iniziale |
| --- | --- | --- |
| Playwright MCP | Esplorare UI locale, riprodurre problemi e verificare upload, coda, anteprime e revisione nel browser | Candidato per la fase 3; affiancare test Playwright ripetibili in CI |
| OpenAI Docs MCP | Consultare documentazione ufficiale quando si implementano integrazioni OpenAI o configurazioni Codex | Facoltativo; attivare quando utile, con sola consultazione documentale |
| Integrazione GitHub | Consultare issue, diff e PR nel repository | Riutilizzare il connettore disponibile o la CLI; evitare una seconda integrazione equivalente |
| MCP filesystem o SQLite generici | Esplorare file e DB | Non necessari ora: shell e strumenti locali coprono il bisogno con un perimetro più chiaro |
| Aggregatori di documentazione | Consultare molte librerie da un solo strumento | Riesaminare se le fonti ufficiali diventano insufficienti; nessuna dipendenza iniziale |

[Playwright MCP](https://github.com/microsoft/playwright-mcp) espone strumenti di
automazione del browser. La nostra raccomandazione è usarlo su un'istanza di sviluppo
con dati sintetici e profilo isolato; non sostituisce i test di regressione versionati.

Il [server documentale OpenAI](https://developers.openai.com/learn/docs-mcp) fornisce
accesso alla documentazione. È distinto dalle API dei modelli: configurarlo non
configura credenziali o provider AI del prodotto.

## Configurazione locale al progetto

Codex supporta server MCP nel file di progetto `.codex/config.toml` per repository
considerati attendibili. Riferimento: [configurazione MCP](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
Se ne introdurremo uno, registrare qui scopo, origine, versione quando applicabile,
strumenti abilitati, credenziali necessarie e modo di rimuoverlo.

Esempio **inattivo**, da integrare nella configurazione del progetto solo quando si
decide di usare questo server; non modifica configurazioni globali:

```toml
[mcp_servers.openaiDeveloperDocs]
url = "https://developers.openai.com/mcp"
enabled = false
```

Per server avviati come pacchetti locali, scegliere una versione verificata e
riproducibile quando vengono introdotti. Non inserire comandi con aggiornamenti
impliciti a ogni avvio nella configurazione condivisa. Non versionare token o chiavi.

Il `.env` del prodotto resta dedicato alla configurazione dell'applicazione; gli
strumenti dell'agente hanno un ciclo di vita separato. Un MCP per esporre in futuro
il convertitore ad altri agenti sarebbe una funzionalità del prodotto distinta,
da valutare dopo API e contratti stabili.
