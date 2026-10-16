# Debito tecnico riconosciuto — validazione differita

Aggiornato16ott2026. **APERTO, riconosciuto, da consolidare appena tempo e
risorse lo consentiranno dopo un prodotto utilizzabile.** Disposizione esplicita
utente: il rinvio permette di sviluppare e usare il convertitore e Web UI con
limiti dichiarati; non chiude i difetti o converte vecchi NO_GO in GO.
[Decisione vigente ADR0008](../decisions/0008-maintained-parser-and-provisional-benchmark.md).

| Area | Stato e conseguenza | Consolidamento richiesto |
| --- | --- | --- |
| A002: D-A002-01–05 | Aperti; reader/valutatore sperimentale, nessuna nuova certificazione COMPLETE/UNSAFE | Criteri e chiusura per ID nel [registro a002](../benchmarks/a002-residual-debt.md), implementazione/prove e review pertinenti |
| A003: comparatore/proiezioni | R007 NO_GO; misure del campione parzialmente corroborate, qualificatore generale non convalidato | Rendere affidabili le misure dichiarate, con sorgenti/gold indipendenti e perimetro esplicito; niente uso come certificazione universale |
| Parser/trace e adattatori | R005 consegnato/non revisionato, nucleo r002 GO circoscritto; nessun adattatore di prodotto completo | Riesaminare soluzione e costo quando ripreso, ricostruire ambienti eliminati ove necessario, nuove prove/review sui byte effettivi |
| Fedeltà di contenuti/asset | Nessuna certificazione automatica completa; errori dei motori possono sfuggire ai controlli operativi | Verifiche di contenuto e copertura delle proprietà effettivamente promesse; valutazione umana e regressioni pertinenti |

Le fasi2/3 sono sviluppo applicativo ancora da realizzare, non implementazioni
fallite da dichiarare chiuse. I tentativi non conclusivi riguardano comparatore,
reader e strumenti di validazione della fase1. Le funzionalità operative avranno
propri criteri e review, senza dipendere dal consolidamento completo sopra.

Responsabile: supervisore del progetto per programmazione e revisione del debito,
autori/reviewer per l'esecuzione. Riesaminare il registro al primo prodotto
utilizzabile e quando una funzione richiede una proprietà non ancora provata.
L'utente consente di dilazionare il lavoro; ciò non equivale a cancellazione o
abbandono. Nessun automatismo può attestare una proprietà dipendente dal debito
senza prove adeguate. Job terminato, integrità del bundle e qualità non verificata
restano distinti; path/errori/file presenti e conservazione dei dati rimangono
controlli di prodotto necessari.

La pulizia selettiva rimuove ambienti/build ricostruibili, non il registro o gli
esiti dei tentativi. La [sintesi a003](../runs/run-a003-fase1-engine-comparison/manifest.md)
dichiara disponibilità e limiti: hash storici non rendono presenti prove eliminate.
Il rinvio non deve diventare una promessa di riproducibilità integrale.
