# Debito tecnico riconosciuto — validazione differita e residui di prodotto

Aggiornato17ott2026. **APERTO, riconosciuto, da consolidare appena tempo e
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

Le fasi2/3 hanno ora il primo incremento applicativo I1 r004 **GO dopo due
review e arbitrato**, nel worktree e con integrazione manuale pendente.
Non sono completate in tutta la roadmap. I tentativi non conclusivi storici
riguardano comparatore, reader e validazione della fase1; il loro rinvio
non è stato usato per ignorare i blocchi operativi di I1.

Responsabile: supervisore del progetto per programmazione e revisione del debito,
autori/reviewer per l'esecuzione. Riesaminare il registro al primo prodotto
utilizzabile e quando una funzione richiede una proprietà non ancora provata.
L'utente consente di dilazionare il lavoro; ciò non equivale a cancellazione o
abbandono. Nessun automatismo può attestare una proprietà dipendente dal debito
senza prove adeguate. Job terminato, integrità del bundle e qualità non verificata
restano distinti; path/errori/file presenti e conservazione dei dati rimangono
controlli di prodotto necessari.

La [pulizia selettiva del 17 ottobre](cleanup-2026-10-17.md) rimuove anche
payload e copie operative superati; conserva registro, esiti, casi decisivi
e sorgenti del prototipo finale. Non tutte le vecchie prove restano disponibili. La [sintesi a003](../runs/run-a003-fase1-engine-comparison/manifest.md)
dichiara disponibilità e limiti: hash storici non rendono presenti prove eliminate.
Il rinvio non deve diventare una promessa di riproducibilità integrale.

## Residui operativi I1 r004 — bassi, aperti e non bloccanti

Riesame al primo prodotto effettuato il 17 ottobre: il debito avanzato sopra
resta aperto, non viene ripreso automaticamente. I residui sotto provengono
dalla review Claude r004 e sono accolti dall'arbitrato GO; non sono risolti.
[Sintesi della run](../runs/run-a004-fase2-prodotto-utilizzabile/manifest.md).

| ID | Limite e motivazione del rinvio | Criterio di consolidamento |
| --- | --- | --- |
| CLA-R4-I001 | Il test UTF-8 guasto usa inventari incoerenti e passa dal rifiuto del manifest; non copre più il ramo previsto. Una sonda con fixture coerente conferma il comportamento corretto del prodotto | Aggiornare entrambi gli inventari e asserire causa UTF-8 e invalid persistito nel DB |
| CLA-R4-I002 | La diagnosi GET prende brevemente worker.lock in esclusiva: l'avvio concorrente può uscire con Errno11 anche senza altro worker. Errore visibile, nessun dato modificato; ripetere l'avvio funziona. Frequenza dello stress non è stima affidabile per l'uso reale | Diagnostica senza contesa di proprietà o retry iniziale limitato e messaggio distinto; test concorrente senza rifiuti spuri |
| CLA-R4-I003 | Una voce non sicura estranea alla pubblicazione può far attribuire a questa un errore d'integrità e perdere recuperabilità dello stesso ID. Byte conservati. Riprodotto con symlink non attribuito o in downloads; il normale flusso provato riconcilia prima lo scratch del job successivo e non produce quel caso. Non si promette recupero da alterazioni arbitrarie dell'archivio | Errori reserve/capacity su voce estranea differiscono publishing con motivo; rimossa la voce, stesso ID completed. Unsafe nel risultato selezionato resta bloccante |

Responsabile del riesame: supervisore; implementatore e due reviewer per fix
e verifica pertinenti. Riprendere al prossimo intervento sulle aree coinvolte
e prima di ampliare distribuzione/gestione dell'archivio. Se I003 appare nel
normale uso, priorità al fix di disponibilità. Non modificare manualmente
DB/storage o aggiungervi file per recuperare risultati o liberare quota.

Osservazioni di usabilità conservate: diagnosis visibile solo in JSON/CLI,
messaggio generico per limite manifest e sonda worker che attesta soltanto
presenza del proprietario, non il recupero del singolo run. Non diventano
nuovi gate né dichiarazioni di correttezza. Nessun ulteriore ciclo automatico
di fix; P14 utente resta PENDING e qualità unverified.
