# ADR0008 — Parser mantenuto e benchmark provvisorio

- Data: 2026-10-12
- Stato decisione: **accettata** come direzione e condizione della consegna ridotta
- Stato implementazione: prodotto ancora da realizzare; priorità approvata al percorso utilizzabile fasi2–8. Trace r005 consegnato, non revisionato; validazione avanzata e debito a002 differiti, nessun GO tecnico retroattivo.
- Origine: accettazione esplicita dell’utente, run a002, dopo14 revisioni autore
- Sostituisce: nessun ADR; integra ADR0001 e applica un’eccezione locale aADR0007

## Disposizione vigente — prodotto prima della validazione avanzata, 2026-10-16

L'utente approva di procedere verso il convertitore funzionante e le funzioni
della roadmap, inclusa Web UI, accettandone l'immaturità e l'assenza di
certificazione automatica completa di contenuti/asset. La risoluzione del
parser/trace e dei debiti del valutatore viene differita e potrà essere
distribuita nel tempo dopo che il prodotto sarà utilizzabile. Nessun nuovo
ciclo automatico di review/fix del trace r005 è richiesto ora.

Questa disposizione **sostituisce i vincoli temporali** di ripresa obbligatoria
all'uscita fase3 e di chiusura del parser prima di qualsiasi consegna/rilascio
sperimentale. Le disposizioni originarie sotto restano storia, non il gate
vigente. Non cancella D-A002-01–05, non promuove i NO_GO precedenti e non
trasferisce il GO del nucleo r002 ai ruoli o agli adattatori. R005 resta una
consegna autoriale non revisionata; nessun fork sperimentale è adottato nel prodotto.

Si realizza la roadmap **per incrementi utilizzabili**, iniziando da un percorso
reale PDF→Markdown con asset, conservazione del sorgente/output originale,
CLI e Web UI sullo stesso servizio. Il primo incremento include le fondazioni
necessarie, non l'intera architettura futura prima di una conversione visibile.
Le funzioni successive di formati, revisione, distribuzione ed esportazione
restano obiettivi autorizzati: dipendenze, hardware e prove ne determinano la
sequenza. Derivati opzionali restano opzionali; nessun servizio a pagamento,
invio remoto o deploy pubblico diventa implicitamente autorizzato.

L'assenza di certificazione semantica non elimina i controlli operativi:
errori dei motori visibili, file e asset consegnati identificati, riferimenti
locali mancanti segnalati, scritture confinate, esecuzioni distinte e download
coerenti. Un job terminato non attesta fedeltà. Qualità non verificata distinta
da integrità/completezza del bundle; incompletezza tecnica nota non nascosta
come conversione riuscita. Anteprima di contenuto importato inerte o confinata,
nessuna esecuzione di HTML/asset non fidati autorizzata dal solo output del motore.

Sono ammessi output e confronti manuali con limiti dichiarati, senza nuove
normalizzazioni distruttive, gold riscritti o etichette COMPLETE/UNSAFE del
misuratore sperimentale come barriera di prodotto. Il valore si verifica su
sorgente/Markdown/asset realmente consultabili, non sui conteggi dei test.
Il debito viene riesaminato quando una funzione richiede attestazioni avanzate,
o a una verifica di roadmap successiva al prodotto utilizzabile; nessuna
scadenza artificiale al termine fase3. Una promessa di certificazione/fedeltà
automatica richiederà ancora le prove pertinenti.

Vantaggi attesi: uso concreto, feedback sui documenti e minore dipendenza dal
verificatore. Costi accettati: verifica manuale, errori del motore non sempre
rilevati e possibili adattamenti futuri dei contratti. Per limitarli conservare
raw, asset, versioni/configurazioni e revisioni; separare il validatore opzionale
dai servizi di conversione. Nuova preparazione prodotto in temp/run-a004-fase2-
prodotto-utilizzabile; piani/review del codice applicativo restano necessari,
Git manuale invariato. Stato decisione approvato, prodotto non ancora realizzato.

### Debito riconosciuto e pulizia autorizzata — precisazione utente16ott2026

Il lavoro non conclusivo su reader/comparatore, parser/trace e validazione è
**debito tecnico riconosciuto da consolidare appena avremo modo, tempo e risorse**.
La priorità al prodotto rinvia il lavoro e ne rende gestibile la programmazione;
non è abbandono né risoluzione degli ID. Il [registro trasversale](../development/technical-debt.md)
conserva stato, responsabilità e criteri di ripresa, insieme al registro a002.

L'utente autorizza anche la pulizia selettiva dei materiali temp non più necessari.
Rimossi ambienti/build Haskell ricostruibili dopo verifica delle dipendenze:
8.777.867.264 byte allocati liberati; sorgenti, patch, report, raw/gold, risultati,
binari finali e modelli/ambienti dei motori mantenuti. I vecchi snapshot non sono
una promessa di ambiente ancora presente. Disponibilità nella sintesi a003;
nessuna modifica dei risultati per ottenere MATCH e nessuna riscrittura Git.

## Contesto

Il reader del benchmark ha progressivamente ricostruito sintassi Markdown,
metadata, code e HTML con decisioni non sempre coerenti. Fix locali hanno
chiuso casi reali ma lasciato classi non convalidate. R014 Claude NO_GO statico,
review OpenAI incompleta per costo amministrativo: nessun arbitrato tecnico
finale positivo. L’utente accetta la consegna sperimentale e richiede che il
problema venga ripreso e risolto, dando priorità a un prodotto valutabile.

## Decisione

Adottare un parser mantenuto per la sintassi del profilo Markdown, con versione,
estensioni e configurazione riproducibili; adattare l’AST al modello osservato
invece di continuare a espandere la grammatica propria. Pandoc, già locale
3.1.3 e disponibile via JSON AST nelle prove a002, è il candidato iniziale
scelto per la verifica dell’adattatore. Nessuna nuova dipendenza o installazione
è attuata qui; l’idoneità a coordinate/provenienza e contenuti va dimostrata.
Se l’AST non offre dati sufficienti, il contratto deve dichiarare UNAVAILABLE
per quelle misure o valutare un parser mantenuto con source-map adeguata;
non inventare coordinate né riprodurre l’intera grammatica per colmare il gap.

Separare parsing sintattico, misure di fedeltà e controlli di confinamento/
risorse. L’AST del convertitore non è un oracolo indipendente della bontà
della sua conversione: gold, sorgenti e valutazione umana restano necessari.
Nessun COMPLETE del reader sperimentale autorizza rendering attivo o accessi.

Chiudere a002 con **accettazione utente del perimetro ridotto**, senza GO
tecnico retroattivo. Le doppie review restano per le implementazioni future.
Riaprire obbligatoriamente il debito dopo la prima applicazione valutabile
(uscita fase3), risolverlo prima degli automatismi di attestazione e del
rilascio. Non blocca il confronto esplorativo fase1 svolto con limiti e
revisione umana; non autorizza saltare le dipendenze della roadmap.

## Alternative considerate

- Continuare fix locali del reader: risultati parziali provati, costo dei giri
  e assunzioni condivise non accettabili come strategia principale.
- Eliminare ogni verifica: impedirebbe confronti motivati e diagnosi di perdite.
- Usare solo AST del motore come gold: confonderebbe sintassi con correttezza.
- Parser mantenuto più riferimenti indipendenti: direzione accettata; source-map,
  profilo e prestazioni da convalidare in un intervento distinto.

## Conseguenze e verifica

Conservare raw, asset, riferimenti, revisioni ed errori. Classificare separatamente
difetti riprodotti, tracce statiche e ipotesi AST. Chiudere il debito solo con
prove su originali/varianti di liste,div,fence,metadata,formule,HTML e risorse;
source-map verificata dove dichiarata, nessuna risorsa silenziosamente omessa,
controlli positivi di code/valori inerti e negativi di HTML reale. Rerun sul
corpus e sui test invalidati, classificando ogni delta rispetto ai dati storici;
due nuove review indipendenti e arbitrato sull’adattatore. Nessuna garanzia
di perfezione su Markdown arbitrario e nessuna sanatoria dei vecchi FAIL.

I budget degli esperimenti devono essere dimensionati al lavoro previsto;
la qualità non si giudica dal consumo CPU. Evitare lookup ripetuti degli
inventari: indicizzare una volta per path/hash, distinguere prove nuove da
ricevute, non rieseguire tutta la storia per ogni sonda. Limiti reali di
risorse e isolamento restano espliciti, nessuna quota aggiunta qui.

## Riesame circoscritto in a003 — 2026-10-15

Dopo sette consegne e due NO_GO r007, l'arbitrato conferma la ricorrenza del
meccanismo descritto nel contesto: stime sintattiche elevate a prove, ruoli
dipendenti dall'annotazione e regole di blocco duplicate. I controesempi ricevuti
si chiudono ma altre interazioni delle stesse classi restano ammesse. Le 23
proiezioni e i 17 assessment del campione restano corroborati; non attestano
correttezza generale del qualificatore.

Si anticipa una **verifica di fattibilità del metodo del comparatore a003**,
prima di altro codice integrato. È l'applicazione circoscritta della direzione
già accettata, non la selezione di un parser o la chiusura del debito a002.
Occorre provare coordinate sul raw, profilo, ruoli indipendenti dalle annotazioni
e copertura necessaria, con controprove e positivi indipendenti; poi piano
modificato, due review e arbitrato. Il prototipo precede build e suite complete.
Nessun ampliamento automatico alle fasi 2/3 o accettazione ridotta di a003;
il termine obbligatorio del debito a002 resta invariato.

Il candidato iniziale rimane Pandoc; le sue capacità effettive di source-map
non sono state convalidate qui. Mancanza di coordinate significa misura
esplicitamente indisponibile o candidato inadatto, non coordinate inventate
né un GO ottenuto rendendo indisponibile l'intero confronto. Il metodo e la
copertura essenziale devono essere valutati prima dell'integrazione.
[Stato a003](../runs/run-a003-fase1-engine-comparison/manifest.md).

## Ripresa anticipata a002 — disposizione utente 2026-10-15

L'utente autorizza ora di anticipare il lavoro anche per il debito a002.
Il riesame a003 comprende la fattibilità del nucleo condiviso e il piano di
risoluzione D-A002-01–05: adattatore del comparatore a003 e adattatore del modello
osservato a002, con prove e criteri di chiusura distinti. La selezione deve
verificare fin dall'inizio le esigenze decisive di entrambi; struttura, profilo
e coordinate possono essere condivisi. Il prototipo precede l'integrazione.

Questa disposizione supera il solo perimetro E01 del riesame precedente e
anticipa il trigger obbligatorio di ripresa a fine fase3. Non occorre attendere
quel termine. Il debito rimane aperto fino a implementazione, nuove prove,
due review indipendenti e arbitrato esplicito per ID; nessun GO retroattivo.
La chiusura storica ridotta a002 rimane valida. OCR, compilazione, hardware e
altri limiti sperimentali non risolti dalla sintassi restano separati;
nessun avvio anticipato del prodotto delle fasi2/3.
[Registro del debito](../benchmarks/a002-residual-debt.md).

## Ricezione della fattibilità r002 — 2026-10-15

Il piano r002 dichiara NOT_VIABLE per a003 e a002 nei tre profili/interfacce
effettivamente provati. Non è una dimostrazione di impossibilità generale.
Il percorso proposto di provenienza nel parser Pandoc e le altre alternative
sono ancora ipotesi da dimostrare; nessun fork o nuovo runtime è stato scelto.
La supervisione prepara due review del metodo e delle prove per valutare
diagnosi, requisiti necessari e prossimo esperimento minimo. Implementazione
dei due adattatori e chiusure del debito restano future.

## Arbitrato della fattibilità r002 — 2026-10-16

Diagnosi negativa circoscritta accolta; il prossimo esperimento richiede errata
locale e criteri di scarto prima di build. I requisiti di coordinate vanno
motivati per trasformazione e consumatore: ruoli/valori possono bastare per
alcune verifiche, altre richiedono intervalli e copertura dei residui. Il
testimone strutturale è un'alternativa da dimostrare per a003, non una chiusura
a002. Risorse nelle celle eccedenti omesse dall'AST rendono decisivo il vincolo
di copertura a002. Nessun fork, migrazione di dialetto o runtime scelto;
direzione mantenuta e chiusure separate restano valide.

## Esperimento di provenienza interna — 2026-10-16

Verificate e accolte le errata E01–E06: il testimone strutturale e il semplice
wrapper dell'AST non soddisfano il contratto pieno. I controesempi perdono
spelling, ID/note o risorse prima dell'interfaccia; non è un'impossibilità
generale di adottare un parser mantenuto. Non si torna a scanner propri o
a un altro emettitore dei soli valori.

Si autorizza un esperimento locale di strumentazione del reader Pandoc3.1.3:
acquisizione/toolchain/build isolati in temp, dati di provenienza registrati
nei rami originali, equivalenza AST fra base e variante costruite dagli stessi
input e verifica dei consumi/residui su un piccolo nucleo prima delle campagne.
L'autore della prova risponde del delta sperimentale; sostenibilità e strategia
upstream restano da dimostrare. La grammatica originale deve rimanere identica.

**Non è adozione permanente di un fork.** Upstream è mantenuto, il delta locale
non lo diventa per questo. Su dati persi/backtracking o mutamento semantico il
metodo va corretto o scartato con prova, senza un secondo parser di recupero.
Il GO riguarda solo l'esperimento; prodotto/due adattatori non implementati,
nuove review del codice e arbitrato necessari prima di integrare o chiudere
gli ID. [Disponibilità e stato](../runs/run-a003-fase1-engine-comparison/manifest.md).

## Ricezione del prototipo trace — 2026-10-16

L'esperimento è stato realizzato:7file di delta locale, AST invariata sui34casi
del nucleo e381raw estesi, origini/rollback e visita dei residui provati nel
perimetro dichiarato. Sono prove autoriali ricevute, con due review del nucleo
da svolgere prima della decisione. Non dimostrano ruoli completi, nuovi stati
audit, fedeltà o equivalenza universale del reader.

Il codice aggiunge anche una visita osservativa con il parser originale delle
celle scartate: il suo isolamento rispetto allo stato semantico è parte del
riesame. La manutenzione del delta rimane locale e la copertura non qualificata
esplicita. Nessuna adozione permanente, estensione del profilo o integrazione
dei due adattatori è approvata dalla sola fattibilità positiva.

Le due review successive sono state arbitrate NO_GO sul nucleo r001: trim
non equivalente al reader originale e due controlli mancanti del verificatore.
Disposto fix locale con nuovi gate attribuiti, senza cambiare la decisione
architetturale o adottare il delta permanente. Ruoli futuri e adattatori non
sono nuovi obblighi del fix. Esiti e seguito nella sintesi della run.

## GO del nucleo e seguito sui ruoli — 2026-10-16

Dopo i fix, due review indipendenti e arbitrato approvano il nucleo r002,
senza adottare permanentemente il delta locale. Un limite basso è esplicito:
alcune fasi restano attestazioni del produttore, non prove semantiche autonome
del verificatore senza attesi. Non richiede un nuovo fix del solo nucleo.

Si autorizza la prosecuzione sperimentale sui ruoli essenziali del campione
a003 e dei casi a002, mantenendo grammatica/AST e nuovi output in temp.
Prima code C01 e risorse HTML/blocchi, poi le altre famiglie pertinenti:
copertura e sufficienza per consumatore precedono gli adattatori integrati.
Nessun GO su contenuto scientifico, COMPLETE/UNSAFE o chiusura del debito.
Il nuovo codice richiederà due review e arbitrato; sostenibilità/adozione
permanente del delta restano da valutare. [Sintesi](../runs/run-a003-fase1-engine-comparison/manifest.md).

## Stato della prova r004 — 2026-10-16

Il completamento sperimentale introduce anche due modifiche osservazionali nella
TagSoup pinned, previste dal mandato locale. Il grafo cambia per quella dipendenza;
non si tratta di tutte dipendenze byte-identiche. Origine/licenza/delta ed equivalenza
finita dei token sono identificati nelle prove in temp. Non è adozione permanente
né modifica approvata della grammatica. Due review mirate devono giudicare il delta,
le chiusure HTML/composizione/selezione e la sufficienza per i due consumatori.
Proposte esterne a003 non adottate; debito a002 ancora aperto.

Le review r004 hanno poi dimostrato che anche un cursore interno può inglobare
sintassi consumata senza token; l'equivalenza dei valori non prova le estensioni.
Arbitrato NO_GO e fix r005: osservazione del termine effettivo nella sorgente
mantenuta, prova indipendente delle estensioni e origine esclusiva dei nodi
post-composizione. Strumentazione osservativa ammessa anche nei punti degli
stati se necessaria, senza cambiare scelte/consumo/valori o adottare il delta
permanente. Nessuna nuova grammatica o integrazione; sufficienza a002 non confermata.

## Collegamenti

- [Fedeltà](0001-document-fidelity.md), [run supervisionate](0007-supervised-development-runs.md).
- [Chiusura a002](../runs/run-a002-fase1-corpus/manifest.md).
- [Registro del debito e criteri](../benchmarks/a002-residual-debt.md).
- [Roadmap](../roadmap.md), [prossimo incremento](../development/next-run-a003-engine-comparison.md).
- [Pandoc: reader→AST→writer e JSON](https://pandoc.org/filters.html), consultato2026-10-12:
  documenta capacità dell’interfaccia, non prova equivalenza completa di r014.
