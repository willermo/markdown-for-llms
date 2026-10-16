# A003 — Mandato di preparazione del confronto motori

2026-10-12. **PREPARATION**: ricognizione e mandato predisposti dal supervisore;
piano, review, conversioni e decisione sui motori non ancora prodotti. Questo
documento non autorizza workload o acquisizioni. La fase1 resta aperta.

## Base e passaggio Git manuale

Base effettiva: `dev` a `9f702a2d368cb8f4a3a6bfb7b9146c6757ecd9df`, padre
`7caff336119b6d7d78978204142cfdc16d620fff`. Verificati commit amministrativo,
24 log aggiunti e delta limitato a documentazione/Gitignore. Indice e working
tree puliti all'ingresso; `origin/dev` locale coincide con `dev`. Nessuna query
remota: questa coincidenza non attesta un push. A002 resta chiusa e accettata
con debito; nessuna nuova review tecnica a002.

Il branch previsto non esiste ancora. Dopo aver controllato che le modifiche
siano soltanto questa preparazione documentale, l'utente esegue:

```bash
git status --short --branch
git diff --stat
git rev-parse HEAD dev
git switch -c feature/run-a003-engine-comparison dev
git status --short --branch
git rev-parse HEAD dev
```

Le modifiche documentali non committate seguono il nuovo branch. Se la base
cambia o il branch esiste già, conservare il lavoro e far valutare il delta;
nessun reset, stash, checkout forzato o merge da ripetere. Commit, merge, push
e promozione restano operazioni manuali dell'utente.

## Risultato richiesto

Un confronto piccolo e riproducibile con sorgenti, Markdown completi e asset
apribili dall'utente, perdite descritte per documento e fenomeno e proposta
motivata di instradamento per formato. Nessun vincitore predeterminato.
Separare capacità dichiarate, presenza locale e comportamento misurato;
la [ricognizione](a003-local-reconnaissance.md) identifica il punto di partenza.

La pianificazione deve proporre il confronto minimo che risponde alla domanda
di prodotto, anziché una campagna sul reader. Conservare i [limiti a002](../benchmarks/a002-residual-debt.md)
e applicare [ADR0008](../decisions/0008-maintained-parser-and-provisional-benchmark.md):
metriche provvisorie, nessuna garanzia generale COMPLETE/UNSAFE. Non modificare
il parser sperimentale, non implementare ora l'adattatore AST e non aprire fase2.

## Campione iniziale e confronto equo

Riutilizzare i canonici di [a002](../../tests/fixtures/conversion-corpus/README.md).
La selezione proposta comprende sette documenti e otto pagine PDF complessive:

| Caso | Domanda | Confronto proposto, da precisare nel piano |
| --- | --- | --- |
| C01-txt-it | Unicode, numeri e codice letterale | Controllo diretto/legacy; TXT non va interpretato come Markdown |
| C02-docx-it | Formule, celle, immagine, note | Percorso diretto/legacy e Pandoc; ML solo se coperto e utile |
| C03-epub-en | Ordine, riferimenti, asset | Percorso diretto/legacy e Pandoc |
| C06-pdf-text-it | PDF testuale, due pagine | Marker e Docling CPU se realmente disponibili |
| C07-pdf-columns-en | Ordine a colonne, due pagine | Stessi profili dei due motori PDF |
| C08-scan-it | OCR, una pagina | Stessi profili; nessun falso successo da testo vuoto |
| C10-pdf-mixed-it-en | Pagine native/raster e lingue, tre pagine | Stessi profili; copertura esplicita di tutte le pagine |

C04-RTF, C05-TeX e C09-scansione EN rimangono fuori dal campione iniziale,
elencati come non misurati in a003; possono sostituire un caso con motivazione,
senza riscrivere input o trasferirvi prove precedenti. Le inclusioni TeX e la
compilazione non sono requisito nascosto di questo incremento. Il piano fissa
la selezione finale e spiega quali fenomeni non sono coperti.

Pandoc locale elenca DOCX/EPUB/RTF/LaTeX come reader, ma non PDF: non inventare
un confronto Pandoc OCR. `pdftotext` o Tesseract possono costituire baseline
diagnostiche separate, se utili; non equivalgono a un convertitore completo
di formule, tabelle e asset. La loro sola presenza non attesta qualità.

Una conversione per coppia documento/profilo nella prima passata. Ripetizioni
solo per errore ordinario corretto o variabilità osservata, con identità e
risorse distinte. Un pilota su pagina deve avere ID/ambito proprio: non attribuire
la riuscita all'intero documento. Congelare la configurazione prima del confronto;
ottimizzazioni successive diventano profili distinti e dichiarati, senza
selezionare a posteriori soltanto i migliori output.

## Riferimenti, prove e fruibilità

Input, gold, trascrizioni, fonti e render canonici sono protetti. I quattordici
hash input/gold selezionati coincidono con il manifest alla ricognizione;
questa verifica non crea una nuova approvazione umana. Recuperare le ricevute
indipendenti pertinenti tramite l'[indice umano a002](../runs/run-a002-fase1-corpus/human-evaluation.md)
e il manifest dell'archivio. Rendere espliciti provenienza, stato e limiti dei
gold; non generarli dall'output del motore o dal suo AST. Le vecchie V/A03
valgono per gli oggetti identificati, non per nuove conversioni.

Per ogni tentativo conservare run ID, hash sorgente, versione di motore/pesi,
configurazione effettiva senza segreti, hardware, argv/cwd, exit, stdout/stderr,
raw Markdown, asset e rappresentazione nativa se disponibile. Errori e output
parziali restano visibili. Una conversione riuscita può avere fedeltà FAIL.
Separare preparazione, startup, inferenza e valutazione; CPU cumulativa dei
figli senza doppio conteggio, wall come misura, RSS campionata con limite
dichiarato, bytes conservati, cache fredda/calda senza cancellare cache storiche.

Tabella obbligatoria per documento/profilo: testo/omissioni, numeri, formule,
ordine, celle/intestazioni, immagini/didascalie, note/riferimenti, copertura
pagine, risorse e fallimenti. Per ogni fenomeno pertinente riportare evidenza
e stato PASS/FAIL/UNAVAILABLE; NOT_APPLICABLE solo con motivo. Per esperimenti
non eseguiti usare NOT_MEASURED e spiegare perché, anche se esiste un mock o
un vecchio risultato. Nessuna media deve nascondere pagine o formule mancanti.

Il pianificatore definisce soglie per classe, denominatori e trattamento delle
ambiguità prima delle prove. Per i casi sintetici verificabili, perdite di numeri,
formule, pagine o asset richiesti impediscono di raccomandare il profilo come
fedele senza esplicita limitazione. Il reader può fornire diagnostica etichettata
provvisoria; il giudizio determinante deve essere riscontrabile nei contenuti.

La consegna prevista è un indice Markdown locale con pochi collegamenti per
caso: originale/render esistente, gold/trascrizione, raw di ciascun candidato,
asset e scheda delle differenze. Nessun server o frontend necessario. Mostrare
gli output come dati non fidati e mantenere i controlli di file/processi;
non introdurre rendering HTML attivo per agevolare l'ispezione. Valutazioni
umane nuove devono riportare autore, oggetto e osservazioni reali, senza firme
inventate o certificazioni attribuite al supervisore.

## Risorse: stima da disporre prima dei workload

Il ledger a002 di chiusura riporta residuo prudenziale negativo; non finanzia
a003. La preparazione leggibile richiesta può continuare. **Nessuna quota
a003 è disposta da questo documento.** Il piano deve verificare la sostenibilità
e il supervisore ottenere la disposizione operativa prima di preparare/eseguire
ambienti, modelli, conversioni o prove di review con runtime.

Scenario proposto: CPU locale, un solo processo di conversione alla volta,
massimo quattro thread; riuso offline di artefatti già disponibili e identificati.
L'immagine Marker e la cache osservate sono possibilità da verificare, non
prerequisiti già soddisfatti. Docling manca negli ambienti/cache ispezionati.

| Voce proposta | CPU cumulativa stimata |
| --- | --- |
| Preparazione/preflight e verifica degli artefatti locali | 3.600 CPU-s |
| Confronto nativo e controlli diagnostici | 1.800 CPU-s |
| Due candidati PDF, otto pagine ciascuno, pilota e retry pertinenti | 21.600 CPU-s |
| Valutazione, test proporzionati e report | 5.400 CPU-s |
| Quattro review reali piano/implementazione e amministrazione | 7.200 CPU-s |
| Riserva per fix e chiusura | 3.600 CPU-s |
| Totale proposto | 43.200 CPU-s |

Sono stime operative, non prestazioni misurate o promessa di completamento.
Stop ordinario proposto a 39.600 CPU-s; riserva 3.600 accessibile solo tramite
disposizione esplicita. Nessun limite di 10 CPU-s per reviewer; inventari piccoli
indicizzati una volta e verifiche proporzionate. Quote ulteriori richiedono
disposizione prima del consumo, non vengono aggiunte per correggere un errore.

Proposta ulteriore: RAM aggregata 12 GiB con stop campionato a 10 GiB; nuovo
storage cumulativo 12 GiB con stop a 10 GiB, inclusi ambienti/archivi/output/log
e file conservati dei tentativi falliti, senza ricopiare i 2,07 GiB di pesi
storici. Verificare nel piano cgroup effettivo e disponibilità al momento delle
prove: i 64 GB nominali non sono un'autorizzazione. Comando singolo massimo
1.800 s wall per proteggere i processi, senza deadline della run; timeout/log/output
sono adattabili entro il mandato effettivamente disposto. Costo servizi €0,
invio documenti remoto 0, nuove acquisizioni/installazioni attualmente vietate.

Se Docling o un modello indispensabile mancano, il piano propone una tranche
separata con versioni, fonti, licenze di codice/pesi, bytes, espansione e quote
di preparazione espliciti, oppure registra NOT_MEASURED. Nessun download di pesi,
installazione, backend in rete o servizio remoto nasce da questa stima. Le sole
letture pubbliche di documentazione non sono conversioni remote.

## Ruoli, perimetro e uscita

Il prossimo ruolo è **pianificatore in nuova chat**: piano versionato e checkpoint,
senza implementare o eseguire benchmark. Poi il supervisore identifica il piano
comune per due review reali indipendenti ChatGPT/Claude e arbitra. Soltanto dopo
GO del piano, branch verificato e risorse disposte viene emesso il mandato di
implementazione completo. Nessuna delega o review simulata in questa chat.

Gli strumenti necessari al confronto possono essere proposti in perimetro;
riusare quelli standard, non creare un nuovo framework di parser/monitor/installer.
Le directory proposte sono `temp/run-a003-fase1-engine-comparison/` per scratch
ed evidenze; risultati reali pubblicabili in `documentation/benchmarks/` e
archivio selettivo in `documentation/runs/run-a003-fase1-engine-comparison/`.
Non scrivere mock o tabelle vuote come benchmark già eseguiti.

Alla consegna completa: report riproducibile, indice per l'utente, matrice dei
formati/profili e proposta ADR di scelta con alternative, licenze, limiti e prima
ipotesi del modello documentale. La scelta può essere diversa per formato.
Applicare la skill `record-architecture-decision` quando esistono prove e una
decisione significativa; nessun nuovo ADR di motore è giustificato dalla sola
ricognizione. L'implementazione del modello documentale appartiene alla fase2.

Gli esperimenti mancanti devono avere un seguito circoscritto. Se mancano prove
essenziali PDF/OCR o la scelta motivata, a003 può consegnare risultati parziali
espliciti ma **non completare fase1 né autorizzare fase2**. GPU non disponibile
e provider remoto fuori perimetro restano NOT_MEASURED con motivo e prossimo
esperimento; nessuna macchina o quota nuova dedotta dal cambio di chat.

Arresti sostanziali: input protetti difformi senza recupero, perimetro/costo non
coperto, esaurimento risorse, impossibilità di confinamento/integrità, rifiuto
effettivo dell'approvazione. Errori ordinari e EPERM/EACCES seguono il
[protocollo](run-lifecycle.md#restrizioni-del-sandbox-e-approvazione-degli-strumenti),
senza eludere guardie o cancellare fallimenti. Raw/gold/corpus, V13/A03, limiti
L01/L02/L04/L05, Chrome NONPASS, helper12gap, H15/E9 e tutta la temp a002 restano
conservati. Nessun deploy o promozione a main.
