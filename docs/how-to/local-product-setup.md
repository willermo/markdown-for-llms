# Preparare il prodotto locale

Questa guida riguarda il package sperimentale `markdown_for_llms`, distinto dai
comandi `markdown-*` legacy. Target: Linux x86_64, CPython managed 3.12.13, CPU,
singolo utente e filesystem locale. Il processo API non importa Marker o torch;
il worker usa lo stesso interprete della venv per i figli isolati.

Segui la [guida uv](uv-environment.md) per origine del managed, inventario di ingresso,
snapshot della run, build sdist→wheel e reinstallazione canonica. Aggiungi gli
extra `--extra app --extra marker-cpu` al sync; per lo sviluppo aggiungi `--group dev`.
Non usare insieme `marker-cpu` e `marker-cu126`. Esegui le prove distribuzione con
profilo `cpu` quando la venv contiene ML; il profilo `api` rifiuta tali dipendenze.
I diagnostici identificano i dieci moduli legacy e i file del nuovo package,
compresi statici, migrazione e inventario modelli. Il solo sync non sostituisce S/B/I.

Usa `uv export --locked --no-default-groups --extra app --extra marker-cpu
--no-emit-project` per preparare i requisiti della venv esterna. Le installazioni
con hash richiedono gli archivi identificati dal lock. EbookLib0.18 è distribuito
come sdist: una wheel locale costruita da tale sdist richiede identità e provenienza
proprie, non eredita il digest dell’archivio sorgente. Non modificare le venv
storiche di benchmark per installare il prodotto.

Prepara esplicitamente Bubblewrap del sistema (`bwrap`) e i dati del profilo
Marker1.10.2/Surya0.17.1. L’inventario
[models.json](../../markdown_for_llms/models.json) identifica i byte osservati dei
cinque checkpoint e due font del profilo: recognition/layout2025_09_23,
detection2025_05_07, table/OCR-error2025_02_18, GoNotoCurrent e GoNotoCJK.
Puoi riusare una copia locale già acquisita con provenienza verificata; imposta
`MDF_MODEL_ROOT` alla sua radice. Il worker verifica i cinquanta file prima
dell’inferenza. Non scarica né ripara modelli durante la conversione.

L’acquisizione dei pesi è una preparazione separata, da fonti ufficiali del
[pin Marker](https://github.com/datalab-to/marker/tree/v1.10.2) e
[Surya](https://github.com/datalab-to/surya/tree/v0.17.1). Il package non include
pesi. Le condizioni del codice e dei pesi sono distinte; l’uso qui collaudato
è locale sperimentale, senza redistribuzione dei modelli. L’inventario registra
hash osservati, non firme o checksum indipendenti pubblicati dal fornitore.

Per questa run i dati già acquisiti si trovano localmente in
`temp/run-a003-fase1-engine-comparison/work/implementation-r001/marker-models-r001`.
È un percorso di lavoro ignorato da Git, non una risorsa garantita da un clone.
La venv prodotto nuova è sotto `temp/run-a004-fase2-prodotto-utilizzabile/work/implementation-r001/venv`.
Non trasferire questi percorsi personali nel manifest esportato.

Avvia API e worker secondo il [tutorial](../tutorials/first-local-pdf.md).
`/health` risponde alla sola liveness; `/api/v1/readiness` indica worker assente,
runtime incompleto o isolamento non disponibile. Bubblewrap deve poter creare
namespace privati: un errore di permessi non abilita un fallback senza isolamento.
Nell’ambiente di sviluppo usa l’escalation prevista dagli strumenti, mantenendo
la barriera; non modificare sysctl, setuid o profili persistenti per il collaudo.

La quota archivio predefinita è20GiB con1GiB libero di riserva, oltre alla venv e
ai circa3,48GB di checkpoint/font. Il profilo del figlio ha16GiB di memoria
virtuale massima per processo e una guardia RSS aggregata campionata; lascia
RAM per sistema/API/browser. Il timeout450s è configurabile e non garantisce
la conversione di50pagine.

Per cambiare quota o timeout di un archivio esistente:

1. Arresta API, worker e CLI che usano l’archivio. Uno stop del worker conserva
   il tentativo come `interrupted`; l’annullamento richiesto dall’utente è `cancelled`.
2. Con il vecchio file ambiente, leggi il campo `config` di `identity.json`.
   Il comando verifica anche radici, identità DB e schema; non modificare il JSON.
3. Esegui, sostituendo il digest con quello letto:

   ```bash
   mdf --env-file /percorso/prodotto.env configure --expected-config DIGEST --storage-mib 30720 --timeout-seconds 600
   ```

4. Aggiorna `MDF_STORAGE_MIB` e `MDF_TIMEOUT_SECONDS` nello stesso file ambiente
   e riavvia tutti i processi. La configurazione precedente viene rifiutata.
   Manifest e configurazioni dei risultati storici restano invariati.

Il lock esclusivo rifiuta l’aggiornamento se un servizio è ancora aperto.
Dopo un crash del worker, riavvialo con la vecchia configurazione per riconciliare
figli e prenotazioni prima del cambio. Un’identità precedente errata o una quota
inferiore allo spazio occupato/impegnato blocca l’operazione. Non serve cancellare
l’archivio; la cancellazione generale resta fuori I1.


Se un risultato resta `publishing`, consulta `mdf show ID` o il dettaglio
nel browser. Un rifiuto di quota/spazio o un guasto I/O/fsync conserva il
risultato completo anche se il rename non è avvenuto e indica il recupero
dello stesso ID. Una pubblicazione normale in corso non richiede intervento;
la lettura non ricalcola la capacità dell’archivio a ogni aggiornamento.
Se il recupero è differito, ripristina l’accesso o lo
spazio libero del volume senza rimuovere membri dell’archivio; se serve quota
applicativa maggiore usa `configure` con tutti i servizi fermi. Per questa
condizione, i record del worker morto possono richiedere prima un avvio con
la vecchia configurazione, che li riconcilia e rilascia. Poi riavvia il worker
con la configurazione corrente: ripete verifiche e sincronizzazioni prima di
completare la pubblicazione. Controlla che lo stesso ID diventi `completed`
e che lo ZIP sia disponibile. Non usare `retry` per recuperare `publishing`:
quel comando crea una nuova conversione per un’esecuzione terminale.

Un tentativo fallito per voce non sicura conserva fonte e parziale; gli altri
risultati e nuovi invii restano utilizzabili entro quota. Puoi creare un nuovo
tentativo con `mdf retry ID`; se il motore riproduce la voce non sicura fallirà
ancora esplicitamente. I1 non offre eliminazione o riparazione dei parziali.
