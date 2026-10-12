# Handover supervisore — prossimo incremento fase1 dopo a002

Agisci esclusivamente come **supervisore in una nuova chat**. Recupera AGENTS,
temp/HANDOVER, prima parte PROJECT-CONTEXT, questo documento, roadmap, ADR0001/
0004/0008, archivio a002 e debito permanente. Usa manage-implementation-run,
verify-conversion-fidelity e record-architecture-decision dove pertinenti.
Non implementare, delegare o impersonare reviewer. Non riaprire il ciclo a002.

## Decisione ricevuta e prossimo risultato

A002 **CLOSED / ACCEPTED_REDUCED_SCOPE_WITH_DEBT** per richiesta utente12ott2026.
Reader r014 resta sperimentale, codice non modificato dalla chiusura. ClaudeNO_GO
statico/GPTreviewincompleta/NO_GO precedenti non diventanoGO. Direzione parser
mantenuto accettata, adattatore da implementare dopo prima app valutabile;
debito obbligatorio secondo [registro](../benchmarks/a002-residual-debt.md).

La fase1 **non è conclusa**. Prossimo incremento proposto a003: confronto motori
e scelta motivata su pochi documenti rappresentativi, con risultati leggibili
e valutabili dall’utente. Niente vincitore predeterminato, né nuova campagna
per correggere il reader. Obiettivo successivo: risultati essenziali fase1,
poi fondazioni fase2 e percorso upload→anteprima→download reale fase3.
Il prodotto Web non è già disponibile.

## Ingresso e azione concreta

Prima di aprire la nuova feature, eseguire il commit della consegna a002 e
l'integrazione manuale in dev secondo i [comandi contestualizzati](../runs/run-a002-fase1-corpus/manual-integration.md).
Il supervisore riceve output e SHA e registra l'integrazione reale; non
considerarla già avvenuta dalla sola chiusura della run.

1. Leggi [manifest a002](../runs/run-a002-fase1-corpus/manifest.md),
   [indice umano](../runs/run-a002-fase1-corpus/human-evaluation.md),
   [ADR0008](../decisions/0008-maintained-parser-and-provisional-benchmark.md)
   e [roadmap](../roadmap.md) nelle sezioni fase1–3. Storia originale archiviata,
   inventario integrità e dipendenze temp nel manifest; non leggere tutti i log.
2. Verifica branch/HEAD/indice/delta, senza reset/checkout/query remota o Git
   mutativo. A002 alla chiusura feature/run-a002-corpus, HEAD/dev/origin-dev
   locale/merge-base80c31da46e5caab990d3596c79c44e524ec1dc95; non integrata.
   Commit/merge/push restano manuali dell’utente; integrare solo la consegna
   esplicitamente accettata e documentata, non presentarla comeGOtecnico.
   Una nuova feature deve partire dalla base dev effettiva: predisponi il
   passaggio Git manuale necessario, non abbandonare modifiche esistenti.
3. Esegui sola ricognizione leggibile delle capacità già presenti: hardware,
   Pandoc/Marker/Docling, package/pesi realmente disponibili, licenze/profili.
   Nessun import pesante, installazione, download, conversione o invio implicito.
   Distingui capacità documentate, disponibilità locale e capacità misurate.
4. Prepara brief/mandato circoscritto del prossimo incremento: selezione piccola
   di casi nativi ePDFtestuale/scansione/misto dove fattibile; gold indipendenti,
   raw/asset/provenienza, errori espliciti, tabella per documento efenomeno,
   confronto umano accessibile. Metriche del reader provvisorie, niente scelta
   definitiva fondata soltanto su di esse. Gli esperimenti mancanti restano
   NOT_MEASURED con motivo, non mock spacciati per conversioni reali.
5. Dimensiona risorse rispetto ai workload e alla disponibilità reale, includendo
   review/amministrazione; evita il tetto10CPU-s per reviewer e lookup ripetuti.
   A002 non rinnova le quote: nuovo budget operativo da disporre prima dei
   workload se quello esistente è insufficiente. Pianificazione/ricognizione
   indipendente coperta può continuare. Non inviare documenti a remoto o
   scaricare pesi pesanti senza perimetro/costi espliciti. Nessuna deadline
   della run dedotta o reintrodotta; nessun nuovo servizio per cambiochat.

La nuova run segue il ciclo ordinario per obiettivo, con piano e review nei
ruoli reali. L’eccezione di accettazione del debito riguarda a002, non ogni
run futura. Snapshot tecnici puntuali e inventari indicizzati una volta, non
freeze/manualpermission per ciascun comando. Verifiche proporzionate e risultati
umani devono guidare decisioni e scelta dei motori, non generare apparato inutile.

## Vincoli da mantenere

Raw/gold/corpus/ricevute/FAIL/STALE/NONPASS storici e V13/A03 conservati.
Pandoc AST per sintassi distinto dai gold di fedeltà. Nessuna attestazione
universale COMPLETE/UNSAFE, nessun aggiramento delle guardie file/processi.
L01/L02/L04/L05/ChromeNONPASS/helper12gap/H15/E9 restano limiti dichiarati.
Nessuna pulizia della temp a002 finché le dipendenze archiviate non coprono
quanto necessario. Nessun deploy, promozione main o motore già scelto.

All’uscita: aggiorna CP/STATE della nuova attività, prepara il mandato concreto
e indica cosa l’utente deve inviare alla chat del ruolo successivo. Non avviare
fase2 prima dei risultati essenziali fase1 e della decisione documentata.
