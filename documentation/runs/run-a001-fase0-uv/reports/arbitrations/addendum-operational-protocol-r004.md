# Addendum operativo r004 — mandato per risultato e controlli pertinenti

2026-10-05. Supervisore, adozione esplicita dell'utente del processo adattato dal
handover Claude. [Valutazione](../../../../development/process-adaptation-2026-10-05.md).
Integra r001–r003 senza cambiare piano/arbitrato r003, requisiti A1–A7 o i risultati
storici. Autorizza lavoro implementativo nei limiti; non è GO finale del codice.

## Variazioni puntuali

1. Prompt35 e next-scope s007 rimangono storici. Per le attività nuove, prompt36
   sostituisce il divieto di modifiche tracciate e l'obiettivo limitato alla sola
   request baseline-s006: completare P1–P7 del core autorizzato, strumenti e prove
   pertinenti, raggruppando gli input pronti. Nessun cambio di pin o dominio.
2. «Ogni FAIL STOP» dei vecchi mandati non governa lo sviluppo nuovo: errori di
   sintassi, generatori, aspettative sbagliate, fixture e codice nel perimetro si
   correggono nella stessa chat. Conservare risultato e versione pertinenti,
   ripetere le sole prove invalidate. Un test fallito non dà diritto a cambiarne
   il requisito. Negativi intenzionali sono esiti attesi delle guardie.
3. Il contesto process-adaptation-context-r001 identifica l'ingresso. Dopo
   modifiche autorizzate diventa storico per quel delta; non richiedere MATCH
   contro codice nuovo. Prima dei gate ufficiali: request concreta, snapshot
   nuovo del supervisore e verify. Mai adattare helper/snapshot per mascherare
   differenze, né cambiare input protetti sotto una vecchia label.
4. D1 si applica per dipendenze effettive: baseline/package/tests compatibili
   possono condividere un freeze che identifica separatamente originali e
   prodotto. Se un input consumato deve essere prodotto prima (es. lock), serve
   quel passaggio; non generare output futuri nella lista. Nessun freeze per
   singolo file e nessuna nuova pianificazione per un fix locale. Image/final
   restano distinti. S→B→I→E conserva schema, contenuto e invalidazioni P2.
5. Togliere ricertificazioni ripetute dell'intero host e di tutta la storia.
   Conservare gli inventari esistenti; verificare input consumati, startup,
   confinamento e delta pertinenti. All'ingresso/uscita delle sonde sulla venv
   baseline verificare l'inventario del target protetto. Un'anomalia concreta
   richiede l'estensione delle verifiche. Nessuna equivalenza globale implicita.
6. Review: due nuove chat indipendenti sul medesimo snapshot, stesso criterio
   tecnico; primo giro completo, fix successivi mirati anche alle regressioni.
   Il supervisore arbitra e consegna il prompt dei fix insieme al NO_GO locale.
   Nuovo piano soltanto per cambi sostanziali. Nessuna review simulata.

## Costi e attività operative

Questa adozione non aumenta budget o autorizza download impliciti. Riuso baseline
s007 in sola lettura; sonde R preliminari e ambiente esplicito come next-scope.md
s007. Restano due gruppi/sei chiamate, 120s figlio/300s esterno/1800s cumulativi;
nuovo gruppo motivato dopo fix ordinario, senza riparare il target protetto.

Budget run500MiB, soglia384MiB, nuovi scratch/report16MiB, riserva prove64MiB e
riserva esterna16MiB invariati; consumi storici inclusi, niente cleanup per passare.
Log1MiB/stream, JSON8MiB, file32MiB, libero516MiB; misure pre/post, non quota atomica.
I nuovi sorgenti del prodotto nel repository non sono cache da nascondere in temp.

Managed Python, resolver/lock, backend/build/sync/install hanno GO tecnico nel
piano, ma i vecchi budget di recupero non coprono implicitamente quelle acquisizioni.
Se mancano comandi/input/costi ammessi, predisporli insieme al lavoro pronto nella
request del prossimo gate. Sospendere solo l'attività dipendente; continuare codice,
documentazione e verifiche già coperte. Non sommare di nuovo riserve esaurite, né
nascondere cache/interpreti fuori dal ledger. Il supervisore decide sull'operazione
concreta, senza richiedere un altro giro di sola preparazione. V7/V8 pesanti hanno
mandato distinto e restano essenziali; nessun peso/font/inferenza V10/V11.

## Input protetti e autorità

S007 target work/baseline-recovery-install-r003 e lock, originali legacy, vecchi
FAIL/parziali, snapshot/receipt e piano/arbitrato sono immutabili. Usa percorsi nuovi
per le sonde e -I -B espliciti anche sui figli; .pth/startup controllati. Nessuna
modifica host/privilegi/rete/profili, nessun invio documenti, cleanup o Git automatico.
Rifiuto sandbox resta IMPEDITA. Mancanza integrità, risorse, confini o nuovo scope
richiede arresto dell'attività dipendente, non PASS o retry cieco.

Il supervisore aggiorna registri comuni; l'autore report e checkpoint propri.
Consegna unica: codice e input pronti più request concreta, oppure impedimento
sostanziale con lavoro indipendente completato. Autorità corrente STATE e prompt36;
checkpoint d'autore s007 conservato, non riscritto dal supervisore.
