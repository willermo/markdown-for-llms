---
name: manage-implementation-run
description: "Gestisce le run del convertitore: mandati per obiettivi, correzioni locali, doppie review indipendenti, arbitrati e handover in temp/. Usare per pianificare, implementare, revisionare o supervisionare una run e i suoi fix; non per semplici domande sul progetto."
---

# Gestire una run supervisionata

## Recuperare il ruolo e lo stato

Leggere `AGENTS.md`, `temp/HANDOVER.md` se esiste, `STATE.md` della run e il prompt
assegnato al proprio ruolo. Leggere il [protocollo](../../../documentation/development/run-lifecycle.md)
per passaggi, responsabilità, snapshot e archivio. Non caricare tutta `temp/`:
recuperare brief, checkpoint del ruolo e soli artefatti pertinenti.

Verificare branch, HEAD, modifiche locali attese e input pertinenti. Un working tree già modificato e identificato è valido: i commit restano manuali dell'utente. Non ricontrollare a ogni consegna l'interprete intero o tutti gli artefatti storici. Se mancano file, non
ricostruire GO o review dalla memoria della chat. Recuperarli dall'archivio o chiedere
il trasferimento del contesto necessario, continuando il lavoro indipendente possibile.
Il bootstrap di una run è disponibile con `python3 scripts/run_context.py init`;
seguire gli argomenti del protocollo. Non sovrascrivere una run esistente.

## Operare nel ruolo assegnato

- **Analisi:** definire brief e criteri, verificare feature branch da `dev` e dipendenze;
  preparare prompt di handover al supervisore. La run può restare in PREPARATION se
  l'integrazione manuale di una fase precedente non è avvenuta.
- **Supervisione:** mantenere stato condiviso, indice, handover, prompt e arbitrati.
  Generare un mandato completo per risultato, con input/output, criteri, costi e arresti sostanziali. Scrivere prompt autosufficienti anche quando implementazione e fix continuano nella stessa chat. Richiedere i due
  report reali ChatGPT/Claude sullo stesso oggetto; eventuali sostituzioni si registrano.
  Arbitrare ogni rilievo con evidenze, poi GO o NO_GO secondo il protocollo. Al NO_GO consegnare insieme il prompt di fix, senza pianificazione intermedia per correzioni locali.
- **Pianificazione:** produrre piano versionato verificabile o piano di fix derivato
  dall'arbitrato. Consegnare al supervisore senza implementare prematuramente.
- **Review:** controllare il medesimo piano/snapshot dell'altro revisore in una chat
  indipendente, senza leggerne il report corrente. Scrivere nel proprio file esito,
  rilievi identificati, prove e limiti; non correggere il prodotto durante la review.
- **Implementazione:** eseguire il piano identificato dal GO. Produrre report,
  evidenze e checkpoint alla consegna completa. Correggere test rossi, sintassi,
  attese e propri strumenti nella stessa chat; preservare input di confronto ed
  esiti invalidati. Creare autonomamente gli snapshot di esecuzione richiesti
  dalle prove e ripetere quelle invalidate, senza consegnare una richiesta di
  freeze al supervisore. Un FAIL non è da solo una richiesta di supervisione.
  Rinviare scostamenti sostanziali reali al supervisore. Il riepilogo
  dell'implementatore non sostituisce le due review.

Non impersonare contemporaneamente autori e due revisori, né sostituire una chat
Claude con un secondo passaggio dello stesso agente dichiarandoli indipendenti.
Se non è disponibile la chat richiesta, consegnare il prompt e mantenere il passaggio
in attesa. Il contesto scritto permette la ripresa senza perdere il lavoro svolto.

Per EPERM/EACCES o altra restrizione del processo seguire la
[distinzione sandbox/approvazione](../../../documentation/development/run-lifecycle.md#restrizioni-del-sandbox-e-approvazione-degli-strumenti):
diagnosi e richiesta di escalation tramite gli strumenti, se disponibile,
rientrano nel mandato già autorizzato mantenendo l'isolamento. Il gate è il
rifiuto effettivo dell'approvazione o un nuovo confine necessario; non il solo
errore di syscall. Nessun aggiramento di una richiesta respinta.

## Passaggi e chiusura

Usare i [template](../../../documentation/development/run-lifecycle.md#template)
pertinenti. Revisioni e nomi file sono progressivi e specifici per ruolo/revisore.
Legare le review agli snapshot prodotti da `scripts/run_context.py`; verificarli
prima/dopo. Una modifica agli oggetti approvati richiede il riesame previsto.

NO_GO sul piano con correzione locale produce errata vincolanti, verificate prima del GO; un nuovo piano serve per cambi di metodo/perimetro o nuove informazioni decisive. NO_GO sull'implementazione produce un prompt di fix nella stessa run; può proseguire la stessa chat implementatrice. I due reviewer restano chat nuove e indipendenti a ogni giro, con criteri comuni: prima review completa del delta, successive su chiusure e regressioni dei fix, senza ignorare requisiti essenziali. Per ogni rilievo verificare raggiungibilità sul codice/input ammessi, impatto, attribuzione alla base e requisito violato. I sintetici sono validi per fedeltà, confini e sicurezza; un difetto non è innocuo perché raro o preesistente. GO finale consente di preparare
i comandi Git **da far eseguire all'utente**: non effettuare commit, merge, push o
promozioni automaticamente. Un deploy dipende dal perimetro operativo autorizzato.

Aggiornare il proprio checkpoint a ogni consegna e prima di cambiare chat. Conservare
evidenze utili nelle destinazioni permanenti del protocollo e verificarle prima della
pulizia selettiva. Lo stato comune appartiene al supervisore, i checkpoint agli autori.
Terminare ogni consegna con esito fattuale, percorsi dei file e prossimo ruolo/azione.

La fonte operativa è il protocollo collegato sopra. Qui la memoria permanente è
documentation/, quella della run è temp/. Il supervisore congela piano e risultato
per le review; l'implementatore identifica gli input e produce gli eventuali
snapshot tecnici necessari durante l'esecuzione. Un cambiamento ordinario non
introduce un passaggio di ruolo. Un piano/arbitrato specifico si varia una volta
con addendum esplicito, non ignorandone i requisiti. Per questa run usare lo stato
corrente e il nuovo mandato, non la sequenza storica dei prompt.
