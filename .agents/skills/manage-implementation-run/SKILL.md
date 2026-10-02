---
name: manage-implementation-run
description: "Avvia, supervisiona o riprende una run di implementazione del convertitore usando contesto locale in temp/, chat separate, doppia review indipendente del piano e del codice, arbitrati GO/NO_GO e handover. Usare per pianificare/eseguire/revisionare una fase o i suoi fix e per passare a una chat fresca; non aprire run per semplici domande sul progetto."
---

# Gestire una run supervisionata

## Recuperare il ruolo e lo stato

Leggere `AGENTS.md`, `temp/HANDOVER.md` se esiste, `STATE.md` della run e il prompt
assegnato al proprio ruolo. Leggere il [protocollo](../../../documentation/development/run-lifecycle.md)
per passaggi, responsabilità, snapshot e archivio. Non caricare tutta `temp/`:
recuperare brief, checkpoint del ruolo e soli artefatti pertinenti.

Verificare branch, HEAD, modifiche locali e impronte previste. Se mancano file, non
ricostruire GO o review dalla memoria della chat. Recuperarli dall'archivio o chiedere
il trasferimento del contesto necessario, continuando il lavoro indipendente possibile.
Il bootstrap di una run è disponibile con `python3 scripts/run_context.py init`;
seguire gli argomenti del protocollo. Non sovrascrivere una run esistente.

## Operare nel ruolo assegnato

- **Analisi:** definire brief e criteri, verificare feature branch da `dev` e dipendenze;
  preparare prompt di handover al supervisore. La run può restare in PREPARATION se
  l'integrazione manuale di una fase precedente non è avvenuta.
- **Supervisione:** mantenere stato condiviso, indice, handover, prompt e arbitrati.
  Generare prompt completi per nuove chat con input e output esatti. Richiedere i due
  report reali ChatGPT/Claude sullo stesso oggetto; eventuali sostituzioni si registrano.
  Arbitrare ogni rilievo con evidenze, poi GO o NO_GO secondo il protocollo.
- **Pianificazione:** produrre piano versionato verificabile o piano di fix derivato
  dall'arbitrato. Consegnare al supervisore senza implementare prematuramente.
- **Review:** controllare il medesimo piano/snapshot dell'altro revisore in una chat
  indipendente, senza leggerne il report corrente. Scrivere nel proprio file esito,
  rilievi identificati, prove e limiti; non correggere il prodotto durante la review.
- **Implementazione:** eseguire il piano identificato dal GO. Produrre report,
  evidenze e checkpoint; rinviare scostamenti sostanziali al supervisore. Il riepilogo
  dell'implementatore non sostituisce le due review.

Non impersonare contemporaneamente autori e due revisori, né sostituire una chat
Claude con un secondo passaggio dello stesso agente dichiarandoli indipendenti.
Se non è disponibile la chat richiesta, consegnare il prompt e mantenere il passaggio
in attesa. Il contesto scritto permette la ripresa senza perdere il lavoro svolto.

## Passaggi e chiusura

Usare i [template](../../../documentation/development/run-lifecycle.md#template)
pertinenti. Revisioni e nomi file sono progressivi e specifici per ruolo/revisore.
Legare le review agli snapshot prodotti da `scripts/run_context.py`; verificarli
prima/dopo. Una modifica agli oggetti approvati richiede il riesame previsto.

NO_GO sul piano torna alla pianificazione; NO_GO sull'implementazione produce un
prompt di pianificazione fix per una nuova chat. GO finale consente di preparare
i comandi Git **da far eseguire all'utente**: non effettuare commit, merge, push o
promozioni automaticamente. Un deploy dipende dal perimetro operativo autorizzato.

Aggiornare il proprio checkpoint a ogni consegna e prima di cambiare chat. Conservare
evidenze utili nelle destinazioni permanenti del protocollo e verificarle prima della
pulizia selettiva. Lo stato comune appartiene al supervisore, i checkpoint agli autori.
Terminare ogni consegna con esito fattuale, percorsi dei file e prossimo ruolo/azione.
