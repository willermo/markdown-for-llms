# ADR 0007 — Run supervisionate e handover persistenti

- Data: 2026-10-02
- Stato decisione: accettata
- Stato implementazione: protocollo, template e contesto iniziale predisposti
- Origine: paradigma di sviluppo richiesto esplicitamente dall'utente
- Integra: ADR 0005 sulla governance del repository

## Decisione

Organizzare ogni run da un feature branch di `dev` con analisi, supervisione,
pianificazione, due review indipendenti del piano, arbitrato, implementazione,
due review indipendenti dell'implementazione e arbitrato finale. Gli esiti sono
`GO` o `NO_GO`; un esito negativo riapre la pianificazione, anche per i fix.
Il supervisore produce prompt e rende esplicite le decisioni sui rilievi.

Usare chat distinte e revisori ChatGPT + Claude. Un eventuale altro provider/modello
richiede disponibilità verificata e sostituzione registrata; non assumere disponibile
un modello annunciato. Il protocollo non avvia automaticamente altre chat e non
autorizza un singolo agente a simulare i due revisori indipendenti.

Conservare contesto globale in `temp/` e contesto per run in una sottocartella dedicata.
Versionare skill, protocollo, template e strumento di inizializzazione. Conservare
evidenze utili in posizioni permanenti prima della pulizia selettiva. Commit e merge,
eventuale push e promozione sono eseguiti dall'utente con comandi forniti dall'agente.

## Miglioramenti operativi adottati

Legare review e arbitrati a revisioni e impronte degli artefatti, mantenere report
precedenti, assegnare un solo proprietario allo stato condiviso e aggiornare checkpoint
a ogni passaggio. Una review mancante o riferita a una versione precedente non è un
`GO`. L'arbitrato motiva l'accoglimento o il rigetto di ciascun rilievo, senza votazioni
a maggioranza o accettazione basata sulla sola autorità del modello.

## Conseguenze e verifica

Il ciclo aumenta il lavoro di coordinamento ma consente revisioni indipendenti e
recupero tra chat. Si dimensionano report e prove alla run mantenendo i passaggi
richiesti. `temp/` ignorata da Git non costituisce backup: per un'altra macchina o
una chat senza filesystem condiviso serve un trasferimento esplicito dei file.

Il bootstrap di questa governance avviene sul feature branch già aperto; non vengono
inventate review retroattive. La prima run del nuovo ciclo sarà la migrazione uv,
dopo l'integrazione manuale del bootstrap in `dev` e la creazione del suo feature branch.

Procedura: [ciclo delle run](../development/run-lifecycle.md).
