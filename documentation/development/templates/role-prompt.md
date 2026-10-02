# Prompt — RUN_ID — RUOLO — rNNN — DESTINATARIO

Agisci come RUOLO per questa run. Leggi `AGENTS.md`, il protocollo in
`documentation/development/run-lifecycle.md` e i file sotto elencati. Verifica
branch, HEAD e snapshot prima di lavorare. Se non hai accesso al repository e ai
file di contesto, richiedi il trasferimento necessario senza inventarne il contenuto.

## Input espliciti

- Directory della run, stato e brief:
- Obiettivo e criteri di accettazione:
- Piano/arbitrato/snapshot pertinenti:
- File necessari e ordine di lettura:

## Compito del ruolo

Il supervisore compila soltanto le istruzioni pertinenti:

- Pianificazione: produrre un piano verificabile, senza implementare il codice.
- Review piano: valutare il piano e il codice di contesto, senza leggere il report
  dell'altro revisore né l'arbitrato corrente; scrivere il proprio report GO/NO_GO.
- Implementazione: eseguire il piano identificato dal GO, verificare e documentare;
  rinviare al supervisore scostamenti sostanziali senza estendere tacitamente il piano.
- Review implementazione: controllare diff, file nuovi e criteri del piano sullo
  snapshot comune; non modificare il prodotto e non leggere la review concorrente.
- Supervisione: arbitrare i due report, aggiornare stato/handover e produrre il prompt
  successivo. Non simulare review mancanti e non eseguire operazioni Git manuali dell'utente.

## Output obbligatori

Percorsi esatti del report, evidenze e `handovers/<ruolo>.md`. Non sovrascrivere
report delle revisioni precedenti o di altri autori. Includere nel riepilogo in chat
esito, limiti, percorsi dei file e prossimo ruolo. Un prompt generato dal supervisore
deve essere completo, senza segnaposto residui, e non includere istruzioni per ruoli diversi.
