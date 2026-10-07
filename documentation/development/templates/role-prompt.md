# Prompt — RUN_ID — RUOLO — rNNN — DESTINATARIO

Agisci come RUOLO per questa run. Leggi `AGENTS.md`, il protocollo in
`documentation/development/run-lifecycle.md` e i file sotto elencati. Verifica
branch, HEAD, modifiche attese e identità pertinenti prima di lavorare. Un
contesto d’ingresso non resta MATCH dopo modifiche autorizzate; lo snapshot di
review/prova richiede invece input stabili. Se non hai accesso al repository e ai
file di contesto, richiedi il trasferimento necessario senza inventarne il contenuto.

## Input espliciti

- Directory della run, stato e brief:
- Obiettivo e criteri di accettazione:
- Piano/arbitrato/snapshot pertinenti:
- File necessari e ordine di lettura:
- Mandato per risultato, directory e costi ammessi (elenco dei mezzi non esaustivo):
- Input protetti, correzioni autonome e arresti sostanziali:
- Distinzione fra input di confronto, input della prova e parametri adattabili;
  deadline ammesse compatibili con la CLI, directory di output e budget cumulativo:
- Identificazione autonoma degli input delle prove; snapshot comune del supervisore
  alla consegna per le review:

## Compito del ruolo

Il supervisore compila soltanto le istruzioni pertinenti:

- Pianificazione: produrre un piano verificabile, senza implementare il codice.
- Review piano: valutare il piano e il codice di contesto, senza leggere il report
  dell'altro revisore né l'arbitrato corrente; scrivere il proprio report GO/NO_GO.
- Implementazione: eseguire il piano identificato dal GO, verificare e documentare;
  correggere errori ordinari nella stessa chat e rinviare soltanto scostamenti
  sostanziali. Decidere anche mezzi reversibili non enumerati entro requisiti e
  confini; fonti già configurate e redirect verificati non richiedono un mandato
  per host. Registrare le decisioni significative per le due review, senza
  estendere obiettivo, costi o divieti espliciti.
  I parametri operativi si adattano negli intervalli ammessi, anche dopo freeze:
  conservare argv storico e registrare quello eseguito, senza nuovo mandato per
  timeout compatibili, output esclusivi o correzioni di launcher/reader propri.
  Se cambia un input della prova, correggere nel perimetro e ripetere le verifiche
  invalidate. Creare autonomamente gli eventuali snapshot di esecuzione richiesti
  dagli strumenti; non attendere un freeze del supervisore e non consegnare una
  sola preparazione. Consegnare il risultato completo con errori e soluzioni adottate.
  Per restrizioni del processo distinguere EPERM/EACCES dal rifiuto effettivo
  dell'approvazione: diagnosticare e richiedere l'escalation dello strumento
  entro il mandato, se disponibile, conservando l'isolamento della prova.
  Non aggirare una richiesta respinta; riferire il motivo reale del blocco.
- Review implementazione: controllare diff, file nuovi e criteri del piano sullo
  snapshot comune; primo giro completo, fix mirati a chiusure e regressioni delle
  dipendenze toccate. Non modificare il prodotto e non leggere la review concorrente.
- Supervisione: ricevere il risultato completo, congelarlo per le due review,
  arbitrare i due report, aggiornare stato/handover e produrre il prompt
  successivo insieme a NO_GO: fix locale senza pianificazione intermedia, oppure
  nuovo piano motivato per cambi sostanziali. Non simulare review mancanti e non eseguire operazioni Git manuali dell'utente.

## Output obbligatori

Percorsi esatti del report, evidenze e `handovers/<ruolo>.md`. Non sovrascrivere
report delle revisioni precedenti o di altri autori. Includere nel riepilogo in chat
esito, limiti, percorsi dei file e prossimo ruolo. Un prompt generato dal supervisore
deve essere completo, senza segnaposto residui, e non includere istruzioni per ruoli diversi.
