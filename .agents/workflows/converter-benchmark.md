# Workflow — Confronto dei convertitori

**Ingresso:** capacità da valutare, corpus e profili hardware disponibili.

1. Definire formati e fenomeni da confrontare: colonne, formule, lingue, qualità
   delle scansioni, figure e tabelle. Identificare campioni e trascrizioni di riferimento.
2. Registrare provenienza/licenza e hash dei campioni, senza aggiungere file privati
   o chiavi al repository. Verificare che il benchmark remoto rientri nel perimetro
   autorizzato di dati e costi; svolgere intanto le prove locali indipendenti.
3. Bloccare versioni di motore, pesi e dipendenze, configurazione, hardware e comandi
   effettivi. Distinguere preparazione/download da conversione e cache fredda/calda.
4. Conservare output grezzi e problemi; misurare testo, ordine, formule, tabelle,
   asset, tempo, memoria e costo. Non omettere fallimenti o selezionare solo esempi riusciti.
5. Esaminare manualmente gli elementi scientifici e un campione delle correzioni.
   Distinguere equivalenza tipografica, sintattica e semantica; dichiarare il metodo
   di confronto e i casi per i quali la metrica non è attendibile.
6. Salvare un report datato in `documentation/benchmarks/` quando esiste un esperimento
   reale, con istruzioni riproducibili, risultati, limiti e proposta di scelta. Non
   creare ora risultati segnaposto che sembrino benchmark già eseguiti.
7. Aggiornare matrice di capacità e ADR pertinenti. Un aggiornamento importante del
   modello richiede un nuovo confronto sui casi interessati, non l'assunzione di equivalenza.

**Uscita:** scelta motivata da prove ripetibili, con risultati negativi conservati.
