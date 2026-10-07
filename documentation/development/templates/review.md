# Review — piano|implementazione — RUN_ID — rNNN — REVISORE

- Autore/provider/modello e chat:
- Prompt di origine:
- Oggetto e revisione esatta:
- Snapshot, HEAD e impronta verificati prima/dopo:
- Indipendenza: indicare se sono stati letti altri report dello stesso giro
- Giro: iniziale / fix, rilievi aperti e parti invalidate:
- Base iniziale e delta del giro (anche file nuovi non committati):
- Esito: da compilare con GO oppure NO_GO dopo la verifica

## Ambito e prove

File letti, comandi eseguiti, risultati osservati e verifiche non eseguite con motivo.

## Rilievi

| ID | Severità e blocco sì/no | Posizione | Raggiungibilità/input ammessi | Attribuzione e requisito | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- | --- | --- |

Se non ci sono rilievi, dichiararlo con il perimetro effettivamente controllato.
Non inventare rilievi per riempire la tabella né approvare prove mai eseguite.

## Motivazione dell'esito e limiti

Collegare esito a requisiti, rilievi e prove. Una review di una versione obsoleta
deve essere segnalata come non valida per il passaggio di stato.

Nei fix verificare chiusure e regressioni delle parti toccate, senza ignorare un
requisito essenziale scoperto nel delta. Sintetici/rari sono pertinenti quando
esercitano input ammessi, fedeltà o confini di sicurezza.
