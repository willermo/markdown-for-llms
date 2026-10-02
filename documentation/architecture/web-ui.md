# Web UI — Percorso utente e sviluppo progressivo

Stato: requisiti iniziali e proposta di realizzazione. La UI della nuova versione
non è ancora implementata. Decisione di riferimento: [ADR 0002](../decisions/0002-fastapi-web-ui.md).

## Esperienza prevista

La Web UI è una parte centrale del prodotto. Deve essere utilizzabile in un browser
locale e distribuita insieme al backend, evitando l'obbligo di una GUI diversa per
ogni sistema operativo. Swagger resta disponibile per esplorare le API.

| Vista | Azioni e informazioni |
| --- | --- |
| Nuova conversione | Caricare file o bundle, scegliere profilo e lingue, vedere se dati saranno elaborati localmente o da un provider remoto |
| Coda e attività | Vedere fase corrente, errori, annullamento e ripetizione; indicare avanzamento indeterminato se il motore non fornisce una percentuale attendibile |
| Archivio | Cercare documenti e conversioni, confrontare versioni, riaprire risultati e scaricare bundle |
| Revisione | Sorgente e risultato affiancati, problemi per blocco, proposta OCR con prima/dopo e motivazione, accettazione o rifiuto |
| Risultato | Anteprima di formule, immagini e tabelle; testo Markdown; download del documento e degli asset |

Per i PDF il pannello sorgente mostra pagina e regione associata al blocco quando
disponibili. Per formati nativi senza coordinate, usare sezioni e identificatori
logici; la UI non deve inventare una posizione nella pagina. Per documenti lunghi
caricare pagine e anteprime progressivamente.

## Stati comprensibili

Distinguere stato di esecuzione, disponibilità degli artefatti e stato di revisione.
Un job terminato può produrre un risultato da controllare. Un errore di conversione,
un asset mancante e una correzione suggerita sono condizioni differenti.

Le correzioni accettate creano una revisione e conservano l'estrazione originale.
Il download rende chiaro quale revisione viene esportata. Non usare un'etichetta
«verificato» basata esclusivamente sulla confidenza dichiarata dall'LLM.

## Scelta frontend

React con TypeScript è una proposta da confrontare con un frontend più semplice
servito dal backend. Il confronto deve includere visualizzatore PDF, collegamento
pagina/blocco, rendering della matematica, revisione di differenze, accessibilità e
manutenzione. La scelta avverrà prima della fase 3 con un ADR dedicato, senza
introdurre oggi dipendenze frontend o una libreria di componenti per inerzia.

## Incrementi e verifica

1. **Fase 3:** upload, coda, archivio e download realmente collegati al backend;
   anteprima di un formato semplice, persistenza anche dopo riavvio.
2. **Fase 4:** visualizzazione delle sorgenti scientifiche e problemi di conversione.
3. **Fase 5:** confronto per blocco, correzioni OCR/AI e revisioni tracciate.

Verificare il percorso anche con job falliti, refresh del browser, documenti lunghi,
formule e asset mancanti. Prevedere uso da tastiera e messaggi leggibili; mantenere
credenziali sul backend e sanificare i contenuti renderizzati. L'accesso dalla rete
e la gestione multiutente dipendono dal perimetro ancora da definire, mentre la
prima distribuzione proposta è locale.
