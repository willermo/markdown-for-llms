# Workflow — Incremento di sviluppo

**Ingresso:** funzionalità o correzione richiesta, con fase della roadmap pertinente.

1. Leggere `AGENTS.md`, stato Git, ADR e codice coinvolto. Preservare le modifiche
   dell'utente e lavorare sul feature branch concordato.
2. Definire un risultato osservabile e il minimo percorso completo che lo dimostra.
   Risolvere le normali scelte implementative nel perimetro autorizzato; registrare
   nuovi tradeoff architetturali con il workflow dedicato.
3. Implementare separando servizi applicativi, adattatori e trasporti HTTP/CLI quando
   il nuovo package è disponibile. Non duplicare logica nella UI o dentro endpoint.
4. Verificare il comportamento pertinente. Per conversione usare la skill di fedeltà;
   per UI provare nel browser anche refresh, job fallito e persistenza. Dichiarare
   quando una prova è soltanto simulata o un componente non è ancora disponibile.
5. Aggiornare reference/guide interessate, `.env.example` se cambia la configurazione,
   e stato della roadmap soltanto quando i criteri sono soddisfatti.
6. Controllare diff e risultati; consegnare cambiamento, motivazione, verifiche e
   limiti. Non eseguire automaticamente merge o promozioni come effetto della procedura.

**Uscita:** incremento revisionabile, con comportamento dimostrato e documentazione coerente.
