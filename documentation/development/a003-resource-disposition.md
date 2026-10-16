# A003 — Risorse necessarie senza spesa monetaria

2026-10-13 Europe/Rome. **Disposizione ricevuta e attiva**, distinta dal GO
tecnico del piano e dall'esito delle conversioni. Il supervisore ne registra
l'applicazione operativa; non ha eseguito workload o acquisizioni.

## Disposizione originale dell'utente

> Io sono sempre dell'idea che possiamo concedere tute le risorse che occorrono, ad ecdcezione delle risorse che consumano denaro. Per cui lascia chje l'implementazione abbia tutte le risorse che le occorrono

Testo effettivamente ricevuto, conservato senza correggere la trascrizione.
Data di registrazione locale; nessun timestamp piattaforma o durata inventati.

## Applicazione al mandato a003

L'implementatore può usare le risorse necessarie al risultato autorizzato,
senza richiedere nuove quote gratuite per ogni comando, retry, preparazione o
review. I tetti proposti CPU/RAM/disco, i pool S0/S1/S2, le riserve sospese e
le relative richieste di riallocazione sono **superati** per il lavoro futuro.
I consumi storici, le stime e gli esiti NONPASS restano conservati; non diventano
misure nuove o successi retroattivi. La disponibilità fisica del sistema resta
un dato da verificare, non una promessa di risorse infinite.

Sono autorizzati setup e acquisizioni **gratuite** pertinenti al confronto:
runtime/wheel, modelli/pesi, font e dipendenze pubbliche necessarie a Marker e
Docling, con fonti primarie, licenze, versioni, hash/byte/provenienza identificati.
Nessun cambio al lock/pin prodotto o al corpus per convenienza: Docling ha
ambiente/configurazione separati; il pin Marker resta quello del piano.
Se le cache esistenti non sono compatibili, preparare gli artefatti corretti in
nuove directory della run, senza retrocedere i checkpoint o alterare la storia.

L'acquisizione è una fase di preparazione esplicita, distinta dalla conversione:
rete verso fonti pubbliche per metadata/payload, senza documenti, gold, dati
privati o credenziali esposti. Poi installazione e conversione offline con
preflight e confinamento verificati. Nessun fallback di conversione remoto.
Il ramo Docling previsto dal piano può essere preparato e misurato senza una
nuova richiesta di budget gratuito: scegliere e registrare un solo rilascio,
closure, OCR/pesi e profilo CPU riproducibile secondo E08, stessi piloti/casi,
soglie e verifiche di fedeltà. Non sono aggiunti nuovi candidati o formati.
Il gate I resta obbligatorio: Marker conserva il proprio profilo cpu; Docling
può avere un profilo puntuale verificato nello strumento standard, con gli stessi
invarianti di origine/archivi/canonica/startup/RECORD/dieci moduli e controllo della
closure effettiva. Nessun profilo esistente indebolito o PASS trasferito fra motori.

Timeout, affinità/thread, RSS, dimensioni di file/log/output e monitoraggio sono
parametri operativi da dimensionare rispetto al workload e alle risorse reali.
I vecchi numeri 900/1800s,6/10/12GiB,32MiB/1GiB e quattro CPU non sono nuove
quote vincolanti dopo questa disposizione. Adattarli dopo diagnosi è coperto,
con configurazione versionata, chiamate reali, fallimenti conservati e prove
invalidate ripetute; niente handover per chiedere più CPU gratuita.

Se i diagnostici impongono quei valori nel codice, sono ammessi adattamenti
mirati ai soli controlli di consumo/configurazione in `run_offline.py`,
`scripts/benchmarks/resources.py`, profili/launcher pertinenti e test correlati.
Usare monitor e strumenti standard esistenti, senza costruire un nuovo runner.
Mantenere watchdog, diagnosi di stallo, raccolta di tutti i figli e controlli
contro espansioni/output incontrollati: soglie giustificate dal caso e dalla
disponibilità reale, non assenza di controllo. Le modifiche saranno incluse
nelle due review codice indipendenti, non presentate come già convalidate.

## Confini mantenuti

- **Nessuna spesa monetaria:** niente API/inferenza a pagamento, cloud a consumo,
  noleggio hardware, acquisti o licenze onerose. Se un mezzo richiede denaro,
  sospendere quella parte e continuare le alternative gratuite pertinenti;
  non impegnare pagamenti, neppure se il costo sembra piccolo.
- Nessuna riduzione dei controlli di rete/file, protezione di canonici/gold,
  provenienza, S/B/I, ambiente managed o isolamento dei processi. Non esporre
  documenti a servizi remoti per ottenere risorse gratuite.
- Nessun privilegio persistente nuovo, cambio sysctl/AppArmor/setuid, elusione
  di un rifiuto effettivo di approvazione o cancellazione di prove per rientrare
  nei consumi. EPERM/EACCES seguono diagnosi/escalation già prevista.
- Metodo, campione, profili riproducibili, protocollo testo, soglie di fedeltà,
  risposta umana, due review codice e ruoli restano quelli approvati. Risorse
  gratuite disponibili non dimostrano completezza, qualità o fase1 conclusa.
- Nessuna fase2/3, UI, deploy, motore già scelto o Git mutativo automatico.
  A002 e il suo debito/storico/temp restano preservati.

La preferenza generale dell'utente è registrata nel contesto comune: usare
autonomamente risorse gratuite necessarie **entro gli obiettivi autorizzati**.
Non autorizza avviare nuove fasi della roadmap o introdurre spese. Per a003
attiva il mandato05 aggiornato e l'addendum operativo r003 (prova locale non versionata)
nella stessa run; il piano e l'errata originali restano immutati.
