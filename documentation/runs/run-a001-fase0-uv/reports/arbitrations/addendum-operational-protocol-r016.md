# R016 — Implementazione continua; snapshot delle prove delegati all'autore

2026-10-06, supervisore. Disposizione richiesta esplicitamente dall'utente dopo
report-r014. Il ciclo è quello di AGENTS e del protocollo corrente: piano → due
review → arbitrato → implementazione completa con fix autonomi → due review del
risultato → arbitrato e chiusura oppure fix. Nessuna review nuova è qui inventata.

## Problema e responsabilità

Mandato49: cinque operazioni PASS, poi I root FAIL su uv_cache.json; reader
corretto con tredici controlli locali dichiarati PASS. L'autore si è fermato
per il freeze s018 imposto dal mandato, non perché fosse incapace di correggere
il difetto. La dipendenza dal supervisore era un errore del mandato e viene rimossa.
L'arresto e le ricevute antecedenti restano descrizioni corrette della loro versione.

## Variazione vincolante

Superati **arbitrato del piano r003 D1, passi 1–5 e divieto di scrivere snapshots/**,
**piano r003 P2, prescrizioni di attesa del supervisore**, e le corrispondenti
clausole dei successivi addenda/mandati, inclusi R014, R015 e prompt48/49, nei
soli punti che impongono consegna, owner supervisore e pausa prima di una prova
o dopo un fix ordinario. I vecchi prompt sono storia, non mandati da concatenare.

L'implementatore è autorizzato a:

- completare il piano approvato e correggere codice, test, fixture e propri
  diagnostici nel perimetro, senza un nuovo piano o un'autorizzazione per il fix;
- identificare gli input realmente consumati e produrre autonomamente gli
  snapshot di esecuzione con `scripts/run_context.py snapshot/verify`;
- usare s018, se ancora assente, per gli input corretti già preparati, oppure
  nuove label non esistenti per ulteriori correzioni; mantenere la cronologia;
- aggiornare inventari, binding e driver propri, acquisire nuove ricevute e
  ripetere tutte le prove invalidate nella stessa chat, senza richieste di freeze.

Il supervisore mantiene **STATE, HANDOVER comuni, eventi, indice, arbitrati e
snapshot comune delle review**. L'autore aggiorna solo report, evidenze e proprio
checkpoint oltre agli snapshot di esecuzione delegati. Non può emettere GO,
simulare review o approvare nuovi requisiti/costi fuori mandato.

Restano R/D4, attribuzione corrente/S/B/I/E, controlli pre-startup/backend e
canonica installata prima di `--no-sync`, fedeltà e confronti senza normalizzazioni.
Una versione nuova non eredita i cinque PASS s017; ripetere le prove dipendenti.
Il codice della nuova implementazione può essere corretto: gli hash antecedenti
ne identificano la versione, non lo rendono una baseline immutabile. Sono protetti
la vera baseline, le copie di riferimento e le ricevute storiche. Una correzione
di packaging/avvio autorizzata aggiorna gli input e la catena delle prove; non
autorizza cambiare algoritmi legacy o scelte di prodotto fuori piano.

## Ripresa concreta e costi

Report-r014 e request s018 sono ricevuti come consegna storica. Il loro stato
WAITING_FOR_STAGE_SNAPSHOT e snapshot_owner=supervisor sono superati da R016.
Nessun s018 viene prodotto dal supervisore e nessuna nuova request di sola
preparazione è dovuta. Prompt50 identifica il risultato e la ripresa concreta.

La fase base S/B/I/E conserva 112MiB cumulativi dal medesimo Hentry553541632
e 7200s complessivi; report-r014 dichiara 487.34544921084307s consumati,
6712.654550789157s residui. Non azzerare quote o duplicare intervalli annidati.
Lo scope ricevuto resta la fonte delle guardie di storage e delle directory.
Questi limiti sono della fase base, non un costo approvato per dev/API/ML/Docker.
R016 autorizza il lavoro restante del piano eseguibile con risorse già ammesse;
non autorizza implicitamente acquisizioni escluse o pesanti, V10/V11, spese,
privilegi persistenti o indebolimento dell'isolamento.

Un limite reale blocca la sola attività dipendente: completare il lavoro
indipendente, diagnosticare, proporre nel report finale un intervento concreto
con costi/input mancanti. Nessun arresto per il solo FAIL, cambio di helper,
argv, label o necessità di un nuovo snapshot tecnico. Le richieste di escalation
degli strumenti già delegate da R015 restano ammesse; rispettarne il rifiuto reale.

## Consegna e prossimo passaggio

Una consegna dell'implementatore al completamento del lavoro autorizzato:
report-r015, evidenze e checkpoint. Se resta un vincolo sostanziale, indicare
esattamente cosa manca e cosa è stato comunque completato; non dichiarare GO
o completamento di un requisito essenziale assente. Il supervisore riceve il
risultato; quando revisionabile, congela l'oggetto comune e scrive i due prompt
di review ChatGPT/Claude in chat nuove. Arbitrato successivo: chiusura manuale
Git su GO, oppure mandato di fix e nuova doppia review. Non riaprire il piano
per le correzioni ordinarie.
