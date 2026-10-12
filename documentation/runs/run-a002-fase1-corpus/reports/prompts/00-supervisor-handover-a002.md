# Handover alla nuova supervisione — run a002, corpus e misurazione

Agisci esclusivamente come **supervisore** della run `run-a002-fase1-corpus`.
L'utente autorizza la preparazione e il ciclo supervisionato di questo incremento.
Non avviare implicitamente il resto della roadmap. Usa la skill
`.agents/skills/manage-implementation-run/SKILL.md` e il protocollo del repository.

## Ingresso autorevole

Leggi nell'ordine:

1. `AGENTS.md`, `temp/HANDOVER.md`, `temp/PROJECT-CONTEXT.md`.
2. `temp/run-a002-fase1-corpus/STATE.md`, `brief.md` e `HANDOVER.md`.
3. `documentation/development/next-run-a002-corpus.md`, roadmap fase 1,
   indice architetturale, ADR e `documentation/open-questions.md`.
4. `documentation/runs/run-a001-fase0-uv/manifest.md`, la receipt
   `evidence/integration-push-2026-10-07.json` e
   `notes/cleanup-assessment-2026-10-07.md` in quel medesimo archivio.
5. Per definire i criteri di fedeltà, ADR 0001, skill
   `.agents/skills/verify-conversion-fidelity/SKILL.md` e workflow benchmark.

Non caricare tutta la storia a001 e non ripetere le sue campagne per il solo
cambio di chat. Le review/arbitrati archiviati sono reali; nessuna review a002 esiste.

## Punto di partenza verificato

- Base osservata `dev` e remoto `origin/dev`:
  `80c31da46e5caab990d3596c79c44e524ec1dc95`.
- Squash uv `b90c6d4c00aed5ceb9122f6f42544736de7ff35a`; feature storica
  `8479f975618034abfcfaeb4c9e47ba925d5e356e`; 405 file MATCH e alberi
  feature/squash identici. Il commit successivo modifica solo sette registri.
- A001: GO finale r003 dopo ChatGPT/Claude GO/GO, integrata e pubblicata.
  Quote temporali NON PASS e lacune storiche conservate senza sanatoria.
  Nuova architettura, inferenza e hardware non collaudati restano tali.
- A002 è solo PREPARATION. Contesto generato con `scripts/run_context.py init`.
  Branch previsto `feature/run-a002-corpus`, non creato dal supervisore uscente.
- Preparazione/handover e piano di pulizia sono nuovi documenti locali attesi;
  possono essere portati sul feature branch e inclusi nella prossima consegna.
  Non richiedere un altro commit di chiusura a001 per avviare la pianificazione.

Verifica branch, HEAD e delta presenti. Se il branch manca, il comando manuale
previsto per l'utente è `git switch -c feature/run-a002-corpus dev`.
Se `dev` è avanzato, registra il nuovo commit base dopo lettura del delta;
non fare reset o ricreare snapshot per nascondere differenze. La preparazione
del mandato/prompt del piano può proseguire; prima dell'implementazione serve
il feature branch corretto. Commit, merge e push restano manuali dell'utente.

## Risultato che devi portare avanti

Il brief definisce l'obiettivo completo: corpus sintetico redistribuibile,
riferimenti controllabili indipendentemente, protocollo e runner di misura,
negativi che rilevano perdite e primo esperimento reale sui percorsi leggeri.
Questo incremento prepara il confronto Docling/Marker/remoto; non conclude la
fase 1 e non sceglie in anticipo un vincitore o lo schema dell'applicazione.
Concorda col piano la dimensione minima sufficiente del corpus e le metriche
per testo, numeri, ordine, formule, celle, asset e riferimenti. Non sostituire la
fedeltà con esistenza dei file, metadati o un punteggio medio.

Pianifica il lavoro indipendente oggi disponibile: synthetic/gold, strumenti,
negativi e baseline nativa. Hardware osservato in questo ambiente: CPU i7-1165G7,
8 CPU logiche, circa 62,5 GiB RAM totale, nessun device NVIDIA esposto alla sonda.
È un'osservazione locale, non una certificazione GPU/Windows/WSL/macOS.
Per dati privati, modelli pesanti, altre macchine e provider a pagamento serve
un perimetro esplicito; chiedi solo informazioni realmente indispensabili,
continuando la preparazione locale. Usa fonti primarie aggiornate se valuti
versioni/capacità/licenze dei candidati, senza acquisire automaticamente pesi.

## Ciclo richiesto dall'utente

1. Questa chat supervisiona, arbitra e scrive i prompt.
2. Scrivi ora un prompt completo per la chat pianificatrice e aggiorna stato/
   checkpoint. Non limitarti a presentare un piano in chat o offrire di continuare.
3. Ricevi il piano e prepara due prompt con criteri comuni per nuove chat
   indipendenti ChatGPT e Claude; nessuna review simulata o delegata implicitamente.
4. Arbitra i due report. GO sul piano: mandato di implementazione per l'intero
   risultato. NO_GO locale: errata/fix mirati, senza ripartire ogni volta dall'analisi.
5. L'implementatore completa obiettivo e fix nella stessa chat: errori ordinari,
   diagnostici, argv, permessi e freeze di esecuzione non sono consegne parziali.
   Se serve escalation degli strumenti per un'azione già autorizzata la richiede,
   mantenendo isolamento e rispettando un effettivo rifiuto di approvazione.
6. Alla consegna completa, snapshot comune e due nuove review indipendenti.
7. Arbitra: GO/GO e GO del supervisore preparano chiusura/Git manuale; NO_GO
   prepara il prompt di fix basato sui rilievi accolti, poi nuove review mirate.

Un mandato copre il risultato, non una catena di autorizzazioni per singoli comandi.
Fissa requisiti e confini veri; lascia all'implementatore i mezzi ordinari reversibili.
Snapshot tecnici attribuiscono le prove, non sono nuove approvazioni. Richiedi nuove
prove solo per gli input modificati; non riconvalidare indiscriminatamente tutto a001.
Una quota reale finita comprende preparazione, workload, registri e riserva di
consegna; evitare cap minimi ripetuti che impongano un nuovo prompt ad ogni sonda.
La pianificazione stima i costi e il supervisore emette una disposizione unica
adeguata; le vecchie quote a001 non si azzerano né si ereditano in a002.

## Residui e pulizia

La valutazione è pronta nell'archivio; nessuna cancellazione eseguita dall'uscente.
Alla chiusura l'utente ha richiesto i comandi per eseguire la pulizia selettiva.
Lo script locale `temp/maintenance-a001-to-a002/cleanup-a001.py` è stato verificato
senza `--apply` su file e Docker. I comandi manuali `--apply` e `--apply --docker`
rimuovono rispettivamente solo le193copie e i tre tag superati, dopo ricontrollo.
Alla ripresa leggere le receipt `cleanup-files-receipt.json` e
`cleanup-docker-receipt.json` nella stessa directory: sono prodotte dall'esecuzione
reale, non dal dry-run. Se presenti, registrare gli esiti; se assenti, non presumere
rimozioni. Non chiedere di ripetere una pulizia già riuscita per il cambio di chat.
193 wheel storiche sono duplicate byte per byte di copie conservate; lista esatta,
hash e keeper sono nel JSON collegato dal piano. Tre immagini a001 r004/r005/r006
sono superate e non associate ai container osservati. R007 e supply finale r004,
managed, .venv, ambienti dev/API, cache utili e prove uniche restano conservati.
Non fare `rm -rf temp`, `git clean`, prune globale Docker, cancellazione cache
condivise o rimozione di oggetti di altri progetti. La pulizia è facoltativa per
avviare a002: non inventare un gate di spazio quando è sufficiente.
L'utente ha autorizzato la pulizia selettiva e preferito eseguire personalmente
i comandi forniti. Ricontrolla gli esiti e lo stato corrente prima di eventuali
sole rimozioni identificate ancora necessarie. Non trasferire vecchie receipt a nuovi input.

## Output di questa chat

- Stato, eventi, artefatti e checkpoint del supervisore a002 aggiornati.
- `temp/run-a002-fase1-corpus/prompts/01-planning-r001-corpus-and-measurement.md`
  autosufficiente per la chat pianificatrice, con obiettivo, criteri, output,
  ipotesi, limiti, stima delle risorse e percorso del report.
- Alla consegna del piano, prompt review in posizioni distinte e arbitrato reale;
  nessuna creazione anticipata di GO o report di benchmark inesistenti.
- Aggiornamento del changelog agli avanzamenti significativi; stato della
  decisione distinto da quello dell'implementazione. Non anticipare la fase 2.
