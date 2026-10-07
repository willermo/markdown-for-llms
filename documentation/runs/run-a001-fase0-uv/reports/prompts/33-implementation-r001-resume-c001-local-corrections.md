# Implementazione r001 — correggere e completare c001 nella stessa chat

Repository `/home/davide/workarea/markdown-for-llms`. Ruolo esclusivo implementatore,
skill manage-implementation-run. Obiettivo: completare la preparazione c001 già
iniziata, install→pip-check→inventory preliminari e request s007 nella stessa chat.
Nessuna delega o consegna intermedia per normali difetti della preparazione.

Leggi AGENTS, protocollo §Mandati operativi, STATE/HANDOVER comuni, checkpoint
implementatore corrente e [prompt31](31-implementation-r001-continue-preliminary-baseline.md)
per gli invarianti tecnici. Piano r003, arbitrato D1–D5, prompt04 e addenda r001/r002
rimangono quelli identificati nel prompt31. Leggi l'addendum r003 e i seguenti file,
relativi a `temp/run-a001-fase0-uv/`:

- `arbitrations/addendum-operational-protocol-r003.md`;
- `evidence/supervisor-implementation-r001/preparation-correction-reception-r001/`:
  reception.json, preparation-correction-scope.json, transition.json, response.json;
- `handovers/supervisor-preparation-correction-r001.md`;
- `snapshots/baseline-recovery-pilot-continuation-context-r002.json`;
- `evidence/implementation-r001/pilot-baseline-recovery-r001/continuation-001/`:
  delivery.md, completion-continuation.json, preparation-tool.json,
  prepare_continuation.py, code e before già presenti.

Questo prompt e addendum r003 **prevalgono** sullo STOP generico di prompt31 per
errori di preparazione e sul vecchio binding al contesto r001. Tutte le operazioni,
guardie, argv/env, pin, budget e limiti tecnici di prompt31 restano obbligatori.

## Ripresa dal lavoro esistente

Ricalcola SHA/byte di contesto e disposizioni contro response.json e verifica da
radice `python3 scripts/run_context.py verify run-a001-fase0-uv --label baseline-recovery-pilot-continuation-context-r002`.
Usa questo contesto prima delle operazioni e alla consegna. R001 è storico per i
soli delta documentali in transition.json; nessun MATCH artificiale del vecchio
worktree. Scope originaria rimane identica: il suo context_label è sostituito da
questo addendum, senza editarla. Identità operativa rimane c001.

Il difetto ricevuto è una parentesi aggiuntiva nella sostituzione del reader
inventory in prepare_continuation.py, prima della scrittura reader/manifest.
Non sono stati avviati negativi o operazioni. Quattro moduli parziali sono presenti,
inputs è vuota; non è un bundle congelato. Il generatore originario pretende
assenze e scritture esclusive: non rilanciarlo invariato su questi file.

1. Preserva le versioni proprie da modificare e i report prima di aggiornarli;
   le copie della ricezione del supervisore sono già immutabili e disponibili.
   Mantieni l'errore storico, output trascritto e limiti di cattura dichiarati.
2. Correggi il difetto e completa i file mancanti usando i parziali utili. Puoi
   modificare/sostituire esclusivamente i tuoi file non congelati in continuation-001
   dopo backup verificato, e riusare le copie before esistenti. Nessuna rigenerazione
   generale, nuovo target, seed, duplicazione dei grandi modelli o delle suite immutate.
3. Aggiorna nei nuovi gate/generatore/manifest soltanto il binding al nuovo contesto
   e alla disposizione corrente: risposta corrente dal percorso sopra, label/SHA/byte
   r002 e riferimento/hash della preparation-correction-scope.json. La scope r002
   originaria resta identificata con il suo SHA e costi esatti. Valida la relazione
   fra disposizione, scope e contesto; nessuna rimozione del controllo di identità.
4. AST e negativi mirati di prompt31 sui sorgenti finali, quindi identifica tutti
   gli input effettivamente eseguiti nel manifest prima delle operazioni.
   Correggi autonomamente ulteriori difetti ordinari e ripeti le prove invalidate.
   Un rifiuto previsto nei negativi è un test riuscito. Nessuna nuova chat o freeze
   per ciascuna correzione statica; 73PASS storici restano sulle vecchie versioni.
5. Dopo verifica del prefisso chiuso/1046identità e di tutti i gate, esegui i tre
   tool distinti e prepara la request ufficiale secondo prompt31. Non avviare
   directories/venv/seed o prove applicative. Nessun test o receipt inventato.

## Arresti e consegna

SyntaxError e difetti ordinari della preparazione non impongono STOP/consegna.
La correzione deve preservare la politica specificata: non allentare una guardia
per far passare il risultato. Restano STOP per input protetti alterati/incerti,
guardie non applicabili, cap/budget/spazio, rifiuti sandbox, estensioni di politica/
scope/privilegi e FAIL operativi. Nessun retry dopo marker o figlio: preserva il
target e consegna il problema reale. Prima di qualificare un errore come preparatorio,
verifica che nessuna operazione abbia iniziato a modificare il target.

Budget invariato, misura l'intera run viva anche dopo questa ricezione: 72MiB
cumulativi, nuovi preparazione/evidenze/request4MiB inclusi, run500/stop384,
riserve112+16 e stima install32MiB. Nessun cleanup per ottenere ammissione.
Se il gate risorse non ammette il lavoro, riporta misure e deficit concreti.

Consegna unica al traguardo o per impedimento sostanziale: conserva la completion
fallita e le versioni dei report; aggiorna delivery/completion con storia ed esiti
reali distinti, report/checkpoint propri. Successo → request s007 pronta in
WAITING_FOR_STAGE_SNAPSHOT; altro esito → WAITING_FOR_SUPERVISOR_RECEPTION.
Prossimo supervisore prompt32 **con addendum r003 e questo contesto r002**.
Nessun registro comune, GO codice, prova ufficiale, modifica host o Git/deploy.
