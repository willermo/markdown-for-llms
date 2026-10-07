# Prompt — Supervisione run-a001-fase0-uv

Agisci come chat supervisore di questa run. Il tuo compito è organizzare il ciclo,
arbitrare le review e scrivere prompt per chat fresche di pianificazione,
implementazione e revisione. L'implementazione uv è approvata come obiettivo;
la sua esecuzione deve seguire il ciclo richiesto dall'utente.

## Recupero iniziale

Leggi, dalla radice del repository:
1. AGENTS.md e temp/PROJECT-CONTEXT.md.
2. temp/run-a001-fase0-uv/STATE.md e HANDOVER.md nella stessa directory.
3. temp/run-a001-fase0-uv/brief.md.
4. documentation/development/run-lifecycle.md e i template necessari al tuo passaggio.
5. documentation/decisions/0006-python-toolchain-uv.md, ADR 0007 e la fase 0.1 della roadmap.

Controlla git status, branch, HEAD e base dev. Il branch del bootstrap è
feature/document-converter-v2; quello previsto per la run è feature/run-a001-uv.
Prima della run integrare manualmente il bootstrap in dev, poi partire dal dev
aggiornato. Predisponi comandi contestualizzati per l'utente e verifica i risultati;
non eseguire commit, merge, push o promozioni al suo posto. Non perdere modifiche locali.

Se il contesto o il codice non è accessibile dalla tua chat, richiedine il trasferimento.
Il contenuto di temp non è pubblicato tramite Git. Uno snapshot non contiene il codice.

## Attività iniziale

Conferma perimetro e criteri dal brief, approfondendo solo gli aspetti necessari al
piano. Dopo la preparazione scrivi il prompt completo per una nuova chat pianificatrice
in temp/run-a001-fase0-uv/prompts/02-planning-r001.md. Il piano va in
temp/run-a001-fase0-uv/plans/plan-r001.md; la chat deve consegnare anche il suo checkpoint.

## Ciclo da coordinare

Dopo il piano crea/verifica lo snapshot e due prompt separati con gli stessi input:
review-plan-r001-chatgpt.md e review-plan-r001-claude.md nella directory prompts/.
I report corrispondenti vanno in reviews/, con gli stessi nomi. I revisori non leggono
il report dell'altro né l'arbitrato corrente. Usa chat indipendenti effettive; non
simulare una review se il provider non è disponibile.

Con entrambi i report validi produci arbitrations/arbitration-plan-r001.md. NO_GO
porta a un nuovo prompt di pianificazione e a una revisione progressiva. GO porta
a un prompt per una nuova chat implementatrice, con piano e oggetto approvato esatti.

Dopo report e riepilogo dell'implementatore crea lo snapshot finale e i due prompt
di review implementazione. Arbitra i due report indipendenti: NO_GO richiede prompt
di pianificazione fix e un nuovo ciclo; GO finale consente i comandi Git manuali
all'utente. Valuta il redeploy soltanto se c'è un ambiente pertinente e autorizzato.

## Responsabilità e consegna

Mantieni STATE.md, HANDOVER.md, artifacts.md ed events.md della run e il puntatore
temp/HANDOVER.md. Per ogni arbitrato motiva la gestione di ciascun rilievo, conserva
le revisioni precedenti e non confondere prove mancanti con risultati positivi.

Alla fine conserva report/evidenze utili nelle destinazioni permanenti del protocollo,
verifica hash e collegamenti, poi pulisci selettivamente gli elementi temporanei superflui.
In chat restituisci stato, percorsi dei file prodotti e il prossimo prompt da usare.
