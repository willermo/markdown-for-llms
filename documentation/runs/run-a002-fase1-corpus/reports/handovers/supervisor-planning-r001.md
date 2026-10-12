# Checkpoint supervisore — prompt pianificazione r001 consegnato

2026-10-07, supervisore Codex/OpenAI. Ruolo esclusivo: supervisione a002.
Letti gli ingressi del prompt00, skill di gestione/fedeltà, protocollo,
ADR/roadmap/questioni aperte e i tre documenti mirati dell'archivio a001.
Nessun recupero indiscriminato della storia né campagna a001 rieseguita.

## Stato verificato e output

Branch `dev`, HEAD/dev e ref locale origin/dev
`80c31da46e5caab990d3596c79c44e524ec1dc95`; feature a002 assente, indice vuoto.
Il remoto reale resta verificato dalla receipt precedente; nessuna nuova query
di rete in questa chat. Delta d'ingresso: sette documenti modificati e quattro
nuovi documenti/JSON di preparazione/pulizia, nessun codice prodotto modificato.
Base non avanzata. Git di scrittura non eseguito.

Scritto [prompt01 completo](../prompts/01-planning-r001-corpus-and-measurement.md):
corpus/gold indipendenti, metriche per fenomeno e soglie motivate, positivi e
negativi, primo esperimento nativo reale, contratti benchmark senza fase2,
stima numerica di risorse per l'intero obiettivo con riserva. Nessun mandato di
implementazione, quota definitiva, piano, review, arbitrato o benchmark creato.
PREPARATION / PLANNING_PROMPT_READY: pianificazione consentita su dev,
feature corretta necessaria prima dell'implementazione. Comando manuale utente:
`git switch -c feature/run-a002-corpus dev`.

Ricevute due receipt reali di pulizia, PASS/apply true. Ricontrollati 193 file
assenti, 91 keeper unici per byte/hash, lista immutata; tre tag Docker assenti e
r007 presente. Prima lettura Docker fallita per permessi del processo, escalation
read-only approvata e riuscita; nessun rifiuto del sistema di approvazione.
Nessuna cancellazione di questa supervisione. Spazio fisico recuperato non
misurato. Originali in maintenance, copie byte-identiche e nota di ricezione
nell'archivio permanente a001; input/snapshot/review storici non riscritti.

## Prossima azione e ripresa

Aprire una nuova chat pianificatrice con prompt01. Output attesi:
`plans/plan-r001.md`, `handovers/planning-r001.md` ed eventuali evidenze proprie.
Sono percorsi attesi, non artefatti già presenti. Nessun agente delegato avviato.

Alla ricezione del piano questa chat supervisore verifica branch/base/delta,
completezza e costi, aggiorna i registri comuni, congela piano e gli artefatti
pertinenti con `scripts/run_context.py`, verifica lo snapshot, poi scrive due
prompt separati per nuove chat ChatGPT e Claude con criteri tecnici identici.
Non creare review o GO prima dei report reali. Leggere/verificare entrambi,
arbitrare ogni rilievo; GO del piano autorizza il mandato completo, NO_GO locale
produce errata/fix mirati. Review del risultato successive, indipendenti.

Se riprende una nuova supervisione, leggere AGENTS, handover globale, STATE,
questo checkpoint, brief, prompt01 e piano/checkpoint del pianificatore solo
se realmente consegnati. Preservare il perimetro utente: nessun resto della
roadmap, ML/remoto a pagamento o Git automatico. Errori ordinari e snapshot
di esecuzione restano autonomi all'implementatore dopo il GO reale sul piano.

Verifiche documentali della consegna in
`evidence/supervisor-planning-r001/document-checks.json`; ricezione pulizia in
`evidence/supervisor-planning-r001/reception.json`. Nessuna suite/build/conversione
eseguita, nessun processo proprio pendente. Il supervisore non lavora fra chat.
