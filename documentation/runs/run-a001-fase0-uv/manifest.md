# Archivio — run-a001-fase0-uv

- Obiettivo/fase: toolchain Python uv, packaging e verifiche legacy, fase0.1.
- Base/HEAD alla chiusura tecnica:66ba82200e5def5a4db76f9bafccb0731b506091;
  feature/run-a001-uv. Commit e integrazione in dev **non ancora eseguiti**.
- Esito: **GO / READY_FOR_MANUAL_INTEGRATION**, arbitrato implementazione r003.
- Data/responsabile:2026-10-07, supervisore Codex/OpenAI. Nessun deploy/main/cleanup.
- Oggetto review finale:final-s003 SHA
  6295835b1407ea44f6c49780d38061c0fd5d6389a2ef1e096da9dd260f778962.

## Risultato e limiti

Piano r003 approvato; prima review completa e due giri mirati, due reviewer
indipendenti per giro. Ultime review ChatGPT/Claude GO, blocchi chiusi. Quote
temporali **NON PASS**: storico minimo8321,418589/overrun41,418589, upper ignoto;
nuovo osservato8946,997655 e addebito8947,997655, senza upper dimostrato.
Fine turno circa8966,24/overrun86,24 è inferenza, non misura. Gate di finalize
tardivo e record FAIL ricostruito. Nessun workload prodotto oltre cap rilevato.
Il GO accetta la deviazione amministrativa e non sana i costi.

S/B/I/E22/22 e fedeltà12casi, host/standalone/V1 e Docker sono prove dei propri
input/stage con equivalenze verificate. Vecchia C-fast137+300 non attribuita al
nuovo harness;12mock mirati e2R reali a001/b002 attestano il delta finale.
Baseline62/5/perdite e FAIL storici conservati. Nuova architettura, inferenza,
GPU/Windows/V10/V11 e universalità delle ABI non verificate.

## Materiale conservato

Elenco completo con origine, destinazione e hash sorgente/archivio in
[archive-files.json](archive-files.json). Copie JSON/TXT/PY sono byte-identiche;
i Markdown cambiano solo i collegamenti locali, con trasformazioni in
[link-adaptations](notes/link-adaptations.json). Nessun codice prodotto spostato
qui; gli script di evidenza sono conservati come dati storici.

- [Arbitrato finale](reports/arbitrations/arbitration-implementation-r003.md).
- [Review finale ChatGPT](reports/reviews/review-implementation-r003-chatgpt.md).
- [Review finale Claude](reports/reviews/review-implementation-r003-claude.md).
- [Report-r020](reports/implementation/report-r020.md).
- [Snapshot finale](evidence/snapshots/impl-r001-stage-final-s003.json).
- [Comandi manuali](reports/prompts/62-user-close-run-a001-manual-git.md).
- [Materiale locale/limiti](notes/materiale-locale-conservato.md).

## Verifica e pulizia

Integrità delle copie, adattamento dei link, delta documentale rispetto al GO e
git diff --check verificati dal supervisore; ricevuta di chiusura in temp.
Nessuna nuova esecuzione delle prove, commit/merge/push o pulizia. Tutti i payload
non selezionati e gli originali rimangono nella run locale. Registrare i commit
reali dopo l'integrazione manuale; non dichiarare già completata la fase in dev.
