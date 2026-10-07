# Prompt — run-a001-fase0-uv — Review del piano r001 — Claude

Agisci come revisore Claude in una **nuova chat indipendente**, distinta
sia dalla chat pianificatrice sia dal supervisore e dall'altro revisore.
Revisiona il piano, non implementare la migrazione. Il tuo ruolo è assegnato
esclusivamente da questo prompt, anche se un handover globale parla di supervisione.

Provider previsto: Claude. Registra provider/modello effettivi e riferimento
chat se disponibile, senza inventare identificatori non esposti. Se il provider
previsto non è disponibile o questa chat ha già scritto il piano/svolto l'altra
review, segnala il problema al supervisore senza impersonare il revisore mancante.
Non leggere report, checkpoint o evidenze dell'altro revisore di questo giro,
né l'arbitrato corrente. Non coordinare i rilievi con l'altra chat.

## Oggetto e identità comuni ai due revisori

- Repository: `/home/davide/workarea/markdown-for-llms`.
- Run: `run-a001-fase0-uv`, fase roadmap 0.1; revisione del piano: r001.
- Piano: `temp/run-a001-fase0-uv/plans/plan-r001.md`.
- SHA-256 piano: `7f9dd0a2ded10c40ef9e3a42ae1424fb08a23059384468ea160dd7173fd3e8fb`.
- Snapshot comune: `temp/run-a001-fase0-uv/snapshots/plan-r001.json`.
- Branch: `feature/run-a001-uv`.
- HEAD/base dev/merge-base verificati dal supervisore:
  `66ba82200e5def5a4db76f9bafccb0731b506091`.
- Sei modifiche documentali di supervisione non committate sono previste e incluse
  nello snapshot. Nessun codice uv implementato; bootstrap già integrato in dev.

Dalla radice verifica branch, HEAD/base, modifiche e snapshot **prima della lettura
tecnica e alla fine della review**:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
python3 scripts/run_context.py verify run-a001-fase0-uv --label plan-r001
```

Il risultato atteso è MATCH. Registra nel report HEAD, impronta worktree e hash
del piano letti dal manifest, oltre agli esiti pre/post. Il vecchio snapshot
`planning-context-r001` è storico: dopo gli aggiornamenti documentali di stato
non è più l'oggetto comune di review. Non ricreare o sovrascrivere gli snapshot.
Se `plan-r001` è STALE o il codice non corrisponde, descrivi la divergenza e
consegna una verifica di identità non valida per l'arbitrato; non attribuire GO
al contenuto differente. Se manca accesso ai file, richiedi il trasferimento
necessario: il manifest non contiene il codice, e Git non trasferisce temp/ o
modifiche non committate. Non ricostruire input o esiti dalla sola chat.

## Input espliciti e ordine di lettura

1. Questo prompt, `AGENTS.md`, `.agents/skills/manage-implementation-run/SKILL.md`,
   `documentation/development/run-lifecycle.md` e template
   `documentation/development/templates/review.md`.
2. `temp/run-a001-fase0-uv/STATE.md`, `brief.md` e
   `prompts/02-planning-r001.md`. La migrazione uv è già approvata; criteri e
   confini del brief restano la specifica. Nessun GO sul piano è ancora emesso.
3. `documentation/README.md`, `documentation/decisions/README.md`, ADR 0001,
   0006 e 0007, fase 0.1 della roadmap e changelog. Il codice applicativo è legacy.
4. Piano `plans/plan-r001.md` integralmente; checkpoint
   `handovers/planning-r001.md`; evidenze `evidence/planning-r001/findings.md`,
   `checks.json` e `local-inventory.json`. Catalogo/lista Python e help uv in
   quella stessa cartella sono input congelati: consulta quelli pertinenti alle
   opzioni/versioni discusse. Il manifest enumera e identifica tutti gli artefatti.
5. `evidence/supervisor-preparation-r001.md` e
   `evidence/supervisor-plan-review-r001/receipt.json` per i controlli di identità
   e la provenienza. Sono controlli amministrativi, non giudizi sul piano.
6. Sorgenti e test necessari alla valutazione: `setup.py`, `requirements.txt`,
   `.gitignore`, `Dockerfile`, `docker-compose.yml`, README nelle sezioni interessate,
   i dieci moduli elencati nel piano; `tests/conftest.py`, unit/integration/governance
   pertinenti e i due smoke manuali alla radice per il conflitto di discovery.
   Leggi altri file soltanto se servono a un rilievo specifico.

Non caricare tutta temp/ e non leggere report dell'altra chat. Le due review
ricevono lo stesso piano, snapshot, evidenze e criteri; cambia soltanto l'identità
del revisore e la destinazione dei suoi output. Il pianificatore ha fatto letture
e verifiche documentali/di identità, senza lock, build, test o conversioni.

## Criteri e compito della review

Valuta se il piano è implementabile, circoscritto e verificabile rispetto ai sette
criteri del brief, motivando tutti i rilievi. Per ogni A1–A7 controlla che i passi
P0–P7 e le prove V0–V12 diano risultati osservabili, prerequisiti/costi, limiti e
recupero. Non ridurre il criterio A6 a una lettura del Dockerfile; non attribuire
al piano prove di implementazione che devono ancora essere svolte.

Controlli comuni richiesti:

- A1/A2: interprete esplicito e rapporto con pyenv, restrizione Python proposta,
  disponibilità/pin toolchain/backend; pyproject e lock come fonti coerenti;
  separazione runtime/dev/server/ML. Valuta il grafo CPU/cu126, indici, sorgente
  Git, marker di piattaforma e congelamento dei backend, inclusi i fallback
  proposti quando la risoluzione/digest non è ancora provata.
- A2/A3: dieci moduli flat e cinque entry point; config.main; origine degli import
  della wheel non editable fuori dai sorgenti; directory dati distinta dal codice,
  avvio -m delle fasi, dotenv e side effect. Help, editable o sys.path dei test
  non sostituiscono la prova reale delle fasi su output inizialmente vuoti.
- A4: baseline e fixture sintetiche, suite selezionata distinta dai test nuovi e
  governance; discovery dalla radice; assenza di HTTP/modelli impliciti e cache
  tokenizer preparata. Consulta la skill di fedeltà per verificare che testo,
  formule, numeri, riferimenti, asset e ordine siano confrontati davvero, senza
  alterazioni globali o normalizzazioni che nascondano regressioni.
- A5: istruzioni uv/README/guide coerenti e provabili, comandi realmente esistenti,
  nuova applicazione presentata come futura; limiti legacy e risultati storici
  distinti da verifiche della migrazione. Verifica il perimetro editoriale proposto.
- A6: motivazione del candidato Marker, compatibilità statica e limiti rispetto
  all'inferenza; Docker dal lock, digest e native, filtro del contesto, CPU/GPU e
  cache. Valuta la sufficienza delle prove obbligatorie V7/V8 e delle condizioni
  per rinviare V10/V11; /health, import e mock non dimostrano conversione reale.
  Non trattare le versioni candidate o la disponibilità GPU come fatti collaudati.
- A7: quattro report reali, due arbitrati, snapshot identificati, responsabilità
  separate, gestione degli scostamenti e integrazione Git manuale dell'utente.
  Un GO di questa review del piano non autorizza da solo l'implementazione.

Le scelte numeriche del piano sono proposte. Verifica affermazioni aggiornabili o
incerte usando help locale congelato, documentazione ufficiale e sorgenti primarie;
registra URL/data e distingui verifiche da inferenze. Una procedura futura non
eseguita non è un fallimento di per sé: giudica se è fattibile e sufficiente per
il requisito, e se le incertezze essenziali sono gestite prima del GO finale.

Questa è una review del piano: non eseguire la migrazione, uv lock/sync/build,
installazioni, suite, conversioni, Docker build/up o startup Marker. Sono ammessi
letture, controlli statici, hash e help/versioni senza download. Prove ulteriori
necessarie vanno formulate come requisito o rilievo con un criterio di risoluzione,
non realizzate modificando l'oggetto sotto review. Non inviare documenti a provider,
scaricare modelli o introdurre benchmark/costi impliciti.

## Output obbligatori

- Report: `temp/run-a001-fase0-uv/reviews/review-plan-r001-claude.md`.
- Evidenze proprie: `temp/run-a001-fase0-uv/evidence/review-plan-r001-claude/`.
- Checkpoint: `temp/run-a001-fase0-uv/handovers/review-plan-r001-claude.md`.

Usa il template review. Il report dichiara autore/provider/modello effettivi,
prompt di origine, oggetto r001, snapshot/hash/HEAD, indipendenza, file e controlli
eseguiti, esiti pre/post, limiti e **GO oppure NO_GO motivato** sul piano.
Assegna ID ai rilievi `CLA-P001`, `CLA-P002`, ecc.; per ciascuno indica
severità, blocco sì/no, posizione, impatto, evidenza e criterio verificabile di
risoluzione. Distingui blocchi di correttezza/requisito da suggerimenti opzionali.
Se non ci sono rilievi, dichiaralo senza inventarli; non approvare prove mai fatte.

Scrivi soltanto i tuoi output, senza sovrascrivere un report già presente, il piano,
input, registri condivisi, snapshot o codice. Nessun commit, merge, push o deploy.
Il checkpoint riporta processi ancora in corso o la loro assenza.

Consegna al supervisore percorsi, esito e limiti. Non leggere la review concorrente
neppure dopo aver consegnato per modificare il tuo giudizio. Il supervisore attende
entrambi i report validi sullo snapshot `plan-r001`, arbitra tutti i rilievi e solo
con l'arbitrato GO produce il prompt per la nuova chat implementatrice.
