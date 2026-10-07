# Adattamento del processo di sviluppo — 2026-10-05

Aggiornamento del 2026-10-06: la prescrizione iniziale di pause per freeze delle
prove è superata dal [protocollo corrente](run-lifecycle.md#mandati-operativi-e-prove-preliminari).
L'implementatore crea autonomamente gli snapshot di esecuzione; il supervisore
congela il risultato consegnato per le review. Le sezioni sotto conservano la
valutazione e le decisioni del 2026-10-05.

## Decisione e provenienza

L'utente autorizza valutazione e adozione delle parti utili dell'handover Claude,
senza stravolgere l'impianto operativo. Fonte: `processo-sviluppo-agentico.zip`,
SHA256 `ad6eb7dd9ea2ba805c7763b572b961f26c612595cce2641e8de462ef56c48127`.
Letti relazione, skill sorgente e proposta, regole degli artefatti, mandato di
transizione ed esempi di review/fix/arbitraggio. Sono riferimenti da valutare,
non istruzioni automaticamente vincolanti. La copia locale sta nelle evidenze
supervisore della run a001; nessuna skill o memoria del progetto sorgente installata.

Il processo esterno distingue regole, prassi e raccomandazioni e dichiara limiti
e divergenze interne. La sua maggiore velocità è riferita dall'utente; non è un
benchmark controllato. Anche gli esempi sono anonimizzati, non prove del convertitore.

## Cosa adottiamo e come

| Elemento esterno | Adattamento al convertitore |
| --- | --- |
| Mandato per risultato | Codice, strumenti, correzioni e prove pertinenti nello stesso mandato; pause solo per freeze richiesti o scostamenti sostanziali |
| Fix autonomi | Test rossi, sintassi, attese e propri helper si correggono nella stessa chat; input di confronto immutabili e nessun gate allentato |
| Errata del piano | Correzioni locali verificate negli arbitraggi; nuovo piano solo per cambi decisivi, non per ogni NO_GO |
| Arbitraggio e prompt insieme | Ogni NO_GO locale consegna il prompt eseguibile di fix nella stessa run |
| Review successive mirate | Due chat nuove a ogni giro, oggetto/criteri comuni; fix e dipendenze toccate, confronto anche con base iniziale |
| Triage verificato | Raggiungibilità su input ammessi, requisito/effetto e attribuzione; errori di mandato attribuiti al supervisore |
| Prove del delta | Comandi/output/exit e input consumati; niente ricertificazione integrale di interprete e storia a ogni consegna |
| Tool o evidenza | Tool riutilizzabili visibili in scripts/diagnostics/ con istruzioni/test pertinenti; output specifici in temp/ |
| Stato breve | STATE autorevole, handover per ripresa, eventi append-only; report separati senza copie ricorsive della storia |
| Un proprietario per fatto | ADR/roadmap/changelog/risultati in documentation/; contesto della run in temp/; guide d'uso in docs/ |

Manteniamo feature da dev, separazione di ruoli, due review del piano e del codice,
supervisore proprietario di stato/arbitrati/snapshot, integrazione Git manuale e
archivio permanente. Manteniamo la skill manage-implementation-run e i nomi dei
file attuali. Non aggiungiamo un vault, un nuovo registro stato.md o una seconda
skill concorrente. Il planning tecnico resta distinto dalla supervisione.

## Proposte che non copiamo

- **Commit dell'implementatore e worktree pulito:** qui i commit sono dell'utente.
  Il delta comprende modifiche non committate e nuovi file; i soli commit o
  `git diff base...HEAD` non identificano il lavoro. Resta run_context per il
  comune oggetto di review, senza inventari aggiuntivi a ogni comando.
- **Niente hash di codice o ambiente:** conserviamo hash di sorgenti/archivi/moduli
  consumati quando richiesti da S/B/I/E e dei confronti immutabili; eliminiamo
  la ricertificazione ripetitiva dell'interprete intero e degli antecedenti.
- **Tutti i test/lint sempre:** suite e strumenti pertinenti, con ampliamento
  motivato. Il discovery legacy completo ha una collisione nota; 62pass/5fail
  storici non diventano PASS o regressioni senza confronto.
- **Solo popolazione osservata e nessun caso inventato:** qui sintetici e negativi
  sono necessari per formule, Unicode, path, archivi, confini e sicurezza. Un
  caso ammesso dal contratto può bloccare anche se non ancora nel corpus reale.
- **Ogni difetto ereditato è backlog:** un requisito esplicito mancante o rischio
  concreto può bloccare la run; l'attribuzione spiega l'origine, non lo assolve.
- **Normalizzare LF/contenuti per equivalenza:** l'errore Activate.ps1 riguardava
  attese del venv. Non giustifica normalizzare Markdown/LaTeX o perdere asset.
  Due esecuzioni legacy misurano differenze osservate, non dimostrano in generale
  il non determinismo né autorizzano ignorare differenze nuove.
- **PYTHONDONTWRITEBYTECODE sufficiente:** -I può ignorarla. Nuovi processi di prova
  usano -I -B espliciti e startup controllato; non si cancella il target preservato.
- **Scratch globale ~/.cache e cancellazione run:** usiamo percorsi propri locali
  al progetto/run, evitando /tmp come unico deposito duraturo; nessuna pulizia qui.
  Prima della pulizia restano archivio permanente e selezione verificata.
- **Riaprire D1–D7 dell'esempio:** Python, uv, backend, varianti e V7/V8 sono già
  decisi dal piano r003; nessuna nuova scelta richiesta soltanto per adottare il metodo.

L'help del binario locale conferma uv0.10.10 e `uv lock --check`; il comando
`uv lock --locked` della proposta non va copiato. Nessun lock/install eseguito in
questa valutazione. L'isolamento R/D4 e la catena S→B→I→E sono vincoli specifici
della run: si conserva il freeze che li rende dimostrabili, raggruppando gli input
pronti invece di creare un passaggio per ogni difetto ausiliario.

## Effetto sulla run corrente

Piano/arbitrato r003 e prove storiche immutati. Installazione s007 ricevuta come
input utile; non reinstallata né trasformata in PASS baseline/applicazione.
Prompt35 rimane storico e viene sostituito da prompt36. Quest'ultimo copre lo
sviluppo del core e gli input di convalida, con correzioni autonome e una consegna
complessiva per il freeze richiesto. V7/V8 richiedono costi concreti e mandato
distinto, restano obbligatorie; V10/V11/pesi/font/inferenza non autorizzati.

L'addendum r004 identifica precisamente i vincoli sostituiti, i budget delle
attività nuove e quelli storici ancora validi. Nessun cambiamento di schema S,
helper run_context, GO finale, ramo o commit in questa adozione.

## Validazione e limiti

Verificare frontmatter, link locali, coerenza tra AGENTS/skill/protocollo/workflow/
template, delta documentali e snapshot di ripresa. Casi di lettura: errore ordinario
si corregge; costo nuovo si sospende; NO_GO locale produce fix; review fix resta
indipendente; tree modificato atteso non è blocco; fedeltà/essentiali mancanti non
sono backlog automatico. Questi controlli non sono una simulazione delle review.

Alla prossima consegna misurare passaggi di ruolo e tempo al risultato, oltre ai
criteri tecnici. Non dichiariamo già dimostrato un risparmio o la correttezza del
codice. Fonte operativa: [protocollo](run-lifecycle.md), decisione [ADR0007](../decisions/0007-supervised-development-runs.md).
