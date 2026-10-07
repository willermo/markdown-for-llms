# Prompt — run-a001-fase0-uv — Pianificazione r002 dopo NO_GO r001

Agisci come pianificatore in una **nuova chat distinta dal supervisore**. Produci
la revisione r002 del piano uv, senza implementare la migrazione. Repository:
`/home/davide/workarea/markdown-for-llms`; fase roadmap **0.1**. L'adozione di uv
è già approvata. Il NO_GO riguarda il piano r001, non l'obiettivo della migrazione.

## Recupero e identità degli input

Leggi nell'ordine:

1. `AGENTS.md`, `temp/HANDOVER.md`, `temp/PROJECT-CONTEXT.md` e
   `temp/run-a001-fase0-uv/STATE.md`/`HANDOVER.md`. Questo prompt assegna il ruolo
   pianificatore anche se il recupero globale è scritto per il supervisore.
2. `.agents/skills/manage-implementation-run/SKILL.md`,
   `documentation/development/run-lifecycle.md` e template
   `documentation/development/templates/plan.md`.
3. `temp/run-a001-fase0-uv/brief.md`, `prompts/02-planning-r001.md`, piano
   `plans/plan-r001.md` e checkpoint `handovers/planning-r001.md`.
4. `arbitrations/arbitration-plan-r001.md`, **specifica vincolante del giro**;
   i due report `reviews/review-plan-r001-chatgpt.md` e
   `reviews/review-plan-r001-claude.md`, i rispettivi checkpoint e le evidenze
   richiamate. Il pianificatore può leggere entrambe le review precedenti.
5. `evidence/planning-r001/findings.md`, `checks.json`, inventario/cataloghi/help
   pertinenti; `evidence/supervisor-arbitration-plan-r001/sources.md`,
   `identity-and-probes.json` e `pyenv-shim-probe.json`.
6. Indice architetturale, changelog, indice ADR, ADR 0001/0006/0007 e fase 0.1
   della roadmap; soltanto sorgenti/test necessari ai rilievi, compreso
   `docker-compose-build.yml` e le istruzioni README/help interessate.

I percorsi della run senza prefisso sono relativi a temp/run-a001-fase0-uv/.
Verifica prima e dopo il lavoro:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
python3 scripts/run_context.py verify run-a001-fase0-uv --label planning-context-r002
```

Branch atteso: feature/run-a001-uv. HEAD/base/merge-base:
`66ba82200e5def5a4db76f9bafccb0731b506091`. Sono previste sei modifiche documentali
non committate di supervisione, incluse nel nuovo contesto. Il risultato atteso è
MATCH per **planning-context-r002**, senza GO. Non sovrascrivere gli snapshot.
plan-r001 e planning-context-r001 sono storici: gli aggiornamenti documentali
dell'arbitrato appartengono al nuovo contesto e non modificano piano/review r001.

Se l'identità cambia, segnala campi/differenze al supervisore prima di consegnare
un piano su input diversi. Se manca il filesystem, richiedi i file necessari,
compresi temp/ e modifiche non committate: un manifest non contiene il codice.

## Obiettivo invariato e modifiche richieste

Conserva tutti e sette i criteri A1–A7 del brief, il perimetro di migrazione legacy
e la separazione fra prove locali, mock, Docker e inferenza reale. Nessuna nuova
Web UI, fase 2 applicativa, riscrittura di algoritmi o scelta OCR definitiva.
R001 resta immutato. R002 è un piano completo utilizzabile da solo, con una matrice
di tracciabilità per **tutti gli undici ID accolti nell'arbitrato**, indicando
sezioni, interventi e verifiche future. Non basta una lista di errata separata.

I tre blocchi da chiudere nel nuovo piano:

- **CLA-P001 — interprete/pyenv:** motiva il pin locale tenendo conto della
  selezione degli shim, con e senza PYENV_VERSION e versione non installata in
  pyenv. Distingui il Python diretto già in PATH da quello selezionato dallo shim;
  il probe controllato è pyenv-shim-probe.json. Definisci un percorso utente managed
  univoco (directory locale oppure predefinita, senza alternative implicite) e
  flag/variabili necessari. V1 controlla sys.executable/origini prima/dopo il pin,
  per python3, uv run e interprete venv; le prove managed future non sono già svolte.
  Mantieni pyenv e altri progetti invariati a livello globale.
- **CLA-P002 — strati e prerequisiti dei test:** per ogni test nuovo/mutato
  indica strato, dipendenze, package installato, comando esatto, prerequisiti e
  comportamento se mancanti. Risolvi l'import FastAPI dei test mock API quando
  V6 usa solo dev: extra marker-server o altra selezione esplicita, senza catena ML.
  Separa il controllo della wheel dalla suite veloce: build/installazioni sono
  preparazione dichiarata, la raccolta non deve provocare rete o build. Se scegli
  marker/deselezioni, evita che import mancanti falliscano prima della selezione;
  dichiara conteggi e nessuno skip silenzioso di verifiche essenziali. I subprocess
  -m dei test di integrazione richiedono il progetto installato nello stesso
  interprete. Specifica comandi per suite veloce, API, packaging, governance e
  discovery senza affidarti ai sys.path di conftest.
- **CLA-P005 — codice dalla directory dati:** usa -P o isolamento equivalente
  nei subprocess delle fasi installate. Motiva gestione di PYTHONPATH, verifica
  origine di moduli e transitive, e compatibilità con script diretti dal clone.
  V5 include un workspace con moduli sentinella omonimi che devono restare
  ineseguiti, preservando i dati/override e l'esecuzione del package installato.

Precisazioni accolte da integrare nello stesso giro, senza ampliamento del prodotto:

- **CLA-P003:** confronto motivato fra wheel PyPI Marker 1.10.2 e sorgente Git
  al commit noto. Scegli una fonte, chiarisci provenienza/hash/metadata e verifiche
  del contratto; se mantieni Git, fissa il backend con build constraints e prove
  ripetibili. Non sostenere che una wheel con hash sia di per sé equivalente al
  tag o che Git al SHA sia di per sé irriproducibile. Distingui i pin torch/transitive
  candidati da combinazioni collaudate; backend degli altri sdist restano da gestire.
- **CLA-P004:** punto esplicito di caricamento .env dal workspace **prima** degli
  override applicativi, precedenza della shell e cambio della ricerca legacy.
  Prova futura con valore .env che modifica un osservabile deterministico e override
  della shell prevalente. Usa solo variabili sintetiche innocue.
- **CLA-P006:** destino di docker-compose-build.yml; se resta, allineamento e
  comando config con -f, mount/cache/profilo verificati. Nessuno startup implicito.
- **CLA-P007:** base/distribuzione e patch motivate, native riferite alla release
  scelta, manutenzione e politica di aggiornamento con ripetizione V7/V8. Digest
  nativi prima dell'uso; non copiare valori estratti da riassunti web.
- **CLA-P008:** inventario completo dei comandi README/guide/help, classificando
  verificabile, illustrativo, da correggere o rimuovere. Perimetro mirato agli
  errori della migrazione/avvio; includi epilogo run_full_pipeline.py e opzioni
  inesistenti. Decidi la modifica operativa di AGENTS.md e il responsabile nel
  futuro piano approvato, mantenendo al supervisore i registri condivisi della run.
- **CLA-P009:** baseline chunk con stessi input/argomenti dell'orchestratore,
  compresa validated anziché il default cleaned. Versioni Python/dipendenze/tiktoken
  registrate; in caso di differenze, confronto incrociato circoscritto prima di
  attribuire la regressione. Nessuna normalizzazione dei contenuti.
- **CLA-P010:** V10 non è autorizzata nel mandato corrente. Rendi espliciti
  rinvio, limite local_marker non verificato dopo migrazione e istruzioni separate.
  Indica quando il supervisore presenterà un eventuale perimetro operativo per
  pesi/font, CPU, costi e hardware. Non chiedere adesso autorizzazioni per svolgere
  questa pianificazione. V7/V8 restano obbligatorie e il loro eventuale download
  pesante deve essere un passo operativo distinto dell'implementazione, non un test
  veloce. V10 torna essenziale se scostamenti semantici/incompatibilità lo richiedono;
  in quel caso riporta il bisogno al supervisore senza dichiarare il requisito passato.
- **GPT-P001:** fixture deterministica con almeno due chunk nella baseline e
  nella wheel, medesimi argomenti, confronto di sequenza/contenuto/metadati/overlap.
  Le eventuali perdite legacy restano inventariate e non vengono corrette qui.

Precisa anche che --no-registry riguarda Windows. Eventuali scelte di versione
diverse da r001 devono essere motivate e identificate come proposte della r002;
non cambiano tacitamente la decisione uv o i criteri del brief.

## Metodo, verifiche e output

Consulta fonti ufficiali/sorgenti primarie per affermazioni aggiornabili, con URL,
data e limiti. Sono ammessi controlli statici e probe standard-library confinati
alla run o a directory temporanee, senza modifiche globali. Mantieni distinto
quanto riprodotto localmente dai comandi da eseguire dopo il GO. Non generare
pyproject/lock, ambienti, codice applicativo, fixture implementative o guide uv;
nessuna installazione, risoluzione, build, suite, conversione o download pesante.

Consegna:

- `temp/run-a001-fase0-uv/plans/plan-r002.md`, completo secondo il template, con
  autore/provider/modello effettivi, origine, identità, perimetro, passi, A1–A7,
  prove, matrice degli undici rilievi, rischi e recupero.
- `temp/run-a001-fase0-uv/evidence/planning-r002/findings.md` e `checks.json`,
  con controlli realmente svolti, hash degli output, MATCH finale e limiti.
- `temp/run-a001-fase0-uv/handovers/planning-r002.md`, checkpoint del ruolo.

Non sovrascrivere un output r002 esistente; segnala il conflitto. Non modificare
stato/handover comuni, indici, eventi, arbitrati, report o snapshot: appartengono
al supervisore o ai rispettivi autori. Nessun commit, merge, push, deploy o GO.

Destinatario finale: supervisore. Consegna percorsi e limiti; serviranno snapshot
plan-r002, **due nuove chat di review indipendenti** e nuovo arbitrato. Il GO
ChatGPT r001 non si trasferisce automaticamente alla revisione r002.
