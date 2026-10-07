# Arbitrato — piano — run-a001-fase0-uv — r001

- Data: 2026-10-02, Europe/Rome. Fase roadmap: **0.1**, migrazione uv.
  Il titolo «fase 2» nella consegna in chat non avvia la fase 2 della roadmap.
- Supervisore: Codex/OpenAI, famiglia GPT-6 indicata dalla sessione; modello
  specifico e ID chat non esposti. Ruolo esclusivo: supervisione/arbitrato.
- Oggetto: [piano r001](../plans/plan-r001.md), SHA-256
  `7f9dd0a2ded10c40ef9e3a42ae1424fb08a23059384468ea160dd7173fd3e8fb`.
- Snapshot esaminato: [plan-r001.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), branch
  `feature/run-a001-uv`, HEAD/dev/merge-base
  `66ba82200e5def5a4db76f9bafccb0731b506091`, impronta worktree
  `111f2830bfbef9b3bf816b02ab8818e8daf2d8a272f37b05e9fc565a004a7b60`.
- Report [ChatGPT](../reviews/review-plan-r001-chatgpt.md): **GO**, GPT-P001 opzionale.
  Autore effettivo dichiarato Codex/OpenAI, famiglia GPT-6, ruolo revisore ChatGPT
  assegnato dall'utente; non viene descritto come interfaccia ChatGPT Web.
- Report [Claude](../reviews/review-plan-r001-claude.md): **NO_GO**, CLA-P001/P002
  bloccanti e CLA-P003–P010 non bloccanti. Provider/modello dichiarati Anthropic,
  Claude Opus 5.5 (`claude-opus-5-5[1m]`), Claude Code; ID chat non esposto.
- Esito del supervisore: **NO_GO sul piano r001**. L'adozione di uv resta approvata;
  nessun GO sull'implementazione o autorizzazione a eseguire il piano r001.

## Validità delle consegne

Entrambi i report e checkpoint sono presenti e dichiarano chat indipendenti,
senza lettura della review concorrente o dell'arbitrato. Provider/interfacce
effettivi sono registrati come dichiarati, senza inventare certificazioni o ID.
L'utente ha consegnato i due esiti come review indipendenti; nessun revisore
è stato impersonato dal supervisore. Gli hash identificano i file, non provano
crittograficamente l'indipendenza o l'identità del modello.

Oggetto, HEAD/base, hash del piano e impronta coincidono nei due report. Le
evidenze registrano MATCH pre/post; il supervisore ha riconfermato MATCH prima
e dopo i propri probe, prima degli aggiornamenti documentali di arbitrato.
Le sei modifiche documentali attese erano invariate. I 17 file dei revisori sono
identificati in [identity-and-probes.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Le prove del pianificatore e i risultati storici dei test restano distinti dalle review.

L'arbitrato usa il protocollo e tutti e sette i criteri del brief. Non decide
per maggioranza fra GO e NO_GO: verifica ogni rilievo e le prove disponibili.

## Decisione sui rilievi

Tutti i rilievi accolti sono **da trattare nel piano r002**, non già risolti nel
codice. La severità riveduta dal supervisore è esplicita; le azioni non autorizzano
implementazione prima della nuova doppia review e del relativo arbitrato.

| ID | Decisione e blocco | Motivazione/evidenza | Azione e criterio di chiusura nel piano r002 |
| --- | --- | --- | --- |
| CLA-P001 | **Accolto, bloccante A1** | La descrizione riguarda solo PYENV_VERSION impostata. Il probe dello shim pyenv 2.6.12 conferma che il pin 3.12.13 assente da pyenv può selezionare il Python di sistema senza avviso. Non è una modifica globale di pyenv, ma cambia l'interprete dei comandi locali e va resa esplicita. Le invocazioni uv managed non risolvono da sole l'ambiguità di python3 | Un percorso utente managed univoco, pin motivato e matrice prima/dopo con/senza PYENV_VERSION per python3, uv run e interprete della venv. V1 osserva sys.executable/origini, non soltanto il numero di versione. Non cambiare pyenv globale |
| CLA-P002 | **Accolto, bloccante A4** | V6 usa solo dev mentre i nuovi test wrapper importano FastAPI dall'extra marker-server. Packaging sotto tests/ non ha una separazione esecutiva definita; il nuovo subprocess -m richiede il package nell'interprete. Il GO ChatGPT non tratta queste condizioni concrete | Matrice per ogni file di test: strato, dipendenze, installazione del progetto, comando, rete e prerequisiti. Comandi esatti per veloce/API/packaging/governance, inclusa raccolta; packaging non avvia build/rete implicitamente nel veloce. Deselezioni esplicite con conteggi, nessuno skip che nasconda una prova essenziale |
| CLA-P003 | Accolto, non bloccante | PyPI espone una wheel 1.10.2 con hash. Il SHA Git già identifica il sorgente e r001 prevede build constraints: Git non è automaticamente irriproducibile. La wheel può ridurre la build, ma versione/hash non provano equivalenza col tag | Confrontare wheel PyPI e Git, scegliere una fonte e motivarla. Per wheel: provenienza, metadata/hash e contratto da verificare; per Git: backend esatto e prova di build constraints. Motivare torch/transitive come candidati non collaudati; non copiare il lock upstream senza verifica |
| CLA-P004 | Accolto, non bloccante; precisazione di correttezza richiesta | Oggi load_dotenv all'import precede apply_env_overrides; spostarlo nel converter può arrivare dopo gli override. P3 non specifica il punto né un osservabile sui valori applicati | Caricamento dal workspace prima degli override, precedenza della shell e differenza dalla ricerca legacy espliciti. V5 osserva un valore applicativo deterministico derivato da .env e poi l'override della shell, senza credenziali |
| CLA-P005 | **Accolto, elevato a bloccante A3 e dati non fidati** | Il subprocess proposto con -m e CWD dati esegue un modulo omonimo dal workspace. Probe sintetico e documentazione Python confermano il comportamento; la fixture priva di .py lo nasconde. L'introduzione di questa esposizione viola il principio del repository sui dati non fidati | -P o isolamento equivalente nei subprocess; prova con sentinelle omonime nel CWD e origini dei moduli installati. Esplicitare il trattamento di PYTHONPATH e la compatibilità con l'avvio dai sorgenti. Il workspace dati non deve fornire codice alle fasi installate |
| CLA-P006 | Accolto, non bloccante | docker-compose-build.yml usa il medesimo Dockerfile con vecchi mount/cache e non è nominato in P6/V7 | Aggiornarlo, deprecarlo o rimuoverlo con motivazione e riferimenti coerenti. Se resta, config esplicita con -f e verifiche di profilo/mount/cache senza avvio implicito |
| CLA-P007 | Accolto, non bloccante | Serve una motivazione della distribuzione/base, oltre al pin. Lo stato Debian bookworm è confermato; digest e manutenzione dei tag Docker riportati dai revisori restano da riconfermare nativamente | Motivare base e patch, verificare native sulla release scelta e definire politica di aggiornamento con V7/V8. Nessun digest copiato da un riassunto web o promessa di supporto non misurato |
| CLA-P008 | Accolto, non bloccante | A5 richiede istruzioni coerenti; epilogo help e comandi README obsoleti sono presenti. Il comando pytest di AGENTS è corretto per il legacy attuale, ma deve essere coordinato con il nuovo flusso prima della consegna uv | Inventario dei comandi README/guide/help con stato provato, illustrativo, da correggere o rimuovere; perimetro mirato. Indicare chi aggiorna l'istruzione operativa AGENTS nel piano approvato. Nessuna modifica ora a parser o istruzioni dell'applicazione |
| CLA-P009 | Accolto, non bloccante | Baseline e nuovo ambiente possono differire per Python/dipendenze; lo script chunk standalone usa cleaned mentre l'orchestratore passa validated | Argomenti e fixture equivalenti per i due percorsi, tiktoken/versioni registrati. Se emerge una differenza, confronto incrociato controllato prima di attribuirla; nessuna normalizzazione del contenuto o nuovo algoritmo |
| CLA-P010 | Accolto come limite operativo; **V10 rinviata non bloccante per la toolchain** | Il mandato attuale è supervisione/pianificazione e non comprende acquisizione di pesi o inferenza. V7/V8 restano obbligatorie per A6; non dimostrano local_marker operativo | R002 dichiara V10 non autorizzata/non eseguita nel mandato corrente, limite local_marker non verificato dopo la migrazione e istruzioni separate. Definisce il punto in cui presentare costi/download/hardware per un eventuale mandato operativo; nessuna domanda o download implicito in questa fase |
| GPT-P001 | Accolto, non bloccante | La prova installata può produrre un solo chunk e non esercitare overlap; il corpus più lungo è un'aggiunta circoscritta al confronto già previsto | Fixture deterministica che produca almeno due chunk nella baseline e nella wheel con stessi argomenti; confronto di ordine, contenuto, metadati e overlap, preservando l'inventario delle perdite legacy |

La nota minore su `--no-registry` è corretta: il flag riguarda Windows. R002 ne
precisa il ruolo senza presentarlo come protezione necessaria su Linux.

## Motivazione complessiva e limiti

CLA-P001 e CLA-P002 sono blocchi validi e non coperti dalla motivazione del GO
ChatGPT. CLA-P005 viene elevato perché introduce esecuzione di codice dalla
directory dei dati; una correzione piccola non rende il requisito facoltativo.
Il supervisore conferma quindi NO_GO, senza invalidare l'approvazione dell'obiettivo
uv né richiedere di cambiare la nuova architettura.

Gli altri otto rilievi sono accolti nel giro r002 come precisazioni circoscritte.
Nessun rilievo è respinto per l'autorità del revisore o marcato risolto senza prova.
Non sono richiesti lock/build/test prima della review r002 per simulare risultati:
il piano deve specificare verifiche future sufficienti, e distinguere letture,
ipotesi, prove obbligatorie e prove rinviate. V7/V8 restano essenziali; V10/V11
non diventano automaticamente autorizzate e non possono essere chiamate PASS.

Conferme e fonti sono in [sources.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
e [pyenv-shim-probe.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Nessuna migrazione, suite, risoluzione lock, installazione, build, conversione,
startup, download di modelli o operazione Git di integrazione. I probe temporanei
sono distinti dai test applicativi e non collaudano la migrazione.

## Passaggio successivo

Stato della run: **PLANNING**, giro r002. Prompt:
`temp/run-a001-fase0-uv/prompts/02-planning-r002.md`, per una nuova chat pianificatrice.
Output attesi: plans/plan-r002.md, evidence/planning-r002/ e
handovers/planning-r002.md. Snapshot d'ingresso: planning-context-r002.

Conservare piano, due review, checkpoint e snapshot r001 senza sovrascriverli.
Gli aggiornamenti documentali dell'arbitrato rendono storico il precedente
worktree plan-r001; il giudizio resta riferito all'identità verificata sopra.
Il nuovo contesto congela anche report, evidenze, arbitrato e prompt r002.
Alla consegna r002 serviranno un nuovo snapshot del piano, due review in nuove
chat e un nuovo arbitrato. Nessun prompt di implementazione prima del relativo GO.
