# Prompt — run-a001-fase0-uv — Pianificazione — r001

Agisci come pianificatore in una nuova chat distinta dal supervisore. Produci il
piano verificabile della migrazione Python a uv; non implementare la migrazione.
Lavora dalla radice del repository `/home/davide/workarea/markdown-for-llms`.
L'adozione di uv è già approvata. Non chiedere una nuova autorizzazione e non
anticipare sviluppo della nuova applicazione o benchmark dei motori.

## Recupero e input

Leggi nell'ordine:

1. `AGENTS.md`, `temp/HANDOVER.md` e `temp/PROJECT-CONTEXT.md`.
2. `temp/run-a001-fase0-uv/STATE.md`, `HANDOVER.md`, `brief.md` e questo prompt.
3. `.agents/skills/manage-implementation-run/SKILL.md` e
   `documentation/development/run-lifecycle.md`, per le responsabilità del ruolo.
4. `documentation/README.md`, `documentation/CHANGELOG.md`,
   `documentation/decisions/README.md`, ADR 0006/0007 e fase 0.1 della roadmap.
5. `temp/run-a001-fase0-uv/evidence/supervisor-preparation-r001.md` e
   `documentation/development/templates/plan.md`.
6. Solo i sorgenti pertinenti: `setup.py`, `requirements.txt`, `.gitignore`,
   `Dockerfile`, `docker-compose.yml`, sezioni di installazione/avvio/test del
   `README.md`, `config.py`, `master_workflow.py`, `unified_converter.py`,
   `marker_api_server.py` e i moduli richiamati. Per i test leggere
   `tests/conftest.py`, unit/integration pertinenti e `tests/governance/`;
   `test_pipeline.py` alla radice serve a valutare il conflitto di discovery.

Verifica subito `git status --short --branch`, branch, HEAD e base `dev`.
Branch atteso: `feature/run-a001-uv`; HEAD/base verificati dal supervisore:
`66ba82200e5def5a4db76f9bafccb0731b506091`. Bootstrap già integrato e pubblicato.
Sono previste modifiche documentali non committate di supervisione, incluse nel
contesto; non eliminarle, committarle o scambiarle per implementazione uv.

Controlla l'identità degli input prima e dopo il lavoro:

```bash
python3 scripts/run_context.py verify run-a001-fase0-uv --label planning-context-r001
```

Questo è uno snapshot del contesto di pianificazione, senza GO. Se risulta STALE,
indica i campi e le differenze al supervisore prima di consegnare un piano riferito
a una base diversa; non sovrascrivere lo snapshot. Se mancano file o accesso al
repository, richiedi il trasferimento necessario senza inventarne il contenuto.

## Obiettivo e confini

Ambiente Python e dipendenze riproducibili con uv, package installabile e avvio
legacy corretto anche fuori dai sorgenti; documentazione e Docker coerenti con il
perimetro approvato. Preserva il comportamento della pipeline salvo le correzioni
necessarie a packaging e avvio. Nessuna nuova Web UI, riscrittura del dominio,
scelta definitiva OCR, modifica globale di pyenv o altri progetti della macchina.

Il brief è la fonte dei sette criteri iniziali di accettazione. Associa ogni
criterio a interventi, prove, risultati attesi e limiti; non ridurne tacitamente
il perimetro per ottenere un GO.

## Compito del pianificatore

Proponi passi ordinati, file da modificare e alternative motivate per:

- `pyproject.toml`, backend di build, `uv.lock`, `.python-version` e `.venv/`
  ignorata. Motiva versione/range Python e versione uv riproducibile; chiarisci
  quale interprete useranno i comandi con `PYENV_VERSION` eventualmente attiva,
  senza modificare la configurazione globale dell'utente.
- Inventario delle dipendenze realmente usate: runtime, gruppo sviluppo e extra
  dei motori/server. Decidi la sorte di `setup.py` e `requirements.txt`, evitando
  fonti divergenti e commenti inline passati come requisiti. L'ambiente base e i
  test veloci devono evitare installazioni implicite della catena ML pesante.
- Transizione minima dei moduli flat, con elenco dei moduli distribuiti nella
  wheel e dei cinque entry point esistenti. `config.main` oggi manca, mentre la
  CLI è nel blocco `__main__`: pianifica come renderla invocabile preservandola.
- Avvio delle fasi orchestrate fuori dal repository: `master_workflow.py` usa
  `BASE_DIR = Path.cwd()` e percorsi di script. Distinguere directory dei dati e
  codice installato. Le prove devono avviare davvero una fase su fixture sintetica
  e controllare contenuto/output e codice di uscita, oltre a import e help.
- Compatibilità di script diretti, `python -m` e comandi installati. I test
  attuali inseriscono la radice in `sys.path`: non bastano a provare il packaging.
  Prevedi una prova della wheel non editable in directory temporanea estranea ai
  sorgenti, senza `PYTHONPATH` aggiunto o copie dei sorgenti per aggirare il problema.
- Docker e Compose: sincronizzazione dal lockfile, versione uv/base riproducibile,
  dipendenze native e Python distinte, `.dockerignore` per `.venv`, `temp`, segreti
  e dati locali. Il Dockerfile installa Marker da `master`: motiva un riferimento
  immutabile e la compatibilità con import, costruttore, provider e output del
  wrapper, oppure esplicita ciò che resta da risolvere. Non scegliere una versione
  arbitraria né considerare `/health` prova sufficiente di conversione riuscita.
- Strategia CPU/GPU ed eventuale ambiente separato del server Marker, entro ADR
  0006. Distingui riproducibilità delle dipendenze, build dell'immagine, pesi e prova
  reale del motore; indica quali verifiche sono essenziali per il GO finale e quali
  richiedono hardware/download con un passaggio operativo esplicito.
- Aggiornamenti mirati di README e guide uv in `docs/`, senza presentare la nuova
  applicazione come disponibile. Prevedi aggiornamento del changelog da parte
  dell'implementatore per il lavoro realizzato, con verifiche e limiti.

Ricontrolla le osservazioni del brief solo dove servono al piano. Se fai affermazioni
su opzioni/versioni o compatibilità di strumenti aggiornabili, usa documentazione
ufficiale e sorgenti primarie, con riferimenti e data nelle evidenze. Distingui dati
verificati, proposte e ipotesi da collaudare. Non generare pyproject/lockfile, creare
ambienti o correggere sorgenti come parte della pianificazione. Eventuali probe
leggeri vanno soltanto nella cartella locale della run e documentati.

## Verifiche da pianificare

Descrivi i comandi o le procedure concrete per ricreare un ambiente pulito dal
lockfile, build/install della wheel, import/entry point e fasi fuori dai sorgenti,
controlli delle dipendenze e coerenza delle istruzioni. Associa ogni prova a costo,
prerequisiti e risultato osservabile; prevedi come registrare un fallimento.

La suite legacy selezionata è `python -m pytest tests/ -q`, da tradurre nel flusso uv
proposto. I 67 test legacy e 6 test governance riportati nel bootstrap sono storici,
non risultati della migrazione. Il discovery dalla radice ha il conflitto noto dei
due `test_pipeline.py`: decidi un perimetro esplicito e una correzione proporzionata
se necessaria, senza chiamare la suite selezionata collaudo completo.

Le prove che eseguono le fasi su fixture devono confrontare il contenuto pertinente
prima/dopo la migrazione (testo, formule, numeri, riferimenti e ordine), senza
correggere in questa run i difetti legacy estranei al packaging. Consulta la skill
di fedeltà se definisci un confronto di conversione. Separa controlli senza modelli,
mock, integrazione Docker e motori reali. Nessuna chiamata cloud a pagamento,
benchmark o download pesante implicito nei test veloci; considera anche l'avvio del
server, che tenta di precaricare i modelli.

## Output obbligatori e consegna

- Piano: `temp/run-a001-fase0-uv/plans/plan-r001.md`, seguendo il template, con
  autore/provider/modello effettivamente noto, prompt di origine, branch/HEAD/base,
  fonti, passi, matrice di accettazione, rischi, migrazione e recupero.
- Evidenze eventualmente raccolte: `temp/run-a001-fase0-uv/evidence/planning-r001/`.
- Checkpoint: `temp/run-a001-fase0-uv/handovers/planning-r001.md`, con output,
  verifiche eseguite/non eseguite, incertezze e nessun processo lasciato implicito.

Non modificare STATE, handover comuni, indici, eventi o arbitrati: spettano al
supervisore. Non produrre review o GO; non eseguire commit, merge, push o promozioni.
Non sovrascrivere un piano r001 già presente: segnala il conflitto al supervisore.

Consegna in chat percorsi, decisioni proposte e limiti. Prossimo destinatario:
supervisore, che leggerà il piano, creerà lo snapshot comune e preparerà i due
prompt per review indipendenti ChatGPT e Claude. L'implementazione attende entrambe
le review e l'arbitrato GO sul piano identificato.
