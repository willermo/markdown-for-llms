# Preparare e aggiornare l'ambiente uv legacy

Per utenti e sviluppatori Linux. Usare managed locale e `uv.lock`, installazione
canonica e preflight corrente. Build, contratto Docker offline e inferenza
sono verifiche distinte; gli esiti delle run restano nei registri di sviluppo.
Questi comandi richiedono preparazione
e preflight; l'acquisizione di ML non è implicita.

Servono uv **0.10.10** identificato, Pandoc host e spazio per managed/cache/venv.
La patch CPython **3.12.13** e Setuptools **84.0.0** sono i pin della run. Le altre
minor Python non sono nel supporto dichiarato. Il pin locale può cambiare la
selezione dello shim pyenv; non installa Python in pyenv. Nessuna modifica globale
è necessaria. Il collaudo Windows resta futuro (path `Scripts` e `--no-registry`
per l'acquisizione Windows, non una protezione Linux).

In una subshell del clone, impostare i path assoluti. L'acquisizione Python è
esplicita, con sorgente/hash/build e consumo registrati prima dell'uso nella run.

```bash
(
  RUN_REPO="$(pwd -P)"
  export UV_PYTHON_INSTALL_DIR="$RUN_REPO/.venv-python"
  export UV_CACHE_DIR="$RUN_REPO/.cache/uv-fidelity/cache"
  uv python install 3.12.13 --no-bin
)
```

Dopo la preparazione esplicita, nella subshell di lavoro definire gli stessi
path, `UV_PYTHON_DOWNLOADS=never` e `UV_PROJECT_ENVIRONMENT` alla venv desiderata.
Acquisire il path reale `RUN_MANAGED_PY` dal catalogo locale, sotto `.venv-python`;
prima di importare il prodotto verificarlo con stdlib. Nessun Python globale è
accettato per il solo numero di versione.

```bash
"$RUN_MANAGED_PY" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/check_python_origin.py" --expected-managed "$RUN_REPO/.venv-python"
uv lock --check --python 3.12.13 --managed-python --no-python-downloads
```

La guardia persistente `python-downloads = "manual"` impedisce download Python
impliciti; non impone la directory e non protegge da override deliberati.
La shell senza variabili uv può trovare un managed esterno: quel percorso viene
rifiutato dal preflight. `uv run --no-sync` usa un ambiente già preparato.
`--project` seleziona il progetto mantenendo il CWD dei dati; `--directory` cambia
CWD. Non usare `uv lock --locked`: il controllo locale è `uv lock --check`.

Per ogni modifica, anche a un solo `.py` senza cambio di versione, preparare una
nuova directory di prove, ricostruire e confrontare i sorgenti. La configurazione
persistente `reinstall-package` e il flag forzano il rebuild del progetto locale;
non aggiornano ambienti inattivi. Il flusso normale senza run usa
`.cache/uv-fidelity/`, esclusa da Git, e S con scope `standalone`.

Prima di S, compilare `package-inputs.json`: schema 1, scope `package-inputs`,
`files` con path assoluti/bytes/sha256 degli artefatti realmente acquisiti;
`backend` con name `setuptools`, version `84.0.0`, `artifacts` (path della wheel
Setuptools), `config_settings: {}` e `environment: {}`. L'inventario include il
lock e l'interprete/stdlib e i pacchetti di ingresso pertinenti. Il diagnostico
rifiuta backend/input ignoti. Non inventare digest per file ancora da acquisire.

`RUN_PROOF` è una directory nuova sotto `.cache/uv-fidelity/` e `RUN_ENV` la venv
assoluta. Nella run sostituire `--standalone` con `--snapshot` e il percorso
dello snapshot di esecuzione creato dall'implementatore secondo il protocollo;
lo snapshot comune delle review resta del supervisore. S standalone non è
prova ufficiale. Dopo S gli input build,
compreso README, restano stabili fino all'esito.

```bash
"$RUN_MANAGED_PY" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/make_source_manifest.py" --repo "$RUN_REPO" --standalone --input-inventory "$RUN_PROOF/package-inputs.json" --output "$RUN_PROOF/S.json"
uv build --python 3.12.13 --managed-python --no-python-downloads --build-constraints "$RUN_REPO/build-constraints.txt" --out-dir "$RUN_PROOF/dist"
```

`uv build` produce sdist e wheel dalla sdist. Identificare i due singoli artefatti
come `RUN_SDIST` e `RUN_WHEEL`; non scegliere una vecchia wheel da `dist`.
Controllare archivi prima dell'installazione, nell'interprete managed già provato.

```bash
"$RUN_MANAGED_PY" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py" --repo "$RUN_REPO" --source-manifest "$RUN_PROOF/S.json" --sdist "$RUN_SDIST" --wheel "$RUN_WHEEL" --profile base --expected-managed "$RUN_REPO/.venv-python" --archives-only --receipt "$RUN_PROOF/B-check.json"
UV_PROJECT_ENVIRONMENT="$RUN_ENV" uv sync --locked --no-default-groups --no-editable --reinstall-package markdown-for-llms --python 3.12.13 --managed-python --no-python-downloads
uv pip install --python "$RUN_ENV/bin/python" --no-deps --no-build --reinstall-package markdown-for-llms "$RUN_WHEEL"
"$RUN_ENV/bin/python" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py" --repo "$RUN_REPO" --source-manifest "$RUN_PROOF/S.json" --sdist "$RUN_SDIST" --wheel "$RUN_WHEEL" --profile base --expected-managed "$RUN_REPO/.venv-python" --receipt "$RUN_PROOF/I.json"
"$RUN_ENV/bin/python" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/check_preflight.py" --repo "$RUN_REPO" --source "$RUN_PROOF/S.json" --receipt "$RUN_PROOF/I.json" --standalone
```

Il gate confronta dieci moduli, input build, archivi, RECORD e installazione;
un mismatch ferma l'avvio. Dopo modifiche generare nuovi S/B/I e ripetere le
prove dipendenti; conservare esiti precedenti senza attribuirli al codice nuovo.
Gli extra RECORD creati da uv, incluso `uv_cache.json`, sono controllati con
hash/dimensioni e percorsi confinati. Per una wheel locale uv può non registrare
un hash in `direct_url.json`: I verifica l'URI, il digest di B e tutti i bytes
della wheel installata, senza dichiarare presente un hash omesso dall'installer.
In questa run il probe di cache modifica soltanto `exceptions.py` in una copia,
verifica il negativo prima del sync e il positivo prima della reinstallazione
canonica, poi ripristina la copia. Non pulire cache globali per farlo passare.

Per dev aggiungere `--group dev` al sync; per API mock aggiungere anche
`--extra marker-server`, in una venv distinta. Ogni ambiente riceve per ultimo la
stessa wheel canonica e il suo preflight (profile `api` per API). Gli extra
`marker-cpu` e `marker-cu126` sono incompatibili e hanno preparazione pesante
separata. Il lock universale può richiedere metadata degli extra anche se il
client non installa ML: contabilizzare rete/cache. Ogni uv pip che può costruire
usa `--build-constraints`; installare una wheel canonica usa `--no-build`.

Per la prova esterna preparare una venv nuova con runtime esportato dal lock
hash-locked, fuori dal clone. La suite non deve prepararla da sola:

```bash
uv export --locked --no-default-groups --no-emit-project --format requirements.txt --output-file "$RUN_PROOF/runtime.txt"
uv venv --python 3.12.13 --managed-python --no-python-downloads "$RUN_ENV"
uv pip sync --python "$RUN_ENV/bin/python" --build-constraints "$RUN_REPO/build-constraints.txt" --require-hashes "$RUN_PROOF/runtime.txt"
```

Poi installare la wheel canonica e verificarla come sopra. In IDE selezionare
`.venv/bin/python` assoluto; un esempio locale è in
[settings.uv.example.json](../../.vscode/settings.uv.example.json). Copiarlo solo
nelle proprie impostazioni workspace e completare l'installazione verificata;
nessuna impostazione globale cambia. I figli delle fasi usano lo stesso
`sys.executable`, con `-I -B -m`, e il workspace dei dati.

## Test locali espliciti senza run

In un clone normale senza `temp/`, selezionare **standalone** deliberatamente.
L'assenza di temp non cambia automaticamente la modalità. Prima di S, creare
un inventario con `make_input_inventory.py --repo "$RUN_REPO" --managed
"$RUN_REPO/.venv-python" --backend-wheel "$RUN_BACKEND_WHEEL" --uv "$RUN_UV"
--tokenizer-cache "$TIKTOKEN_CACHE_DIR" --output "$RUN_PROOF/package-inputs.json"`,
usando il managed verificato con `-I -B`. I path dei tool sono sotto
`scripts/diagnostics/run-a001-fase0-uv/`; il nome storico non impone una run.
`RUN_PROOF` è una directory nuova ignorata, per esempio `.cache/uv-fidelity/`.

Generare S con `make_source_manifest.py --standalone --repo "$RUN_REPO"
--input-inventory "$RUN_PROOF/package-inputs.json" --uv "$RUN_UV" --output
"$RUN_PROOF/S.json"`. Ricostruire sdist/wheel, fare audit, installare la canonica
e creare I con `verify_distribution.py` e gli stessi repo/S/archivi:
I deriva lo scope da S, senza un flag `--standalone` del verificatore.
Il Git locale del clone è registrato anche se non esiste una branch dev;
una copia sintetica inizializzata senza commit registra HEAD/dev null.

Per i test, `RUN_TEST_MODE=standalone` oppure `--test-mode standalone` sceglie il
gate locale; valori invalidi o scelte discordanti sono errori. `run_tests.py`
accetta la stessa selezione e richiede una receipt R corrente, con preflight
prima di pytest, cache/Pandoc pronti e guardie rete prima della collection.
La selezione veloce è quella della reference. I risultati S/I/E rimangono
**standalone**, senza diventare prove ufficiali. Il default **official** esige
S run-stage e snapshot verificato del run_id valido dichiarato.
S o I stantii/mismatched fermano la prova prima degli import applicativi:
ricostruire e verificare esplicitamente, senza sync nascosto nella collection.

Per **C-docker** selezionare invece `RUN_TEST_MODE=official|standalone`
nell'ambiente: il solo flag pytest `--test-mode` non seleziona la modalità
del driver Docker. Scelte discordanti restano errori. Prerequisiti e procedura
sono nella [guida Docker CPU](marker-legacy-docker.md).

[Reference e test](../reference/toolchain-legacy.md),
[Docker](marker-legacy-docker.md), [indice](../README.md).
