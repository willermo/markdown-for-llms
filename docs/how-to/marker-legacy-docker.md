# Preparare e verificare Marker in Docker

Build CPU, contratto offline e inferenza sono verifiche distinte. Servono Docker,
Compose/buildx, il lock universale e spazio esplicito per OCI/APT/wheel/venv.
Non avviare servizi per verificare build o contratto. Il client base non acquisisce
ML implicitamente. Gli esiti delle run sono nei registri di sviluppo.

## Input pubblici e provenienza

Usare manifest **linux/amd64 pinned** delle fonti configurate
`registry-1.docker.io/library/python` (CPython3.12.13-slim-bookworm) e
`ghcr.io/astral-sh/uv` (0.10.10), non tag mobili o digest illustrativi. La
manutenzione delle basi richiede scelta esplicita e ripetizione delle prove.
Scegliere uno snapshot Debian/bookworm e security disponibile, con firme attive.
`RUN_PYTHON_IMAGE`, `RUN_UV_IMAGE` sono riferimenti `@sha256` verificati con
imagetools; `RUN_APT_SNAPSHOT` è il timestamp `YYYYMMDDTHHMMSSZ` scelto.

Il tool versionato `scripts/diagnostics/run-a001-fase0-uv/prepare_docker_inputs.py`
ha due operazioni: **supply** e **context**. Richiede il dev managed verificato
3.12.13 con packaging, `-I -B`, repo/input/output espliciti e directory nuove
pubbliche fuori dai dati operativi e da Git. Non dipende da temp, HOME personale
o una run. La receipt è un sibling dell'output, conservata anche sui FAIL.

Supply deriva l'export uv nativo locked CPU/server senza dev/cu126, seleziona
wheel compatibili con l'ABI, verifica SHA/size del lock, ZIP/payload/RECORD.
L'unica sdist ammessa è EbookLib0.18 con backend84 esplicito hash-pinned;
requisiti ignoti fermano la preparazione/build. Per APT esporta il keyring dalla
base pinned ferma, verifica gpgv InRelease e il SHA/size dei Packages, esegue
il resolver apt-get nativo offline in container proprio e verifica i .deb
selezionati contro quei Packages firmati. Il container monta soltanto quel
repository pubblico readonly, networknone/cap-dropALL/no-new-privileges.

Per un clone nuovo aggiungere **--acquire** al comando supply soltanto dopo
ammissione delle risorse. Acquisisce sole URL del lock/fonti configurate e
redirect verificati, OCI pinned e snapshot APT; ha deadline, storage e rete
finiti. Il traffico OCI è un upper dichiarato, non misura wire. Per riuso locale
fornire --wheel-dir/--apt-repository: controlla nuovamente tutto prima dell'uso,
senza fidarsi di copie opache o vecchie receipt. Senza --acquire un input mancante
è FAIL, non download per riparare una build. Pesi/font remoti non sono supply.

Esempio di **riuso** di dati pubblici già disponibili; per la prima preparazione
omettere i due input di riuso e scegliere esplicitamente --acquire e quote:

```bash
"$RUN_DEV_PY" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/prepare_docker_inputs.py" supply   --repo "$RUN_REPO" --output "$RUN_INPUTS" --uv "$RUN_UV"   --wheel-dir "$RUN_PUBLIC_WHEELS" --apt-repository "$RUN_PUBLIC_APT"   --python-image "$RUN_PYTHON_IMAGE" --uv-image "$RUN_UV_IMAGE"   --apt-snapshot "$RUN_APT_SNAPSHOT" --network-budget-bytes 0 --reuse-in-place
"$RUN_DEV_PY" -I -B "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/prepare_docker_inputs.py" context   --repo "$RUN_REPO" --source "$RUN_SOURCE_MANIFEST" --mode standalone --output "$RUN_IMAGE_CONTEXT"
```

`--mode standalone` richiede S standalone generato con la [guida uv](ambiente-uv.md);
per prove di run selezionare official e S dello snapshot corrente. Nessun S/output
futuro nei propri input. Context verifica S corrente e copia soltanto dieci moduli,
metadata/lock/constraints/README/licenza, Dockerfile/.dockerignore e diagnostici
ammessi. Genera image-source.json **nella copia usa e getta**, mai nel clone.
Sentinelle sintetiche nuove per .env/.git/venv/temp/config/dati/cache verificano
il filtro effettivamente consumato dal builder. Nessuna copia ricorsiva del clone.

## Build Compose offline

Entrambi i Compose richiedono RUN_IMAGE_CONTEXT e RUN_VERIFIED_WHEELS e RUN_VERIFIED_APT (directory verificate dal tool, output
supply), basi pinned, timestamp APT e RUN_HOST_NETNS=`readlink /proc/self/ns/net`.
RUN_IMAGE_TAG identifica una nuova immagine di prova. RUN_DATA_ROOT può indicare
un workspace esterno assoluto; in assenza, il path relativo dei source_documents
si risolve rispetto al file Compose, **non** rispetto al contesto usa e getta.

```bash
docker compose -f "$RUN_REPO/docker-compose.yml" config
docker compose -f "$RUN_REPO/docker-compose-build.yml" config
docker compose -f "$RUN_REPO/docker-compose.yml" -f "$RUN_REPO/docker-compose.gpu.yml" config
docker compose --progress plain -f "$RUN_REPO/docker-compose.yml" build marker-api
```

build.network è none; additional_contexts fornisce verified-wheels e verified-apt locali.
Il secondo Compose ha lo stesso build risolto; il runtime può avere differenti
limiti di memoria, che non sono una misura di throughput. Override GPU è soltanto
config statica: non avviare CUDA/device o inferenza per verificarla.

Dockerfile usa APT firmato locale readonly soltanto nel RUN, poi export uv
locked/pip sync hash-locked offline da supply locale, no-build-isolation e
constraints84. Backend/import/get_requires EbookLib sono attestati prima della
build e i veri hook nativi tracciati. image_package.py costruisce sdist e wheel
dalla sdist, audit B/RECORD prima installazione, canonica e I full builder.
Runtime contiene venv/prove, senza src, UV, supply APT/wheel, cache build, .deb o
compilatori. Setuptools84 è una dipendenza runtime Torch del lock, non gruppo dev.

## C-docker permanente

Preparare un'immagine locale per ID sha256 e impostare RUN_IMAGE,
RUN_IMAGE_MANIFEST=image-source.json della copia, RUN_IMAGE_INPUTS=receipt supply,
RUN_BACKEND_WHEEL, RUN_IMAGE_CONTEXT_RECEIPT (receipt del producer contesto), RUN_TEST_OUTPUT (output nuovo), RUN_SOURCE_MANIFEST e
RUN_INSTALLATION_RECEIPT correnti. Nel clone normale selezionare
RUN_TEST_MODE=standalone; official richiede snapshot corrente. Per C-docker
la selezione è `RUN_TEST_MODE=official|standalone` nell’ambiente: il solo flag
pytest `--test-mode` non seleziona il driver Docker. Scelte discordanti
sono rifiutate. Il daemon locale
è un perimetro esplicito distinto dal runner che lo maschera nei test host.

```bash
"$RUN_DEV_PY" -I -B -m pytest "$RUN_REPO/tests/docker/test_marker_contract.py" -q
```

Il test invoca run_docker_contract.py versionato con CLI repo/input/output.
Container per ID/pullnever/networknone, nessun mount, cap-dropALL,
no-new-privileges, healthcheck disabilitato, pids128/mem2GiB/cpu2. Export streaming
del filesystem fermo: startup/.pth/customize/pyvenv/backend84 controllati **prima**
di Python. Stdin fidato; full I runtime/RECORD/canonica prima import, versioni
esatte del grafo locked e native APT, nove sottocasi offline invariati: CPU/import,
signature/provider full/WeasyPrint sintetico/MarkdownOutput/wrapper/errori/
constructor mock/cache-font. Niente create_model_dict reale, download, inferenza
opzioni GPU o normale CMD. Receipts conservano input/argv/esiti/processi/container
propri anche sui FAIL; cleanup soltanto degli oggetti diagnostici propri.

## Limiti operativi

Startup con pesi, inferenza PDF/non-PDF e GPU richiedono fonti/hash/licenze,
RAM/disco/cache/durata/hardware e un perimetro distinto. Non sono test veloci.
/health misura liveness anche dopo caricamento modelli fallito; il wrapper
restituisce immagini vuote e può riportare opzioni senza applicarle. Contratto
mock e WeasyPrint non certificano fedeltà Marker o servizio host.

Cache modelli=/data/cache/models, HF=/data/cache/huggingface, font=/data/fonts,
TMPDIR=/data/tmp; FONT_PATH e la mappa Surya sono espliciti. Non riusare volumi
operativi nelle prove. Nessun prune/down con rimozione globale dei volumi.
Aggiornamento/deploy richiede un perimetro proprio e nuove prove dipendenti.

[Ambiente](ambiente-uv.md), [reference](../reference/toolchain-legacy.md), [indice](../README.md).
