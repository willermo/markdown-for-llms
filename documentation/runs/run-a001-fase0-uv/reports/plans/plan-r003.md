# Piano — run-a001-fase0-uv — r003

- Autore/provider/modello: **Codex / OpenAI**, famiglia **GPT-6** indicata dalle
  istruzioni della sessione; identificatore specifico del modello e ID chat non
  esposti. Ruolo esclusivo: pianificatore, in questa nuova chat distinta dalla
  supervisione e dalle review r001/r002. Non sono state delegate review o altre attività.
- Data: **2026-10-02, Europe/Rome**. Fase roadmap: **0.1**.
- Origine: [prompt r003](../prompts/02-planning-r003.md), ricevuto dall'utente;
  giro di fix del [NO_GO r002](../arbitrations/arbitration-plan-r002.md), specifica
  vincolante. L'adozione di uv resta approvata.
- Identità: branch `feature/run-a001-uv`; HEAD, `dev` e merge-base
  **`66ba82200e5def5a4db76f9bafccb0731b506091`**.
- Ingresso: [planning-context-r003 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
  82 artefatti espliciti e 79 file, impronta worktree
  `cfa665ec5bc59fe0f6ff37988452eb6ccb5b271c90e5bcc8e0e5c95f7c80888e`.
  MATCH iniziale; esito finale e hash degli output in
  [checks.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md). MATCH non è un GO.
- Le sei modifiche documentali del supervisore sono conservate: changelog,
  indice architetturale, ADR 0006/0007, indice decisioni e roadmap. Il bootstrap
  è già integrato/pubblicato dall'utente; non ripeterne le operazioni Git.
- Requisiti: [brief A1–A7 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), ADR
  [0006](../../../../decisions/0006-python-toolchain-uv.md),
  [0007](../../../../decisions/0007-supervised-development-runs.md),
  [0001](../../../../decisions/0001-document-fidelity.md),
  [roadmap](../../../../roadmap.md) e
  [protocollo](../../../../development/run-lifecycle.md).
- SHA-256 del manifest d’ingresso: `ac95089d1715e1061e34af369763df12bd6c4ad39288d03376aee6a8b08b930c`.
  Piano r002 antecedente, letto integralmente: SHA-256
  `df23588de1247820a797e033ba7e93f60ed7d6b5f1f64e1948f012646039ca71`.
  [Transition r002 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
  identifica i soli sei metadata aggiornati; i 55 input e 20 output review sono
  conservati. Gli snapshot precedenti sono storici, nessun GO si trasferisce.
- Piano completo secondo il [template](../../../../development/templates/plan.md).
  R001/r002, le loro review e i loro arbitrati restano storici e immutati.
  Il pianificatore non emette GO né chiusure operative dei rilievi. Le sezioni P0–P7 e V0–V12 costituiscono la proposta autonoma di questo giro.

## Obiettivo e perimetro

Rendere ambiente, dipendenze e avvio della pipeline **legacy** riproducibili con
uv: pin Python esplicito, lock versionato, runtime/dev/motori distinti, wheel
completa e cinque entry point verificati fuori dai sorgenti. Separare il codice
installato dal workspace dei dati, mantenendo JSON, override, output e algoritmi
legacy salvo le correzioni necessarie a packaging e avvio. Allineare Docker,
entrambi i Compose esistenti, README, help e istruzioni operative.

Il documento Markdown completo con asset resta il risultato principale del
prodotto; questa migrazione non modifica il carattere opzionale dei derivati né
corregge le perdite legacy del cleaning/asset. Non introduce nuova Web UI, backend
applicativo/worker della fase 2, storage, nuovo schema del dominio, OCR definitivo,
fallback remoto, revisione AI, nuove strategie CLI o benchmark dei motori.
FastAPI in questo piano identifica esclusivamente il wrapper Marker già esistente.

L'intervento non modifica pyenv globale, PATH persistente, altri progetti o dati
privati. Nessun commit, merge, push, deploy o GO da parte del pianificatore.
**V10/V11 non sono autorizzate nel mandato corrente**; V7/V8 rimangono essenziali
per A6 nella futura implementazione, con un passo operativo distinto per i download
pesanti. In questa chat sono consentite solo letture, hash, AST e controlli documentali; non sono stati creati ambienti, lock o fixture applicative.

### Fonti e distinzione fra fatti e proposte

Le [evidenze r003 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md) registrano letture,
fonti primarie datate, inferenze e controlli reali. Il loro
[checks.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md) contiene gli hash degli input,
l’inventario corretto di 74 fence README (63 shell/Python/YAML e 11 esempi di
dati/output) e nove inline, i flag degli help congelati e i dieci moduli.
Il [catalogo r002 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md) resta storico:
i suoi 58 record sono corretti ma incompleti. Non correggere retroattivamente
quel catalogo o il primo confronto incompleto ChatGPT. I cinque blocchi indentati
e l’inline Git sono riconciliati con
[static-findings.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Tutte le prove runtime qui descritte restano future.

Il probe controllato [pyenv-shim-probe.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
è evidenza del supervisore su pyenv 2.6.12. Il primo probe in
[identity-and-probes.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
aveva il bin del Python già selezionato davanti agli shim: non dimostra la loro
selezione. I test 67 legacy / 6 governance sono risultati storici, non rieseguiti.
La lettura AST osserva 55 funzioni unit, 5 integration, 6 governance; non determina
il numero di nodeid parametrizzati. Tutti gli esiti V0–V12 sotto sono **futuri**.

## Implementazione proposta


### R — Preflight del runner dopo GO, prima di P0/V0 — CLA-P002 r002

È preparazione operativa futura. Nessun namespace o Firejail è stato avviato
in pianificazione. Target: questo host Linux del clone, UID utente corrente,
baseline pyenv diretta e futuri interpreti managed/venv, Pandoc, Git, cache
tokenizer e directory di prova. Perimetro: un processo isolato e i suoi figli,
nessuna rete configurata, nessuna modifica a sysctl/AppArmor/rete host,
nessun sudo, installazione di privilegi o profilo persistente.

L’implementatore prepara un diagnostico standard-library verificabile in
`scripts/diagnostics/run-a001-fase0-uv/check_runner.py` (file del repository
visibile alla review, **da creare solo dopo GO**). Non contiene conversioni,
installazioni o accessi remoti. Il supervisore identifica prima della sonda
target, binari/versioni, comando e policy di esecuzione. Il GO sul piano non
scavalca un vincolo operativo su operazioni privilegiate: se il runner esistente
richiede un perimetro ulteriore, si registra IMPEDITA e si torna al supervisore.

Variabili future: `RUN_BOOTSTRAP_PY` è il path assoluto pyenv 3.12.3 osservato;
`RUN_RUNNER_CHECK` il diagnostico sopra; `RUN_HOST_NETNS` il link letto di
`/proc/self/ns/net` fuori dal runner; `RUN_RUNNER_OUT` una directory nuova
d’evidenze. Usare argv/valori assoluti, non concatenazioni di shell non fidate.

| Ordine / candidato | Comando futuro della sonda |
| --- | --- |
| primario, unshare | `unshare --user --map-root-user --net -- "$RUN_BOOTSTRAP_PY" -I "$RUN_RUNNER_CHECK" --host-netns "$RUN_HOST_NETNS" --repo "$RUN_REPO" --output "$RUN_RUNNER_OUT/unshare.json"` |
| alternativa concreta sullo stesso host, Firejail già presente secondo review | `firejail --noprofile --net=none "$RUN_BOOTSTRAP_PY" -I "$RUN_RUNNER_CHECK" --host-netns "$RUN_HOST_NETNS" --repo "$RUN_REPO" --output "$RUN_RUNNER_OUT/firejail.json"` |

Il manuale locale letto staticamente descrive net=none; presenza/setuid del
binario e sysctl della review sono antecedenti, **non prove della disponibilità**.
Il rifiuto unshare è ancora un’inferenza. Se la prima sonda fallisce, conservare
stderr/exit e provare l’alternativa solo entro il perimetro consentito.
Firejail non diventa una semplice etichetta per aggirare il confinamento:
serve la medesima prova di equivalenza seguente.

Il diagnostico, **prima di qualunque tentativo di socket Internet**, deve
osservare namespace rete diverso da quello host, sole interfacce lo e nessuna
rotta/indirizzo verso reti esterne IPv4/IPv6; leggere /proc e netlink locale.
Poi eseguire connect con timeout breve verso gli indirizzi di documentazione
192.0.2.1 e 2001:db8::1: deve fallire senza pacchetti verso l’host, perché non
esiste una rotta. Il figlio `sys.executable -I` ripete namespace/interfacce/
route/socket: stesso isolamento e stessi fallimenti. Nessun DNS o endpoint
pubblico. Un socketpair **AF_UNIX** deve invece funzionare: è locale e serve
anche al loop API; non abilita socket Internet.

Registrare CWD, uid, executable/prefix/stdlib, binari Pandoc/Git leggibili,
cache tokenizer leggibile con hash, scrittura/cancellazione del solo file
sintetico in TMPDIR e workspace di prova, environment pertinente senza segreti.
Non usare --private-tmp se nasconde ambienti/cache preparati. Rifare il
preflight con gli interpreti definitivi dopo P1; la sonda bootstrap conferma
la fattibilità prima della baseline. Guardie pytest/requests completano il
runner, senza sostituirlo. Proxy, uv --offline o sola monkeypatch non sono
equivalenti.

Output: receipt con argv, target, timestamp, exit/stdout/stderr, hash del
diagnostico e namespace host/padre/figlio, prerequisiti e PASS/FAIL/IMPEDITA.
Solo PASS ammette P0/V0. Se entrambi i candidati sono negati o non equivalenti,
**V0 rimane IMPEDITA**: nessuna baseline fuori runner; il supervisore deve
identificare e autorizzare un altro runner Linux con trasferimento verificato.
Non si inventa qui l’identità di una macchina remota.

Docker è separato: il namespace del client non confina il daemon. V7 build
usa il suo perimetro di rete/preparazione; V8 ammette solo daemon locale su
Unix socket, immagine locale identificata, --pull=never e --network none.
Non esporre il socket Docker nei figli dei test veloci/API/packaging.

### P0 — Ingresso autorizzato, baseline e ordine del lavoro

1. Il supervisore riceve piano/evidenze/checkpoint r003, verifica il contesto,
   congela un **nuovo snapshot plan-r003** e prepara prompt per **due nuove chat
   indipendenti ChatGPT e Claude**. L'implementazione attende entrambi i report
   reali e il nuovo arbitrato GO riferito a quegli artefatti.
2. Il preflight R deve essere PASS prima della baseline. L'implementatore
   verifica Git, piano e arbitrato approvati; registra la patch
   preesistente del supervisore senza reset, cambio branch o aggiornamenti a r001.
   Prima di cambiare i sorgenti conserva una copia identificata dei dieci moduli legacy
   della base in una directory baseline della run, priva di dati privati. Hash dei dieci originali e degli input di V0, impronta run_context dello
   stage baseline e origine sono dichiarati; tale copia serve solo al confronto, mai alla prova
   della wheel. Registra anche eventuale immagine Marker precedente senza avviarla.
3. Ricostruisce, se necessario, un ambiente baseline separato con l'interprete
   diretto pyenv **3.12.3**, i quattro runtime e pytest, evitando il setup difettoso.
   Registra binario, Python, dipendenze complete, tiktoken, encoding/cache e Pandoc.
   Questa preparazione può installare pacchetti solo nella futura implementazione;
   produce un manifest di confronto, non una seconda fonte di produzione.
4. Crea F1–F6 di P4, conserva input e output baseline con argomenti, CWD, exit,
   stdout/stderr, hash e difetti. Esegue gli script diretti originali; documenta
   il fallimento dell'orchestratore legacy da un CWD esterno. Una baseline mancante
   per il contenuto esaminato deve essere recuperata prima del GO finale.
5. Svolge P1/P2/P3, prepara esplicitamente gli ambienti di verifica, poi V1–V6.
   Esegue P5/P6 e V7/V8 nel passo Docker distinto; completa P7/V9 e il report.
   Versioni alternative, nuovo layout o incompatibilità sostanziali tornano al
   supervisore prima di cambiare la proposta approvata.

### P1 — Interprete, pyenv e dipendenze

File futuri: `pyproject.toml`, `uv.lock`, `.python-version`,
`build-constraints.txt`, `.gitignore`; rimozione di `setup.py` e `requirements.txt`
dopo l'allineamento di tutti i consumatori.

| Scelta conservata / precisata in r003 | Motivazione e limite |
| --- | --- |
| uv **0.10.10**, `tool.uv.required-version = ==0.10.10` | mantiene la versione r001 disponibile localmente, con help e catalogo congelati. Il binario distribuito agli utenti va identificato con release/checksum; nessun auto-update |
| CPython **3.12.13**, `.python-version` esatta | mantiene il candidato r001 e la stessa minor della baseline. Il catalogo di uv 0.10.10 lo enumera; nessun binario managed è stato provato. Non è l'ultima patch: la fonte Python.org già letta negli antecedenti segnala 3.12.15 |
| `requires-python = >=3.12,<3.13` | supporto dichiarato ristretto rispetto al metadata legacy >=3.9; altre minor non collaudate. Restrizione da documentare, non errore da nascondere |
| Managed **solo in `$RUN_REPO/.venv-python`** | percorso utente unico, locale e ignorato, selezionato sempre con `UV_PYTHON_INSTALL_DIR` assoluta. Nessuna alternativa implicita alla directory predefinita o a pyenv |
| `setuptools.build_meta`, `setuptools==84.0.0` | dieci moduli flat espliciti senza spostamenti di dominio; disponibilità e Requires-Python confermati staticamente. Build da provare. Hatchling/uv_build non necessari |
| Nome/versione `markdown-for-llms`, `1.0.0` | conserva identità legacy; non pubblica una release. Rimuovere metadata example.com ingannevoli, senza inventare contatti |
| `default-groups = []` | ambiente utente base leggero; sviluppo e server solo su richiesta esplicita |

Non cambio tacitamente versioni rispetto a r001. La scelta **di fonte Marker introdotta in r002**
è conservata e identificata in P5. Aggiornare Python/uv alla patch corrente sarebbe una revisione
esplicita dei candidati con nuovo catalogo, pin e V1–V8; non è una conseguenza
automatica dell'approvazione uv.

#### Effetto locale del pin e percorso managed univoco — CLA-P001 r001

Il pin esatto è intenzionale: un prefisso `3.12` lascia variare la patch managed e
non identifica l'ambiente della run. Ha però un effetto osservabile sui comandi
**che passano dallo shim pyenv** nel clone. Una installazione uv nella directory
locale non installa la versione in pyenv e non ripara quello shim.

| PATH del solo caso di prova | PYENV_VERSION | Prima del pin | Dopo pin 3.12.13 assente da pyenv |
| --- | --- | --- | --- |
| shim pyenv prioritario | assente | pyenv globale 3.12.3 | nel probe pyenv 2.6.12: `python3` ripiega su `/usr/bin/python3`, exit 0 senza avviso; `pyenv version-name` invece exit 1 |
| shim prioritario | `3.12.3` | pyenv 3.12.3 | pyenv 3.12.3, variabile prioritaria al file locale |
| bin diretto `/home/davide/.pyenv/versions/3.12.3/bin` prioritario | assente o `3.12.3` | bin diretto 3.12.3 | stesso bin diretto: lo shim non è invocato |
| `uv run` con richiesta/flag managed e ambiente preparato | assente o `3.12.3` | futuro ambiente managed, prima del file usare richiesta CLI esplicita | stesso managed 3.12.13; non dipende dalla selezione pyenv |
| interprete assoluto della venv managed | assente o `3.12.3` | futuro base managed 3.12.13 | stesso base managed 3.12.13 |

Le prime righe sono osservazioni pregresse circoscritte; le ultime due sono
**attese di V1 da verificare**, non prove svolte. Il numero 3.12.3 da solo è
insufficiente: sistema e pyenv nel probe hanno la stessa versione. V1 registra
`sys.executable`, realpath, `sys.prefix`, `sys.base_prefix`, `sys.path`, origine
dello stdlib e dei moduli importati, `shutil.which('python3')`, versione pyenv e
soli valori delle variabili pertinenti. Nessun dump di credenziali.

Percorso utente Linux, da documentare dopo GO: dentro una **subshell** del clone,
definire `RUN_REPO` assoluta; esportare solo lì
`UV_PYTHON_INSTALL_DIR="$RUN_REPO/.venv-python"`. Acquisizione esplicita iniziale:

```bash
UV_PYTHON_INSTALL_DIR="$RUN_REPO/.venv-python" uv python install 3.12.13 --no-bin
```

Il download è preparazione, separato dai test; registrare URL/build standalone,
hash e path effettivi. Successivamente esportare nella stessa subshell
`UV_PYTHON_DOWNLOADS=never` e usare `--python 3.12.13 --managed-python
--no-python-downloads` in lock/sync/build/venv e nelle verifiche `uv run`.
Mai `--default`, `--global`, shell update, modifica PATH persistente, comandi
pyenv install/local/global o cancellazione di interpreti condivisi.
`--no-registry` riguarda **Windows**: non è una protezione Linux e non è incluso
nel comando Linux. Su Windows la futura guida specifica separatamente quel flag
e i path `Scripts`; il collaudo della run resta Linux.

README, guida e AGENTS prescrivono l'interprete uv/venv per i comandi del progetto.
Non promettono che `python3` della shell cambi al managed; descrivono lo shim
e la scelta esplicita. Chi vuole usare un Python pyenv diretto per la baseline
lo invoca al path assoluto già verificato, con ambiente della sola prova.
`.venv/`, `.venv-python/`, `.venv-marker/`, `.venv-marker-gpu/` sono ignorate e
filtrate da Docker; uv.lock è versionato.


#### Guardia persistente e origine — CLA-P003 r002

Prevedere nel futuro pyproject `[tool.uv] python-downloads = "manual"`.
La chiave è documentata al tag 0.10.10 e nei sorgenti/estratti congelati delle
review; V1 deve provarne il parsing reale. Il valore vieta acquisizioni implicite
nelle operazioni di progetto e lascia l’acquisizione al solo python install.
Non impone la directory: non esiste una chiave equivalente a
UV_PYTHON_INSTALL_DIR nel progetto, e un interprete compatibile già presente
altrove può ancora essere trovato.

L’uso supportato richiede quindi entrambe le protezioni: directory assoluta
locale esportata nella sola subshell **e preflight dell’origine prima di ogni
preparazione/avvio**. Leggere il path reale del CPython installato sotto
`$RUN_REPO/.venv-python`, chiamarlo `RUN_MANAGED_PY`, verificare versione esatta,
binario/stdlib e symlink confinati a quel managed. Nessuna ricerca che accetti
il primo Python 3.12.13 globale. Dopo l’install esplicito, selezionare
`--python 3.12.13 --managed-python --no-python-downloads`.
Verificare anche sys.base_prefix di ogni venv e l’origine dello stdlib.
La richiesta uv per versione è sempre accompagnata dalla directory locale e
dal confronto con RUN_MANAGED_PY, che viene invocato direttamente per le sonde.
Per comandi di progetto da un workspace dati usare --project con variabili
locali oppure l’interprete venv assoluto già verificato.

V1 aggiunge una shell senza alcuna variabile UV_*, VIRTUAL_ENV e PYENV_VERSION,
con PATH/prerequisiti e soli valori non segreti dichiarati nella receipt.
Nessun utente/config/path globale viene alterato. Da copia-probe identificata
con pin/pyproject invariati: caso senza venv pronta, comando uv run --no-sync
di sonda stdlib dentro il runner R già PASS; caso con venv locale pronta,
medesima sonda. Questa prova omette --no-python-downloads e --offline:
non deve mascherare la guardia persistente con un flag che vieta già il download.
Il runner impedisce egress anche in caso di regressione; un tentativo di
download, rilevato nei log, è FAIL, non un successo attribuito a manual.
La sonda controlla l’origine contro il managed atteso **prima di ogni import
applicativo** e dà exit nonzero esplicito se manca o è diversa. Un’installazione
globale compatibile già disponibile è quindi rifiutata dal preflight, non
considerata una prova riuscita per il solo numero di versione.

Attese: errore leggibile nel caso non preparato oppure unico ambiente/origine
attesi; nessun download altrove. Inventariare prima/dopo la directory managed
locale e quelle predefinite/UV_PYTHON_INSTALL_DIR ereditate (sola lettura di
nomi/metadata/hash pertinenti, mai cleanup). Se il comando uv riesce a scegliere
un’origine diversa, conservare il riscontro e il rifiuto del preflight; **quel
percorso nudo non è supportato** e non può eseguire test o applicazione.
Una shell preparata con la directory locale e i flag deve riuscire.
Manual da solo non viene presentato come guardia dell’origine o dei pacchetti.
Prima validare la chiave con uv 0.10.10 e log di configurazione; una prova
aggiuntiva con --offline/--no-python-downloads non sostituisce il caso senza
questi flag. Nessuna installazione fuori dal path scelto.

README/guida/AGENTS e configurazione IDE futura dichiarano questo prerequisito:
interprete assoluto `.venv/bin/python` collegato al managed locale, sync
esplicita dopo le modifiche, preflight, nessun generico “uv run pytest” in una
shell non preparata. Il piano non introduce settaggi globali IDE/pyenv e non
promette resistenza a --no-config, variabili/flag deliberatamente sostituiti
o cancellazione delle guardie. Fallimento preflight = nessuna esecuzione.

#### Inventario delle dipendenze

| Selezione | Requisiti diretti proposti | Verifica futura |
| --- | --- | --- |
| runtime base | `requests>=2.31.0`, `tiktoken>=0.5.1`, `tqdm>=4.66.0`, `python-dotenv>=1.0.0` | versioni effettive nel lock; client/fasi senza FastAPI, torch, Marker, pytest/lint |
| gruppo `dev` | pytest>=7.4.0, pytest-cov>=4.1.0, pytest-mock>=3.11.0, black>=23.0.0, flake8>=6.0.0, mypy>=1.5.0, httpx | strumenti dai minimi legacy, pin effettivi dal lock; httpx per mock API, nessuna catena ML |
| extra `marker-server` | fastapi, uvicorn, python-multipart | import wrapper e upload mock; selezione **esplicita in ambiente API**, non presunta dal solo dev |
| extra `marker-cpu` | `marker-pdf[full]==1.10.2`, `surya-ocr==0.17.1`, `torch==2.7.1`, beautifulsoup4 | Linux x86_64, indice torch CPU esplicito; candidati da risolvere/provare |
| extra `marker-cu126` | stesso Marker/Surya/bs4, torch==2.7.1 dall'indice cu126 | profilo GPU candidato, incompatibile con CPU nello stesso sync |
| provider/transitive motore | mammoth, ebooklib, openpyxl, python-pptx, weasyprint, filetype, Pillow, transformers, pydantic, pdftext ecc. | da metadata/lock; bs4 esplicita perché usata dalla registry. Non copiare il lock Poetry upstream né dichiarare la combinazione collaudata |
| esterne/native | Pandoc host; Docker/Compose; librerie della release Debian scelta; GPU driver/Toolkit | inventari separati dai pacchetti Python |

Colorama/rich/enhanced non hanno uso negli import inventariati: non trasferirli
al runtime. Gruppo dev non finisce nel Requires-Dist della wheel. Il motore può
trascinare strumenti upstream come pre-commit: inventariarli separatamente dal
nostro dev. Non installare torchvision/torchaudio senza necessità nel grafo;
l'eventuale coppia ufficiale torch 2.7.1 / torchvision 0.22.1 resta candidata.
Indici PyTorch nominati con `explicit=true`, selezione per extra e conflitto
CPU/cu126; nessun backend auto o pip successivo che cambi variante. Il client
Marker HTTP non richiede l'extra motore.

Un solo pyproject/lock, due ambienti principali: `.venv` client e `.venv-marker`
server CPU, quest'ultimo solo con `UV_PROJECT_ENVIRONMENT` assoluta nella singola
invocazione/subshell e marker-server+marker-cpu. GPU separato e non collaudato.
Il lock universale considera anche extra non installati: risoluzione/metadata
degli extra possono usare rete, pur senza installare ML nel base. Registrare
quel costo nella preparazione. Un conflitto reale richiede nuova proposta al
supervisore per un progetto server distinto, non rimozione silenziosa di A6.

`pyproject.toml` è la dichiarazione, `uv.lock` la risoluzione. Eliminare setup e
requirements manuali dopo tutti i consumatori; gli export di verifica dal lock
sono artefatti generati identificati. Nessun shim duplicato o elenco manuale
alternativo salvo consumatore necessario documentato e riesame del supervisore.

Build isolata: fissare Setuptools esatto in build-system e in
`build-constraints.txt`; usare gli stessi pin in
`tool.uv.build-constraint-dependencies` alla radice per lock/sync/run, e il file
con --build-constraints per uv build/pip. Controllare la corrispondenza delle due
rappresentazioni nel report: sono vincoli di build, non due fonti runtime.
Prima della risoluzione/installazione inventariare anche
sdist e build requirements dinamici. Per altri backend, incluso poetry-core se
serve a una transitive, fissare versioni esatte compatibili dopo lettura del
metadata e registrare hash/backend/input/output e due build pulite ripetibili.
Il lock runtime non congela quei backend. Preferire wheel del target; Marker
deve usare la wheel P5 (`no-build-package = ['marker-pdf']`). Se un altro sdist
non ha backend identificato/vincolato, fermare quella preparazione, non risolverlo
implicitamente con backend mobile. Non promettere wheel byte-identiche senza
aver controllato timestamp e contenuto.

### P2 — Wheel flat e cinque entry point

Distribuire esclusivamente questi dieci `py-modules`, con elenco esplicito:

```text
config, logging_config, exceptions, unified_converter, master_workflow,
clean_markdown, validate_markdown, chunk_markdown, batch_monitor, marker_api_server
```

I primi nove richiedono solo runtime; il decimo richiede marker-server e importa
Marker pigramente. Wheel e sdist non includono test, fixture, helper governance,
setup, dati, configurazioni operative, `.env`, temp/cache o segreti. Metadata
README/licenza possono essere inclusi. Verificare archivi, non solo find_packages.

| Console script invariato | Target |
| --- | --- |
| markdown-pipeline | master_workflow:main |
| markdown-config | config:main |
| markdown-clean | clean_markdown:main |
| markdown-validate | validate_markdown:main |
| markdown-chunk | chunk_markdown:main |

Estrarre la CLI attuale di config in `main()` e chiamarla anche da `__main__`.
Conservare create-default/validate/show, JSON, messaggi e limiti della validazione
legacy. Clean e validate non hanno parser/help: eseguirli su fixture senza
argomenti. Help reale solo pipeline/config/chunk; unified_converter e batch_monitor
sono moduli aggiuntivi, senza creare altri console script.

Script diretti dal clone restano invocabili con l'interprete del progetto
**sincronizzato e con package installato nello stesso interprete**. Lo script
diretto usa naturalmente la directory del sorgente fidato per il proprio import;
le fasi figlie usano il package di quel medesimo ambiente. Dopo modifiche ai sorgenti di una installazione non editable applicare
il protocollo seguente di ricostruzione, reinstallazione e confronto prima
della prova. Un semplice sync senza regola di rebuild non dimostra freschezza. Non usare
conftest o copie dei sorgenti per far funzionare un interprete privo del package.
Un layout con namespace/shim ridurrebbe i nomi generici, ma anticiperebbe gli
spostamenti della fase 2: in questa run restano flat, con origini provate.


#### Catena sorgenti → distribuzione → installazione → esito — CLA-P001 r002

**Meccanismo scelto:** `[tool.uv] reinstall-package = ["markdown-for-llms"]`
persistente, oltre a `--reinstall-package markdown-for-llms` nei comandi sync
riprodotti. La [documentazione cache al tag 0.10.10](https://raw.githubusercontent.com/astral-sh/uv/0.10.10/docs/concepts/cache.md)
attribuisce a questa configurazione la ricostruzione/reinstallazione della
directory locale a ogni sync; l’help congelato conferma il flag. Non scelgo
cache-keys, per evitare di dipendere da un elenco incompleto o da soli timestamp.
Nessuna pulizia di cache globale o reinstallazione di altri progetti.

La configurazione non aggiorna ambienti inattivi e **--no-sync non costruisce**.
Per ogni modifica, anche a un solo .py con versione/metadata invariati, sono
obbligatori preparazione esplicita e preflight di contenuto. La wheel canonica
V2 da sdist viene reinstallata come ultimo passo di ogni ambiente della
verifica; nessun editable, conftest o copia manuale dei moduli vale come prova.

| Ambiente host previsto | Selezione sync e invalidazione | Ultimo passo e prova |
| --- | --- | --- |
| uso normale `$RUN_REPO/.venv` | base, no default groups, no editable, reinstall-package dopo ogni modifica | wheel canonica aggiornata senza deps/build, preflight dieci moduli prima dell’uso documentato |
| `base-a/base-b` | base, stessa regola; due venv non dimostrano da sole rebuild | medesima wheel canonica, inventario runtime e origini |
| `dev-a/dev-b/dev-env` | group dev, stessa regola, package nello stesso sys.executable dei figli | wheel canonica e preflight prima C-fast/C-package/governance pertinente |
| `api-env` | group dev + marker-server, stessa regola | wheel canonica e preflight prima C-api e raccolta C-all |
| host opzionale `.venv-marker` | marker-server + marker-cpu, stessa regola se preparato nel perimetro pesante | wheel canonica e dieci hash; server host resta non verificato senza prova dedicata |
| host GPU `.venv-marker-gpu`, se autorizzato in futuro | marker-server + marker-cu126, stessa regola, mai insieme CPU | medesima catena; nessuna installazione/GPU implicita qui |
| wheel-env fuori clone | nessun sync del clone; runtime da export hash-locked | reinstallare la **nuova wheel identificata**, --no-deps --no-build --reinstall-package; V3/V4/V5 |
| ambienti dei confronti incrociati | legacy diretto identificato o nuova wheel, versioni controllate | manifest separato; non aggiornare baseline originale né lock di produzione |

Preparare dopo GO `scripts/diagnostics/run-a001-fase0-uv/verify_distribution.py`,
stdlib, nel repository e sottoposto a review. Definire la sua CLI:
`--source-manifest PATH --sdist PATH --wheel PATH --profile base|api|cpu
--expected-managed PATH --receipt PATH`; gira nell’interprete in prova con -I.
Un’opzione `--archives-only` permette V2 senza import applicativi.
I test/harness lo richiamano da path fidato assoluto. Non contiene installazioni
o fallback, non modifica i moduli ed esce nonzero su qualunque differenza.

**Manifest sorgenti S dello stage**, fuori dalla produzione:
ID univoco, branch/HEAD/dev/merge-base, snapshot run_context e sua impronta
worktree, SHA-256 del manifest snapshot, dieci .py **con byte/sha256**, path e
stato Git. Identificare inoltre pyproject e .python-version, uv.lock,
build-constraints.txt e la rappresentazione tool.uv, configurazione backend
(build-system/config-settings/env realmente usati), README e licenze consumati,
eventuale MANIFEST.in/setup.cfg/setup.py/src come input o assenza esplicita,
uv/binario Python/stdlib/piattaforma e inventario completo delle dipendenze.
Niente segreti in env/snapshot. README/licenze non sono un optional se il
backend li legge. Metadata statici name/version impediscono dipendenze
nascoste da Git; eventuali nuovi input dinamici richiedono nuovo manifest.

**Build B:** output in directory nuova per stage. Registrare input S, argv,
backend effettivo/versione/hash artefatto e requisiti PEP 517 iniziali/dinamici,
constraint/lock e log. Acquisire sdist V2, poi wheel realmente costruita dalla
sdist, non da un vecchio dist. Ispezione tar/ZIP senza estrazione pericolosa:
path assoluti/.., duplicati, symlink non attesi o file operativi = FAIL.
Dieci .py esattamente una volta, stessi byte/hash di S nella sdist e wheel;
METADATA, WHEEL, entry_points e licensing identificati.
RECORD deve attestare digest sha256 URL-safe-base64 e dimensioni della wheel:
decodificare e confrontare digest con quello esadecimale di S e con byte ZIP.
Le eccezioni PEP RECORD, file RECORD stesso e script generati, vanno nominate;
non estendere eccezioni ai dieci moduli. Hash esterno della wheel da solo
non prova il legame col clone.

**Installazione I:** per ciascuna venv sync selezionata conservare log del
rebuild, poi installare esplicitamente la wheel B già verificata come ultimo
passo; le deps restano dal lock. Identificare file wheel realmente consumato,
SHA-256 e B/S. Il preflight -I usa importlib.metadata/distribution e
sysconfig/spec per trovare site-packages, legge senza importare i dieci .py,
confronta SHA-256 e byte con S e B e verifica RECORD installato per i moduli,
Name/Version/entry point, direct_url non editable e inventario.
Il base deve hashare marker_api_server.py **senza importare FastAPI o Marker**.
Origin/spec padre e figli devono risolvere ai file verificati; i moduli
effettivamente importati nei profili appropriati devono confermare __file__.
Nessun codice copiato manualmente nei site-packages.
Container: stessa verifica della wheel consumata e di **/opt/venv**, §P6/V7/V8.

**Esito E:** receipt per ogni C-fast/C-api/C-package, V3/V4/V5 e immagine V7/V8,
con S/B/I, hash diagnostico/harness/test/fixture/config/cache e origine/namespace runner.
Il preflight precede collection/import pertinenti; per C-package copre sia
RUN_DEV_PY sia RUN_WHEEL_PY. Prima/dopo ogni prova verificare snapshot dello
stage ed input hash; differenza durante la prova = esito invalidato, mai PASS
attribuito al worktree finale. La suite parent può importare il clone, ma
ogni origine usata e la copia installata devono coincidere con S. I log/report
sono in spazi esclusi; codice nuovo, diagnostici e fixture di regressione
restano in file del repository visibili alla review.

#### Stadi, invalidazione e ripetizione

Il supervisore mantiene gli snapshot condivisi. L’implementatore gli consegna
manifest e receipt per snapshot stage-baseline, stage-package, stage-tests,
stage-image e finale, con label progressive; nessuno sovrascritto. Per prove
intermedie si usa lo snapshot concordato dello stage, non il vecchio contesto
planning-context-r003 dopo le modifiche previste. Una receipt può derivare da
un altro stage solo con confronto esplicito degli input pertinenti; il report
finale conserva l’impronta originaria e una motivazione di equivalenza, senza
“trasferire” il PASS a tutto il nuovo worktree.

| Input modificato | Esiti invalidati e preparazione/prove da ripetere |
| --- | --- |
| uno dei dieci moduli | S/B/I e verifiche V2–V6 dipendenti, hash di tutti gli ambienti; rebuild/reinstall prima di C-fast/API/package e V3–V5; nuova immagine V7 e V8 se include il modulo |
| metadata/build backend/constraints/README/licenze | V2/B/I, tutte le distribuzioni e preflight, V3–V6 interessate, V7/V8; lock --check, nuova risoluzione solo se necessaria |
| uv.lock/runtime/interprete/uv | V1, build/installazioni, preflight e prove V2–V8 interessate, confronti contenuto; mai attribuzione dalla sola versione |
| test/harness/guardia/diagnostico | preflight e C/discovery/V pertinenti; nessun rebuild del prodotto se i suoi input sono identici, confronto hash motivato |
| F1–F6/argomenti/config/tokenizer/Pandoc | baseline pertinente e V4/V5/confronti; fixture non ritoccata per fare coincidere output nuovi |
| Dockerfile/.dockerignore/Compose/digest/apt/native | V7/V8 e contesto/config/build; V1–V6 non trasferite senza confronto degli input host immutati |
| sola documentazione non consumata dalla build | V9/link/whitespace; preservare receipt precedenti con confronto degli input, non rifare tutto per abitudine |

A fine implementazione consolidare gli input finali e rifare le prove richieste
da questa matrice. Baseline originale conserva per sempre i dieci originali
e l’ambiente pyenv 3.12.3; non chiamarla “codice nuovo” dopo un rebuild.
Nuovo codice in una copia isolata per diagnosi è dato di prova identificato,
mai una modifica di prodotto nascosta in temp/.

#### Verifica discriminante della cache, futura e separata dai test veloci

Dopo GO, come **preparazione V1/V2**, creare copia isolata del sorgente nuovo
identificato, lock/metadata/backend identici, cache uv propria già popolata
da un primo sync e installazione canonica. Non cambiare metadata/versione.
Scegliere il solo exceptions.py; aggiungere una costante innocua specifica
del probe. Registrare hash prima/dopo, S-probe separato, nessuna fixture/data
privata. Prima del secondo sync il preflight confrontato con S-probe deve
dare **FAIL per il modulo stantio**: controllo negativo essenziale.

Eseguire sync non editable nella **stessa copia, stesso ambiente e stessa
cache già popolata**, con la configurazione reinstall-package e il flag;
controllare log di rebuild e hash della nuova copia installata **prima**
della reinstallazione canonica, per non mascherare il comportamento di sync.
Solo exceptions.py cambia; altri nove hash e metadata restano uguali.
Poi rigenerare sdist/wheel e reinstallare la wheel probe, confermando S-probe/B/I.
Per validare il controllo negativo anche verso installazioni alterate, può
essere usata una receipt vecchia in confronto al nuovo S: deve fallire,
senza modificare manualmente i moduli installati.

Questo probe non è la prova esterna A3 né un test veloce con build/rete.
Cache backend/deps preparate; una mancanza non provoca fetch implicito nel
probe offline. Ripristinare bytes originali **nella sola copia**, dimostrare
i dieci hash/metadata/lock uguali a S, rifare sync/rebuild/install e preflight.
Verificare clone originale/snapshot e assenza di cambiamenti al managed/cache
globali. Conservare le due receipt negative/positive e il ripristino; la
copia non viene usata dalle verifiche finali. Prima dei test finali deve essere
provata l’identità degli ambienti di produzione con S finale.

### P3 — Workspace, dotenv e avvio isolato — CLA-P004/005 r001

File coinvolti: `master_workflow.py`, `config.py`, `unified_converter.py` e test
di avvio. Catturare `workspace = Path.cwd().resolve()` nell'istanziazione, prima
di applicare override; non usare BASE_DIR all'import come origine dei dati/codice.
I percorsi dati relativi, JSON, report di validation e log restano nel workspace;
i percorsi assoluti configurati rimangono assoluti. `get_directory_path` resta
il contratto legacy; i subprocess hanno lo stesso CWD dei dati. Niente scritture
nel clone o nel site-packages per le prove installate.

#### Caricamento .env prima degli override

Introdurre un piccolo helper condiviso in **config.py** per
`load_dotenv(dotenv_path=workspace / '.env', override=False)`, senza nuova
configurazione applicativa. Rimuovere il `load_dotenv()` implicito all'import di
unified_converter. L'orchestratore chiama l'helper **prima** di
`apply_env_overrides(self.config)` e prima di costruire UnifiedDocumentConverter.
L'inizializzazione standalone del converter chiama il medesimo helper prima di
leggere `_get_conversion_config`, preservando gli override già implementati dal
converter; non aggiunge qui un nuovo schema o nuove regole di parsing.

Conservare l'ordine legacy tra CLI e override applicativi: CLI inizializzata,
poi override ambiente dell'orchestratore. `override=False` dà precedenza alle
variabili già presenti nella shell rispetto a .env. La differenza deliberata è
la **ricerca esplicita nel workspace**, senza risalire dal modulo/site-packages
o cercare .env nei parent. L'helper è innocuo se il file manca e non scrive file.
Test in processi separati evitano che l'ambiente caricato in un caso contamini
il successivo. Non promettere che config --show applichi override che oggi non
applica; V5 osserva la pipeline realmente istanziata.

#### Subprocess di fase

Mappa cleaning/validation/chunking ai rispettivi moduli; lanciare
**`[sys.executable, '-I', '-m', modulo, ...]`**, con `cwd=workspace`, timeout,
stdout/stderr, argomenti chunk e codici di uscita legacy conservati. `-I` include
`-P`, ignora PYTHONPATH/PYTHONHOME e user site, ma mantiene l'ambiente applicativo
(CHUNK_SIZE, TIKTOKEN_CACHE_DIR ecc.) e il CWD dei dati. Un semplice `-P` da solo
non neutralizza un PYTHONPATH che contiene la directory dati.

Costruire l'ambiente figlio dall'ambiente necessario, rimuovendo esplicitamente
PYTHONPATH/PYTHONHOME; nessuna aggiunta del clone o del workspace al module path.
L'interprete figlio non dipende da PATH/pyenv: è sys.executable del padre.
Sostituire il controllo di file CWD/*.py con un preflight in **subprocess -I**
che controlla spec/origine dei moduli di fase nello stesso interprete. Un find_spec
nel solo padre può essere mascherato da sys.path di pytest; non basta. Errore
leggibile quando l'interprete non contiene il progetto installato.

Per avvio da wheel prescrivere console script con PYTHONPATH/PYTHONHOME ripuliti
oppure `python -I -m master_workflow`; mai il semplice `python -m` nel workspace
non fidato. Da clone sincronizzato, invocare lo script diretto al suo path
assoluto con ambiente ripulito; è l'origine fidata iniziale, senza fallback figlio
al codice della directory dati. Nessun nuovo launcher generato a runtime.

V5 verifica moduli applicativi e transitive requests/dotenv/tqdm/tiktoken sotto
site-packages della wheel; lo stdlib appartiene al base interpreter identificato.
Non limita il controllo ai tre file di fase. Sentinelle omonime nei dati e
PYTHONPATH avverso restano ineseguiti. Questo protegge gli import Python delle
fasi; non pretende di rendere innocui tutti i documenti/LaTeX del legacy.

Preservare i prerequisiti globali Pandoc e tiktoken anche per --step; V5 li richiede.
Aggiornare solo il suggerimento pip obsoleto al percorso uv. `uv run --project
"$RUN_REPO"` consente l'ambiente del clone mantenendo il CWD dei dati;
`--directory` lo cambia e non è un sinonimo. V3/V5 usano la wheel esterna e i
suoi binari, senza uv --project o editable come surrogato della prova installata.

### P4 — Strati dei test, prerequisiti e fedeltà

Questi nomi di file e casi sono la proposta r003 da realizzare dopo GO. Non esistono
ancora le fixture/helper nuovi. Separare la **preparazione** (lock, sync, build,
installazione, cache) dall'esecuzione/raccolta, entrambe senza rete o build.

#### Matrice di ogni test nuovo o modificato — CLA-P002 r001

| File/componente | Strato e casi previsti | Dipendenze/package/interprete | Comando futuro | Prerequisito mancante |
| --- | --- | --- | --- | --- |
| tests/unit/test_config_cli.py, nuovo | unit: CLI main/help, create/show equivalenti, invocazioni diretta/-I -m/console (almeno 3 casi) | base+dev, progetto installato in dev-env; subprocess nello stesso interprete; nessun server | C-fast o `"$RUN_DEV_PY" -I -m pytest tests/unit/test_config_cli.py -q` | FAIL/preflight esplicito; niente installazioni o importorskip |
| tests/unit/test_phase_launch.py, nuovo | unit: interprete/argv -I e ambiente figlio; preflight mancante; CWD/report (almeno 3 casi) | base+dev, package installato; subprocess reale per spec, fake della sola fase ove dichiarato | C-fast o `"$RUN_DEV_PY" -I -m pytest tests/unit/test_phase_launch.py -q` | FAIL, distinguere fake dal processo reale |
| tests/integration/test_pipeline.py, mutato | 5 funzioni legacy: seed converted, output vuoti, skip_existing=False; fasi locali realmente eseguite | base+dev, package installato **nello stesso sys.executable** che lancia -I -m; cache tiktoken pronta; niente Marker/API | C-fast o `"$RUN_DEV_PY" -I -m pytest tests/integration/test_pipeline.py -q` | FAIL/preflight installazione/cache; non affidarsi a conftest |
| tests/api/test_marker_wrapper.py, nuovo (spostato dalla proposta unit r001) | API mock: successo sui due endpoint, eccezione converter, ImportError, metadata/opzioni legacy, health; almeno 6 nodeid | base+dev+**marker-server**, httpx da dev; progetto installato in api-env; nessuna ML | C-api | FAIL se FastAPI/extra manca; directory ignorata prima dell'import da C-fast |
| tests/packaging/test_installed_distribution.py, nuovo | packaging: inventario/origini, CLI, fasi e confronto contenuto, dotenv, sentinelle, caso errore; almeno 6 funzioni | harness pytest in dev-env, wheel non editable in wheel-env con solo runtime; subprocess **RUN_WHEEL_PY** | C-package | FAIL indicando manifest/venv/wheel/cache/Pandoc assente; non costruire o installare dai test |
| tests/docker/test_marker_contract.py, nuovo | driver V8: subcasi import/firme/provider/native/output/constructor mock, ciascuno nel report | parent dev-env senza ML, Docker CLI; immagine CPU già costruita con package e marker-server+marker-cpu; diagnostica via stdin | C-docker | FAIL/preflight se Docker/immagine/harness manca; nessun build/pull/up da test o collection |
| scripts/diagnostics/run-a001-fase0-uv/{check_runner,check_python_origin,verify_distribution}.py, nuovi dopo GO | diagnostici stdlib: runner/origine e catena S/B/I, negativo modulo stantio e ripristino nella preparazione | bootstrap/managed/venv assoluti; nessun import ML/API per hash; non distribuiti nella wheel | §R, V1/V2 e preflight V3–V8 prima dei C pertinenti | FAIL/IMPEDITA con receipt, mai installare o riparare dal diagnostico |
| tests/conftest.py, mutato | helper comuni, reset singleton e allestimento deterministico; guardia rete/preflight per suite veloce | base+dev; nessun FastAPI/Marker importato a livello globale | raccolta C-fast/C-all e test pertinenti | errori di preparazione espliciti; helper non scarica/cache/build |
| configurazione pytest in pyproject, nuova | testpaths unit/integration/governance; ignores smoke radice; norecursedirs esplicite temp/tmp/documentation/runs/ambienti/artefatti/dati, mantenendo esclusioni standard | pytest da dev, progetto installato; nessuna selezione con -m che importi prima i file esclusi | C-discovery/C-all | raccolta fallita = FAIL, non suite passata |

Restano invariati tests/unit/test_cleaning.py (18 funzioni), test_validation.py
(18), test_chunking.py (19) e tests/governance/test_run_context.py (6). I sys.path
legacy possono restare nel parent; le prove dei figli -I e della wheel sono
indipendenti da quelle aggiunte. I due smoke radice `test_pipeline.py` e
`test_conversion.py` restano manuali, esclusi esplicitamente dal discovery
ordinario; non si correggono algoritmo o rete per promuoverli a test veloci.

Config pytest: `testpaths` contiene solo i tre strati veloci; `addopts` esclude
**soltanto i due smoke radice** con --ignore. Il comando con `tests/` ignora API,
packaging e Docker prima dell'import. Una selezione con marker `-m` non è usata
per aggirare import mancanti. `pytest .` veloce ripete questi ignore. C-all,
con api-env, raccoglie tutti gli strati sotto tests e dalla radice (smoke esclusi),
senza che packaging/Docker importino ML o facciano preflight operativo all'import.
I preflight di esecuzione restano nelle fixture/test degli strati interessati.

Per CLA-S2 r002 nominare in norecursedirs: temp, tmp, documentation/runs,
.venv, .venv-python, .venv-marker, .venv-marker-gpu, build, dist, .cache,
__pycache__, .git e directory dati/output note (source_pdfs, source_documents,
source_ebooks, converted_markdown, cleaned_markdown, validated_markdown,
chunked_markdown, docker-data, .docker); conservare anche le esclusioni standard
pytest. Se si usano ulteriori directory verificarle per ruolo/contenuto: niente
wildcard indiscriminata sui test. Gli ignore nominati restano quelli del piano.

Registrare per ogni comando nodeid, numero raccolto/eseguito, PASS/FAIL/skip e
motivo delle directory escluse. Base AST attuale: 60 funzioni legacy +6 governance;
target aggiunte veloci >=6 funzioni, API >=6 nodeid, packaging >=6 funzioni,
driver Docker >=1 con subcasi enumerati. I conteggi effettivi dipendono dai
parametri e sono prodotti dopo implementazione, non inventati ora. Riconciliare
la partizione C-fast + C-api + C-package + C-docker con C-all; annotare il doppio
passaggio dei sei unittest governance, senza contarli due volte come copertura.
**Zero skip di verifiche essenziali**: un prerequisito mancante lascia la prova
FAIL/IMPEDITA e A3/A4/A6 aperto; gli ignore non rendono facoltativi gli altri strati.

Per mock API patchare get_converter/startup prima del lifespan TestClient;
non chiamare create_model_dict. Verificare JSON/markdown atteso, alias endpoint,
errore/import mancante e cleanup temporanei. Per CLA-S6 r002 registrare versioni **locked** FastAPI/Starlette/anyio/httpx,
log lifespan e warning effettivamente osservati. Non pin alle ultime versioni
citate da una review e non refactoring di on_event, né soppressione generica
warning; incompatibilità = FAIL al supervisore.
Il wrapper resta con immagini vuote,
opzioni solo riportate e health di liveness: i test li caratterizzano senza
aggiungere readiness o nuove funzionalità.

Preflight tokenizer fuori collection: cache tiktoken pronta con hash/versione,
encoding effettivo registrato. Mancanza significa impedimento, non fetch dal test.
Guardia di rete pytest per socket/requests anche nella raccolta: bloccare
connessioni AF_INET/AF_INET6; consentire soltanto socketpair locale AF_UNIX
necessario ad asyncio/anyio, con caso positivo locale e negativo Internet.
Non disabilitare la guardia per TestClient (CLA-S4 r002). I subprocess di
fase usano cache verificata e solo fixture senza HTTP. Per la prova offline dei
subprocess usare il runner Linux della sezione V6, con network namespace senza
egress ereditato dai figli, senza cambiare il launcher -I -m di produzione.
Se il sistema non offre tale isolamento, dichiararlo e procurare il runner;
uv --offline da solo non impedisce requests. Nessun mock del tokenizer nelle
prove di fedeltà. Le sole acquisizioni leggere del tokenizer sono preparazione
esplicita; pesi/font OCR sono esclusi.

#### Fixture e confronto — CLA-P009/GPT-P001 r001

Applicare la [skill di fedeltà](../../../../../.agents/skills/verify-conversion-fidelity/SKILL.md).
Preservare sorgente, baseline, output nuovo e manifest separati. Le fixture sono
sintetiche con provenienza, senza documenti privati o chiamate cloud.

| Fixture | Contratto della prova futura |
| --- | --- |
| F1 Markdown stabile | sezioni in ordine, Unicode IT, numeri 123,45/-7/2026, `$E=mc^2$`, riferimento [1], lunghezza sufficiente alla validation; cleaning reale e confronto byte/invarianti |
| F2 contenuto sensibile | formule con graffe/indici e multilinea, codice, link/bibliografia e immagine; inventario perdite legacy rispetto al sorgente, nessuna correzione regex |
| F3 validation | F1/F2 direttamente in cleaned, soglie sintetiche dichiarate; copie Markdown byte-identiche e report con campi deterministici |
| F4 HTML Pandoc | piccolo HTML senza URL remoti: Unicode, numeri, due sezioni, formula testuale E = mc², tabella e riferimento; summary failed=0, output/ordine e exit, nessun Marker |
| F5 asset | piccola immagine deterministica, hash e riferimenti prima/dopo; asset non copiati restano perdita inventariata, nessuna dichiarazione di bundle completo |
| F6 multi-chunk | Markdown lungo deterministico, paragrafi numerati univoci e sezioni; stessi byte/filename/argomenti nella baseline e wheel; **almeno due chunk in entrambe**, sequenza, contenuto, metadati e overlap confrontati |

F6 usa inizialmente almeno 160 paragrafi di circa 40 parole, testo ASCII/Unicode
e numeri distinti; la preparazione baseline misura i token e adegua **la fixture
prima di congelarla**, finché il percorso CLI con custom/1000/100 produce almeno
due chunk. Una fixture che produce un solo chunk è inadeguata, non un PASS.
Congelare byte/hash e argomenti, poi usare quella stessa fixture nella wheel.
Non ritoccare il testo dopo aver visto l'output nuovo.

Baseline chunk: script originale con `--input-dir <workspace>/validated_markdown
--output-dir <workspace>/chunked_markdown --target-llm custom --chunk-size 1000
--overlap 100`, coerente con l'orchestratore. Non usare il default standalone
cleaned. Allineare anche JSON, env/CLI, filenames e posizione dei dati. Confrontare
file dei chunk in ordine, frontmatter, results/metadata, chunks_index, token/word
count, heading_context, posizioni e overlap_prev/next; soltanto eventuali campi
precisamente identificati come tempo/path della prova sono confrontati a parte.
Markdown/LaTeX non vengono normalizzati.

Nel percorso semantic legacy l'overlap può essere zero: misurare suffisso/prefisso
effettivo e metadati, registrando quel limite senza introdurre overlap nell'algoritmo.
Per esercitare anche un confine con overlap non vuoto, usare **la stessa F6 e gli
stessi parametri** in un confronto complementare del metodo già esistente
`MarkdownChunker.chunk_by_sliding_window`, con baseline/script di riferimento e
wheel, entrambi >=2 chunk e intersezione osservabile >0. Confrontare la sequenza
integrale restituita e i confini rispetto ai token sorgente; non aggiungere
`--chunking-strategy`, che non esiste. Un difetto legacy negli indici/overlap
rimane inventariato; la nuova esecuzione deve riprodurlo senza nuove perdite.

Se i risultati differiscono, non attribuirli subito al packaging. Prima eseguire
un confronto incrociato limitato alla fixture divergente: sorgenti legacy con
Python 3.12.13 e dipendenze locked uguali alla wheel. Confronto vecchio/nuovo in
quello stesso ambiente distingue codice da ambiente. Se resta ambigua la causa,
stessi sorgenti legacy con dipendenze locked su 3.12.3 distinguono la patch; per
tokenizzazione variare soltanto tiktoken/versione/cache a Python fissato. Ogni
combinazione ha proprio ambiente/manifest, senza alterare il lock di produzione.
Compatibilità mancante impedisce l'attribuzione e va al supervisore.

### P5 — Fonte Marker e contratto candidato — CLA-P003/010 r001

**Fonte proposta in r002 e conservata in r003, versione 1.10.2:** usare
`marker-pdf[full]==1.10.2` dalla **wheel PyPI**, evitando build Marker da Git.
Il confronto è con Git al commit r001
`5a41cdbac6232a10aaf4fad29f8e6c1a0c2a0808`, associato al tag v1.10.2 dalle
evidenze primarie pregresse. Il commit resta il riferimento dei sorgenti letti,
non la fonte installata proposta. La serie 2.0 non è un upgrade implicito.

| Fonte | Proprietà e verifiche |
| --- | --- |
| wheel PyPI 1.10.2, **scelta** | file `marker_pdf-1.10.2-py3-none-any.whl`, SHA-256 pubblicato `f631737dd46d3927142b4b14c7b488962c5af278dc2c3ae7dc5b03a47f5909fb`; URL/file/metadata/hash nel manifest e lock; `no-build-package` impedisce il fallback sdist di Marker. Scaricare/verificare realmente solo nella futura preparazione |
| Git al SHA completo, alternativa non scelta | identifica il sorgente immutabile; è riproducibile con backend, build requirements, toolchain e input fissati. Il pyproject al tag usa poetry-core senza pin; il lock runtime non lo fissa. Tornare a Git richiederebbe versioni esatte dei backend/build constraints e due build pulite dimostrate, con revisione dello scostamento |

L'hash della wheel certifica l'identità dell'artefatto scaricato rispetto al valore
atteso, **non l'equivalenza al tag**. Prima dell'uso, registrare metadata della
wheel (Name, Version, Requires-Python, Requires-Dist, extra full, licenza, WHEEL,
RECORD) e origine PyPI; confrontare con il pyproject del commit/tag. Confrontare
hash/contenuto dei file di contratto converters/pdf, models, renderers/markdown,
providers/registry e converters/__init__ con i sorgenti del riferimento. Differenze
si documentano e si rivalutano; non richiedere falsa identità byte dei metadati
generati, ma verificare sul **codice installato** tutte le assunzioni del wrapper.
Nessuna provenienza Git inventata in direct_url.json della wheel PyPI.

V8: import PdfConverter/create_model_dict, signature bind con artifact_dict,
processor_list=None, renderer=None; provider PDF/EPUB/DOCX/PPTX/XLSX/HTML/immagini,
bs4/filetype; MarkdownOutput reale con markdown/images/metadata; wrapper fake
deve estrarre il testo dal campo markdown e page_count. Un fallback str(object)
per un tipo non previsto non prova il contratto. Constructor con download_font
e componenti di modelli sostituiti prima dell'uso; nessun create_model_dict reale.
Import/constructor mock/provider detection non equivalgono a OCR o rendering completo.

Torch 2.7.1 e Surya 0.17.1 restano **candidati non collaudati**: ammessi dai vincoli
diretti e dalle wheel ufficiali CPU/cu126 per cp312 Linux x86_64. Transformers
e altre transitive sono resolved pins futuri; i minimi statici non certificano
la combinazione. Lock, native e test V7/V8 devono passare prima del GO finale.
Gli altri sdist, incluso Surya se non si usa una wheel, seguono P1: backend esatti
e build constraints; la scelta PyPI di Marker non elimina quel problema generale.

Quattro livelli distinti: dipendenze locked; immagine costruita/importabile;
pesi/font identificati; inferenza reale. Il wrapper attuale restituisce immagini
vuote, non applica le opzioni riportate, tollera errori di startup e health resta
healthy. P7 documenta questi limiti, senza dichiarare il percorso local_marker
operativo solo per V7/V8.

**V10 CPU e V11 GPU sono rinviate e non autorizzate nel mandato corrente.** Dopo
i risultati V7/V8 e prima di proporre uno startup normale, il supervisore presenterà
un eventuale perimetro operativo separato: pesi/font da acquisire, fonti/hash/licenze,
rete/disco/RAM, durata CPU/costi e hardware target, cache nuove/verificate, corpus
e criteri di uscita. Nessuna richiesta di autorizzazione adesso per pianificare.
Se emergono incompatibilità o scostamenti semantici che richiedono V10, la prova
torna **essenziale per quel percorso**: l'implementatore segnala il bisogno al
supervisore, non la salta né dichiara A6/compatibilità reale superate. In assenza
di tali scostamenti, l'arbitrato può concludere la toolchain con limite esplicito
**«local_marker non verificato dopo la migrazione»** e istruzioni V10 separate.

### P6 — Docker, entrambi i Compose e manutenzione — CLA-P006/007 r001

File futuri: `Dockerfile`, `.dockerignore`, `docker-compose.yml`,
**`docker-compose-build.yml` conservato e allineato**, override GPU documentato
`docker-compose.gpu.yml`. Conservare il Compose build evita di rompere un percorso
esistente; renderlo una configurazione CPU alternativa con le proprie risorse,
non un secondo stack di dipendenze. Riferimenti e help devono nominarlo.

1. Mantenere come **base candidata della run** Python **3.12.13-slim-bookworm**
   Linux amd64, stessa patch host; uv 0.10.10. Bookworm consente un insieme native
   Debian esplicito per lo stack Marker/WeasyPrint legacy e un cambiamento di
   distribuzione controllabile. Trixie è l'alternativa corrente, da collaudare
   come aggiornamento distinto; Alpine introdurrebbe musl/wheel diverse. Non
   sostenere che il tag legacy mobile 3.12-slim fosse già bookworm: la fonte
   official-images letta negli antecedenti associa quel tag a trixie.
2. Dichiarare la manutenzione: bookworm è in **LTS fino al 30 giugno 2028**,
   amd64 compresa, dopo il supporto regolare terminato l'11 luglio 2026. La patch
   3.12.13 non è nella serie Docker corrente mantenuta (3.12.15); l'esistenza del
   tag storico non promette nuove rebuild. È un punto iniziale riproducibile per
   la run, non una promessa di immagine aggiornata per un deploy.
3. Prima del FROM/COPY acquisire con strumenti **nativi** i manifest delle immagini
   Python e uv (`docker buildx imagetools inspect`), distinguere index e manifest
   linux/amd64, registrare entrambi i digest, data e piattaforma e usare digest
   reali. Non copiare i valori del riassunto web della review. Tag indisponibile
   o digest non verificabile = stop P6 al supervisore, nessun fallback mobile.
4. Build multi-stage, stesso digest Python nel runtime; uv copiato dalla propria
   immagine pinned. Python container `/usr/local/bin/python`, **--no-managed-python**
   e download Python vietati: nessun secondo managed host nel container.
   Sync dal lock, no dev, marker-server+marker-cpu, package **non editable** in
   `/opt/venv`; sync finale completo se il primo layer usa --no-install-project.
   Come ultimo passo installare in /opt/venv la wheel canonica prodotta dalla
   sdist del medesimo contesto identificato; uv pip --no-deps --no-build e
   --reinstall-package, nessun codice copiato a mano. Nel builder manifest
   S/B e confronto dieci moduli contesto/sdist/wheel/RECORD; nel runtime
   confronto /opt/venv e receipt immagine prima del probe V8. Cache/layer
   non sostituiscono questi confronti. Nuovo input di build invalida V7/V8.
   Nessun pip extra fuori lock. Copy ammesso per pyproject/lock/build constraints,
   .python-version e dieci moduli, **README/licenze/altri file realmente consumati
   dai metadata** (CLA-S3 r002); codice separato da `/data`. Avvio runtime con interprete venv
   `-I -m marker_api_server`; non copia del solo server per mascherare packaging.
5. Native bookworm: candidati runtime ca-certificates/curl, libpango-1.0-0,
   libpangoft2-1.0-0, libharfbuzz-subset0, fontconfig/fonts-dejavu-core; libgomp1
   e altre librerie solo se richieste dallo stack installato. La fonte WeasyPrint
   63.0 orientava r002; la guida primaria 63.1 è stata recuperata nella review
   ChatGPT r002. I dati statici non sono un collaudo delle native:
   verificare nomi/versioni ABI sulla release e WeasyPrint realmente locked,
   `weasyprint --info`, import/provider e rendering HTML sintetico locale senza
   Marker/modelli in V8. curl è health; Pandoc host non serve al wrapper per
   supposizione. Compiler/Git/dev native solo builder se un sdist ne dimostra
   bisogno; Poppler/X/GL/cairo legacy da conservare/rimuovere con evidenza.
6. Congelare repository apt bookworm e bookworm-security con timestamp snapshot
   **effettivamente disponibile**, scelto e registrato nella preparazione (data
   obiettivo 2026-10-02), firme attive, versioni dpkg e dipendenze. LTS non garantisce
   ogni pacchetto: registrare limiti del set usato. Nessun mirror mobile implicito
   in caso di errore; valid-until solo per la consultazione storica quando necessario.
7. `.dockerignore` a lista ammessa della build, con esclusione verificata di .git,
   .venv*, temp/tmp, .env/.env.*, local.env, config JSON operativo, segreti/dati,
   source_*, output/log, cache modelli/tokenizer, build/dist e docker-data/.docker.
   L'eventuale .env.example è l'unica eccezione esplicita se davvero necessaria.
   V7 dimostra che sentinelle innocue non entrano nel contesto inviato al builder.
8. Entrambi i Compose CPU: stesso build arg/extra, TORCH_DEVICE=cpu; temporanei
   sotto `/data/tmp` mediante TMPDIR, input separati, venv mai coperta dai mount.
   Cache/font del Marker/Surya installato sotto path espliciti verificati, volumi
   di prova distinti; non assumere sufficiente `/root/.cache/datalab/models`.
   Verificare sul codice installato variabili e path; eventuale cache dati/font
   non identificata blocca la documentazione operativa, non si conserva un mount
   casuale. Non riutilizzare/cancellare i volumi dell'utente nelle prove.
9. Compose build conserva le risorse memory 8G/reservation 4G come limite candidato
   di quel file, da mostrare nel config, senza attribuirgli throughput collaudato.
   Rimuovere env non riconosciute/MAX_PAGES/WORKER_MULTIPLIER se non provate sul
   server pinned o etichettarle illustrativamente; LOG_LEVEL secondo uso reale.
   GPU override sceglie extra cu126/TORCH_DEVICE=cuda e reservation NVIDIA
   `capabilities: [gpu]`, validata con config; non device_requests non supportato
   nel Compose README. Configurazione GPU non prova hardware o inferenza.
10. Politica di aggiornamento: nessun latest/upgrade automatico. Prima di un
    eventuale deploy, o per aggiornamento patch/base/uv/native/lock, il responsabile
    operativo porta al supervisore la nuova identità e ripete **V7/V8**, più V1–V6
    se cambia interprete/runtime. Cambio distribuzione/Marker semantico richiede
    riesame e possibile V10. Manifest precedenti conservati; il pianificatore
    non registra deploy o digest ancora sconosciuti come fatti.

Healthcheck conserva liveness HTTP; `compose config`, build e probe offline non
eseguono startup normale. La build V7 può acquisire dipendenze ML/native (GB):
all'inizio di quel passo l'implementatore presenta al supervisore stima, target,
rete/disco e comandi. Se non coperti dal mandato implementativo, il supervisore
definisce il perimetro prima del download. A6 resta aperto se V7/V8 impedite.
Nessun peso/font scaricato implicitamente da build, test o startup diagnostico.

### P7 — README, guide, help e responsabilità — CLA-P008 r001 / CLA-P004 r002

L’inventario corretto in [findings r003 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
e [checks r003 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md) copre **tutti i 74 blocchi**:
63 shell/Python/YAML, incluse le cinque bash indentate/lista, e 11
configurazioni/output/alberi/Markdown. Nove inline, incluso Git a L1538.
Linguaggi: 60 bash, 2 YAML, 1 Python, 7 JSON, 3 senza linguaggio, 1 Markdown.
Le cinque guide docs attuali non hanno comandi. Epilogo pipeline: sei
run_full_pipeline.py inesistenti; opzioni AST reali catalogate.
I vecchi 58 record e il primo riscontro ChatGPT restano invariati.

Regola dell’estrazione futura: scanner a stati delle fence backtick/tilde,
con lunghezza delimitatore e chiusura coerenti, normalizzazione dei soli
prefissi di lista/quote/indentazione per riconoscere il contenitore; mantenere
coordinate e byte originali. Non cercare soltanto fence a colonna zero.
Inline fuori dalle fence: distinguere comandi da nomi file, chiavi JSON/opzioni.
L’estrazione è statica e non esegue i blocchi.

Quattro classi: verificabile (prova futura innocua identificata), illustrativo
(pseudocodice, configurazione/esempio/output non catturato, servizio/hardware
non verificati), da correggere, da rimuovere. Tutti gli 11 non operativi sono
**illustrativi nella baseline r003**, motivati individualmente nelle evidenze.
V9 può rendere verificabile il JSON completo confrontandolo con create-default
V4, i frammenti con config/override effettivi, report/frontmatter/alberi con
output e packaging reali. Campi inesistenti/promesse errate si correggono anche
negli illustrativi; i numeri d’esempio non sono risultati delle prove.

Congelare il pre con hash README. Nel post includere nuove guide/help e assegnare
a ogni blocco/inline precedente mantenuto/modificato/rimosso/spostato, più i
nuovi elementi. Riconciliare totali e delta motivati: **non aspettarsi 74 nel post**.
Ogni comando mantenuto ha profilo, CWD/prerequisiti, nuova forma e prova/esito
oppure limite illustrativo/non eseguito. V9 non avvia Marker/cloud/GPU senza
mandato operativo pertinente.

Interventi mirati:

- Sostituire venv/pip/requirements/Poetry/dev extra con flusso uv managed unico,
  guardia manual e preflight origine; dopo modifiche ricostruzione/reinstallazione
  mirata P2 e confronto S/B/I prima di --no-sync. Avvio sicuro da dati,
  interprete/package installato e test per strato, anche per IDE/AGENTS.
- Correggere **epilogo master_workflow --help**, tutte le sei invocazioni di
  `run_full_pipeline.py` inesistente, usando markdown-pipeline con opzioni reali;
  rimuovere `--chunking-strategy semantic`, senza aggiungerla al parser.
- Correggere source_ebooks verso directory realmente configurate, path del
  validation_report (workspace) e esempio del costruttore UnifiedDocumentConverter
  (ConfigManager, non config.conversion_settings). Help/script -m coerenti.
- Tenere gli script downstream assenti (analyze_documents, prepare_training_data,
  embed_chunks, add_to_vector_store, build_rag_system, fine_tune_model) solo come
  **pseudocodice esterno illustrativo**. Non installarli o farli sembrare disponibili.
- Distinguere Docker config/build, startup pesi e inferenza; allineare entrambi
  i Compose, cache/mount, health e limiti GPU. Rimuovere il dummy PDF e la promessa
  di verifica reale mediante test_conversion.py; rinviare a V10 con PDF valido.
- Sostituire le pulizie globali prune/down -v con recupero circoscritto dei soli
  artefatti di prova; rimuovere dump di chiavi/.env come diagnostica. I comandi
  amministrativi OS/Toolkit restano prerequisiti illustrativi, senza modifiche
  globali automatiche in questa run.

Guide italiane dopo implementazione, applicando write-diataxis-docs:
`docs/how-to/ambiente-uv.md`, `docs/how-to/marker-legacy-docker.md`,
`docs/reference/toolchain-legacy.md`; aggiornare gli indici docs pertinenti.
Documentare versioni, extra, cinque script, dieci moduli, dati/.env/precedenze,
cache tokenizer, Linux collaudato e limiti, senza presentare la nuova app come
disponibile. .env.example solo con variabili legacy effettive e valori innocui;
UV_PROJECT_ENVIRONMENT e toolchain non diventano configurazione applicativa.

**AGENTS.md:** nella futura implementazione approvata, **l'implementatore**
sostituisce la sola istruzione della suite legacy con C-fast e prerequisiti
installazione/cache, aggiungendo rimando alla guida per C-api/C-package/C-docker e
governance, preflight runner R prima della baseline, managed/origine,
ricostruzione/reinstallazione P2 e dieci hash prima dei comandi.
La sola frase “risincronizzare” non è istruzione sufficiente. Mantiene il limite della suite selezionata e i 67 storici come storia.
Non riscrive governance/ruoli. **Il supervisore** mantiene STATE, handover comuni,
indici/eventi/arbitrati/snapshot e aggiornamenti di stato degli ADR. L'implementatore
registra nel changelog solo il lavoro realizzato e verifiche/limiti, preservando
le voci preesistenti; il pianificatore r003 non modifica nessuno di questi file.

## Criteri di accettazione e verifiche

### A1–A7 — tutti i criteri del brief, invariati

| ID / requisito | Interventi | Prove, attese e limiti |
| --- | --- | --- |
| **A1** Ambiente pulito ricreabile usando il lockfile; scelta Python esplicita | P1 | V1: guardia/origine senza variabili, probe cache P2, due base/dev nuovi con stesso inventario selezionato e lock invariato; matrice python3/uv run/venv prima/dopo con/senza PYENV_VERSION; origini managed esplicite. Preparazione rete dichiarata |
| **A2** Dipendenze di sviluppo separate dal runtime; package installato con moduli corretti | P1/P2/P5 | V1–V3: 10 moduli/5 script, metadata e assenza dev/ML/server nel base; API extra esplicito. Build/install sono preparazione, no raccolta con rete |
| **A3** Import ed entry point verificati fuori dalla directory del codice | P2/P3 | V3–V5: catena S/B/I verificata prima delle prove, wheel esterna, origini/transitive, cinque CLI reali, fasi -I con dati e sentinelle/PYTHONPATH avverso, exit/output. Conftest non prova packaging |
| **A4** Suite pertinente eseguita, distinguendo regressioni legacy, mock e prove reali | P0/P4 | §R PASS prima di V0; V0/V5/V6/V8 con S/B/I/E: baseline, F1–F6, confronti byte, >=2 chunk e overlap, conteggi per strato, mock/API/Docker distinti. Cache/Pandoc e assenza rete dichiarate; nessuno skip essenziale |
| **A5** README/guide aggiornati senza comandi inesistenti e senza dichiarare collaudi non fatti | P7 | V4/V9: inventario completo e help coerenti, flusso uv provato, istruzione AGENTS operativa aggiornata. Pseudocodice/V10 non eseguiti dichiarati |
| **A6** Build Docker coerente col perimetro del piano e limiti dei motori documentati | P5/P6 | **V7/V8 obbligatorie**: build CPU dal lock, digest nativi, dpkg/native, contesto, due config Compose, contratti offline. local_marker non verificato dopo migrazione; V10/V11 rinviate salvo incompatibilità che renda V10 essenziale |
| **A7** Due review del piano, arbitrato GO, implementazione e due review con arbitrato finale | P0/consegne | V12: quattro report nuovi reali, due arbitrati relativi agli snapshot correnti; r001 storico. Manualità Git dell'utente, nessun GO del pianificatore |

### Catalogo V0–V12 e condizioni di uscita

| ID | Strato/procedura | Prerequisiti e costo futuro | Stato necessario alla consegna implementativa |
| --- | --- | --- | --- |
| V0 | baseline F1–F6, script/suite, impronta/manifest originali; runner R PASS prima di P0 | ambiente separato, tokenizer pronto, Pandoc per F4; minuti dopo preparazione | essenziale per contenuto pertinente; baseline impedita resta aperta |
| V1 | matrice interpreti con shell senza variabili, guardia/origine, lock check, sync base/dev e probe cache discriminante | download Python/pacchetti separato; minuti | PASS, hash lock e origini coerenti |
| V2 | catena S/B: sdist e wheel dalla sdist, dieci hash tar/ZIP/RECORD/metadata/backend | backend constraints/cache preparati; minuti, no ML implicita | PASS, 10 moduli/5 script e input identificati |
| V3 | catena S/B/I: wheel esterna, dieci hash installati, runtime hash-locked/build constraints, import/pip check | installazione dichiarata prima dei test | PASS, origini sotto wheel-env; no sorgenti/copied modules |
| V4 | help reali, config create/show, clean/validate, -I -m e script clone | ambienti installati, fixture sintetiche | PASS, CLI e contenuti effettivi; help da solo non basta |
| V5 | fasi esterne F1–F6, .env/shell, sentinelle, caso negativo, contenuto | wheel/Pandoc/cache/runner offline e baseline | PASS, dati preservati/nessuna sentinella eseguita/nessuna perdita nuova |
| V6 | C-fast/API/packaging/governance/discovery, conteggi e assenza fetch/build | preparazioni per strato già concluse; secondi/minuti | PASS per ogni strato pertinente, no skip essenziale |
| V7 | manifest nativi, contesto, config CPU/build/GPU, build CPU/inventari | Docker/registry/apt/ML, GB e minuti; passo operativo distinto | **PASS obbligatorio A6**, config da sola insufficiente |
| V8 | container offline import/native/contratti; modelli/font mock | immagine V7 identificata, Docker; minuti | **PASS obbligatorio A6**, non inferenza |
| V9 | inventario e prove documentali/operative consentite | ambienti pronti, docs/help nuovi | PASS o esempi illustrativi espliciti; nessun comando inesistente operativo |
| V10 | CPU startup/pesi/font e PDF+non-PDF reale/contenuto/errori | **non autorizzata**, futuro perimetro dal supervisore | NON_ESEGUITA nel mandato; local_marker non verificato. Essenziale se incompatibilità/scostamenti lo richiedono |
| V11 | CUDA/driver/device, stessa verifica reale e risorse | **non autorizzata**, hardware e mandato separati | NON_ESEGUITA; nessun supporto GPU collaudato dichiarato |
| V12 | snapshot/doppia review/arbitrato per piano e implementazione | supervisore/revisori in chat nuove | passaggio reale obbligatorio, non sostituibile con i test |

### Procedure future V1–V3: preparazione dichiarata

Tutti i comandi di questa sezione sono **da eseguire dopo il GO**, non ora.
Le variabili RUN_* appartengono alla sola subshell della prova, mai HOME/CODEX_HOME.
`RUN_REPO` è la radice assoluta del clone; `RUN_VERIFY` una nuova directory esterna
creata con `mktemp -d /tmp/a001-uv-verify-XXXXXX`; `RUN_WHEEL` il singolo artefatto
identificato da manifest/hash (non la prima wheel di un vecchio dist).
Preimpostare nella subshell `UV_PYTHON_INSTALL_DIR="$RUN_REPO/.venv-python"`,
`UV_PYTHON_DOWNLOADS=never`, cache uv locale ignorata e CWD clone. L'acquisizione
iniziale Python descritta in P1 precede questi comandi.

V2 precede l’installazione canonica delle venv V1/V3: freeze S dello stage
con il supervisore, build in out-dir nuovo e ispezione prima di consumare gli
artefatti. Il comando uv build predefinito produce sdist e wheel dalla sdist;
l’help 0.10.10 lo conferma. Conservare anche il log della provenienza effettiva.
Non usare --clear contro un dist operativo.

```bash
uv lock --python 3.12.13 --managed-python --no-python-downloads
uv lock --check --python 3.12.13 --managed-python --no-python-downloads
uv build --python 3.12.13 --managed-python --no-python-downloads --build-constraints "$RUN_REPO/build-constraints.txt" --out-dir "$RUN_VERIFY/dist"
```

Solo dopo verifica V2 definire RUN_SDIST/RUN_WHEEL ai singoli file identificati
da B, RUN_SOURCE_MANIFEST a S e RUN_CHECK al diagnostico P2, tutti assoluti.
Prima del freeze archivi il controllo --archives-only confronta S con tar/ZIP/
RECORD; se una differenza emerge, stop senza installazione. Ogni sync sotto
ricostruisce il progetto, poi l’installazione canonica collega I alla wheel B.
Le righe sono preparazione con cache/backend già identificati, fuori dalle suite.

```bash
UV_PROJECT_ENVIRONMENT="$RUN_REPO/.venv" uv sync --locked --no-default-groups --no-editable --reinstall-package markdown-for-llms --python 3.12.13 --managed-python --no-python-downloads
UV_PROJECT_ENVIRONMENT="$RUN_VERIFY/base-a" uv sync --locked --no-default-groups --no-editable --reinstall-package markdown-for-llms --python 3.12.13 --managed-python --no-python-downloads
UV_PROJECT_ENVIRONMENT="$RUN_VERIFY/base-b" uv sync --locked --no-default-groups --no-editable --reinstall-package markdown-for-llms --python 3.12.13 --managed-python --no-python-downloads
UV_PROJECT_ENVIRONMENT="$RUN_VERIFY/dev-a" uv sync --locked --no-default-groups --group dev --no-editable --reinstall-package markdown-for-llms --python 3.12.13 --managed-python --no-python-downloads
UV_PROJECT_ENVIRONMENT="$RUN_VERIFY/dev-b" uv sync --locked --no-default-groups --group dev --no-editable --reinstall-package markdown-for-llms --python 3.12.13 --managed-python --no-python-downloads
```

Server host CPU, **solo se preparato nel perimetro operativo identificato**:

```bash
UV_PROJECT_ENVIRONMENT="$RUN_REPO/.venv-marker" uv sync --locked --no-default-groups --extra marker-server --extra marker-cpu --no-editable --reinstall-package markdown-for-llms --python 3.12.13 --managed-python --no-python-downloads
```

L’eventuale ambiente host GPU futuro usa lo stesso comando con
UV_PROJECT_ENVIRONMENT=.venv-marker-gpu e marker-cu126 al posto di marker-cpu;
non viene preparato nel mandato corrente. Nessuno startup host è implicito.

Per **ognuno** di .venv/base-a/base-b/dev-a/dev-b e poi dev-env/api-env,
porre RUN_ENV al path della riga appena eseguita ed eseguire le due righe
seguenti. Stessa coppia per .venv-marker e .venv-marker-gpu se la loro
preparazione è autorizzata (extra server+CPU oppure server+cu126).

```bash
uv pip install --python "$RUN_ENV/bin/python" --no-deps --no-build --reinstall-package markdown-for-llms "$RUN_WHEEL"
"$RUN_ENV/bin/python" -I "$RUN_CHECK" --source-manifest "$RUN_SOURCE_MANIFEST" --sdist "$RUN_SDIST" --wheel "$RUN_WHEEL" --profile "$RUN_PROFILE" --expected-managed "$RUN_REPO/.venv-python" --receipt "$RUN_RECEIPT"
```

RUN_PROFILE=base per .venv/base/dev, api per api-env, cpu per server CPU.
Il parametro diagnostico cpu copre il contratto packaging del server; per
il candidato GPU, se autorizzato, usare il controllo di file api e registrare
separatamente la variante CUDA nell’inventario. RUN_RECEIPT è nuovo e diverso
per ogni ambiente/preflight, con S/B/I/stage. Pip install della wheel vieta
build/sdist; non ha bisogno di vincoli backend aggiuntivi, usa solo B validata.

Wheel esterna V3, in directory fuori clone, senza test/dev/FastAPI:

```bash
uv export --locked --no-default-groups --no-emit-project --format requirements.txt --output-file "$RUN_VERIFY/runtime.txt"
uv venv --python 3.12.13 --managed-python --no-python-downloads "$RUN_VERIFY/wheel-env"
uv pip sync --python "$RUN_VERIFY/wheel-env/bin/python" --build-constraints "$RUN_REPO/build-constraints.txt" --require-hashes "$RUN_VERIFY/runtime.txt"
uv pip install --python "$RUN_VERIFY/wheel-env/bin/python" --no-deps --no-build --reinstall-package markdown-for-llms "$RUN_WHEEL"
uv pip check --python "$RUN_VERIFY/wheel-env/bin/python"
"$RUN_VERIFY/wheel-env/bin/python" -I "$RUN_CHECK" --source-manifest "$RUN_SOURCE_MANIFEST" --sdist "$RUN_SDIST" --wheel "$RUN_WHEEL" --profile base --expected-managed "$RUN_REPO/.venv-python" --receipt "$RUN_VERIFY/wheel-env-preflight.json"
```

Prima creazione lock unica; poi --locked/check con hash prima/dopo.
**GPT-P001 r002:** ogni altro uv pip install/sync/compile che possa costruire
deve passare --build-constraints con file assoluto, oppure --no-build/sole
wheel e stop su sdist. Una wheel di progetto già costruita non è un input
sorgente. Non usare pip correttivo fuori lock. Runtime export hash-locked
e backend constraints hanno funzioni diverse: i primi hash non congelano
il backend. Inventariare per ogni build artefatti di ingresso, backend
effettivo/versione/hash, requirements dinamici ed output. Stop se ignoti.
Due sync nuovi confrontano inventari selezionati e origini; cache wheel e
venv nuova non equivalgono a rebuild. Il preflight deve legare ogni I a S/B.

V1 confronta **prima/dopo il file pin** la matrice P1, usando directory probe
temporanee con metadata equivalenti, senza rimuovere il pin definitivo o toccare
pyenv globale. Il caso prima del pin per uv usa richiesta esplicita 3.12.13 e flag managed,
con directory locale selezionata e confronto dell’origine con RUN_MANAGED_PY.
Nessuna selezione viene inferita dal solo numero di versione.
Per python3, sia PATH ereditato sia shim prioritario nel **solo subprocess**;
PYENV_VERSION assente/3.12.3. Per uv run ambiente già preparato con
`--locked --no-sync --offline --no-default-groups --python 3.12.13 --managed-python
--no-python-downloads python -I -c <probe stdlib>`, e per interprete assoluto
`<venv>/bin/python -I -c <probe stdlib>`. Registrare l'assenza della patch in pyenv,
origini prima/dopo e fallimenti, senza inferire la selezione dal solo --version.

V1 esegue inoltre il caso senza variabili di P1 e il probe discriminante P2,
come preparazione separata. Sonda senza variabili, dentro R già PASS:
`uv run --no-sync python -I "$RUN_REPO/scripts/diagnostics/run-a001-fase0-uv/check_python_origin.py"
--expected-managed "$RUN_REPO/.venv-python"`; il diagnostico stdlib futuro
rifiuta origini diverse prima degli import applicativi. Il path atteso è
argomento, non variabile uv ereditata. Rimuovere tutte le UV_* e non passare
--offline/--no-python-downloads in questo caso; registrare log della guardia
manual e inventari managed prima/dopo. Qualunque tentativo di download è FAIL.
Il caso senza venv usa una copia-probe con sola configurazione identificata;
con venv locale il comando identico deve usare il managed atteso o fallire.
Nessuna creazione o spostamento di directory managed globali per simulare il caso.

V3 usa `RUN_WHEEL_PY="$RUN_VERIFY/wheel-env/bin/python"` e `-I -c` da workspace
esterno: import dei nove moduli base e transitive, __file__/spec, metadata
distribution/entry_points e eventuale direct_url non editable. Nessuna origine
clone nei figli. I log legacy aperti all'import sono attesi solo nel workspace
scrivibile e inventariati. Marker server import richiede l'ambiente API, non base.

### Procedure future V4/V5: esecuzione installata e contenuto

Con CWD nuovo workspace dati e ambiente ripulito da PYTHONPATH/PYTHONHOME:
binari assoluti wheel-env per markdown-pipeline/config/chunk --help; config
--create-default/--show e confronto JSON con -I -m config e script diretto clone
nell'interprete progetto installato. Clean/validate si eseguono realmente senza
argomenti. Fixture converted pronta e output inizialmente vuoti, JSON sintetico:

```bash
"$RUN_VERIFY/wheel-env/bin/markdown-pipeline" --step cleaning --force
"$RUN_VERIFY/wheel-env/bin/markdown-pipeline" --step validation --force
"$RUN_VERIFY/wheel-env/bin/markdown-pipeline" --step chunking --force --llm custom --chunk-size 1000 --overlap 100
```

Ripetere su workspace indipendente con interprete assoluto `-I -m master_workflow`.
Per F4 usare --step conversion con una sola sorgente HTML e backend Pandoc,
summary failed=0, nessun client HTTP usato. Caso negativo senza converted per
cleaning: exit 1 e niente output dichiarato riuscito. Salvare stdout/stderr anche
dei figli, file e confronto contenuto completo con baseline.

V5 .env: workspace con **CHUNK_SIZE=1200 e OVERLAP_SIZE=80**, senza queste variabili
nella shell; pipeline chunking su F6, osservabile `chunking_parameters.chunk_size`
e `overlap` nel metadata e `settings` nel pipeline_state = 1200/80. Secondo processo
pulito con shell CHUNK_SIZE=1600/OVERLAP_SIZE=120 prevale =1600/120. .env immutato;
CLI identica in entrambi i casi, valori default overriden nell'ordine legacy.
Confrontare ogni caso con la propria baseline equivalente e verificare helper
prima degli override; le variabili sono sintetiche, nessuna credenziale.

V5 sentinelle: tutti **dieci** i nomi applicativi di P2, compresi
master_workflow.py, unified_converter.py, batch_monitor.py e marker_api_server.py,
più requests.py, tiktoken.py, tqdm.py, dotenv.py: **14 file** sintetici,
ciascuno scrive un marcatore univoco se eseguito e solleva errore.
Il workspace conserva JSON/.env/asset/output e PYTHONPATH avverso.

Modalità padre/figli: console assoluta con PYTHONPATH/PYTHONHOME ripuliti;
interprete assoluto -I -m master_workflow con PYTHONPATH volutamente al workspace;
script-clone a path assoluto fidato, ambiente ripulito e package installato
identico nello stesso interprete. Per ogni modalità registrare origini nel
padre e nei figli delle fasi -I -m, non soltanto l’assenza di un marcatore.
I tre step reali devono produrre contenuti/override attesi, nessun marcatore
in tutti i 14 casi. Le sentinelle non si cancellano/rinominano per passare.

Nel base usare find_spec/origine e **lettura/hash** di marker_api_server.py,
senza importare FastAPI/Marker. L’avvio/import server runtime è limitato
all’api-env appropriato, con startup/converter mock prima del lifespan.
Per batch_monitor/unified_converter controllare anche import isolato delle
copie installate senza invocazioni HTTP. Nessun nuovo console script.
La compatibilità script-clone resta separata dalla wheel esterna A3.

Prima di V3/V4/V5 e dei relativi processi ripetere il preflight S/B/I P2;
S e B sono quelli dello stage corrente. Spec/RECORD o sole sentinelle non
sostituiscono il confronto col sorgente. V5 comprende il controllo negativo
discriminante P2 già fatto nella preparazione, senza build/rete nella suite.

### Procedure future V6: comandi esatti per strato e discovery

**Preparazione separata**, dal clone, con variabili/managed della sezione V1:

```bash
UV_PROJECT_ENVIRONMENT="$RUN_VERIFY/dev-env" uv sync --locked --no-default-groups --group dev --no-editable --reinstall-package markdown-for-llms --python 3.12.13 --managed-python --no-python-downloads
UV_PROJECT_ENVIRONMENT="$RUN_VERIFY/api-env" uv sync --locked --no-default-groups --group dev --extra marker-server --no-editable --reinstall-package markdown-for-llms --python 3.12.13 --managed-python --no-python-downloads
```

Dopo ciascuno dei due sync eseguire l’installazione canonica e il preflight
P2 con RUN_ENV rispettivamente dev-env/api-env. Non iniziare test su copie
non confrontate. Rieseguire R con questi interpreti e Pandoc/cache definitive.

Definire `RUN_DEV_PY="$RUN_VERIFY/dev-env/bin/python"`,
`RUN_API_PY="$RUN_VERIFY/api-env/bin/python"`, `RUN_WHEEL_PY` come in V3,
`RUN_BASELINE` al manifest/output V0 identificato, `RUN_IMAGE` al digest ID V7.
La suite packaging riceve RUN_WHEEL_PY/RUN_BASELINE esportate nella sola prova;
Docker riceve RUN_IMAGE. Prima dei test: package installato in ciascun interprete,
pip check/inventari, cache tiktoken/hash, guardia rete e Pandoc per packaging.
I comandi sotto partono dal clone, ma Python -I e i figli installati non dipendono
dai sys.path di conftest. Usare ambienti pronti; nessun uv sync implicito.

Runner per V0/V4/V5/V6: usare il candidato **già PASS in R**, con namespace
ereditato dai figli. Shell futura primaria:
`unshare --user --map-root-user --net -- bash --noprofile --norc`;
alternativa confermata:
`firejail --noprofile --net=none bash --noprofile --norc`.
Prima della collection nella shell ripetere il diagnostico R e i preflight
S/B/I pertinenti. Dentro si eseguono i comandi seguenti, con RUN_* e cache
tiktoken esportati nella sola subshell, CWD clone fidato.
Se l’ambiente cambia o il runner non passa più, prove IMPEDITE; nessuna
fallback non confinata. AF_UNIX locale ammesso per API, Internet bloccato.
C-docker usa daemon locale su Unix socket fuori dalla shell solo se necessario:
driver con immagine locale, --pull=never e --network none. Il socket daemon
non viene concesso agli altri strati e non ne prova l’isolamento.

Prima di **C-fast**: dieci hash dev-env rispetto S/B.
Prima di **C-api**: dieci hash api-env, poi import API nel profilo giusto.
Prima di **C-package**: dieci hash dev-env **e wheel-env**; RUN_WHEEL_PY/
RUN_BASELINE/S/B/receipt assoluti già pronti.
Prima V7/V8: builder/wheel e /opt/venv, con impronta S-stage-image.
Ogni esito C riceve la receipt corrente; una mismatch impedisce la prova.
Collection e test non syncano/buildano/installano, neppure per “riparare” un
package assente. Governance/discovery registrano anche gli input propri.

| Etichetta | Comando futuro esatto |
| --- | --- |
| **C-fast** | `"$RUN_DEV_PY" -I -m pytest tests/ --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q` |
| **C-api** | `"$RUN_API_PY" -I -m pytest tests/api/ -q` |
| **C-package** | `"$RUN_DEV_PY" -I -m pytest tests/packaging/ -q` |
| **C-docker** | `"$RUN_DEV_PY" -I -m pytest tests/docker/test_marker_contract.py -q` |
| **C-governance** | `"$RUN_DEV_PY" -I -m unittest discover -s tests/governance -v` |
| **C-discovery selezionata** | `"$RUN_DEV_PY" -I -m pytest tests/ --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker --collect-only -q` |
| **C-discovery default** | `"$RUN_DEV_PY" -I -m pytest --collect-only -q` |
| **C-discovery radice veloce** | `"$RUN_DEV_PY" -I -m pytest . --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker --collect-only -q` |
| **C-all raccolta tests** | `"$RUN_API_PY" -I -m pytest tests/ --collect-only -q` |
| **C-all raccolta radice** | `"$RUN_API_PY" -I -m pytest . --collect-only -q` |

I due smoke radice esclusi in addopts non collidono e non fanno HTTP. C-all importa
solo harness, non Marker: wheel/immagine mancanti **non** fanno build/download in
collection. Registrare lista esclusi e conteggi, non dire solo “nessun errore”.
Se root collection trova altri script/test, identificarli nel report senza
allargare gli ignore alla cieca. Un comando `pytest tests/` senza gli ignore e
senza ambienti/preparazione di tutti gli strati non è la suite veloce documentata.

La variante utente uv di C-fast, con ambiente già sync e UV_PROJECT_ENVIRONMENT
selezionato, è `uv run --locked --no-sync --offline --no-default-groups --group dev
--python 3.12.13 --managed-python --no-python-downloads python -I -m pytest tests/
--ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q`.
Per API includere **--extra marker-server** e api-env; non usare soltanto dev.
uv --offline protegge l'orchestrazione tool, la guardia rete protegge i test.
Eventuali black/flake8/mypy interessano soltanto file nuovi/modificati, senza
refactoring globale o test che specchino la sola implementazione.

### Procedure future V7/V8: build pesante distinta, probe offline

V7 prima della build, dentro il perimetro operativo definito:

```bash
docker buildx imagetools inspect python:3.12.13-slim-bookworm
docker buildx imagetools inspect ghcr.io/astral-sh/uv:0.10.10
docker compose -f docker-compose.yml config
docker compose -f docker-compose-build.yml config
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config
docker compose --progress plain -p a001-uv-v7 -f docker-compose.yml build marker-api
```

L'help locale congelato conferma --progress come opzione di **docker compose**, prima di
build; nessuna build/config è stata eseguita ora. Usare digest verificati prima
della build e controllare
profilo CPU, native, mount/cache/TMPDIR/input/risorse di **entrambi** i file. Override
GPU: solo config nel mandato ordinario; nessun startup o download pesi.

V7 contesto: target diagnostico con sentinelle **non segrete** in ogni categoria
esclusa, manifest del contesto effettivamente filtrato e verifica assenza; niente
sola affermazione sul COPY finale. Conservare log build, hash lock/constraints,
versioni Python/uv, dpkg, inventari pacchetti/varianti torch, digest immagine
costruita, moduli installati, assenza dev/compiler/cache non necessari nel runtime.
Il profilo CPU non deve contenere librerie CUDA per fallback d'indice.

Prima della build V7 acquisire S e hash dei dieci file del contesto reale;
produrre sdist/wheel canoniche nello stage builder e confrontarle con S e
RECORD. Prima del probe V8 leggere/hashare i dieci moduli in /opt/venv mediante
Python -I/stdlib via stdin, confrontare al manifest S/B fidato fornito al driver;
nessun mount di moduli host dentro il container per riparare una differenza.
Receipt con ID immagine/digest, B e S, e nuovo preflight ad ogni rebuild.

V8 C-docker lancia solo `docker run --pull=never --rm --network none -i --entrypoint
/opt/venv/bin/python "$RUN_IMAGE" -I -` con diagnostica fidata su stdin, senza
mount dei dati utente. Non usare il normale CMD/uvicorn/startup. La diagnostica
importa il codice installato e produce ogni subcaso:

- origini/versioni/server, Marker/Surya/torch CPU, PdfConverter/create_model_dict,
  signature bind, registry e provider full, bs4/filetype, WeasyPrint/native;
- rendering WeasyPrint di HTML sintetico con font di sistema già installati e
  nessuna risorsa remota, per verificare native senza inferenza Marker;
- MarkdownOutput reale e invocazione del wrapper con converter fake, markdown
  esatto/page_count/errori/cleanup. Nessuna dipendenza httpx/dev nell'immagine:
  richiamare la funzione async con UploadFile sintetico; il confine HTTP mock è
  già coperto da C-api nell'ambiente API;
- constructor con download_font e risoluzione predictor/processor di modelli
  sostituiti **prima** della chiamata; elenco delle patch e tentativi di rete
  bloccati registrati. Non chiamare create_model_dict o scaricare font per errore.

Un'importazione mancante, variante torch errata, backend non congelato o rendering
native fallito lascia V8 FAIL, anche se health avrebbe risposto healthy.
Provider detection non è collaudo completo di tutti i formati. Nessun prune/down -v
o cleanup di cache/volumi operativi; fermare solo oggetti di prova identificati.

### V9, V10/V11 e registrazione degli esiti

V9 compara inventario pre/post e help, prova gli esempi consentiti di ambiente,
CLI, dati sintetici e suite nel profilo corretto; controlla link locali,
git diff --check e coerenza degli stati. Installare/buildare è preparazione
registrata, non un effetto della verifica documentale. Istruzioni real-engine
V10/V11 in sezione distinta, etichetta non verificate e nessuna esecuzione ora.

Per un futuro V10 autorizzato: acquisizione pesi/font prima della prova, cache
identificata, startup ordinario CPU, PDF sintetico **valido** e almeno un documento
non-PDF full. POST /convert e /marker, Markdown/contenuto non vuoto, numeri/formule/
ordine/page_count, caso corrotto con errore; manifest input/output/asset/modelli/
font/config senza credenziali. Dopo preparazione, disabilitare egress dove possibile,
LLM remoto spento. Immagini vuote del wrapper restano limite, non bundle completo.
V11 aggiunge driver/Toolkit/GPU realmente usata/torch.version.cuda e risorse;
assenza hardware resta NON_ESEGUITA, senza fallback CPU/cloud presentato come GPU.
Queste sono istruzioni separate, **non autorizzazione o PASS nel mandato attuale**.

Ogni prova registra ID, **impronta worktree run_context dello stage**, SHA del
manifest snapshot e catena S/B/I/E con input/output identificati, oltre a branch/HEAD/hash, comando/argomenti/CWD/interprete,
profilo, prerequisiti/costi effettivi, tempi, stdout/stderr, exit atteso/osservato,
file/hash e confronto contenuto. Prima del rilascio delle evidenze, confronto con S finale: esiti invalidati sono
identificati come SUPERATI/DA_RIPETERE nel registro, senza cancellare gli originali.
Stati della singola prova: PASS, FAIL, IMPEDITA, NON_ESEGUITA con
motivo, distinguendo mock/reale. Codice 0 senza contenuto atteso è FAIL; health
non è prova di conversione. Perdita nuova o baseline incerta va al supervisore.

### Matrice 1 — Undici rilievi accolti r001

La matrice indica **interventi e prove future**, non rilievi già risolti nel codice.

| ID accolto | Sezioni/interventi r003 | Criterio di chiusura futuro e responsabilità |
| --- | --- | --- |
| **CLA-P001 r001** bloccante | P1: pin esatto motivato, percorso managed locale unico, matrice shim/bin/PYENV_VERSION, nota Windows | implementatore V1 osserva origini prima/dopo per python3/uv run/venv; P7 documenta; nessun pyenv globale |
| **CLA-P002 r001** bloccante | P4/V6: matrice di ogni test nuovo/mutato, API extra, separazione preparazione/test, ignores prima import e collection inerte | implementatore esegue C-fast/API/package/governance/discovery, progetto nello stesso interprete dei figli, conteggi/zero skip essenziale; V8 driver separato |
| **CLA-P003 r001** | P1/P5: wheel PyPI scelta e confronto Git; hash/metadata/contratto, altri sdist/backend vincolati; torch candidato | preparazione verifica realmente artefatto e provenienza; V2/V7 backend constraints, V8 sul codice installato; incompatibilità al supervisore |
| **CLA-P004 r001** | P3: helper workspace prima override, shell prevalente, cambio ricerca legacy | V5 .env1200/80 e shell1600/120 su metadata/pipeline_state; processo separato e dati innocui |
| **CLA-P005 r001** bloccante | P2/P3: sys.executable -I -m, ambiente figlio ripulito, preflight isolato, compatibilità clone installato | V3/V5 origini di moduli e transitive, sentinelle omonime e PYTHONPATH avverso mai eseguiti, contenuto/override conservati |
| **CLA-P006 r001** | P6: Compose build conservato/allineato al medesimo lock, profilo/mount/cache/risorse | V7 config esplicita **-f docker-compose-build.yml**, inventario entrambe configurazioni, nessuno startup implicito |
| **CLA-P007 r001** | P6: patch/base bookworm motivata, LTS/patch storica, native release-specifiche, politica aggiornamenti | implementatore acquisisce digest nativi prima dell'uso, apt/dpkg reali; V7/V8 e loro ripetizione agli aggiornamenti |
| **CLA-P008 r001** | P7: inventario completo statico e finale, README/guide/help, run_full_pipeline/opzione inesistente, AGENTS responsabile | implementatore V4/V9 verifica comandi consentiti e corregge AGENTS operativo; registri condivisi solo supervisore |
| **CLA-P009 r001** | P0/P4: validated esplicito e argomenti uguali, Python/dipendenze/tiktoken/cache identificati | V0/V5 confronto byte e incroci circoscritti se diverge, senza normalizzazione o attribuzione prematura |
| **CLA-P010 r001** | P5/P6/V10: rinvio operativo, local_marker non verificato, punto di proposta mandato pesi/font/CPU/hardware | supervisore dopo V7/V8 presenta eventuale perimetro separato; V10 essenziale se cambia semantica/incompatibilità, bisogno segnalato senza requisito passato |
| **GPT-P001 r001** | P4 F6: >=2 chunk in baseline/wheel, identici input/argomenti, confronto sequenza/metadati/confini e complemento overlap esistente | implementatore V0/V5 documenta multi-chunk e overlap effettivo; conserva difetti legacy senza correzioni qui |

I tre blocchi r001 sono stati affrontati **nel disegno r002**; nessuno è
chiuso operativamente. CLA-P003 r001 conserva il dettaglio ulteriore di
GPT-P001 **r002** sui pip/build; CLA-P008 r001 è completato da CLA-P004
**r002** sull’inventario. GPT-P001 r001 resta il multi-chunk, distinto dal pip.

### Matrice 2 — Sei rilievi accolti r002

| ID r002 / disposizione | Intervento r003 | Verifica futura / criterio per il supervisore |
| --- | --- | --- |
| **CLA-P001 r002**, unico blocco A3/A4 | P2: reinstall-package persistente/flag, wheel canonica, S/B/I/E per ogni ambiente host, dieci moduli e metadata/backend; P6 immagine; P7 istruzioni | V1 probe solo modulo in copia con cache popolata/metadata invariati, negativo prima sync e ripristino; V2 sdist/wheel/RECORD vs S; V3–V6 prima C-fast/API/package e V4/V5 installed vs S/B; V7/V8 /opt/venv. Impronte stage e invalidazioni motivate, mismatch mai PASS |
| **CLA-P002 r002**, accolto non bloccante | §R prima P0/V0: sonda innocua unshare e alternativa locale Firejail da confermare, target/argv/limiti | namespace/route/no egress e figli, AF_UNIX, Python/Pandoc/cache/temp disponibili; receipt PASS o V0 IMPEDITA; nessuna modifica host/privilegio persistente |
| **CLA-P003 r002**, accolto | P1: python-downloads manual nel progetto, acquisizione solo locale esplicita e preflight origini; P7/AGENTS/IDE | V1 parsing, shell senza variabili e inventari prima/dopo: errore esplicito o origine attesa, nessuna acquisizione altrove; manual non garantisce path né resiste a override deliberati |
| **CLA-P004 r002**, accolto A5 | P7: fence indentate/lista, tutti 74 blocchi e 9 inline, 11 non operativi illustrativi identificati; conserva antecedenti | Findings/checks statici corretti contro static-findings supervisore; V9 riconcilia pre/post e prove/etichette senza imporre totale finale uguale |
| **CLA-P005 r002**, accolto | P3/V5: dieci applicativi + quattro transitive, padre/figli/modi sicuri, origin/hash | 14 marcatori assenti e contenuti/override uguali; server spec/hash nel base, runtime solo api-env mock; nessun import FastAPI implicito |
| **GPT-P001 r002**, accolto | P1 e V1–V6: build-constraints esplicito in ogni pip che può costruire; wheel install --no-build | Runtime hash-locked, backend effettivi/versioni/input/output identificati; stop su backend ignoto o sdist vietata. Non confondere col multi-chunk GPT-P001 r001 |

### Matrice 3 — Sette suggerimenti/disposizioni r002

| Voce | Disposizione dell’arbitrato mantenuta | Sezioni / prova o limite r003 |
| --- | --- | --- |
| **CLA-S1 r002** | Rinviato, non bloccante | P1 mantiene lock universale: non introdurre tool.uv.environments o restrizioni di piattaforma implicite. Linux x86_64 è target di collaudo di questa run, non nuova promessa di esclusività |
| **CLA-S2 r002** | Accolto, precisione discovery | P4/V6 nominano temp/tmp/documentation/runs e ambienti/artefatti/dati, conservano esclusioni standard; liste e conteggi per C-all/partizione |
| **CLA-S3 r002** | Accolto condizionato a input backend | P2/P6/V7 includono README/licenze/altri metadata realmente consumati nel manifest e contesto Docker; nessun dato operativo incluso per comodità |
| **CLA-S4 r002** | Accolto | §R/P4/C-api permettono socketpair AF_UNIX locale, con positivo e negativo Internet; egress e isolamento figli restano bloccati |
| **CLA-S5 r002** | Accolto | V1/V6 eseguono uv lock --check separato con hash prima/dopo. --no-sync, anche combinato con --locked, non attesta coerenza del lock; run usa solo ambiente già validato |
| **CLA-S6 r002** | Accolto come caratterizzazione | P4/C-api registra versioni locked FastAPI/Starlette e warning/lifespan; niente refactoring on_event, pin alle “ultime” o soppressioni generiche |
| **CLA-S7 r002** | Accolto come limite | P2/P5/P7/V9: host .venv-marker non verificato se senza prova dedicata; non trasferire PASS Docker al server host, non avviare startup/inferenza via documentazione |


## Rischi, migrazione e recupero

| Rischio | Trattamento e recupero |
| --- | --- |
| python3 via shim cambia con pin non installato in pyenv | matrice osservabile e flusso managed unico; mai usare shell python3 come prova del progetto |
| Python 3.12.13/base storica rispetto a patch corrente | identità riproducibile dichiarata per run, non promessa di manutenzione; aggiornamento esplicito prima di deploy con prove ripetute |
| Grafo CPU/cu126/native/sdist non risolvibile | conservare errore/metadata, nessun fallback o pip fuori lock; supervisore rivaluta perimetro/progetto server |
| Copia non editable stantia o receipt di worktree diverso | reinstall-package, catena S/B/I/E, dieci hash e invalidazione; mismatch ferma test/avvio, mai cache clean globale |
| Runner negato prima della baseline | §R prova primario e alternativa; V0 IMPEDITA se entrambi falliscono, nessun cambiamento policy/privilegi |
| Shell senza variabili trova managed altrove | manual + prerequisiti origine/path; preflight rifiuta prima import, nessun download altrove o promessa anti-override |
| Nome flat o PYTHONPATH dal workspace | -I, preflight nel figlio, transitive/sentinelle e wheel non editable; namespace rimandato |
| Test parent conftest maschera package assente | package nello stesso interprete, preflight -I e prova esterna; build/install non nei test |
| .env anticipato/tardivo o ricerca site-packages | helper esplicito prima override e due osservabili deterministici, processi separati |
| Python/dipendenze/tokenizer alterano contenuto | manifest e confronto incrociato di sole fixture divergenti, nessuna normalizzazione |
| Download da tokenizer/font/startup | preparazione cache dichiarata, guardia rete, constructor mock, build/inferenza distinte |
| Wheel Marker non equivalente al tag o V10 necessaria | provenance/confronto file/contratti installati; riportare incompatibilità al supervisore, niente dichiarazione local_marker collaudato |
| Difetti legacy cleaning/asset/batch/overlap | inventario prima/dopo, nessun nuovo danno; non correggere algoritmi o chiamare completo output incompleto |
| V7/V8 impedite o review assente | criterio essenziale aperto, checkpoint e prossimo passo; nessun GO/chiusura artificiale |

Migrazione dati: solo workspace sintetici; niente riscrittura automatica dei
documenti/configurazioni dell'utente. JSON legacy compatibile; cambio ricerca .env
documentato. Gli ambienti nuovi non disinstallano pyenv o ambienti di altri progetti.
Preservare base, patch dei soli file della run, artefatti/manifest e output baseline.
In caso di interruzione, salvare checkpoint/log e fermare soltanto processi/container
di prova identificati, senza reset, stash implicito di temp, git clean o prune.
Eliminare selettivamente ambienti temporanei solo dopo archivio delle prove.

Ripristinare sorgenti non ricrea il precedente Marker da master: un'immagine
precedente è riferimento solo se digest/versioni sono disponibili e verificati.
La build legacy oggi non è una baseline di inferenza affidabile. Conservare
immagini/cache operative senza avviarle o scaricare modelli come recupero implicito.

## Consegne e handover

Pianificazione r003: questo piano, [findings — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[checks — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), inventario statico e
[checkpoint planning-r003 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md). Letture/hash/controlli
documentali non dimostrano il funzionamento della migrazione; nessun processo
applicativo o download avviato e nessuna attività continua dopo la consegna. Checks identifica i quattro
output nuovi; non contiene il proprio hash per evitare autoreferenzialità.
Il supervisore ne calcolerà il digest nel futuro snapshot plan-r003.

Dopo GO sul nuovo piano, l'implementatore consegna codice/lock/constraints/docs,
report §R/P0–P7/A1–A7, manifest S/B/I/E e receipt del runner,
backend/build/installazione, probe discriminante/ripristino, baseline/F1–F6, V0–V9 con
conteggi/esiti/limiti e checkpoint. V10/V11 restano NON_ESEGUITA finché un mandato
operativo esplicito le autorizza; incompatibilità che le rende essenziali va
segnalata prima della dichiarazione finale. R001/r002 non vengono corretti retroattivamente.

**Destinatario: supervisore.** Verificare identità, completezza, undici ID r001/sei r002/sette disposizioni,
congelare plan-r003 includendo evidenze/checkpoint, predisporre due nuove chat
di review indipendenti e arbitrare i due report reali. Il supervisore aggiorna
stato/changelog/indici nel proprio ruolo; questo pianificatore non li modifica.
Nessun GO r001/r002 si trasferisce. Nessuna implementazione prima del nuovo arbitrato;
commit/merge/push/promozione restano manuali dell'utente dopo GO finale.
