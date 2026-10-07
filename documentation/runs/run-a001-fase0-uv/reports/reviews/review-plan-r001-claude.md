# Review — piano — run-a001-fase0-uv — r001 — Claude

- Autore/provider/modello e chat: **Claude Opus 5.5** (ID modello `claude-opus-5-5[1m]`),
  provider **Anthropic**, sessione Claude Code nell'estensione VS Code. Nessun ID chat è
  esposto. Il percorso dello scratchpad contiene `d47bf2eb-59f7-486c-add0-5b9565aa8f10`,
  ma non è confermato come identificativo della chat. Ruolo: revisore Claude, in una
  chat distinta dal pianificatore (Codex/OpenAI), dal supervisore e dall'altro revisore.
- Prompt di origine: `temp/run-a001-fase0-uv/prompts/review-plan-r001-claude.md`
  (sha256 `b92457792273d497f1f265176f6b8fd4792feb0c90f95e1f2f3c8fd19c47108c` nel manifest),
  incollato dall'utente in questa chat.
- Oggetto e revisione esatta: `plans/plan-r001.md`, revisione **r001**, sha256
  `7f9dd0a2ded10c40ef9e3a42ae1424fb08a23059384468ea160dd7173fd3e8fb`. Il valore letto
  dal manifest coincide con quello ricalcolato.
- Snapshot, HEAD e impronta verificati prima/dopo: snapshot `plan-r001`; HEAD, `dev`
  e merge-base `66ba82200e5def5a4db76f9bafccb0731b506091`; branch `feature/run-a001-uv`;
  worktree_sha256 `111f2830bfbef9b3bf816b02ab8818e8daf2d8a272f37b05e9fc565a004a7b60`.
  `verify`: **MATCH prima** (exit 0); **MATCH dopo** (exit 0), con identità Git invariata.
- Indipendenza: **non** ho letto report, evidenze o checkpoint del revisore ChatGPT né
  alcun arbitrato. La cartella `reviews/` era vuota all'inizio e l'ho controllata solo per
  verificare l'assenza del mio file di destinazione. Nessun coordinamento con altre chat.
- Esito: **NO_GO** sul piano r001. I motivi sono i rilievi bloccanti CLA-P001 e CLA-P002.

## Ambito e prove

**File letti.** Prompt di review; `AGENTS.md`; skill `manage-implementation-run` e
`verify-conversion-fidelity`; `run-lifecycle.md`; template review e plan; `STATE.md`,
`brief.md`, `prompts/02-planning-r001.md`; `documentation/README.md`, indice ADR,
ADR 0001/0006/0007, roadmap (fase 0.1 e quadro), changelog. Il piano r001 è stato letto
**integralmente**, insieme al checkpoint `handovers/planning-r001.md` e alle evidenze
`findings.md`, `checks.json`, `local-inventory.json`, `python-catalog.txt`,
`python-list.txt` e agli help uv congelati (consultati per i flag citati). Ho letto
`supervisor-preparation-r001.md` e `supervisor-plan-review-r001/receipt.json`.
Sorgenti letti: `setup.py`, `requirements.txt`, `.gitignore`, `.env.example`, `Dockerfile`,
`docker-compose.yml`, `docker-compose-build.yml` e le sezioni del README su
installazione, uso, test, Docker, timeout, troubleshooting e contributing. Dei dieci
moduli ho letto integralmente `config`, `master_workflow`, `unified_converter` e
`marker_api_server`, e le parti rilevanti di `clean_markdown`, `validate_markdown`,
`chunk_markdown`, `logging_config` e `batch_monitor`. Per i test ho letto
`tests/conftest.py`, il test di integrazione completo, le importazioni e i conteggi
dei test unitari, gli smoke `test_pipeline.py` e `test_conversion.py` alla radice e un
estratto di `ebook2md.sh`.

**Comandi e controlli eseguiti.** Sono elencati in
[checks.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).

- Identità Git, snapshot e hash prima e dopo la review.
- Help locali di uv 0.10.10 per `lock`, `venv`, `pip install` e `pip check`, salvati
  come evidenze. Tutti i flag usati in V1–V3 esistono.
- Probe pyenv in una directory dello scratchpad, fuori dal repository:
  [pyenv-probe.txt — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
- Consultazione di fonti primarie con data e URL: PyPI per marker-pdf 1.10.2 e
  setuptools 84.0.0; GitHub API per tag e commit di Marker; pyproject di Marker al tag;
  Docker official-images e Docker Hub; pagina della release Debian bookworm.

**Affermazioni del piano confermate.** Il commit `5a41cdb…` corrisponde al tag firmato
v1.10.2. I vincoli Marker (Python ≥3.10, torch ≥2.7,<3, surya ≥0.17.1,<0.18, extra
full) sono confermati. Il tag `python:3.12.13-slim-bookworm` esiste. Setuptools 84.0.0
esiste e richiede Python ≥3.10. Python 3.12.15 è la patch mantenuta. I riferimenti di
riga ai sorgenti in `findings.md` corrispondono a quanto ho letto. È corretta anche la
diagnosi su `config.main` assente, su BASE_DIR legato al CWD, sul seed con
`skip_existing` nei test di integrazione e sui limiti del wrapper.

**Non eseguito, per vincolo del prompt.** Nessun lock/sync, installazione, build,
pytest/unittest, conversione, Docker build/up/run, avvio Marker o download. Nessuna
prova di implementazione viene attribuita al piano. Le procedure V0–V11 sono
valutate soltanto per fattibilità e sufficienza.

### Valutazione per criterio

| Criterio | Valutazione | Rilievi |
| --- | --- | --- |
| A1 — ambiente ricreabile dal lock, Python esplicito | V1 è solido: due ambienti nuovi, lock invariato, inventari. Il rapporto con pyenv è però descritto solo con `PYENV_VERSION` attiva | **CLA-P001** |
| A2 — dev separato dal runtime, moduli corretti | Il gruppo dev con `default-groups = []`, i dieci moduli espliciti, i cinque entry point, le verifiche V2/V3 dei metadati, l'assenza di dev/ML nel runtime e gli extra CPU/cu126 in conflitto sono un impianto corretto. La fonte di Marker è migliorabile | CLA-P003 |
| A3 — import ed entry point fuori dai sorgenti | Wheel non editable in una venv esterna, `-I`, `direct_url.json`, fasi reali con output vuoti e caso negativo con exit 1: la prova è adeguata e non si limita all'help. Restano l'ordine di caricamento di `.env` e il sys.path dei subprocess `-m` | CLA-P004, CLA-P005 |
| A4 — suite, mock e prove reali distinti | Fixture F1–F5, baseline prima delle modifiche, nessuna normalizzazione, confronto byte per byte e per invarianti, cache tiktoken preparata, smoke esclusi dalla raccolta: tutto coerente con la skill di fedeltà. La composizione di V6 non è eseguibile così com'è scritta | **CLA-P002**, CLA-P009 |
| A5 — istruzioni senza comandi inesistenti | Il perimetro è sensato e la nuova applicazione resta presentata come futura. Manca un inventario completo dei comandi del README e delle istruzioni collegate | CLA-P008 |
| A6 — Docker coerente e limiti dei motori | V7 (build dal lock) e V8 (contratti offline in container) sono obbligatorie. Il piano non riduce A6 alla lettura del Dockerfile, separa liveness da conversione e identifica i quattro livelli. Il rinvio di V10/V11 è accettabile se dichiarato. Restano lacune su un compose secondario e sulla base Debian | CLA-P006, CLA-P007, CLA-P010 |
| A7 — quattro report, due arbitrati, Git manuale | V12 e le consegne sono coerenti con ADR 0007 e con il protocollo. Il GO sul piano non autorizza da solo l'implementazione | nessuno |

## Rilievi

| ID | Severità e blocco sì/no | Posizione | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- |
| CLA-P001 | **Media — bloccante: sì** | plan-r001.md:104, 119–138; matrice A1 (:480) | Il piano tratta solo il caso con `PYENV_VERSION` impostata. Con pyenv 2.6.12 e `PYENV_VERSION` non impostata, che è la condizione della shell di questa review, un `.python-version` = `3.12.13` non installato in pyenv fa sì che `python3` nella radice esegua **silenziosamente `/usr/bin/python3`** al posto della 3.12.3 globale di pyenv. Il piano dice che il pin «consente gestione uv senza selezionare la vecchia pyenv», ma non dichiara l'effetto sui comandi non uv: `python3 -m pytest`, gli script legacy e AGENTS.md:63. La premessa «shell con PYENV_VERSION=3.12.3» non è generale. Impatto: la scelta dell'interprete diventa implicita e dipende dalla shell, contro A1 e contro il vincolo di non alterare l'uso di pyenv dell'utente. Il requisito di `UV_PYTHON_INSTALL_DIR` su «tutti» i comandi managed rende inoltre ambiguo il percorso documentato per gli utenti | [pyenv-probe.txt — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md); `~/.pyenv/libexec/pyenv-version-name`; `pyenv-which:62–80`; `findings.md`, tabella del contesto (PYENV_VERSION osservata solo nella shell del pianificatore) | Il piano r002: (1) documenta l'interprete effettivo di `python3`, `uv run` e `.venv/bin/python` con e senza `PYENV_VERSION`; (2) motiva il contenuto di `.python-version` (patch esatta con avvertenza, oppure un'alternativa) tenendo conto della risoluzione pyenv; (3) aggiunge a V1 un controllo osservabile: `sys.executable` di `python3` nella radice, con e senza `PYENV_VERSION`, prima e dopo l'introduzione del file; (4) README e guida riportano il comportamento e un unico percorso utente per l'interprete managed, nella directory predefinita di uv oppure in una locale, senza variabili implicite |
| CLA-P002 | **Media — bloccante: sì** | plan-r001.md:145–146, 283–316, 502, 609–616 | (a) `tests/unit/test_marker_wrapper.py` richiede FastAPI (extra `marker-server`), ma i comandi obbligatori di V6 usano solo `--group dev` (FastAPI assente, httpx presente). Ne segue un errore di raccolta oppure uno skip, mentre il piano vieta di silenziare gli skip. (b) `tests/packaging/test_installed_distribution.py`, sotto `tests/`, verrebbe raccolto da `pytest tests/`, e non è definito se costruisca o installi una wheel (backend dall'indice, quindi rete) né come resti fuori dai test veloci. (c) Dopo P3 i test di integrazione lanciano `-m modulo` in subprocess, che non eredita il `sys.path` di conftest: serve il progetto installato nell'interprete, e questo prerequisito non è dichiarato. Impatto: V6, obbligatoria per A4, non è eseguibile come scritta; mock API e prove di packaging restano senza una procedura univoca; possibile rete implicita nei test veloci | Piano, P1 (inventario) e P4; V6; AGENTS.md, sezione Verifica; `master_workflow.py:239,282,338` | Per ogni nuovo file di test il piano indica lo strato (unit, integrazione, packaging, Docker), il comando esatto (per esempio `--extra marker-server` o un gruppo dedicato) e il comportamento quando manca il prerequisito: fallimento esplicito, oppure deselezione tramite marker con conteggio riportato. Indica anche l'assenza di rete per lo strato veloce. V6 elenca i comandi e i conteggi per strato, e il prerequisito «progetto installato» dei test di integrazione è dichiarato |
| CLA-P003 | Media — bloccante: no | plan-r001.md:147–149, 184–190, 348–363; rischio :695 | Il piano installa Marker da Git (il commit è verificato) senza valutare che `marker-pdf==1.10.2` è su PyPI come **wheel pura con sha256** (pubblicata il giorno del tag firmato). Il build da Git usa `poetry-core` **senza vincolo** (dal pyproject al tag), richiede git e rete nel builder e lascia il lock senza hash dell'artefatto: è un rischio che il piano stesso elenca. La scelta di torch 2.7.1 è motivata come «prima minor ammessa», non come combinazione collaudata; il `poetry.lock` upstream esiste al tag | [checks.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), fonti PyPI, pyproject al tag e GitHub API | Il piano confronta in modo motivato PyPI e Git. Con PyPI: `marker-pdf[full]==1.10.2` con hash nel lock e controllo di provenienza (versione e metadati della wheel coerenti col tag). Con Git: vincolo di build esatto per poetry-core, verificato in V2/V7. La versione di torch/transformers è motivata da una fonte primaria (per esempio il lock upstream) oppure dichiarata come scelta non collaudata |
| CLA-P004 | Media — bloccante: no | plan-r001.md:257–266 (P3.3–P3.4) | Oggi `.env` viene caricato all'import di `unified_converter`, quindi **prima** di `apply_env_overrides`, nel costruttore dell'orchestratore (r. 51), e prima del converter (r. 57). Il piano sposta la cattura della directory dati all'istanziazione, ma non fissa che `.env` resti caricato prima degli override. Se il caricamento finisce nel costruttore del converter, i valori CHUNK_SIZE, OVERLAP_SIZE, MAX_WORKERS, CONVERSION_TIMEOUT e MARKER_* presi dal `.env` smettono di entrare nella configurazione, senza alcun errore. La prova di `.env` non definisce un osservabile che passi dagli override. Va dichiarato anche il cambio rispetto alla ricerca legacy di `load_dotenv()`, che parte dalla directory del modulo | `master_workflow.py:47–57`; `unified_converter.py:36`; `config.py:263–319` | Il piano specifica il punto di caricamento rispetto agli override e la precedenza. V5 include un `.env` sintetico con una variabile che cambia un output deterministico (per esempio i parametri di chunking in `chunks_index.json`) e un override di shell che prevale. Il cambio di ricerca è documentato |
| CLA-P005 | Bassa — bloccante: no | plan-r001.md:249–256 (P3.1) | `[sys.executable, '-m', modulo]` con `cwd` = directory dati antepone quella directory a `sys.path`. Un `config.py`, `exceptions.py`, `logging_config.py` o anche `tqdm.py` nel workspace verrebbe importato al posto del modulo installato. Gli script legacy (directory dello script) e i console script (bin della venv) non hanno questa esposizione. V5 usa un CWD «privo di .py» e quindi non la rileva. Impatto: codice eseguito dal workspace dati, contro il principio dei dati non fidati | Semantica di `-m` in Python; il piano fissa Python 3.12, dove è disponibile `-P` | Usare `-P` (o un equivalente) nei subprocess di fase e aggiungere a V5 un caso negativo con un modulo sentinella omonimo nel CWD, che non deve essere importato |
| CLA-P006 | Media — bloccante: no | P6 (:386–389), V7 (:627) | `docker-compose-build.yml` usa lo stesso Dockerfile, con volumi `/app/temp` e `/root/.cache/datalab/models` e limiti `deploy.resources`. Il piano non lo nomina e V7 non lo valida. Dopo le modifiche (build arg ed extra, `/data`, cache Surya) diventerebbe incoerente o costruirebbe un profilo non previsto | `docker-compose-build.yml:1–35` | P6 decide in modo esplicito se aggiornarlo, deprecarlo o rimuoverlo, con motivazione e riferimenti. Se resta, V7 esegue `docker compose -f docker-compose-build.yml config` e ne registra l'esito |
| CLA-P007 | Bassa — bloccante: no | plan-r001.md:391–395 (P6.1); rischio :702 | La scelta di `3.12.13-slim-bookworm` non motiva la distribuzione. Verificato: `3.12-slim`, la base effettiva del Dockerfile legacy, oggi corrisponde a trixie; per la serie 3.12 la libreria ufficiale mantiene solo la 3.12.15. Il tag 3.12.13 esiste (ultimo push 2026-08-05) ma non riceve più ricostruzioni. Debian indica che il supporto regolare di bookworm è terminato l'11 luglio 2026 (LTS fino al 2028). I pacchetti nativi vanno riconfermati per la release scelta | [checks.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), fonti official-images, Docker Hub e Debian | Motivare bookworm rispetto a trixie, oppure adottare trixie. Dichiarare nei limiti che il tag della patch non viene ricostruito e come lo si aggiorna, ripetendo V7/V8. Verificare la lista dei pacchetti nativi sulla release scelta |
| CLA-P008 | Media — bloccante: no | P7 (:448–456), V9 (:505) | A5 richiede che non restino comandi inesistenti, ma P7 limita le modifiche ad alcune sezioni. Fuori da quel perimetro restano: `python -m master_workflow --chunking-strategy semantic` (opzione inesistente), `ls source_ebooks/` (directory non configurata) e `device_requests` (già individuato dal piano). `AGENTS.md:63` indica ancora `python -m pytest tests/ -q`. L'epilogo di `markdown-pipeline --help`, usato in V4, cita `python run_full_pipeline.py` | README.md:864, 1411, 1437; `master_workflow.py:515–545`; `config.py:22–30`; AGENTS.md:63 | V9 produce un inventario completo dei comandi del README e delle nuove guide, con stato provato, illustrativo o rimosso. Il piano decide in modo esplicito su AGENTS.md, indicando chi aggiorna l'istruzione di governance, e sull'epilogo dell'help (correzione minima o limite dichiarato) |
| CLA-P009 | Bassa — bloccante: no | P0 (:77–92); V0/V5 (:496, :600) | La baseline gira su Python 3.12.3 con dipendenze risolte al momento; il nuovo ambiente su 3.12.13 con il lock. In caso di differenze il piano afferma di separare gli effetti, ma non prevede una prova incrociata. Inoltre `chunk_markdown.py` eseguito da solo legge per default la directory `cleaned`, mentre l'orchestratore gli passa `validated` | `chunk_markdown.py:670–684`; piano, tabella dei rischi (:691) | Se emergono differenze, eseguire una combinazione incrociata (sorgenti legacy con le versioni del lock e/o Python 3.12.13) prima di attribuirle. La baseline del chunking usa gli stessi argomenti dell'orchestratore. Le versioni di tiktoken vengono registrate |
| CLA-P010 | Bassa (organizzativo) — bloccante: no | plan-r001.md:506, 510–518 | Il rinvio di V10 è legittimo per la sola toolchain. Però il master di Marker è ora la serie 2.0 (findings), quindi rifare oggi la build legacy non dà una baseline funzionante, e il passaggio a 1.10.2 è il rischio principale per il percorso `local_marker` dell'utente. Il piano non dice quando va chiesto il perimetro operativo (pesi, CPU) | `findings.md`, sezione Marker; README, scenari local_marker | L'arbitrato del piano registra se il perimetro di V10 CPU è autorizzato e da chi. In alternativa, la consegna finale e il README dichiarano il percorso `local_marker` «non verificato dopo la migrazione», con istruzioni di verifica |

Nota minore, non classificata come rilievo: `uv python install --no-registry` agisce solo
sul registro Windows, quindi su Linux non ha effetto. Lo conferma l'help congelato.

## Motivazione dell'esito e limiti

Il piano è ampio, ordinato e in gran parte verificabile. Mantiene tutti e sette i criteri
del brief e separa dipendenze, immagine, pesi e inferenza. Impedisce che help, mock o
`/health` vengano scambiati per prove reali. Prevede una baseline prima delle modifiche
e un confronto del contenuto conforme alla skill di fedeltà, senza normalizzazioni.
Non anticipa la nuova applicazione e lascia all'utente commit e merge. Le affermazioni
aggiornabili che ho potuto verificare (commit di Marker, vincoli upstream, esistenza del
tag Python e di setuptools 84.0.0, patch mantenuta) sono corrette.

L'esito è **NO_GO** per due lacune di correttezza su criteri esplicitamente richiesti.

- **CLA-P001 (A1):** l'effetto del nuovo `.python-version` su pyenv è stato osservato
  in modo riproducibile nella shell di questa review: ripiego silenzioso su Python di
  sistema. Il piano lo esclude implicitamente, perché parte da una premessa
  (`PYENV_VERSION` impostata) che non vale in generale.
- **CLA-P002 (A4):** V6, che è obbligatoria, non è eseguibile così come è scritta
  per i test nuovi che il piano stesso introduce, e lascia aperti strato, rete e skip.

Entrambe si risolvono con integrazioni circoscritte in un piano r002. I rilievi non
bloccanti possono essere recepiti nello stesso giro o rinviati dall'arbitro con
motivazione. Questo NO_GO non giudica negativamente l'impianto complessivo e non
attribuisce al piano esiti di prove ancora da eseguire.

Limiti della review: nessuna risoluzione reale del grafo (Marker, Surya, torch CPU/cu126,
marker di piattaforma), nessuna build e nessun manifest Docker nativo. I contenuti web sono
stati estratti con uno strumento di riassunto: digest e date vanno riconfermati con
strumenti nativi. Il `poetry.lock` upstream non è stato letto per intero. Il probe pyenv
è stato eseguito su questa macchina con pyenv 2.6.12; altre versioni di pyenv non sono
state verificate. Questa review vale solo per lo snapshot `plan-r001`: una modifica al
piano o agli input richiede una nuova revisione.
