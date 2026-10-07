# Review — piano — run-a001-fase0-uv — r002 — ChatGPT

- Data: **2026-10-02, Europe/Rome**.
- Autore: Codex; ruolo richiesto **revisore ChatGPT**. Provider effettivo **OpenAI**,
  famiglia **GPT-6** indicata dalla sessione; identificativo specifico del modello
  e riferimento/ID chat non esposti. Interfaccia effettiva **Codex, sessione
  IDE/API con strumenti del workspace**, non interfaccia ChatGPT Web. Questa
  differenza d'interfaccia è registrata; nessun provider/revisore sostitutivo inventato.
- Prompt di origine: [review-plan-r002-chatgpt.md](../prompts/review-plan-r002-chatgpt.md),
  fornito anche esplicitamente dall'utente in questa nuova chat.
- Oggetto: [plan-r002.md](../plans/plan-r002.md), revisione **r002** della run
  `run-a001-fase0-uv`, fase roadmap **0.1**. Nessuna implementazione oggetto di GO.
- Snapshot comune: [plan-r002.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
- Branch: `feature/run-a001-uv`.
- HEAD, `dev` e merge-base: **`66ba82200e5def5a4db76f9bafccb0731b506091`**.
- Impronta worktree del manifest:
  **`46d68f51823734faf1dfc1a9f004d052ed76a395c04851428149905038f3ae0a`**.
- SHA-256 piano, ricalcolato e uguale alla voce del manifest:
  **`df23588de1247820a797e033ba7e93f60ed7d6b5f1f64e1948f012646039ca71`**.
- Identità **pre/post: MATCH**, tutti i cinque comandi richiesti con exit 0.
  Verifica ripetuta alla consegna in `checks-final.json`.
- Indipendenza: questa chat svolge soltanto la review ChatGPT r002; non ha
  pianificato, supervisionato, implementato o svolto le review r001. Letti i due
  report **r001** come antecedenti prescritti. **Non letti** report, checkpoint,
  evidenze dell'altro revisore **r002**, né l'arbitrato r002. Nessun agente creato
  e nessun coordinamento con la chat concorrente.
- Esito: **GO sul piano r002**, con **un rilievo non bloccante GPT-P001 r002**.
  Nessun nuovo blocco essenziale. Il supervisore deve arbitrare il rilievo insieme
  all'altro report valido; questa singola review non autorizza l'implementazione.

## Ambito e prove

### Identità e provenienza

Prima della lettura tecnica sono stati eseguiti dalla radice:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
python3 scripts/run_context.py verify run-a001-fase0-uv --label plan-r002
```

Le catture sono in [identity-pre.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Dopo la lettura tecnica, alle **17:30:27 UTC**, gli stessi comandi hanno restituito
identità invariata e MATCH, con hash piano invariato:
[identity-post.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Il controllo finale dopo gli output è registrato separatamente in
[checks-final.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).

Git mostra sempre le sole sei modifiche documentali previste del supervisore:
CHANGELOG, indice documentation, roadmap, metadata ADR 0006/0007 e indice decisioni.
Nessun codice uv realizzato. Il manifest identifica **55 artefatti espliciti** e
**79 file tracciati**: non contiene i loro sorgenti, che sono stati letti nel
workspace. I sette artefatti di pianificazione sono identificati nello snapshot;
riconfermati anche i cinque hash del registro `checks.json` e il relativo sidecar:
[input-checks.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).

La [receipt del supervisore — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
documenta identità/provenienza e distingue il contesto storico
`planning-context-r002` dall'oggetto attuale. Non è stata usata come giudizio
tecnico. Il NO_GO dell'[arbitrato r001](../arbitrations/arbitration-plan-r001.md)
rimane antecedente vincolante; nessun GO o controllo r001 è trasferito a r002.
Report, evidenze e checkpoint propri erano assenti all'ingresso.

### Input consultati

- Prompt assegnato, `AGENTS.md`, skill `manage-implementation-run`, ciclo
  supervisionato e template review; handover globale letto per contesto, con
  prevalenza del ruolo esclusivo assegnato dal prompt corrente.
- `STATE.md`, brief A1–A7, prompt di pianificazione r002 e arbitrato r001.
- Indici architetturali, ADR 0001/0006/0007, fase 0.1 della roadmap e CHANGELOG.
  L'adozione uv approvata è distinta dal piano approvato e dalla migrazione realizzata.
- **Tutte le 940 righe del piano r002**; checkpoint, findings, checks/sidecar,
  inventario statico e help del pianificatore r002. Nessun esito statico del
  pianificatore è stato interpretato come lock/installazione/prova runtime.
- Due review r001, relativi checkpoint, evidenze pertinenti di chunking/fonti/help
  e probe pyenv; probe e sorgenti pertinenti dell'arbitrato r001. Distinto il primo
  probe con bin diretto dalla successiva prova storica con shim prioritario.
- `setup.py`, `requirements.txt`, `.gitignore`, Dockerfile, **entrambi i Compose**,
  README e cinque indici docs; `.env.example`, senza leggere un `.env` privato.
- I dieci moduli flat, con attenzione a import, configurazione, dotenv, avvio
  delle fasi, output, chunking e wrapper Marker; conftest, tre file unitari,
  integrazione, governance e i due smoke radice `test_pipeline.py` e `test_conversion.py`.
  Lettura e AST, senza importare il progetto o raccogliere test. Non è un audit
  generale della qualità degli algoritmi legacy.
- Skill `verify-conversion-fidelity`, per giudicare contenuti, formule, numeri,
  riferimenti, asset e ordine senza normalizzare perdite.

### Controlli realmente eseguiti

Letture, hash, AST, confronto con help congelati, browser su fonti primarie e
solo comando locale di versione **`uv --version`: 0.10.10**. Evidenze:
[static-checks.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[toolchain-static.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
e [sources.md — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
I flag proposti nelle procedure uv sono presenti negli help 0.10.10 pertinenti;
ciò non dimostra la selezione reale managed, il parsing dell'intera configurazione
futura o la risoluzione delle dipendenze.

Il riscontro AST trova **55 funzioni test unitari, 5 d'integrazione e 6 governance**;
questi conteggi non sono nodeid parametrizzati e non sono risultati di suite.
Il conflitto dei due `test_pipeline.py` e gli import/HTTP degli smoke radice rendono
pertinenti gli ignore prima della collection previsti da P4.

Confrontato il catalogo README con il file effettivo: **58 blocchi operativi**,
tutti con coordinate/testo/linguaggio corrispondenti; 3 verificabili, 18 illustrativi,
34 da correggere, 3 da rimuovere. Le altre 11 fence sono dati/configurazioni JSON,
output, alberi o esempio Markdown. Il primo confronto generico di tutte le 69
fence con le 58 operative dava false; è conservato e chiarito dal confronto
corretto in [readme-inventory-check.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Verificati anche gli otto comandi inline e i sei esempi dell'epilogo inventariato.

Fonti ufficiali confermano staticamente le scelte candidate e i contratti citati.
La fonte PyPI conferma nome/hash della wheel Marker, **non** equivalenza col tag:
[metadata Marker 1.10.2](https://pypi.org/pypi/marker-pdf/1.10.2/json).
La [guida WeasyPrint 63.1](https://doc.courtbouillon.org/weasyprint/v63.1/first_steps.html)
è stata recuperata in questa review, colmando il precedente limite di accesso;
native effettive e rendering restano da provare. URL/data, altri accessi falliti,
distinzione fra fatti e inferenze sono nel registro fonti. Non riusati digest
Docker ottenuti da riassunti storici come se fossero manifest nativi verificati.

**Non eseguiti**, per il mandato della review: uv lock/sync/run/build, installazioni,
suite o collection, conversioni, Docker config/manifest/build/run/up, startup
Marker, creazione di namespace o altri probe runtime. Nessun modello/font/cache
tokenizer scaricato, nessun benchmark o costo remoto, nessun documento inviato a
provider. Nessuna modifica a piano, sorgenti, input, snapshot, arbitrati o registri
condivisi; nessun commit, merge, push o deploy.

### Riscontro A1–A7

| Criterio | Valutazione del disegno r002 | Prerequisiti, osservabili e condizioni di uscita |
| --- | --- | --- |
| **A1** | **Adeguato.** P1/V1 rendono unico il managed locale, mantengono pin esatti e dichiarano la restrizione Python 3.12. La matrice shim/bin con e senza PYENV_VERSION tratta il fallback silenzioso storico e non promette di riparare pyenv. | Acquisizione Python separata, poi download vietati. V1 osserva executable/realpath/prefix/base_prefix/path e origine stdlib prima/dopo il pin; due ambienti nuovi e lock invariato. Help/catalogo non bastano. Nessuna modifica globale; candidato indisponibile torna al supervisore. |
| **A2** | **Adeguato con dettaglio esecutivo da chiarire in GPT-P001 r002.** Runtime/dev/server/ML separati; default-groups vuoto, conflitto CPU/cu126 e indici espliciti. P1 disciplina tutti gli sdist/backend, non solo Marker. P2 dichiara dieci py-modules e cinque target; config.main è da estrarre. | Lock universale può acquisire metadata ML anche per il base: costo di preparazione dichiarato. V1–V3 verificano inventari base/dev, build-system, backend, wheel/sdist e ambiente esterno senza dev/ML. Sdist/backend ignoto o grafo insolubile = stop, nessun pip correttivo fuori lock. Il comando pip sync va allineato alla prescrizione P1. |
| **A3** | **Adeguato.** P3 fissa il workspace dati, carica `.env` esplicito prima degli override, preserva precedenza shell e usa figli sys.executable **-I -m**, con preflight nel medesimo interprete installato. Console installata e script clone hanno contesti di lancio distinti e dichiarati. | V3/V5 provano wheel non editable fuori clone, origini applicative/transitive e sentinelle CWD/PYTHONPATH avverso; PYTHONPATH/PYTHONHOME ripuliti nei contesti prescritti. Pandoc/tokenizer preparati. Metadati chunk e pipeline_state osservano .env 1200/80, poi shell 1600/120. Modulo mancante fallisce senza fallback al workspace. |
| **A4** | **Adeguato.** P4 assegna ogni test nuovo/mutato a uno strato e ambiente. API usa dev+marker-server; figli reali usano un progetto installato. Ignore prima degli import, collection inerte, conteggi/partizione e zero skip essenziale sono espliciti. F1–F6 e confronti incrociati coprono fedeltà effettiva. | Cache tokenizer e baseline preparate e identificate; build/install fuori collection/test. Runner Linux offline con preflight; unshare negato o prerequisito assente = IMPEDITA/FAIL, non skip. V6 confronta liste nodeid, non numeri storici. V0/V5 usano stessi byte/validated/argomenti, >=2 chunk reali e overlap complementare. Divergenze restano aperte fino a spiegazione controllata. |
| **A5** | **Adeguato.** P7/V9 collegano inventario completo di comandi, help finali e guide uv. Trattano run_full_pipeline.py inesistente, --chunking-strategy inesistente, dati/report, pseudocodice e nuova applicazione futura. Responsabilità per l'istruzione operativa AGENTS assegnata all'implementatore. | V9 prova soltanto esempi consentiti nei profili preparati; startup/inferenza/cloud/GPU rinviati sono identificati come non verificati. Inventario finale per comando, link e whitespace. Il supervisore mantiene stati/arbitrati; la documentazione non può dichiarare integrate modifiche non integrate. |
| **A6** | **Adeguato come piano, con esito tecnico futuro.** P5 sceglie wheel PyPI Marker 1.10.2 con hash, provenance/metadata, confronto file col sorgente e contratto installato. P6 comprende entrambi i Compose, lock, digest nativi, bookworm/patch storica/manutenzione, apt congelato, contesto e GPU separata. **V7/V8 essenziali** verificano immagine CPU/import/contratti/native/rendering offline. | Preparazione Docker/ML può richiedere GB e tempi importanti, distinta dai test veloci. Tag/digest/apt/backend/cache reali ancora da acquisire. V8 evita CMD/startup e font/pesi reali, patchando prima del costruttore; config/health non bastano. V7/V8 impedite lasciano A6 aperto. V10/V11 non autorizzate; V10 diventa essenziale se necessaria per un'incompatibilità, senza dichiarare il criterio passato. |
| **A7** | **Adeguato.** P0/V12 richiedono quattro report reali nei due giri piano/implementazione, due arbitrati, snapshot identificati, nuove chat e registrazione di scostamenti/sostituzioni. La storia r001 resta conservata. | Il supervisore attende entrambi i report r002, arbitra tutti i rilievi e prepara una nuova chat implementatrice soltanto dopo GO. Modifica sostanziale ritorna alla supervisione. Commit/merge/push/promozione manuali dell'utente; nessun deploy implicito. |

### Coerenza P0–P7, V0–V12 e recupero

L'ordine P0 → P1/P2/P3 → preparazione/P4 e V1–V6 → P5/P6 e V7/V8 → P7/V9
è implementabile. **P0/V0 precedono le modifiche**: baseline al commit identificato,
Python pyenv 3.12.3 diretto e sole dipendenze legacy necessarie; originali,
fixture/hash, versioni, CWD, argomenti, exit e output conservati. Non si usa setup
legacy difettoso per mascherare una baseline mancante. P1 congela la toolchain e i
backend; P2 rende installabile il codice che P3 lancerà isolato. Dopo modifiche
non editable il piano richiede reinstallazione esplicita prima dei test.

P4 non presuppone FastAPI nel dev né un progetto visibile al figlio grazie ai
sys.path di conftest. C-package usa un harness dev e RUN_WHEEL_PY esterno già
preparato; C-docker usa immagine locale V7 già costruita. C-all raccoglie harness
in api-env senza startup, build o download. `uv --offline` da solo non impedisce
egress dei subprocess: runner/namespace e V8 --network none coprono quel requisito.
Il daemon Docker locale via socket Unix viene trattato separatamente dal namespace
del client. L'assenza di permessi non viene trasformata in prova superata.

| Prova | Risultato osservabile richiesto e sufficienza |
| --- | --- |
| **V0** | Manifest baseline F1–F6, versioni/dipendenze/tokenizer/cache/Pandoc, input e output originali. Fixture F6 ampliabile prima del congelamento, poi identica per entrambi. Baseline essenziale assente resta aperta. |
| **V1** | Lock/check, due sync base e due dev su managed dichiarato; confronto hash lock/inventari ed executable/stdlib nella matrice prima/dopo pin. Dimostra selezione e riproduzione, non solo versione stampata. |
| **V2** | Sdist e wheel dalla sdist, archivio effettivo dei dieci moduli, entry point/metadata e assenza di dati privati/test/helper. Backend di tutte le build identificati, pin e build ripetibili. |
| **V3** | Runtime esportato dal lock con hash, wheel non editable e pip check; probe -I fuori clone e senza codice copiato. Risoluzione nuova non permessa. GPT-P001 r002 precisa l'applicazione dei vincoli se pip costruisce sdist. |
| **V4** | CLI/help e config create/show/validate con contenuto osservato. Non richiedere --help dove lo script legacy non lo supporta, né trattare il solo help come collaudo delle fasi. |
| **V5** | Cleaning/validation/chunking reali con --force, workspace/output vuoti, validated esplicito, HTML Pandoc F4; byte/invarianti, report e negativo exit 1. Prove dotenv/precedenza e sentinelle applicative/transitive nello stesso interprete. |
| **V6** | C-fast/API/package/docker/governance e discovery selezionata/default/radice/C-all con liste, partizione e conteggi reali. Prerequisiti pronti, collection inerte e zero skip essenziale. C-docker può essere eseguito fuori namespace solo con le protezioni Docker prescritte. |
| **V7** | Manifest nativi Python/uv distinti index/piattaforma, config di entrambi i Compose e override GPU, build CPU dal lock, dpkg/cache/varianti/risorse/mount, manifest del contesto filtrato con sentinelle. Rete/build costose sono preparazione distinta. |
| **V8** | Codice installato nell'immagine locale, network none/pull never, import reali, firme/provider, torch CPU, WeasyPrint rendering sintetico e MarkdownOutput reale con converter fake. Costruttore patchato prima della chiamata, senza create_model_dict/font reali. Ogni subcaso deve avere un esito; health non può sostituirlo. |
| **V9** | Inventario pre/post per comando e help finali, esempi autorizzati realmente provati, guide/link/whitespace. Gli esempi di inferenza non diventano autorizzati perché compaiono nel README. |
| **V10** | **Non autorizzata.** Futuro CPU con pesi/font/cache/provenienza, rete/disco/RAM/tempi/costi espliciti. Se necessaria per incompatibilità o semantica richiesta, A6 rimane aperto finché il supervisore definisce e ottiene quel mandato. |
| **V11** | **Non autorizzata.** Futuro GPU richiede hardware/driver/Toolkit/CUDA osservati, risorse e costi distinti. Non dedurre funzionamento da config GPU o dalla risoluzione dell'extra. |
| **V12** | Report/checkpoint implementatore, snapshot esatto, due review reali d'implementazione e arbitrato, conteggi/esiti/limiti. Integrazione Git eseguita dall'utente solo dopo GO e verificata prima di registrarla. |

Per la fedeltà F1–F5 distinguono contenuto stabile, perdite legacy del cleaning,
copie di validation/report deterministici, HTML Pandoc e asset. Il confronto non
normalizza Markdown, formule, codice, link o numeri. F5 verifica l'asset al punto
in cui esiste senza promettere una copia legacy che il codice non fa.
**F6 aggiunge almeno due chunk reali** con input validated/argomenti identici,
ordine, contenuto, frontmatter/index/metadati, token/word count, confini e overlap.
Il CLI semantico può produrre overlap zero: il piano lo registra e aggiunge una
prova del metodo sliding esistente con overlap positivo, senza inventare un flag.
In caso di divergenza i confronti legacy su Python/dipendenze nuovi, e se necessario
Python vecchio con dipendenze comuni, consentono di distinguere migrazione,
toolchain e tokenizer. Campi temporali/percorso sono circoscritti e confrontati
separatamente; nessuna esclusione generica per far coincidere gli output.

Il disegno V8 è fattibile rispetto ai sorgenti primari letti: BaseConverter
scarica font nel costruttore e PdfConverter risolve predictor/processori; patcharli
prima è necessario. Importare create_model_dict senza chiamarlo, verificare
signature e usare MarkdownOutput reale limita la prova al contratto wrapper.
WeasyPrint con HTML/font di sistema prova le native senza inferenza Marker.
Queste prove restano obbligatorie anche senza V10/V11. Immagini vuote, opzioni
non applicate e health di sola liveness sono limiti legacy caratterizzati, non
funzionalità corrette dalla migrazione. La dicitura **«local_marker non verificato
dopo la migrazione»** rimane necessaria se non si svolge il futuro V10 autorizzato.

Il recupero è esplicito: conservare output/log/manifest e tornare al supervisore
per grafo insolubile, digest/tag indisponibile, backend non congelato, import
ostile, regressione contenuto, native fallite o V10 necessaria. Nessun fallback
mobile, cambiamento globale, rimozione tacita di A6 o GO ottenuto tramite skip.
Risorse/tempi dichiarati non sono benchmark. Le modifiche sostanziali dei candidati
richiedono nuova identità e riesame; gli aggiornamenti Docker ripetono V7/V8 e
gli altri controlli pertinenti. Cleanup limitato agli oggetti di prova, senza
prune, down -v o cancellazione di cache/volumi operativi.

### Disposizione degli undici rilievi accolti r001

La valutazione seguente riguarda **come il piano li affronta**, non una chiusura
di implementazione o una riscrittura dell'arbitrato r001.

| Rilievo precedente | Disposizione sul piano r002 | Posizione e riscontro autonomo |
| --- | --- | --- |
| **CLA-P001 r001**, bloccante nell'arbitrato | **Adeguatamente affrontato; nessun blocco residuo di piano.** | P1:120, V1:684. Percorso managed unico, pin esatto e matrice shim/bin/PYENV_VERSION; origini stdlib/executable richieste. Il probe storico con shim conferma perché non basta la versione. Acquisizione reale e matrice restano future. |
| **CLA-P002 r001**, bloccante | **Adeguatamente affrontato; nessun blocco residuo di piano.** | P4:324, V6:739. Matrice di tutti i test nuovi/mutati, API dev+marker-server, stesso interprete installato per figli, wheel-env esterna. Comandi per strato/collection, ignore prima import, prerequisiti e isolamento Linux effettivo, conteggi e zero skip essenziale. |
| **CLA-P003 r001** | **Parzialmente affrontato nel dettaglio esecutivo; nessun blocco residuo.** | P1:200/P5:429 definiscono backend di tutti gli sdist, wheel PyPI scelta, hash/provenienza/confronto e torch candidato. V7/V8 verificano il codice installato. V3:671 non esplicita il file build-constraints per pip sync: nuovo **GPT-P001 r002**. Il requisito generale e lo stop su backend ignoto sono già vincolanti. |
| **CLA-P004 r001** | **Adeguatamente affrontato.** | P3:263, V5:720. Helper `.env` workspace prima degli override, niente ricerca genitori, shell prevalente; chunk metadata e pipeline_state con valori 1200/80 e 1600/120. Il sorgente attuale conferma l'ordine oggi legato al load_dotenv all'import. |
| **CLA-P005 r001**, bloccante nell'arbitrato | **Adeguatamente affrontato; nessun blocco residuo di piano.** | P2/P3:283, V3/V5:728. Figli sys.executable -I -m, preflight isolato nel medesimo interprete, variabili Python ripulite, parent console/clone dichiarati, origini applicative/transitive e sentinelle CWD/PYTHONPATH avverso. -I ha la semantica necessaria secondo la CLI Python ufficiale. |
| **CLA-P006 r001** | **Adeguatamente affrontato.** | P6:485, V7:804. docker-compose-build.yml conservato e allineato al medesimo lock/Dockerfile; config esplicita -f di entrambi, inventari di risorse/mount/cache/TMPDIR/input e override GPU. Nessuno startup implicito. |
| **CLA-P007 r001** | **Adeguatamente affrontato.** | P6:494/V7/V8. Bookworm esplicita, patch storica dichiarata, digest nativi index/piattaforma, apt/dpkg congelati e rendering native. Politica di manutenzione con nuova identità e ripetizione prove. Fonti primarie sostengono LTS e distinzione dall'alias slim corrente, non i futuri digest. |
| **CLA-P008 r001** | **Adeguatamente affrontato.** | P7:565/V9. Catalogo operativo README completo riconfermato, sei esempi nell'epilogo che citano run_full_pipeline.py inesistente, opzione chunking inesistente e responsabilità AGENTS; guida uv, pseudocodice/nuova app e comandi rinviati distinti. Inventario finale per comando richiesto. |
| **CLA-P009 r001** | **Adeguatamente affrontato.** | P0/P4:379, V0/V5. Baseline identificata, input validated e argomenti orchestratore uguali, versioni Python/dipendenze/tokenizer/cache registrate. Confronti incrociati condizionati alle divergenze, senza attribuzione prematura o normalizzazione delle perdite. |
| **CLA-P010 r001** | **Adeguatamente affrontato.** | P5:474, P6, V10/V11:850. Non autorizzate nel mandato; punto della futura proposta del supervisore dopo V7/V8, costi/pesi/font/hardware separati. V10 essenziale quando necessario, altrimenti limite local_marker dichiarato; nessun requisito passato per rinvio. |
| **GPT-P001 r001** | **Adeguatamente affrontato.** | P4 F6/V0/V5. >=2 chunk baseline/wheel, sequenza/contenuto/metadati/confini e overlap effettivo; prova complementare sliding con overlap positivo. È ora azione accolta vincolante, senza trasferire il precedente GO ChatGPT r001. |

## Rilievi

Un solo **nuovo rilievo del giro r002**. Non è il GPT-P001 r001 sul multi-chunk,
che viene valutato separatamente nella tabella precedente.

| ID | Severità e blocco sì/no | Posizione | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- |
| **GPT-P001 r002** | **Bassa; blocco no.** Chiarimento esecutivo del requisito già previsto, non ampliamento del mandato. | P1:200 e procedura V3:671. | P1 prescrive il file `--build-constraints` per uv build/pip; il comando esatto `uv pip sync --python … --require-hashes runtime.txt` non lo passa. Gli sdist runtime non sono categoricamente vietati. Se pip deve costruirne uno, eseguire la sola riga non esplicita come applicare i pin dei backend già richiesti; hash runtime e lock non vanno assunti come vincoli di build. | [Estratti del piano e criterio — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md); flag disponibile nell'help uv 0.10.10 pip sync/install, [toolchain-static.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md). Nessuna installazione eseguita e nessun sdist problematico accertato. | Esplicitare `--build-constraints "$RUN_REPO/build-constraints.txt"` nel pip sync di V3 e negli altri pip che possono costruire; oppure prescrivere sole wheel con arresto su sdist e inventario verificato. Conservare manifest di artefatti/backend/versioni/hash. Riallineare la riga esatta alla regola P1; verificabile staticamente prima dell'implementazione e sui manifest di preparazione successivi. |

Il rilievo non è bloccante perché **P1 già impone** l'identificazione e il pin di
ogni backend e lo stop quando non è congelato. Non manca una condizione essenziale
di uscita: manca l'esplicitazione nella riga eseguibile di V3. Non asserisco che
una sdist runtime sia stata selezionata né che un'installazione sia fallita.
Non propongo ulteriori prove runtime fuori dal catalogo e non ho altri nuovi
rilievi o suggerimenti opzionali nel perimetro controllato.

## Motivazione dell'esito e limiti

**GO sul piano r002 identificato da questo snapshot/hash.** I tre blocchi r001
sono affrontati con interventi tecnicamente pertinenti e prove sufficientemente
concrete: managed e origini; composizione/esecuzione dei test; isolamento delle
fasi installate. Le altre azioni accolte restano nella specifica e hanno prove
osservabili. Il dettaglio pip di GPT-P001 r002 va arbitrato, ma non rimuove il
requisito vincolante di P1 o uno stop essenziale.

Questo GO valuta implementabilità, perimetro e verificabilità. **Non sono provati**
lock/grafo, managed/stdlib, build/backend, installazione esterna, F1–F6, suite e
namespace, immagine/digest/apt/native o contratto Marker installato. Fonti/help
forniscono ragioni per tentare le verifiche, senza certificare i risultati.
L'indisponibilità dei runtime/probe futuri non è automaticamente un difetto del
piano: r002 la tratta come FAIL/IMPEDITA che mantiene il criterio aperto.
V7/V8 rimangono obbligatorie; V10/V11 restano non autorizzate e local_marker non
verificato dopo la migrazione fino all'eventuale prova separata pertinente.

**Consegna al supervisore:** questo report, [evidenze proprie — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
e [checkpoint — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md). Attendere anche l'altro
report reale valido su **plan-r002**, arbitrare tutti i rilievi e preparare il
prompt per una nuova chat implementatrice soltanto con il nuovo arbitrato GO.
La review concorrente non verrà letta dopo la consegna per cambiare il giudizio.
