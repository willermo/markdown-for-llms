# Implementazione r001 — promozione lock e ingressi prodotto, mandato46

2026-10-06. **WAITING_FOR_STAGE_SNAPSHOT**, owner supervisor, label riservata
`impl-r001-stage-product-s016`. Promozione e acquisizione runtime base completate;
inventory reale e 22 operazioni successive predisposte. Nessuno snapshot nuovo,
S ufficiale, build/installazione del progetto, I/E o test applicativo eseguiti.
Prossimo ruolo: supervisore in nuova chat con [prompt47](../prompts/47-supervisor-handover-product-inputs-and-sbi.md).

Richiesta: [request.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Evidenze: [delivery — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[controlli finali d'ingresso — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[inventory — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).

Ingresso preservato in `evidence/implementation-r001/product-inputs-r001/before/`:
pyproject originale, report-r011 e checkpoint autore. Branch
`feature/run-a001-uv`; HEAD/dev/base `66ba82200e5def5a4db76f9bafccb0731b506091`;
indice vuoto. Delta core/documentale preesistente riconosciuto, nessun Git write.
Piano r003 SHA `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`,
arbitrato SHA `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`;
r014 SHA `2ee454ee9820745ba0c1d354528410811ec2d04ad319304d10d89473c0b4e0ff`.
Scope63176byte SHA `3fd09c86e4cd4d124c3df035cf84ee102b97fee7c372a5e27689d8463134e4c7`.
Identità ricalcolate in entry/static-validation; GO del solo piano e NO_GO storici.

S015 conservato immutato e trattato come storico. Verificati soltanto gli input
pertinenti non variati, le cinque variazioni documentali dichiarate e i16 input
protetti dello scope. Dieci moduli prodotto immutati rispetto all'ingresso;
nessun riesame dell'intera baseline/interprete storico o nuovo MATCH S015.
Stato comune, changelog, eventi, arbitrati e snapshot non modificati.

La promozione usa copia byte identica dopo confronto TOML: unica differenza
`tool.uv.dependency-metadata` EbookLib0.18, `requires-dist = [lxml,six]`.
Root pyproject2737byte SHA
`60ef6f17da3f6787b14137749c5143323117dd2e9b257748cc556def5060b856`;
lock287148byte SHA
`b3288f1d51b880d9c683a43607d9eedc5b8c87465c8a1d357818c6b61d12f05f`.
Nessuna rigenerazione, fonte/pin/extra/marker/conflitto cambiato:143package/141nomi
e audit ricevuto restano legati alla medesima copia esatta.

Origine uv0.10.10/binario SHA verificati; managed CPython3.12.13 GNU/build20260310
con binario/BUILD/os.py e ricevute origin/inventory ricevute. Site managed e
bootstrap enumerati prima/dopo; config/credential assenti, environment chiusi,
nessun merge con os.environ. Checksum managed resta inferito dal contratto uv
ricevuto, senza nuovo hash del flusso originale. Rootlock/rootvenv/work/esterno
assenti all'ingresso; rootvenv ed esterno tuttora assenti.

Acquisizione: argv/env/cwd completi in `prefix_operation.json` e ricevute
`acquisition-r001/` e `acquisition-r002/`. Native uv sync locked, no default
groups/project/editable/build, managed assoluto, downloadPython disabilitato,
stessa cache e solo scratch `runtime-acquisition`. Shell false/close_fds true,
log normali. R001 exit1 per DNS su wheel tiktoken nel contesto ristretto,
0.584295861s monitorati; nessun download riuscito dichiarato da quel tentativo.
R002 identico argv su rete pubblica autorizzata, escalation approvata: exit0,
0.573736246s monitorati,9 package installati. Non è una prova offline R/D4.

Runtime effettivo: requests2.34.2, tiktoken0.14.0, tqdm4.70.1,
python-dotenv1.2.4, certifi2026.7.22, charset-normalizer3.5.2, idna3.20,
regex2024.11.6 e urllib3 2.8.0. Nessun progetto/dev/API/ML/backend aggiunto.
217 voci RECORD installate verificate contro byte/hash; origin -I -B del nuovo
runtime PASS managed, senza import app o invocazione dei console script.
Hook `_virtualenv.pth`/`_virtualenv.py` creati da uv letti integralmente prima
del retry/avvio controllato del target; origine, cfg e digest specifici verificati.

Cache nativa espansa inventariata,9 radici runtime identificate da METADATA
effettivi e file canonici inclusi negli ingressi. Archivi wheel necessari a
input riproducibili letti separatamente agli URL/hash esatti del lock:
9 HTTPS200,2777517byte body osservati,0 redirect, certificati/hostname verificati;
lettore bounded senza proxy, resolver o installer alternativo. SHA upstream
confrontati con byte degli archivi realmente presenti. ZIP/RECORD verificati e
file confrontati con scratch installato; RECORD installato e RECORD ZIP distinti.
Traffico/wire/header/redirect delle richieste native uv **NOT_MEASURED** con log
normali; nessun claim wire0 o garanzia atomica. Lettore monitorato1.582309202s.

Correzioni locali conservate in `failures-and-corrections.json`: guardia troppo
restrittiva sugli hook uv prima del retry; successiva canonicalizzazione dei
path RECORD dei console script prima dell'inventory S. Nessuna reinstallazione
o nuovo download per quei fix. Script del primo validator e inventari precedenti
preservati; vecchio acquire.py esatto non salvato prima del fix, limite esplicito,
senza attribuire PASS al suo codice finale. Le ricevute native identificano gli
argv reali. Statiche r002/r003/r004 superate dai binding r005, non da prove prodotto.

Input inventory schema1/scope package-inputs/files/backend:805 file presenti,
canonici, senza link/escape; backend wheel84 e file dell'ambiente backend ricevuto
confrontati byte per byte. Config-settings/build environment backend vuoti;
environment operativo pubblico separato e completo nei template. Formato del
produttore S richiede path assoluti nell'inventory; lista request/freeze relativa.
Il binario uv esterno è un prerequisito hash-checked in inventory/scope ad ogni
operazione; non è copiato o passato come artifact assoluto a run_context.
Managed/binari fuori run sono identificati e verificati senza aggirare il
confinamento degli artifact della run. README/licenza/build/modules/diagnostici
sono coperti dalla lista e dai file Git raccolti dallo snapshot futuro.

Scelte r013 per la ripresa concreta, in `operation-decisions.json` e template:

- Build uv nativa offline: default sdist e wheel dalla stessa sdist, backend84
  già ricevuto con `--no-build-isolation`, constraints84 e supply locale.
  Guardia reale prima dello startup dell'interprete backend. Dentro la stessa
  invocazione R: confronto corrente/S/snapshot, import backend attestato e
  get_requires sdist/wheel con risultati reali, poi build uv. Nuovi requisiti
  non vuoti fermano il launcher. Raw hook process trace separato; assenza della
  traccia di entrambi build_sdist/build_wheel non è PASS. Tutto ciò è futuro.
- Root/base-a/base-b: sync locked offline del runtime con no-install-project e
  no-build, seguito da reinstallazione della stessa wheel canonica, senza
  setuptools nel runtime base. Build una volta nel backend attestabile. Il
  discriminante V1 config-only/rebuild-cache resta obbligatorio e non eseguito;
  questi sync non ne sono prova. Nessun requisito del prodotto/pin cambiato.
- `make_source_manifest.py --uv` esplicito identifica uv fuori dal PATH chiuso;
  default precedente conservato. Nuovi guard/source/backend diagnostics non
  modificano i dieci moduli. Nessun S standalone prodotto.
- I nomi/hash archivi sono letti solo dopo B reale; B/RECORD audit precede gli
  install. Esterno: export hash-locked, venv managed, pip sync offline,
  canonica/pip-check/I/E. Ogni workload usa run_offline con R/D4 dello stesso
  interprete, namespace e comando dei figli, environment nativo via execve.
  Target/preflight reali derivati e verificati prima di ciascun avvio, output
  esclusivi per tentativo. Nessun target futuro dichiarato già attestato.

Launcher `after_freeze.py --validate` PASS statico,22 operazioni; help/parser,
AST, due guardie negative filesystem senza startup target, inventory consumer,
contenuti wheel/runtime e dieci hash verificati. Nessun nuovo R/D4 o suite;
bootstrap/backend pre-build e hook trace sono predisposti e **non attestati ora**.
La ripresa esegue un'operazione alla volta solo con snapshot supervisor esistente
e MATCH; verifica tutti i prerequisiti PASS e il budget cumulativo, conserva fail.

Pool112MiB unico, Hentry553541632; niente residuo32MiB aggiunto o reset dopofreeze.
Ledger lstat run/managed/rootvenv/tmpesterno, directory/link compresi senza seguirli.
Costi prima della consegna: H579092480, Delta25550848, residuo91889664byte;
misura definitiva dopo request in `delivery-final-checks.json`, include nuovi
registri. Formula total_with_reserves687759360 < stop939524096, pool1GiB;
repository oltre1GiB libero e /tmp oltre128MiB. Nessun cleanup/spostamento fuori
ledger; monitor0.5s/gap massimo0.500135844s, non quota atomica/picco continuo.
9.713200615s misurati nei comandi con timer, più30s di preparazione addebitati
conservativamente (durata separata non misurata):39.713200615s,7160.286799385s
residui sul totale7200. Controlli finali di consegna addebitati separatamente nel
delivery; launcher li somma prima del seguito. Paid0, nessun nuovo costo pesante.

FAIL storici/baseline62pass5fail/perdite/s009byteFAIL/lacune/s013logcap preservati.
S/B/I/E prodotto, V1–V9/dev/API/suite/comparativi, V7/V8 costo distinto, review
ChatGPT/Claude reali e arbitrato/GO finale aperti; V10/V11 esclusi. Non installabilità
multiABI né GO prodotto. Nessun commit/staging/merge/push/deploy/servizio automatico.
