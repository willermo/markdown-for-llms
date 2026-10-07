# Implementazione r001 — report-r016, mandato51

2026-10-06. **PARTIAL_CORE_RESOURCE_BOUNDARY_DOCKER_PASS**.
Suite host, supplementi CLI/fedeltà, build CPU e V8 sono stati eseguiti.
Il piano non è dichiarato concluso: V1 finale ha quattro fasi interne PASS
e ricevuta esterna FAIL. Ripetizione e binding finali host richiedono storage
oltre la quota incrementale core R017.

[Delivery — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[misura finale — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[confine core — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[Docker — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).

## Identità e responsabilità

Implementatore esclusivo. Piano r003
462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b;
arbitrato r003 f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d;
R016/R017 e authorized-scope di completion-resources ricevuti. GO del piano,
NO_GO storici conservati, nessun GO finale.

Branch feature/run-a001-uv; HEAD/dev/base
66ba82200e5def5a4db76f9bafccb0731b506091; indice vuoto verificato.
Dieci moduli applicativi invariati nel mandato51.
Pyproject2737byte/SHA60ef6f17da3f6787b14137749c5143323117dd2e9b257748cc556def5060b856;
lock287148byte/SHAb3288f1d51b880d9c683a43607d9eedc5b8c87465c8a1d357818c6b61d12f05f
invariati. Nessun agente, Git write, commit/merge/push/deploy, nuova fase,
modifica dei registri comuni/arbitrati o cleanup globale.

Snapshot tecnici autonomi e output distinti per i fix. Corrente:
impl-r001-stage-product-s053; S ufficiale Docker in S-docker-s053.json,
MATCH prima/dopo build, probe e consegna. Non è il freeze finale di review.
Gli stage precedenti restano storici, STALE dopo delta nuovi non riscritto.
I nuovi snapshot Docker catturano tutti i file worktree e l'inventory
completa806 come artifact; ogni record è ricalcolato prima del consumer.
Non replicano tutto lo storico della run; questo resta identificato nei
suoi artefatti immutati. Il checkpoint vivo è escluso dallo stage corrente.

## Risultati core

La ricezione22/22 S/B/I/E PASS resta legata a stage20/21 e ai propri input.
[Indice22 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
conserva esiti/exit/argv/env/cwd/identità. I12casi di fedeltà ricevuti
hanno **67chiamate reali**, ricalcolate dal workload: corregge il riepilogo64
di report-r015 senza riscriverlo.

Dopo README aggiornato: nuova B46 nativa uv, sdist e wheel dalla stessa
sdist, backend84/get_requires prima della build, rawtrace dei veri
build_sdist/build_wheel, audit tar/ZIP/metadata/RECORD PASS prima di installare.
Primo tentativo conserva archivi e FAIL del CLI audit per argomento
expected-managed mancante; il secondo passa. Nomi/digest/dimensioni reali
in actual-B-artifacts-s046.json, senza filename dimostrativi.

Canonica B46 reinstallata offline/no-deps/no-build nei sei prefix
root/base-a/base-b/esterno/dev/API. Sei I/E s047 e sei successivi I/E s049
PASS; riuso B dopo confronto moduli/input build esatti. Letture filesystem
di consegna confermano tuttora dieci bytes nei sei prefix e archivi B46:
**non sono nuovi I/E s053**.

Dev/API:36wheel locked/10.659.742byte acquisiti, marker/ABI confrontati con
export uv nativo, hash/ZIP/RECORD verificati, venv nuove realmente installate:
35/45dipendenze rispettivamente più canonica. Coverage.pth ricevuto ha causato
il primo FAIL conservato: hook letto dalla wheel hash-locked e ammesso solo
in questi prefix con COVERAGE_* assenti nell'env chiuso. Collection e test
non installano/scaricano.

| Strato reale | Esito | Identità |
| --- | --- | --- |
| Suite veloce AGENTS | PASS134test +300subtest, zero skip | proof-suite-fast-s045-r001 |
| API mock | PASS7test, zero skip, senza ML | proof-suite-api-s045-r001 |
| Packaging installato | PASS11test, zero skip | proof-suite-packaging-s045-r001 |
| Governance | PASS6test inclusi nei134, Git solo fixture sintetiche | nodeid nella receipt veloce |
| Discovery default | PASS collection134nodeid | proof-suite-discovery-default-s045-r001 |
| Discovery tests/ e radice | PASS collection153nodeid ciascuna | proof-suite-discovery-{tests,root}-s045-r001 |
| CLI/figli/negativi | PASS ultimo r003 | proof-cli-child-supplement-r003 |
| V1 precedente r005 | PASS4fasi ai propri input | proof-cache-discriminant-r001 |
| V1 finale r006 | quattro fasi interne PASS, **outer FAIL** | proof-cache-discriminant-final-r001 |

Le suite restano su S45/B20/rispettivi I/E: nessun trasferimento automatico
a diagnostici nuovi o B46. Il primo C-fast aveva134+300 PASS ma E FAIL per
spec del clone cached da conftest nello stesso processo. Fix run_tests.py:
postcheck in figlio isolato fresco nello stesso interprete/namespace,
env catturato prima di pytest; ripetizione S45 con E PASS.

Supplementi: config-only in tre forme senza strumenti pubblici/PipelineState/
fasi, cinque console (tre help e clean/validate con workload reale),
contenuti/report completi per filename univoco, origini/transitive da
execve/openat dei soli figli propri, sentinelle/PYTHONPATH e negativi.
Python wheel esterna, -I/-B, moduli installati. Dotenv nel parent, requests
nel parent, tiktoken nel figlio chunk: non inventati import nei figli che
non li fanno. Due FAIL di assert troppo forti conservati. Nessun algoritmo
fuori scope corretto. Baseline62PASS/5FAIL, perdite legacy, CRLF/racepyc,
s009/bootstrap/s013 e NO_GO storici preservati.

V1 r006: sourcecopy sintetica/backend84, metadata/lock protetti entro ogni
trial. Config-only e flag distinti, negativi I stantia/E vecchia prima del
sync, dieci hash immediatamente dopo sync prima della canonica, B/I/E per
fase e restore bytes effettivo. Il target esterno aveva reso immutabile
METADATA della canonica precedente: README nuovo cambia correttamente i
metadata. Dopo quattro PASS interni R finale lo rifiuta. **FAIL esterno
mantenuto**, nessun PASS retroattivo. Fix ordinario: seed canonica corrente
nel probe oppure identificare precisamente quella mutazione, conservando
startup/versioni/dipendenze/full B/I, poi nuova R/V1.

La quota blocca la ripetizione:573.440byte residui alla diagnosi contro
2.297.856byte della sola evidenza runner precedente, oltre a cache/manifest/
output dei trial. Misura dopo consegna in final-checks. Nessun nuovo tentativo
oltre cap, rimozione o trasferimento dello storico per liberarlo. Restano
aperti I/E host sullo stage finale e ripetizione pertinente fast/packaging
dopo verifier/canonica nuovi.

## Docker CPU V7/V8

**PASS diretto offline**, immagine finale locale
sha256:27b5f840f99007fdf7ed9b9478820a146b364cbf43433ad6a84b1cf4714d4b93,
tag run-a001-marker-cpu:r005, product-s053.
Versione r004 e risultati preservati; dopo eliminazione della supply APT
dal runtime sono state ripetute build e V8.

- OCI linux/amd64: index/manifest/config/layer digest e size verificati prima
  dei pull nativi Python3.12.13-slim-bookworm/uv0.10.10.
- Debian/security20261005: gpgv reali, Packages vincolati al Release,
  resolver APT nativo offline in container proprio,29deb SHA/size verificati
  e tutte le versioni confrontate al dpkg runtime.
- Supply94wheel CPU mancanti più base/backend ricevuti: ZIP/payload/RECORD
  verificati. Torch175.833.687byte/hash esatto del lock, fonte configurata R2;
  HEAD/GET403 preservati, GETRange con digest/provenienza verificati, nessun
  cambio versione/ABI. EbookLib0.18 sdist/backend legacy84/get_requires[]
  verificati prima della vera build; niente backend/requisiti ignoti.
- Buildx nativo networknone/pullfalse, basi digest, contesto sintetico
  realmente filtrato in /src e sentinelle escluse. Startup prima interpreti;
  R/D4 build: namespace diverso/solo lo/AF_UNIX positivo/socket host assenti;
  figli osservati nello stesso ns.
- Sync offline registry non consumava find-links: due FAIL conservati.
  Fix export uv locked CPU/server e pip sync require-hashes da supply locale,
  no-build-isolation/constraints84. Nessun lock/fonte/cache registry simulato
  o acquisizione nuova.
- Sampler sovrascriveva argv con cmdline vuota finale: FAIL conservato,
  fix transizioni pid/starttime/argv e ripetizione, vere rawtrace EbookLib/B.
- Audit B prima della canonica, I full builder e **nuovo I full runtime
  prima degli import**: tutti i payload/RECORD/origini/URI-digest B/S.
  **106distribuzioni runtime esattamente uguali al grafo CPU locked**,
  progetto compreso.
- Supply APT readonly nel solo RUN BuildKit; runtime senza /src, UV,
  supply wheel/APT, cache builder/.deb, compilatori/gruppo dev/CUDA/nvidia.
  Backend84 runtime è dipendenza Torch del lock.
- Ultimo V8 r003: export streaming del filesystem container fermo,
  .pth/customize/virtualenv/pyvenv/backend84 verificati prima di Python.
  Container proprio networknone/pullnever/nohealthcheck/no mount/capdropALL/
  no-new-privileges/pids128/mem2GiB/cpu2, solo stdin fidato -I/-B.
  **9sottocasi PASS**: I/hash/RECORD, import CPU, signature, provider full,
  WeasyPrint nativo (PDF3623byte), MarkdownOutput/wrapper, errore500 atteso,
  constructor mock, path cache/font. Marker1.10.2/Surya0.17.1/Torch2.7.1+cpu;
  zero tentativi rete, nessuna inferenza/peso/font/create_model_dict reale/
  normale CMD.
- Tre config Compose native PASS: CPU, alternativa build, GPU **solo config**;
  args namespace/basi/APT/supply verificati, mount/risorse conservati, no up.
  Driver V8 dedicato9sottocasi: non dichiarato pytest Docker o inferenza.

Container propri raccolti e rimossi individualmente; immagini/cache identificate
conservate. Completamento daemon da risposta solve/export e ID/metadata immagine,
non dalla sola terminazione client. History API vuota dichiarata, nessun job
inventato. Quinto monitor ometteva r004 preesistente: ogni sample ricalcolato
conservativamente aggiungendone la size, entro16GiB con riserva; receipt originale
immutata. Ultimo V8 misura SizeRw/cache/entrambe immagini proprie. Gap attivo
max0,6635s; intervallo iniziale3,8953s comprende export/copie del container fermo;
picco SizeRw campionato2355byte, nessuna garanzia di picco atomico.

## Matrice finale e documentazione

| Criterio | Stato |
| --- | --- |
| A1 | Origini/base/dev/API ai loro stage; V1 e binding finali host **aperti** |
| A2 | B/payload/profili verificati; I/E finali host da riallineare |
| A3 | Import/CLI/fasi/sentinelle esterne PASS identificati, non trasferiti a S53 |
| A4 | Suite/fedeltà PASS ai loro input; revalidazione finale pertinente aperta |
| A5 | Guide/inventario/link/whitespace verificati, esempi e prove distinti |
| A6 | **V7/V8 PASS CPU offline**, nessun collaudo inferenza/GPU/server host |
| A7 | GO piano ricevuto; review/arbitrato/GO implementazione futuri |

| Verifica | Esito/limite |
| --- | --- |
| V0 | Baseline originale62/5/perdite preservate, confronto F1–F6 ricevuto |
| V1 | PASS storico r005, r0064PASS interni/outerFAIL, retry **bloccato quota** |
| V2 | B46/audit/rawhooks PASS; input B esatti fino S53 |
| V3 | Base22/esterno/6prefix I/E identificati, finali host aperti |
| V4/V5 | CLI/figli/negativi/contenuti/dotenv/sentinelle PASS identificati |
| V6 | 134+300/7/11, governance6, discovery134/153/153 PASS S45; revalidation aperta |
| V7/V8 | Build r005/probe r0039subcase/config3/inventari **PASS** |
| V9 | Statici PASS9docs/10blocchi/74pre/9inline/31link/whitespace |
| V10/V11 | NON_ESEGUITA, escluse, nessuna inferenza/GPU dedotta |
| V12 | Supervisore/due nuove chat indipendenti ChatGPT/Claude/arbitrato/GO futuri |

Guide uv/Docker/reference aggiornate in italiano; argv reali distinte dagli esempi,
nessuna literal execution inventata. README input B46 immutato dopo quella build.
Nuova applicazione Web/DB/worker non disponibile. Registri comuni al supervisore.

## Costi e prossimo ruolo

Core: Hentry553.541.632, quota285.212.672byte, pool1GiB/stop896MiB/riserva16MiB;
esclusi solo due subtree Docker autorizzati. Tempo conservativo
**2656,685468308591s**, residuo**4543,314531691409s**/7200, entry1523,5855519899271
conservata; nested export incluso in V1 escluso una volta, nessun reset.
Misura finale e residuo dopo report/checkpoint in final-checks: lstat nofollow,
H=max(logical,allocated), spazio e formula controllati. Campionamento non atomico,
file/JSON/stream limitati, log normali.

Docker separato: Hupper11.247.741.479byte, residuo5.932.127.705/16GiB alla
misura Docker. Tempo568,7285651748534s, residuo6631,271434825147s/7200.
Payload applicativi letti + upper OCI dichiarato +10%/5MiB:
**496.040.214,5byte**, entro1GiB; wire non misurati. Shared layer/basi contati
conservativamente. Output finali riducono il residuo di pochi KiB; final-checks
rimisura. Paid0, niente download in build/probe.

Supervisore: ricevere risultato e **confine costo core effettivo**. Proposta
32MiB extra core nel boundary, totale304MiB dal medesimo Hentry, senza rete/
tempo/reset; formula889.085.952byte sotto stop939.524.096. Per fix/ripetizione
V1, I/E host e verifiche finali pertinenti, conservando lo storico. Il fix è
ordinario; il costo oltrepassa R017. Nessuna prova core nuova col residuo attuale.
Nessun GO finale dedotto dai PASS.
