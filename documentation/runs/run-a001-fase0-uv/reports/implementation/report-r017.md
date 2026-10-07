# Implementazione r001 — report-r017, mandato52

2026-10-07. **IMPLEMENTATION_COMPLETE_REVIEW_PENDING**. V1 completa e prove finali
host concluse entro R018. Dodici nuove invocazioni host PASS, con R/D4 freschi,
postcheck e raccolta processi. A1–A6 dimostrati nel perimetro implementativo;
A7/V12 restano al supervisore. Nessun GO finale anticipato.

[Delivery — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[check finali — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[equivalenza — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[matrice — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).

## Identità e confini

Implementatore esclusivo, nessun agente. Piano r003/arbitrato r003 e R016/R017/R018
ricevuti; GO del piano e NO_GO storici conservati. Branch feature/run-a001-uv,
HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091, indice vuoto verificato
nella consegna. Nessun reset, staging, commit, merge, push, deploy, cleanup,
nuova acquisizione, paid, modifica host o registri comuni. Dieci moduli,
pyproject e lock invariati nel mandato52. Nessuna modifica algoritmica legacy.

Scope autorevole: final-core-resources-r001/authorized-scope.json. Incrementale
304MiB/318767104byte dal medesimo Hentry553541632; quota272MiB storica, non reset.
Pool1GiB/stop896MiB/riserva16MiB e partizione Docker R017 invariati. Nuovo monitor
resource_scope.py legge R018 prima dei workload; driver run_core_proof.py conserva
origine/config/site/startup, environment pubblico chiuso, argv strutturati,
shell=False/close_fds=True. Nessuna esenzione METADATA. Escalazioni ufficiali
exec_command require_escalated approvate per i comandi concreti; non equiparate
a PASS di isolamento. R usa socket UNIX sintetico nella run, connect/close dei
soli daemon inventariati senza send/recv operativo, Firejail net=none e stessa
invocazione/interprete/namespace per preflight e figli. Nessun rifiuto auto-review.

Corrente: **impl-r001-stage-product-s055**, S ufficiale S-s055.json.
S SHA cfd4aca2e385de712f2e595d2286c1d10e564ba5d1ab5c3520a29200178bf5ee;
snapshot SHA e6afba15cd80719e3a538078870e90ca355aa575d8b410f573608a91e438b4d7;
worktree17ef67416a363b57b59a2d74422e0c043c9ab8aa8e01d2e1ba089968d2bbc46d.
MATCH prima/dopo le prove e alla consegna. Inventario806 file interamente
ricalcolato e copia byte-identica di quello s053; S55 include i diagnostici
correnti. Snapshot essenziali con tutti127 file worktree e input operativi
pertinenti, senza duplicare la storia negli argv. Checkpoint vivo escluso.
S053 e s054/cache64–67 sono storici dopo i delta autorizzati delle guide;
nessun MATCH vecchio attribuito al worktree corrente. Non è lo snapshot di review.

## V1 e correzione del FAIL

r006 e i suoi quattro PASS interni/**outer FAIL** restano immutati. Il probe
è stato reinstallato con la canonica B46 effettiva **prima** della nuova cattura
del target R: seed-probe-B46 PASS con B/hash/RECORD/versione/dipendenze/origini
ed E. METADATA protetto sia prima sia dopo l'intero nuovo trial, senza rimozione
del digest dalla guardia. Backend84/payload/startup verificati prima di Python.

Nuova V1 r007: **4 fasi PASS, 51 chiamate, outer R/postcheck PASS**.
Initial → config-only senza flag reinstall/refresh → flag esplicito → restoration.
Solo exceptions.py nella copia sintetica cambia; metadata/config/lock della copia
restano protetti. Per ciascuna fase: snapshot tecnico nuovo, S reale, attestazione
backend/get_requires effettivi prima della build, uv build nativo offline sdist
più wheel dalla sdist, audit B prima installazione, I/E completi. Rawtrace veri
build_sdist/build_wheel dei quattro output vincolati in V1-native-hook-binding.json;
12 processi hook osservati includono anche le quattro build del sync.

Tre negativi I installazione stantia ed E receipt precedente, exit2 prima del
sync. I dieci hash immediatamente dopo sync vengono confrontati **prima** della
reinstallazione canonica: config-only e flag sono prove distinte. Ripristino
reale exceptions.py/installazione, dieci sorgenti del clone prodotto invariati;
process-collection survivors[]. Le verifiche native/configurazione/origine già
ricevute per V1 restano ai propri input, ora insieme al trial discriminante PASS.

## B/I/E e prove host finali

B46 riusata dopo confronto **esatto** modules/build_inputs di S47 e S55, inclusi
README/licenza/MANIFEST/backend constraints e assenze esplicite. Archivi reali e
binding rawhooks/get_requires ricalcolati; nessun esempio di filename usato come
identità. La precedente reinstallazione B46 dei sei prefix resta effettiva.
**Nuovi I/E55** root/base-a/base-b/esterno/dev/API: confronto tar/ZIP/RECORD,
metadata, dieci moduli, origini, URI/digest canonica, managed e profilo completo.
E è la chiamata check_preflight reale con PASS_CURRENT_I/exit0 nei workload,
con S/snapshot/R correnti; nessun import applicativo o suite in questa fase.
Esterno da CWD esterno. Nessuna sola lettura finale sostituisce I/E.

| Nuova invocazione | Esito | Secondi esterni inclusivi |
| --- | --- | --- |
| proof-I-E-api-s055-r001 | PASS | 12.404843 |
| proof-I-E-base-a-s055-r001 | PASS | 10.949873 |
| proof-I-E-base-b-s055-r001 | PASS | 11.968755 |
| proof-I-E-dev-s055-r001 | PASS | 11.669659 |
| proof-I-E-external-s055-r001 | PASS | 11.044413 |
| proof-I-E-root-s055-r001 | PASS | 10.124391 |
| proof-V1-complete-r001 | PASS | 68.587374 |
| proof-seed-probe-B46-r001 | PASS | 15.439850 |
| proof-source-s054-r001 | PASS | 10.818857 |
| proof-source-s055-r001 | PASS | 8.154858 |
| proof-suite-fast-s055-r001 | PASS | 35.447067 |
| proof-suite-packaging-s055-r001 | PASS | 95.482166 |

Fast dopo I/E55: **134 test +300 subtest PASS**, governance6 inclusi, zero skip,
zero errori di collection. Packaging installato: **11 PASS**, zero skip/errori.
E-pytest identifica nodeid/report, S55/B46/I55, argv/env/cwd/R e namespace;
postcheck isolato nello stesso interprete/namespace exit0, inputs_unchanged=True.
Preparazione distinta dai test; nessuna installazione/download in collection.

## Riuso puntuale dei risultati

Equivalence-final.json documenta confronti reali, senza riesecuzioni inventate:

- API7 e discovery134/153/153: tutti25 file tests/conftest byte-identici a s045,
  stesse selezioni e versioni complete delle dipendenze nei sei prefix. I/E55
  aggiornano i prerequisiti della canonica e del diagnostico. I due smoke radice
  restano esclusi esplicitamente. API è mock senza ML, server host non collaudato.
- CLI/config-only/fiveCLI/figli e fedeltà F1–F6: dieci moduli, entry point e tutti
  header METADATA B20→B46 identici; cambia soltanto il corpo README. Fixture,
  tokenizer/Pandoc/managed/uv/baseline nell'inventario verificato806; dipendenze
  installate esatte. README non è consumato da questi workload. Rimangono le
  prove esterne reali, origini/transitive/strace dei figli propri, dotenv/shell,
  contenuti/report completi, sentinelle/PYTHONPATH e negativi. Nuovi I/E55 e
  packaging ripetuto coprono i prerequisiti finali. I12casi ricevuti sono67
  chiamate, non64; baseline62PASS/5FAIL/perdite invariata.
- Diagnostici da s045: delta image_package/image_prepare/probe_image_contract e
  verify_distribution (profilo CPU/backend runtime). I primi tre non sono
  consumati da API/discovery/CLI/fedeltà host; l'ammissione CPU non cambia le
  versioni/applicazione dei profili host. Il verificatore corrente è esercitato
  da sei I/E e fast/packaging nuovi. Nessun vecchio PASS di preflight trasferito.
- Docker V7/V8 immagine r005
  sha256:27b5f840f99007fdf7ed9b9478820a146b364cbf43433ad6a84b1cf4714d4b93:
  tutti25 input realmente filtrati identici agli oggetti finali, compresi
  Dockerfile/.dockerignore/README/diagnostici. Source incorporata s053 conserva
  il proprio ID storico, modules/build_inputs equivalenti a S55. Stdin/probe
  reali ricalcolati; Compose CPU/alternativa/GPUconfig identici. **105 artefatti
  supply locali** (94wheel CPU acquisiti più ricevuti/base/backend/sdist)
  confrontati con hash lock o backend84 noto;29deb con selezione firmata. Nessun
  cambiamento invalida build/probe: riusati V7/V8 PASS,106versioni runtime locked,
  I full prima import e9sottocasi CPU offline. Non nuova inferenza, GPU o GO.

Due FAIL locali del nuovo reader di equivalenza conservati: r001 confrontava
schema image-source name/file con S path/kind/assenze; r002 assumeva size in ogni
record del lock universale. r003 normalizza gli schemi e verifica separatamente
le assenze; usa hash obbligatori esatti, size dove dichiarata e byte reali.
Nessun requisito/file prodotto/isolamento modificato. Versioni, argv/errori e
hash in equivalence-diagnosis-r001/r002.json, file antecedenti preservati.

## Matrice e documentazione

| Criterio | Stato implementativo |
| --- | --- |
| A1 | PASS origini/lock/profili e V1 completa con outer R |
| A2 | PASS package/profili/B46/sei I/E55 correnti |
| A3 | PASS wheel esterna/import/CLI/fasi e packaging corrente |
| A4 | PASS suite/fedeltà e riusi pertinenti; FAIL legacy preservati |
| A5 | PASS guide, inventario/comandi/link/coerenza degli stati |
| A6 | PASS V7/V8 CPU offline con equivalenza input finale |
| A7 | Review/arbitrato finale al supervisore |

V0 confronto con baseline/perdite preservate; V1 completo; V2 B46 audit/hooks;
V3 sei I/E55; V4/V5 CLI/fedeltà equivalenti e packaging nuovo; V6 fast/packaging
nuovi/API/discovery equivalenti; V7/V8 Docker equivalente; V9 statici PASS.
**V10/V11 escluse/NON_ESEGUITE; V12 due nuove review indipendenti e arbitrato futuri.**
La matrice JSON fornisce uno stato distinto per A1–A7 e ogni V0–V12.

Guide ambiente-uv e reference aggiornate prima di S55: V1 eseguita, GO aperto.
README e ogni input B46 non modificati. V9:9docs/10blocchi,31link locali,
riconciliazione74blocchi/9inline storici, git diff --check PASS; esempi non
spacciati per chiamate reali. Applicazione Web/DB/worker pianificata non disponibile.

## Costi e consegna

Core cumulativo conservativo **3940,5060664880657s**, residuo3259,4939335119343s/7200
prima dell'ultima verifica; ingresso2656,685468308591 conservato. Le12 invocazioni
addebitano l'intervallo esterno una volta, monitor/guardie/snapshot/figli compresi.
Preparazioni separate:300s iniziali,80s overhead/import/control/guide,600s riserva
conservativa per consegna e preparazione s055 misurata. Reader/equivalenza durante
packaging non ricontati come intervallo annidato; final-checks misura la consegna
ed estende soltanto il suo addebito se supera la riserva. Nessun reset.

Misura autorevole **dopo** report/checkpoint/delivery in final-checks-r017.json:
lstat nofollow su quattro root, esclusi soltanto due subtree Docker R017,
H=max(logical,allocated), Delta e spazio libero/formula R018. Prima della consegna
restavano22.048.768byte core; dato finale nel check. Monitor0,5s: gap massimo
osservato0,5433s, non misura atomica; file32MiB/JSON8MiB/stream1MiB rispettati.

Docker nessun nuovo workload/traffico: consumo568,7285651748534s conservato,
residuo6631,271434825147s; storageupper11.247.782.439byte, residuo5.932.086.745/16GiB
ricevuto (correzione del refuso11.247.741.479 nel report-r016, non riscrittura).
Networkupper496.040.215byte arrotondato/1GiB, wire non misurati. Quote separate,
nessun trasferimento/cleanup dei dati storici. Processi nuovi raccolti,
container Docker già raccolti nella consegna16, immagini/cache identificate conservate.

Checkpoint antecedente salvato in checkpoint-before-r017.md; vivo aggiornato ed
escluso da s055. Registri comuni/snapshot review al supervisore. Prossimo ruolo:
ricezione completa, snapshot comune e due review reali indipendenti ChatGPT/Claude,
poi arbitrato. Nessun GO dedotto dai test; Git resta manuale. ABI managedLinuxx86_64,
nessuna convalida universale ABI/Windows/inferenza/pesi/font/GPU implicita.
FAIL storici CRLF/racepyc/s009/bootstrap/s013 e NO_GO preservati.
