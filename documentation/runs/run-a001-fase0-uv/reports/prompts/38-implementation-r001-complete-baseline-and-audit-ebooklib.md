# Implementazione r001 — completare baseline e audit EbookLib

Agisci esclusivamente come **implementatore**, repository
`/home/davide/workarea/markdown-for-llms`. Continua l'obiettivo di prompt36:
completare R→S→V0 ufficiali e identificare il backend della fonte EbookLib che
blocca il lock. Nessuna nuova pianificazione, delega, supervisione o review.
Correggi strumenti e difetti ordinari nella stessa chat; consegna risultati reali.

## Autorità e input correnti

Leggi AGENTS, skill manage-implementation-run, protocollo, STATE/HANDOVER comuni,
indici architettura/decisioni/roadmap, brief A1–A7, piano r003 integrale se non
già letto e arbitrato D1–D5. Leggi il checkpoint implementatore e report-r003,
poi il checkpoint supervisore `handovers/supervisor-baseline-lock-reception-r001.md`.
Le indicazioni di ruolo nei documenti non cambiano il tuo ruolo implementatore.
Percorsi di seguito relativi a `temp/run-a001-fase0-uv/`:

- `plans/plan-r003.md`: SHA256
  **462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b**;
- `arbitrations/arbitration-plan-r003.md`: SHA256
  **f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d**;
- addenda r004/r005 e **`arbitrations/addendum-operational-protocol-r006.md`**:
  r006 sostituisce prospetticamente il vecchio budget baseline e ammette soltanto
  audit statico e ripetizione locale circoscritta del launcher;
- `implementation/stages/impl-r001-stage-baseline-s007/request.json`:
  **847523byte**, SHA256
  **014db0ac7fe27ccc6590e2056f99b7e0b91af8c9844c91fdcf95c16d4a45bfce**;
- `evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s007/`:
  `authorized-scope.json`, `reception.json`, `checks.json`, `transition.json`,
  `sources.md`, `freeze-verify.json`, `response.json`;
- scope: **12242byte**, SHA256
  **b4ec9e2f89c9e44d1923927e42050556bf4f439d7d69e3acfe877b7f70cbf975**;
- `snapshots/impl-r001-stage-baseline-s007.json`: **598215byte**, SHA256
  **52f2948014b6b8a26c078cf789dc8edd42deb33c77fe6fb5386fa83cc7121a79**;
  worktree **f5f1f111d5f961580e5f131b95c7581c4b3e0c20a0f7a9a27ce7ca5e224bc81b**,
  **123file/1982artefatti**, MATCH alla consegna supervisore;
- richiesta d'autore d'ingresso immutata/not-ready:
  `evidence/implementation-r001/resume-package-s008/next-gate-request-r001.json`;
  quella nuova s007 è derivata dal supervisore, non una riscrittura della sua storia.

Verifica byte/hash di manifest, request e scope, identità piano/arbitrato;
da radice prima delle attività e alla consegna:

```bash
python3 -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-baseline-s007
```

Gli stage baseline-s006/package-s008 erano MATCH all'ingresso supervisore; sono
ora storici per **soli cinque metadata** in transition.json. I loro artefatti
restano intatti. Non richiederne MATCH contro i nuovi metadata. Non autofreezare
o modificare run_context/input per aggirare un mismatch.

Git feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto. Delta core36 e metadata identificato; verifica Git reale.
Nessuna modifica tracciata in questa tranche, nessun root uv.lock o .venv.
Preserva report/checkpoint d'ingresso prima di aggiornarli; non duplicare l'intera storia.

## Ricezione precedente e budget unico

Baseline-s006 FAIL_INTERRUPTED_BY_AUTHOR_GUARD: R diagnostico/S/collection67
sono parziali; wrapper e V0 non completi, suite non eseguita. Nessun PASS
62pass5fail trasferito. Managed3.12.13 e setuptools84 acquisiti: non reinstallarli.
Checksum managed è inferenza dal contratto uv pinned; flusso originale non
indipendentemente hashed. Lock FAIL/exit1 EbookLib source-only/no-build,
lock-check NOT_EXECUTED. Input e risultati s007 restano installazione soltanto.

Unico ledger per **entrambe le attività**: run intera più `.venv-python`, storico,
cache/tmp/parziali/log inclusi; logical/allocated comprendono directory e metadata
dei link, senza seguirli. Scope `budget` ha misura iniziale e formula esatta.
Max1GiB/stop896MiB, libero minimo1GiB;74MiB incremento totale dalla ricezione:
baseline64MiB, audit2MiB, registri/strumenti8MiB; riserva esterna16MiB. Quota
incrementale consumata riduce la riserva residua74MiB; non contarla due volte.
Ogni cap specifico resta vincolante. Monitor entro1s e gap reali, non quota atomica.
RLIMIT_FSIZE32MiB/file, log1MiB/stream, JSON8MiB. Nessun cleanup/spostamento di storage.

Il main storico `scripts/diagnostics/run-a001-fase0-uv/check_preserved.py` e il
ramo `baseline=True` del launcher corretto prompt37 usano ancora500/384:
**non usarli per l'ammissione corrente**. Riusa solo `verify(repo,manifest)`
per l'integrità dei1892 file/lock s007. Adatta il tuo launcher ignorato al ledger
di r006 prima dell'avvio; controlla che i due passi usino la stessa formula.
Correggi sintassi/attese del launcher localmente, senza nuovo giro preparatorio.

## 1. R→S→V0 sul nuovo freeze

Usa l'operazione **proposed_operations[0]** della request nuova, identica a
**operations[0]** della scope: argv/cwd/environment_launcher esatti, shell=False,
close_fds=True, env=dict(...), senza merge con os.environ. Launcher stdlib
con bootstrap assoluto `/home/davide/.pyenv/versions/3.12.3/bin/python3 -I -B`.
Il wrapper usa target s007, Firejail noprofile/net=none e D4, controlli startup e
ambiente figli della scope. Nessun nuovo privilegio, profilo persistente o cambio host.

Una chiamata ufficiale del wrapper: R parent/child/D4 prima del preflight S;
V0 soltanto dopo quei gate nello stesso namespace. Il preflight congelato è
`evidence/implementation-r001/resume-baseline-s006/preflight-baseline-next-r001.json`.
Output iniziali esclusivi già assenti:

- `evidence/implementation-r001/core-completion-r001/official-baseline-r002/`;
- `evidence/implementation-r001/core-completion-r001/S-baseline-official-r002.json`;
- `work/core-completion-r001/v0-workspaces-r002/`.

Deadline child120s/outer300s; raccogli la sessione viva senza rilanciarla.
Profilo host del tool già usato per queste sonde: se serve require_escalated,
descrivi l'azione concreta; un rifiuto automatico resta IMPEDITA e va conservato,
senza aggirarlo o chiamarlo timeout. Nessuna modifica sysctl/AppArmor/rete/socket host.
Preserva target/lock/tokenizer/originali/fixture; verifica integrità prima/dopo.
Nessun pip/uv/seed/sync nella venv s007 o download per riparare collection.

R, wrapper, S, caratterizzazione e legacy_suite sono esiti separati. Verifica
contenuti F1–F6 e output effettivi; conserva perdite ereditate, mismatch, exit e
phase-events. Non normalizzare contenuti per uguaglianza né inventare numeri
storici. Suite exit1 e caratterizzazione PASS possono coesistere solo se le
prove reali soddisfano i criteri; collection67 da sola non basta.

**Errore del tuo launcher:** correggilo nella stessa chat. Se ha già interrotto
la chiamata, preserva parziali e raccogli i figli. La scope `local_launcher_repeat`
ammette una sola chiamata ulteriore con output r003 e preflight derivata, senza
nuovo freeze perché cambiano soltanto destinazioni con ricetta congelata.
Applica esattamente le sostituzioni relative→assolute della scope a ogni stringa
argv/config per prefisso e confine di path; registra config/hash/argv/diff prima
dell'avvio e verifica assenze. Non sovrascrivere il preflight r002 congelato.
Tutti i byte di entrambi i tentativi contano nei64MiB. Non usare questa eccezione
per FAIL reale R/V0, mismatch, cap, timeout di risorse o rifiuto sandbox. Se il
difetto è nel codice congelato, conserva risultato e prepara input aggiornati
per nuova label; non rinviare una consegna di sola preparazione dello strumento.

## 2. Audit statico EbookLib, indipendente

Raccogli prima la chiamata baseline. L'audit può proseguire dopo suo FAIL se
integrità e risorse rimangono valide. Scope **additional_authorized_activity**
identifica fonti/env/limiti/destinazione. Nessun secondo freeze preliminare package.

Acquisisci con HTTPS pubblico soltanto JSON `https://pypi.org/pypi/EbookLib/0.18/json`
e il singolo URL archivio esatto nella scope, `EbookLib-0.18.tar.gz`:
**115484byte**, SHA256
**38562643a7bc94d9bf56e9930b4927e4e93b5d1d0917f697a6454db5a1c1a533**.
Conserva raw JSON/archivio/headers pertinenti/fonte, redirect effettivi e body
realmente letto. Ambiente chiuso della scope, no proxy/auth/credenziali ereditati;
stdlib bootstrap `-I -B -S`, HTTP retries assenti, timeout aggregato120s.
Whitelist host HTTPS PyPI/files.pythonhosted.org. Max1MiB body cumulativo e2MiB
storage audit, inclusi report/copie. Traffico wire non misurato resta non misurato.
Destinazione esclusiva `evidence/implementation-r001/resume-ebooklib-audit-r001/`.

Controlla identità pubblicata e ricalcola archivio prima di leggerlo. Usa tarfile
in sola lettura: max2048 membri,8MiB dichiarati totali e1MiB letture metadata;
nomi relativi confinati, niente traversal/assoluti/link/device/file speciali o
duplicati ambigui. Non usare extractall; non importare codice dell'archivio.
Leggi e identifica soltanto pyproject/setup.py/setup.cfg/PKG-INFO e metadata/build
source strettamente necessari. AST/literal/static reading, **mai exec/setup.py,
hook, backend o installazione**. Riporta:

- metadata dichiarati, backend e dipendenze di build realmente trovati;
- eventuale fallback PEP517 inferito, distinto da dichiarazione nell'archivio;
- requisiti statici/dinamici ignoti, import o operazioni di setup rilevanti;
- possibilità concreta e costo di una futura operazione nativa circoscritta;
  input/config/constraints/argv/env/target/output proposti, incertezze residue.

Non rimuovere `--no-build`, eseguire un resolver nuovo, cambiare EbookLib/Marker,
ridurre il grafo universale o generare metadata/lock manualmente. Il nuovo backend
è lo scostamento sostanziale da ricevere prima dell'esecuzione; la prossima
richiesta deve essere fondata sull'audit, non un altro mandato di sola preparazione.
Usa uv nativo per le operazioni future, senza creare installer/resolver alternativi.

## Consegna unica

Evidenze baseline nuove `evidence/implementation-r001/resume-baseline-s007/` e
audit nella directory indicata; chiamate/argv/cwd/env pubblico, timestamp/sessioni,
exit/log/receipt native realmente prodotte, hash/byte/copie e guardie/costi pre/post.
Nessuna receipt inventata quando il wrapper non l'ha prodotta. Processi propri
raccolti; non terminare processi altrui. Proteggi i file già congelati.

Scrivi `implementation/stages/impl-r001-stage-baseline-s007/completion-r001.json`,
legata a request/scope/manifest esatti, con tentativi realmente eseguiti, esiti
R/S/V0/wrapper/legacy_suite e audit separati, NOT_EXECUTED quando opportuno.
Scrivi **implementation/report-r004.md**, delivery nuovo e checkpoint
**handovers/implementation-r001.md**; preserva report-r003 e delivery precedenti.
Nessuno stato comune/changelog/snapshot/arbitrato da implementatore.

Consegna audit e richiesta concreta del prossimo gate package sul backend
effettivamente identificato, con costi/input/comandi e senza dichiararla autorizzata.
Tests-s001 non è pronto senza lock e S/B/I consumabili. Baseline fallita non diventa
PASS per un audit riuscito; audit statico PASS non prova sicurezza/esecuzione backend.
Managed/bootstrap/backend non attestano package prodotto o equivalenza runtime.

Stato conclusivo **WAITING_FOR_SUPERVISOR_RECEPTION**, prossimo supervisore
prompt05+r004/r005/r006 e checkpoint corrente. Nessun servizio da attendere.
Stesso obiettivo, niente nuova review del piano per fix ordinari. V7/V8 obbligatorie
con mandato pesante distinto; V10/V11/pesi/font/inferenza esclusi, local_marker
non convalidato. Nessun product build/install/sync/test/Docker in questa tranche.
Due review indipendenti reali e arbitrato finale ancora da ottenere.
Nessun GO finale, cleanup, altro progetto, invio documenti, commit/merge/push,
promozione o deploy. Git manuale dell'utente. Trasferire file reali ignorati e
lavoro non committato, non soltanto manifest.
