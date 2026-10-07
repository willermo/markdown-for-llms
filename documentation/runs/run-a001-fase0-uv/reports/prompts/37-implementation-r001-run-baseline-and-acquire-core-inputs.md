# Implementazione — baseline ufficiale e input managed/lock del core

Agisci esclusivamente come **implementatore**, repository
`/home/davide/workarea/markdown-for-llms`, skill manage-implementation-run.
Continua lo stesso obiettivo di prompt36: esegui R→S→V0 ufficiali e le acquisizioni
standard ammesse, poi consegna risultati e input reali del prossimo gate.
Nessuna delega, supervisione o review simulata. Nessun nuovo giro di pianificazione.

## Autorità e freeze

Leggi AGENTS, skill, protocollo, STATE/HANDOVER, brief A1–A7, indici architettura/
decisioni/roadmap, piano r003 integrale se non già letto e arbitrato D1–D5.
Prompt36 e r004 restano per obiettivo/fix; **r005** ammette la nuova tranche di
risorse e precisa ambiente e ledger. Prompt04/05 e vecchi scope sono storici nei
punti modificati esplicitamente; non autorizzano altri costi o prove.
Leggi checkpoint implementatore `handovers/implementation-r001.md`, report-r002,
delivery core-completion-r001 e checkpoint supervisore core-reception-r001.
Percorsi seguenti relativi a `temp/run-a001-fase0-uv/`:

- piano `plans/plan-r003.md`: SHA256
  462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b;
- arbitrato `arbitrations/arbitration-plan-r003.md`: SHA256
  f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d;
- `arbitrations/addendum-operational-protocol-r005.md`;
- `implementation/stages/impl-r001-stage-baseline-s006/request.json`,
  SHA256 5b8c44155d37a58868a8c4bb35b6d47aad96a36adbdcc1053a4e920536b9ad14;
- `implementation/stages/impl-r001-stage-package-s008/request.json`,
  SHA256 fad6bb929443ebda2681f6829f7bd945155c365ef2481bb7cd468393def204bf, nuova richiesta supervisore derivata dalla richiesta costi r002;
- `evidence/supervisor-implementation-r001/impl-r001-stage-baseline-s006/`:
  entry/transition/checks/authorized-scope/response/freeze-verify.json;
- stessa directory con suffisso `impl-r001-stage-package-s008/`:
  authorized-scope/response/freeze-verify.json e sources.md.

Scope baseline SHA256 **4717e9da1f81da120ff572175356cf80ccf143f6d5ba366605e0056f68a9de08**;
scope package SHA256 **d509dad9160345ab736f3af3cc9fec571860ce097cac386c32fec080d6f3cb02**. Input immutabili: non modificarli.
La richiesta costi d'autore resta r002/not-ready alla sua consegna; autorizzazione
nuova nelle scope/r005 e nei freeze reali. Nessuna falsa modifica della sua storia.

Snapshot baseline-s006: **594062byte**, SHA256
**32bb6220199d2e11f2240dbf79c9b62e65b22a039119473c648d7423fff79e2b**, 123file/1966artefatti.
Snapshot package-s008: **28172byte**, SHA256
**13598aa7be6118e5661a79a7a17be19c2807ce41bd020de47d1470f148ae0835**, 123file/15artefatti.
Worktree comune **d8c661ebba0ad817bc860dadea790cfeb3583661ed7712b8618fdc0e35eab530**, MATCH alla consegna supervisore.
Entrambi sono input, non GO del codice né PASS baseline/installazione del prodotto.

Prima di operazioni e alla consegna ricalcola i byte/hash di manifest e scope,
poi da radice esegui separatamente:

```bash
python3 -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-baseline-s006
python3 -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-package-s008
```

Mismatch di input congelati resta FAIL: niente helper allentato/autofreeze.
Git feature/run-a001-uv, HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto. Il delta core36 e metadata di ricezione è identificato, non una
vecchia lista di modifiche. Non scrivere root uv.lock o modificare tracciati in
questa tranche: lock viene prodotto nella copia ignorata; nuovo freeze successivo
quando gli input consumabili e l'eventuale delta sono effettivamente pronti.

## 1. R→S→V0 ufficiali, una chiamata del wrapper

Usa **proposed_operations[0]** della request baseline, argv/cwd/env esatti.
Il wrapper avvia Firejail, ripete R parent/child/D4; soltanto su PASS esegue il
preflight S e poi V0 nello stesso namespace. Non produrre S o avviare la suite
fuori da questa precedenza. Scope next-scope s007 e r004/r005 restano per confini,
non trasferiscono i PASS preliminari al wrapper congelato corrente.

Launcher stdlib con argv strutturati, `shell=False`, `close_fds=True`,
`env=dict(op['environment_launcher'])`, nessun merge con os.environ. Prima verifica
hash della request; usa op['argv'] senza shell/eval. Bootstrap assoluto già
identificato `/home/davide/.pyenv/versions/3.12.3/bin/python3 -I -B` per il launcher.
Tool host require_escalated già ammesso per queste sonde/runner locali esistenti:
nessun nuovo privilegio/profilo, syscall host o modifica del daemon. Rifiuto
sandbox: conserva azione/ragione ed esito IMPEDITA; nessun aggiramento o esito inventato.

Output esclusivi già assenti: S-baseline-official-r001.json,
core-completion-r001/official-baseline-r001 e work/core-completion-r001/v0-workspaces-r001.
Nessun retry su questi output se esistono. Deadline wrapper figlio120s,
esterno300s, preflight60s; raccogli la sessione viva, non rilanciarla.

Preserva target/lock s007 con inventario prima/dopo; nessun pip/uv/seed lì.
Usa check_preserved.py main soltanto per questa fase con budget storico500/384,
riserve64+16MiB/free516MiB. Originali10+4, fixture, pytest ini e suite originali
sono gli input congelati: niente modifiche per ottenere uguaglianza.
-I -B e startup/origine controllati anche nei nuovi processi figli.

Distingui R, S, caratterizzazione V0 e legacy_suite: 62pass5fail storico non è
un esito da inventare o da promuovere a PASS. Nuovi risultati osservati e contenuti
integrali sono la prova; divergenze vanno preservate e spiegate. Mismatch e
collection inattesa non consentono avvii/installazioni per ripararla.

## 2. Acquisizioni native nel budget ammesso

Soltanto dopo chiusura e raccolta del tool baseline, verifica package-s008 e
input source_inputs, binario uv e assenze della scope package. Se baseline
IMPEDITA/FAIL, sospendi i suoi derivati; l'acquisizione indipendente può proseguire
solo se integrità s007/fonti, confini e risorse rimangono validi. Nessuna concorrenza.

La scope package contiene **operations** nell'ordine:
managed → lock → lock-check → backend-venv → backend-wheel-only.
Usa ciascun argv/cwd/environment esattamente con subprocess strutturato;
una chiamata distinta per operazione, risultato/receipt e identità prima della
successiva. Non installare altro né concatenare sync/build/test del prodotto.
Puoi completare e correggere i launcher/guardie propri non congelati nella stessa
chat: nessuna consegna di sola sintassi, preserva errore/correzione/prove pertinenti.
Non cambiare sorgenti, pin o strumenti congelati senza nuova label.

Prepara le directory nuove della scope con creazione esclusiva: copia solo i
16 source_inputs nella lock-project-r001, verificando hash e provenienza prima
e dopo; niente symlink/input dinamici. Cache/tmp/config nuovi e environment chiuso;
controlla assenze/config /etc indicate prima/dopo, nessun contenuto segreto nei log.
Non usare --no-config per perdere gli indici/guardie del pyproject. Niente override
pin/fonti, proxy/credenziali/config daemon o ambiente Python ereditati.

- **managed**: CPython3.12.13 GNU/key/build20260310 e checksum della scope;
  .venv-python locale, --no-bin, ambiente manual solo per l'install esplicito.
  Verifica catalogo effettivo/URL/redirect/checksum e origine/stdlib/versione
  installati prima di riusarli. Nessuna variante o fallback pyenv/interprete host.
- **lock**: no-build, Python managed esplicito, never/no-python-downloads;
  grafo universale/indici CPU-cu126, soltanto metadata. PEP658/range o piccole
  wheel entro quote; backend dinamico/sdist non ammessa o payload ML/native
  pesante ⇒ stop del passo, nessun grafo/meta inventato o fallback di pin.
- **lock-check**: --check offline con lock/hash/inventario appena prodotti;
  conserva hash prima/dopo. Lock resta nella copia, non root uv.lock.
- **backend-venv / backend-wheel-only**: ambiente nuovo managed, setuptools84
  solo wheel/no-deps/no-build. Origine/site iniziale controllati; conserva fonte,
  hash dell'archivio, METADATA/RECORD/startup prima di build o nuovi avvii che li
  consumano. Nessun altro backend o installazione runtime/dev/ML.

Budget prospettico cumulativo: **run intera + .venv-python**, max1GiB/stop896MiB,
304MiB incremento di picco ammesso e80MiB riserve (64 prove+16 esterne) nel gate;
libero minimo1GiB. Tutte le cache/tmp/log/report/storico contano, nessun deposito
fuori ledger o cleanup per passare. Quote dichiarate32MiB managed rete/32MiB metadata/
8MiB backend sono stime ammesse, non quota atomica di traffico attestata: conserva
le misure realmente disponibili e i loro limiti. Non acquisire payload pesanti.

Guardie launcher: RLIMIT_FSIZE32MiB/file ereditato, log1MiB/stream eJSON8MiB,
misure logico/allocato di entrambi i root pre/post e scansione periodica entro1s,
stop alla quota incrementale o soglia cumulativa con riserve. Timeout scope:
managed600s/lock900s/altri120s. Monitor/costo della scansione non garantiscono una
quota atomica. Termina/raccogli solo il proprio processo/gruppo dopo timeout/cap,
mai processi altrui; conserva raw/parziali. Il monitor non deve diventare un nuovo
installer/resolver: l'operazione è sempre uv nativo.

Il main check_preserved storico usa384MiB: dopo l'acquisizione non usarlo come
ammissione del pool nuovo. Riusa verify(repo,manifest) per **sola integrità s007**,
poi misura separatamente il ledger della scope; non modificare il helper congelato.

## Consegna e prossimo gate

Evidenze nuove in evidence/implementation-r001/resume-baseline-s006/ e
resume-package-s008/: chiamate reali, environment pubblico, timestamp/exit/sessioni,
receipt/log/copie/hash/byte, budget e guardie, source/copia/output distinguibili.
Completion separate nelle due directory implementation/stages/<label>/,
legate a request/scope/manifest esatti; ogni passo non avviato NOT_EXECUTED.
Non inventare una receipt nativa: la tua registra l'operazione osservata e i
limiti. Baseline PASS caratterizzazione non è suitePASS/ABI/GO codice; acquisizione
PASS non è B/I/E o package finale. Fallimenti storici e /tmp mancanti conservati.

Preserva report/checkpoint d'ingresso; scrivi report autore nuovo
implementation/report-r003.md e delivery con risultati separati. Aggiorna solo
handovers/implementation-r001.md con output reali e request esatta del prossimo
gate se pronta; nessun registro comune/changelog/arbitrato/snapshot autonomo.
Non creare una request tests-s001 pronta senza i suoi input consumati. Usa i
nuovi V0/lock/origine/backend per concretizzare costi e comandi package/tests
successivi; nessuna build/installazione del prodotto aggiuntiva in questa tranche.

Arresti sostanziali: integrità/origine/config inattese, limiti/spazio/timeout,
nuovo scope/costo/backend/pin, impossibilità dei confini o rifiuto sandbox.
Sospendi il dipendente, continua l'indipendente ammesso. Normali errori del
launcher si correggono nella chat; nessun retry cieco su operazione già iniziata
con target/parziali o marker preservati. Un fix di codice congelato richiede
nuova label prima delle prove dipendenti, senza nuovo piano.

Stato finale WAITING_FOR_SUPERVISOR_RECEPTION; prossimo supervisore prompt05
con r004/r005 e checkpoint corrente. Nessun servizio automatico da attendere.
V7/V8 obbligatorie con mandato pesante distinto; V10/V11/pesi/font/inferenza,
invio documenti, modifiche host/privilegi/rete/profili, altro progetto e deploy
esclusi. Nessun GO finale/commit/merge/push/promozione; Git manuale dell'utente.
Trasferire anche sorgenti/copie/work/evidenze ignorati reali, non soltanto manifest.
