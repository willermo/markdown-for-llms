# Ripresa implementativa — propagazione TMPDIR e preparazione baseline s003

Agisci come **implementatore** della run `run-a001-fase0-uv` nel repository
`/home/davide/workarea/markdown-for-llms`. Continua il mandato del prompt04,
preservando l'impedimento reale s002. **Ora soltanto preparazione s003:** un
argomento nel wrapper Firejail, verifica pura dell'argv, inventari e request.
Nessuna nuova R/S/V0 prima del freeze e della risposta del supervisore.
Chat distinta dal supervisore, nessun subagente, review simulata o auto-snapshot.

## Letture e identità in ingresso

Leggi AGENTS, CHANGELOG, PROJECT-CONTEXT/HANDOVER globali, STATE/HANDOVER della
run, skill manage-implementation-run e protocollo run-lifecycle. Riprendi la
skill di fedeltà per la futura baseline; non eseguirla come autorizzazione a
probe applicativi in questo passaggio. Brief A1–A7, piano r003 **integrale**
se chat fresca, arbitrato r003 D1–D5 e mandato04 rimangono vincolanti.

Leggi report/checkpoint propri aggiornati, impediment-r002.json dello stage
s002 e receipt/inside/runner dei due bootstrap, mediante lettura strutturata
degli inventari grandi. Recupera gli intervalli eventualmente troncati.
Prompt08 è storia di una sequenza fermata, non comando per ripetere gli output.

Nella cartella `evidence/supervisor-implementation-r001/runner-recovery-r002/`
leggi **decision.md**, reception.json, sources.json/local-manual.json e
proposed-commands-baseline-s003.json. Le copie report/handover ricevute e
run_offline-received-s002.py conservano l'oggetto della disposizione.
Leggi il [checkpoint supervisore — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
e la [risposta — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Percorsi senza prefisso relativi a `temp/run-a001-fase0-uv/`.

| Oggetto immutabile | SHA-256 |
| --- | --- |
| plans/plan-r003.md | `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b` |
| arbitrations/arbitration-plan-r003.md | `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d` |
| prompts/04-implementation-r001.md | `fc93453d9fa57dc362d9e47d4576716f5bcdbd8cc7a07c7361e43949d1042035` |
| snapshots/impl-r001-stage-baseline-s002.json | `c73d1547ad79ce19cd2b481b6f252ceb1cb21148a15c9fdf70768a664c0fc34f` |
| implementation/stages/impl-r001-stage-baseline-s002/request.json | `985cfb1c8c5b9430b23ac7b90f5d2ba3d41386cfc0707f2d06c69745ba07e19d` |
| implementation/stages/impl-r001-stage-baseline-s002/impediment-r002.json | `48108730a78b4dbde3905f19927918865297e5aee02f482ec8cc968fff4618ea` |

Contesto di ingresso **snapshots/runner-recovery-context-r002.json**;
SHA manifest/worktree e lista esatta nella risposta del supervisore, prodotta
dopo il freeze del contesto per evitare autoreferenzialità. Verifica schema,
hash, file/artefatti e **MATCH prima di modificare il wrapper**:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
git diff --cached --name-only
python3 scripts/run_context.py verify run-a001-fase0-uv --label runner-recovery-context-r002
```

Branch feature/run-a001-uv; HEAD/dev/merge-base
66ba82200e5def5a4db76f9bafccb0731b506091, indice vuoto. Sei metadata documentali
modificati e 17 file preparatori nuovi all'ingresso; prodotto legacy non migrato.
GO piano r003 e NO_GO r001/r002 conservati; nessuna r004 o review piano nuova.
Il contesto di recupero congela la disposizione, **non è lo stage s003**.
Alla ricezione, s002 era STALE soltanto CHANGELOG, tutti 109 artefatti invariati;
dopo supervisione anche sei metadata identificati. Non pretendere MATCH storico
né rigenerare s002. Il delta autorizzato del wrapper renderà storico anche
il contesto di ingresso: registra la transizione senza attribuirgli MATCH finale.

## Risultati ricevuti e delta assegnato

Entrambi R-bootstrap s002 sono reali, exit2/IMPEDITA, senza rifiuto della review
automatica. Unshare isola IP ma mantiene raggiungibili sintetico e sei path
daemon in padre/figlio: inadeguato D4. Firejail nega IP/path/sintetico e mantiene
socketpair positivo e 1.393 prerequisiti corrispondenti per processo, ma
**TMPDIR è nullo in padre e figlio**. Nessun PASS R trasferibile; R-baseline,
S/V0/P1–P3 non avviati. I sette test preparatori precedenti non sono R/V0.

Il tag Firejail 0.9.72 e il manuale installato supportano `--env=nome=valore`.
La possibile rimozione TMPDIR al lancio setuid rimane inferenza; non serve un
trace né modifica del loader. Fonti, accessi falliti e limiti nella decisione.
Non chiamare la documentazione prova della nuova invocation.

1. Conserva **tutti** request/manifest/output/receipt s001/s002. Prima di
   aggiornare report/checkpoint preserva le versioni attuali con nomi nuovi,
   ad esempio `implementation/report-r001-before-baseline-s003.md` e
   `handovers/implementation-r001-before-baseline-s003.md`; verifica hash con
   le copie ricevute. Nessuna correzione retroattiva delle prove negative.
2. Ricalcola wrapper ricevuto SHA
   `f8b6d1dc003a773f52fa6c31433a7c4ec4a9ee9861c15b5e1b358dbd43873edf`
   e confrontalo con la copia conservata. Aggiungi **un solo argomento**
   `--env=TMPDIR=<target["tmpdir"]>` al prefisso Firejail in
   `scripts/diagnostics/run-a001-fase0-uv/run_offline.py`, prima di `--`.
   Usa il valore assoluto identificato dal target, non un valore libero.
   Mantieni noprofile/net=none, tutte le blacklist note/canoniche/alias e
   sintetica, clean_env, close_fds, timeout, verifiche snapshot/input e cleanup.
   Il percorso comune deve includerlo in R-bootstrap, R-baseline, S e V0.
3. Non cambiare check_runner né i gate TMPDIR/TIKTOKEN/socketpair/path/IP/
   namespace/figli/prerequisiti. Non impostare TMPDIR dopo l'osservazione R,
   non simulare la receipt e non reinserire proxy/PYTHONPATH/PYTHONHOME o altre
   variabili rimosse. Producer S/driver V0/fixture/originali/metadata di prodotto
   e dipendenze rimangono identici. Preserva ramo unshare del wrapper byte per
   byte nel diff. Altri delta necessari → documentali prima della consegna.
4. Verifica il delta con sintassi/diff e una verifica **pura** dell'argv, statica
   o unitaria: argomento esatto dal target prima di `--`, blacklist e comando
   conservati, ramo unshare invariato e percorso comune per i quattro passi.
   Nessun avvio Firejail/unshare, socket, namespace, import applicativo o
   conversione. Se estendi il test preparatorio, identifica il delta e il suo
   output come prova della costruzione argv, mai PASS del runner.

## Inventari e richiesta reale s003

Mantieni `/tmp/a001-uv-baseline-pi8cvs6x`: venv, tmp, workspace, uv-cache e
tiktoken-cache. Nessuna ricreazione/installazione/acquisizione implicita.
Ricalcola gli input/assenze pertinenti, compresi i 1.940 tecnici e 825 file
cache del vecchio inventario, identificando il solo delta wrapper e l'eventuale
test puro aggiunto. Ricontrolla dieci moduli originali/quattro build input
copiati e hash HEAD, fixture/provenienza, interpreter/tool/deps/guardie/cache.
Conserva il significato di byte mancanti negli inventari vecchi; puoi osservare
i byte attuali senza inventare misure storiche.

Crea `evidence/implementation-r001/preparation-baseline-s003/`, con nuovi
runner-target-bootstrap.json, runner-target-baseline.json, baseline-inputs.json,
inventario/hash/assenze, diff wrapper e verifiche pure. Aggiorna i riferimenti
al nuovo wrapper nei target e negli input della request, evitando hash vecchi.
Niente sovrascrittura degli input s002 o output futuri creati in anticipo.

Per inventario host: stesso clone/host UID/GID1000 e tool `exec_command` con
`sandbox_permissions="require_escalated"`, review automatica per ciascuna
lettura, justification concreta, nessuna prefix_rule ampia. Si possono leggere
stat/versioni/hash, namespace corrente, policy, spazio e path daemon noti/
alias/canonici/XDG pertinenti; niente socket/connect/daemonAPI, namespace
applicativo, segreti o scansione dei dati privati. Conserva parametri/exit/
stdout/stderr della lettura. Le osservazioni storiche non sostituiscono il
contesto corrente; la compatibilità R resta da dimostrare dopo freeze.

S003 ha **Firejail unico candidato attivo**, quattro passi proposti in
[proposed-commands-baseline-s003.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Unshare s002 già realmente inadeguato abilita la scelta del fallback solo se
host/binario/ramo unshare/inventario path sono invariati e riconfermati; non
riavviarlo identico. Registra questa equivalenza limitata e le receipt negative.
Se quelle condizioni cambiano, riconsegna lo scostamento al supervisore.
Non importare PASS Firejail s002 né trasformare TMPDIR in controllo opzionale.

Completa la **request implementatore**
`implementation/stages/impl-r001-stage-baseline-s003/request.json`, schema1 D1,
run/revision/stage/label/autore/Git, **WAITING_FOR_STAGE_SNAPSHOT**:

- Lista esatta file/hash/byte/assenze, input esterni e guardie, nuovi target,
  comando/cwd/timeout/output di ciascun passo, profilo tool/costi/invalidazioni.
  Confronta i quattro argv con i dati proposti, mantieni may_execute_now=false.
- Artefatti reali da congelare: piano/arbitrato/mandato04/prompt09/request nuova,
  originali e inventari/target stabili, decisione/fonti/argv del supervisore,
  contesto recupero e manifest/impedimento/receipt negative s002 pertinenti.
  Path relativi confinati nella run; fixture/diagnostici visibili nei files.
- Nessun self-hash della request, digest futuro s003 o S che rinvia al proprio
  snapshot; escludi report/checkpoint mutabili, response/checks di handoff
  successivi e output futuri. L'ordine rimane input → snapshot → S → B/I/E.
- Quattro argv con profilo require_escalated, review per ciascun comando,
  senza prefix_rule ampia. Costi precedenti R/S circa 2 minuti, V0/audit circa
  30 minuti e output 50 MB, rete di acquisizione 0: aggiorna spazio/stime e
  distingui risultati effettivi. Non autorizzano V7/V8 o pesi/font/inferenza.

Aggiorna soltanto report/checkpoint propri e CHANGELOG per il lavoro **effettivo**
prima della consegna; non aggiornare STATE/HANDOVER comuni/indici/eventi/ADR.
Checkpoint: WAITING_FOR_STAGE_SNAPSHOT, richiesta s003 esatta, nuovi hash/diff,
test puri distinti, processi propri e R s003 NON_ESEGUITA, S NON_GENERATO,
V0 NON_ESEGUITA. Consegna al supervisore tramite prompt05. Non creare snapshot
o eseguire probe dipendenti mentre il supervisore è assente.

## Sequenza futura, vietata durante questa preparazione

Dopo request reale, supervisore valida input/delta/costi, aggiorna metadata,
crea/verifica **impl-r001-stage-baseline-s003** e consegna un prompt completo
di esecuzione distinto. Solo allora: R-bootstrap Firejail **completo**, poi
R-baseline completo, S, V0. Ogni comando ripete R/preflight nello stesso
namespace del programma; TMPDIR esatto padre/figlio è richiesto prima del
programma, come tutti i gate IP/path/sintetico/socketpair/FD/figli/cache/hash.

Output futuri (ora devono essere assenti): le quattro directory runner
`runner-bootstrap-r-firejail-s003`, `runner-baseline-r-firejail-s003`,
`runner-baseline-sources-firejail-s003`, `runner-baseline-v0-firejail-s003`
in evidence/implementation-r001; S in
`implementation/stages/impl-r001-stage-baseline-s003/sources.json`, risultati
`evidence/implementation-r001/baseline-results-s003/` e workspace esterno
`/tmp/a001-uv-baseline-pi8cvs6x/workspace/v0-s003`.

Host wrapper con sole connect/close diagnostiche daemon noti, zero send/recv/API;
sintetico proprio positivo host pre/post e negativo padre/figlio dentro,
socketpair positivo reale, niente egress, cleanup e namespace coerenti.
Nessuna sonda abbreviata del solo TMPDIR o baseline fuori runner. Tool review
negata o gate incompleto → IMPEDITA e consegna motivo/azione al supervisore,
senza fallback altro canale/terminale/macchina per aggirare il rifiuto.
Nessun sudo/privilegio/setuid/profilo nuovo o modifica host/sysctl/AppArmor/
rete/permessi/socket. S003 diverso dopo freeze → s004 o label successiva D1.

F6 contenuti/token/multichunk/overlap, audit integrale F1–F6 e suite legacy con
conteggi/exit restano futuri; P1–P3 dopo baseline valida. V7/V8 obbligatorie
future e costo operativo distinto; V10/V11/pesi/font/inferenza esclusi.
Review codice doppia e arbitrato finale aperti. Nessun invio remoto implicito,
deploy o Git di integrazione: commit/merge/push/promozioni dell'utente.
Temp ignorata e ambienti esterni non trasferiti da Git: preservare il lavoro.
