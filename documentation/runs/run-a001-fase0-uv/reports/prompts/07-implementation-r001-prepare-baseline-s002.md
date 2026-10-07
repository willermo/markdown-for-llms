# Ripresa implementativa — preparare baseline s002 nel contesto host identificato

Agisci come **implementatore** della run `run-a001-fase0-uv`, repository
`/home/davide/workarea/markdown-for-llms`. Continua il mandato del prompt04 e
la tua consegna IMPEDITA. Il supervisore ha identificato un nuovo confine
operativo concreto D4; **non ha eseguito R né creato lo stage baseline s002**.
Questo passaggio autorizza la preparazione della richiesta s002, prima delle
nuove prove. Ruoli separati, nessun subagente o auto-snapshot.

## Letture e oggetto

Leggi AGENTS, CHANGELOG, PROJECT-CONTEXT/HANDOVER globali, STATE/HANDOVER della
run, skill manage-implementation-run e protocollo. Riprendi la skill di fedeltà,
brief A1–A7, piano r003 integrale se chat fresca e arbitrato r003 D1–D5.
Prompt04 resta il mandato generale; prompt06 e s001 sono storia della prima
ripresa, non istruzioni per rieseguire su output esistenti.

Leggi checkpoint/report propri aggiornati, impediment-r001.json e le due
receipt reali; poi, nella cartella
`evidence/supervisor-implementation-r001/runner-recovery-r001/`, **decision.md**,
reception.json, context-identification.json e proposed-commands-baseline-s002.json.
Le copie report-implementation-received.md e handover-implementation-received.md
identificano la consegna ricevuta senza congelare i tuoi file aggiornabili.

Leggi [checkpoint supervisore corrente — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
I percorsi senza prefisso sono relativi a `temp/run-a001-fase0-uv/`.

| Oggetto conservato | SHA-256 |
| --- | --- |
| plans/plan-r003.md | `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b` |
| arbitrations/arbitration-plan-r003.md | `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d` |
| snapshots/impl-r001-stage-baseline-s001.json | `ba686193b00dc068510167608a6ddb439e415210c5fb02e6cd14a6d4117a4c96` |
| implementation/stages/impl-r001-stage-baseline-s001/request.json | `39eefbd38a2d2448210fef013d3fdb00e0cdf4abf8e1b6c62497c320568c1763` |

Il contesto di ingresso di questo passaggio è
**snapshots/runner-recovery-context-r001.json**: identità finale nella
response.json della cartella supervisore e nel checkpoint/HANDOVER comuni,
scritti dopo il freeze. Verifica manifest/schema/hash e MATCH in ingresso:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
git diff --cached --name-only
python3 scripts/run_context.py verify run-a001-fase0-uv --label runner-recovery-context-r001
```

Branch feature/run-a001-uv, HEAD/dev/base
66ba82200e5def5a4db76f9bafccb0731b506091, indice vuoto; sei documenti modificati
e 17 file preparatori nuovi. Non reset/cambio branch/commit/merge/push.
GO piano r003 conservato; nessuna r004 o review del piano aggiuntiva. Nessun
GO codice, R-baseline/S/V0 mancanti, P1–P3 non iniziati. NO_GO r001/r002 storici.

## Risultato ricevuto e nuovo confine

Due wrapper R-bootstrap s001 exit2/IMPEDITA con EPERM **prima di invocation**.
Nessun inside.json/runner.json, namespace R o socketpair positivo. Non è prova
che unshare/Firejail host siano indisponibili. La syscall esatta resta ignota
nelle receipt prive di traceback; niente diagnosi inventata.

Alla ricezione, worktree s001 STALE solo per CHANGELOG aggiunto dopo le prove;
109 input/assenze e 58 artefatti invariati. Nell'inventario dei 1.942 file, i
due diversi sono report/checkpoint propri aggiornati con le copie precedenti
conservate; 1.940 restano uguali. Dopo questa supervisione, s001 è storico anche
per i sei metadata identificati: non chiamarlo MATCH o rigenerarlo.

Contesto selezionato: **lo stesso host UID/GID 1000**, medesimo clone, Python
bootstrap e ambienti esterni, con tool
`tools.exec_command(..., sandbox_permissions="require_escalated")`.
Il supervisore lo ha effettivamente identificato in sola lettura: exit0,
namespace rete net:[4026531833], Seccomp=0/NoNewPrivs=0/CapEff=0 nel processo
osservato, binari/hash/policy presenti. Non è una prova R e non garantisce
che i runner passino. Userns/AppArmor restano attivi, nessun valore modificato.

**Sola preparazione ora:** letture/versioni/hash/metadata/stat/assenze e scrittura
dei nuovi inventari/request propri. Si può usare quel profilo per l'inventario
host, conservando argv/exit/output e senza socket/connect/namespace applicativo.
Niente sudo, chmod, setuid nuovo, modifica sysctl/AppArmor/rete/socket/profilo
persistente. Firejail è quello esistente, identificato per hash, non installato.

La review automatica del tool resta attiva **per ciascun comando**. Per la
preparazione, justification deve descrivere solo letture del target D4; niente
prefix_rule ampia. Un'approvazione di lettura non vale per una futura sonda.
Se il tool rifiuta, registra azione e motivo e consegna al supervisore; non
spostare il comando in un altro canale per aggirare il rifiuto. Non occorre una
nuova approvazione dell'adozione uv: qui il supervisore concretizza D1/D4.

## Preparare input stabili e request s002

1. Preserva integralmente request/snapshot/receipt/output s001 e gli originali.
   Mantieni `/tmp/a001-uv-baseline-pi8cvs6x`: venv, tmp, workspace, uv-cache e
   tiktoken-cache. Ricalcola gli input tecnici pertinenti e 24 assenze. Nessuna
   installazione/acquisizione o ricreazione implicita di venv/cache. Se un
   input non coincide, identifica il cambiamento prima di dichiararlo pronto.
2. Crea una directory nuova
   `evidence/implementation-r001/preparation-baseline-s002/`. Produci lì nuovi
   `runner-target-bootstrap.json`, `runner-target-baseline.json` e
   `baseline-inputs.json`, più inventario innocuo dell'esecuzione host e le
   evidenze hash. Non sovrascrivere gli inventari s001 o riutilizzare la loro
   ownership osservata nel sandbox come dato del contesto host.
3. Nel profilo host selezionato aggiorna effettivamente: UID/GID, namespace
   osservati/policy senza segreti, binari/versioni/hash, interpreter/prefix/
   stdlib, dipendenze e file/cache tokenizer, Pandoc, directory/assenze.
   Identifica gli 11 path daemon noti e alias/target canonici; integra eventuale
   DOCKER_HOST locale/rootless/XDG pertinente, senza API o credenziali. Solo
   stat e letture ora. Il namespace effettivo sarà riletto da R, non riparato
   per far coincidere un numero storico. Nessuna scansione dei dati privati.
4. Mantieni i cinque diagnostici e fixture tecniche identici, salvo un difetto
   concreto identificato e riportato nella nuova richiesta. Non cambiare il
   wrapper per rimuovere il socket sintetico, né sostituire il socketpair reale
   con mock. I dieci originali/quattro metadata copiati restano la baseline;
   non introdurre P1–P3 o modifica della pipeline senza baseline valida.
5. Usa gli **otto argv proposti** in
   [proposed-commands-baseline-s002.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
   Sono derivati dagli argv s001, con nuove label/target/input/output e profilo
   esplicito `require_escalated`. Ora `may_execute_now=false`: questi file
   sono dati da completare/confrontare nella richiesta, non comandi da lanciare.
   CWD e timeout restano quelli identificati; nessuna stringa eval o shell
   composta da valori non fidati. Se sono necessarie altre variazioni, documentale.
6. Scrivi **implementation/stages/impl-r001-stage-baseline-s002/request.json**
   schema1 D1, autore implementatore, stato WAITING_FOR_STAGE_SNAPSHOT, run/
   revision/stage/label/Git, file/hash/byte o assenze, artefatti relativi confinati,
   execution_boundary e profilo del tool per ogni passo, costi e invalidazioni.
   Riporta il contesto reale appena letto e la decisione del supervisore, non
   affermare R PASS o compatibilità operativa già verificata.
7. Elenco da congelare: piano/arbitrato/mandato04/prompt07/request nuova,
   originali e inventari stabili, fixture nei files del repository, nuovi target
   e baseline-inputs, decisione/identificazione/argv del supervisore; conserva
   antecedenti s001/impedimento e il manifest del recupero per la catena di
   provenienza. Identifica ciascun path/hash prima della consegna. Escludi
   S/receipt/output futuri e report/checkpoint mutabili; nessun self-hash
   della request o riferimento al futuro digest s002 negli input congelati.
8. Aggiorna soltanto report/checkpoint propri e changelog del lavoro effettivo
   prima della consegna. Conserva prima le versioni attuali se verranno
   aggiornate. Nel checkpoint: **WAITING_FOR_STAGE_SNAPSHOT**, richiesta s002,
   processi propri, contesto/perimetro/costi e R/S/V0 ancora non eseguiti.
   Consegna al supervisore con prompt05; non eseguire probe dipendenti o
   creare snapshot, né aggiornare STATE/HANDOVER/eventi/indici/metadata comuni.

Il freeze s002 resta al supervisore **dopo questa request reale**. Il contesto
di recupero corrente congela la disposizione e gli antecedenti, non sostituisce
lo stage s002. Cambiamenti preparatori rendono storico il suo worktree: conserva
la transizione, non falsificare MATCH. Il supervisore completerà i metadata
tracciati prima del nuovo freeze e consegnerà identità/prompt di esecuzione.

## Perimetro delle future prove dopo il freeze, già concretizzato

R-bootstrap unshare via tool `require_escalated`; se negato/inadeguato,
Firejail esistente con noprofile/net=none e blacklist dei path inventariati e
del socket sintetico proprio. Entrambi i candidati hanno output esclusivi
`runner-bootstrap-r-<candidato>-s002`. Un candidato completo PASS ammette
R-baseline dello stesso candidato, poi S e V0 con R/preflight e comando
nella **medesima invocazione/namespace**. Argv integrali nel JSON proposto,
che la request identificherà per la successiva risposta di freeze.

Il confine host del wrapper consente solo connect/close diagnostici ai daemon
noti e socket sintetico nuovo: **zero send/recv/API daemon o richieste operative**.
Le sonde IP documentali avvengono solo dentro il runner dopo namespace distinto
e assenza di egress/rotte. Le prove applicative restano sempre nel runner, non
nel processo host non confinato. Socketpair positivo reale, sintetico positivo
fuori prima/dopo e negativo padre/figlio dentro, daemon negati, origini/hash/
cache, cleanup e namespace figli restano obbligatori. Nessuna attenuazione.

S futura: `implementation/stages/impl-r001-stage-baseline-s002/sources.json`,
con copia baseline/host e snapshot s002 effettivo. V0 futura:
`evidence/implementation-r001/baseline-results-s002/` e workspace esterno
`/tmp/a001-uv-baseline-pi8cvs6x/workspace/v0-s002`. Non crearli durante la
preparazione. Non ripetere i comandi s001 su directory esistenti.

Qualsiasi rifiuto policy o gate incompleto: conserva exit/stdout/stderr e
IMPEDITA, nessun S/V0/P1–P3 fuori dal percorso. Un rifiuto della review automatica
si riporta esplicitamente; nessun fallback terminale/remoto o altra macchina
senza un nuovo perimetro del supervisore. F6 runtime ancora non misurato;
tuning dopo freeze s002 richiederà s003 prima di un PASS definitivo.

Stime precedenti R/S circa 2 minuti, V0/audit circa 30 minuti e output 50 MB:
aggiornale con spazio corrente e distinguile dai risultati. Rete/acquisizioni
previste 0 byte. I sette test preparatori rimangono mock/sintetici; nessun test
applicativo nuovo deve sostituire R. Audit contenuti integrali F1–F6, suite
legacy con conteggi/exit separati e tutti i requisiti prompt04/06 restano futuri.

V7/V8 obbligatorie future, perimetro/costo pesante distinti; V10/V11, pesi/font/
inferenza esclusi. Nessun invio remoto implicito, deploy o Git di integrazione.
Due review indipendenti del codice e arbitrato finale ancora necessari.
Temp ignorata e ambienti esterni: preserva il lavoro effettivo quando cambia chat.
