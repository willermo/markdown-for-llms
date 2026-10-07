# Report implementazione — run-a001-fase0-uv — r001

- Autore: **Codex / OpenAI / famiglia GPT-6**, interfaccia IDE/API.
  Modello specifico e ID chat non esposti. Nuova chat nel solo ruolo
  implementatore assegnato dall'utente; nessun subagente o altro ruolo assunto.
- Data: **2026-10-03, Europe/Rome**. Fase **0.1**.
- Prompt: [04-implementation-r001.md](../prompts/04-implementation-r001.md).
- Piano [r003](../plans/plan-r003.md), SHA-256
  `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`.
- Arbitrato [r003](../arbitrations/arbitration-plan-r003.md), SHA-256
  `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`.
- Branch `feature/run-a001-uv`; HEAD/dev/merge-base
  `66ba82200e5def5a4db76f9bafccb0731b506091`; indice vuoto.
- Stato: **PARZIALE — IMPEDITA / RETURN_TO_SUPERVISOR**.
  Stage `impl-r001-stage-baseline-s002`: entrambi i bootstrap R eseguiti,
  exit2/IMPEDITA. R-baseline NON_ESEGUITA, S NON_GENERATO, V0 NON_ESEGUITA.
  S001 conservata; prossimo lavoro da identificare e congelare come baseline s003.
- Nessun GO emesso. Questo report non è la consegna finale alle review del codice.
  Snapshot finale, catena S/B/I/E e prove applicative ancora da produrre.

## Identità e recupero

Eseguiti tutti i sei comandi di ingresso prima delle modifiche: identità
attesa, sei modifiche documentali di supervisione, indice vuoto e MATCH.
[Entry checks — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md) conserva
la ricattura prima delle scritture; primo controllo anche nel tool `c92dd5`.
Manifest implementation-context-r001 SHA-256
`c9f8a4c099e7101ccdf1aa5d0c4c386679d301cd47312d841483fcc5c5950cda`,
worktree `34ad9560a9f3a58f3b02cdd698f6885a3171430c91297fc2bedb32c95d7d4223`:
entrambi confrontati con checks/handover del supervisore.
Ricalcolati gli artefatti dei tre manifest: 90/109/120 invariati all'ingresso.

Letti nell'ordine richiesto contesti, skill/protocollo/template, indici e ADR,
brief, tutte le 1400 righe del piano e 315 dell'arbitrato, arbitrati antecedenti,
checkpoint/findings/checks del pianificatore, entrambi i report e checkpoint
r003, receipt/sources/transition/checks del supervisore e tre manifest.
Output troncati recuperati per intervalli o campi strutturati pertinenti.
Consultati sorgenti e test pertinenti, packaging/Docker/Compose e help congelati;
nessun segreto letto. Applicate manage-implementation-run e
verify-conversion-fidelity; guide Diátaxis P7 ancora da iniziare.

Il contesto d'ingresso diventa storico dopo le modifiche preparatorie
autorizzate. Non è stato rigenerato né usato per chiamare MATCH il nuovo lavoro.
Il freeze successivo appartiene al supervisore; piano/review/arbitrati e
snapshot originali restano invariati.

## Preparazione realizzata e verificata

| Elemento | Esito osservato | Evidenza |
| --- | --- | --- |
| Ingresso Git e contesto | MATCH, valori attesi | entry-checks.json |
| Originali | Dieci moduli byte-identici alle copie; quattro metadata copiati | baseline-originals.json, baseline-build-inputs.json |
| Host/binari/policy | pyenv 3.12.3, uv 0.10.10, Pandoc 3.1.3, unshare 2.39.3, Firejail 0.9.72 | host-inventory.json |
| Daemon noti | 11 path/alias/canonici inventariati; nessun connect | daemon-target.json |
| Venv | Nuova, esterna, creata con pyenv assoluto `-I -m venv` | baseline-location.json, host-inventory.json |
| Dipendenze baseline | 17 pacchetti risolti e installati, più pip da ensurepip | baseline-compile-r002.json, baseline-install.json |
| Vincoli della preparazione | Indice PyPI esplicito, hash obbligatori, `--no-build`, nessun backend sdist | baseline-requirements.in/.txt e log |
| Inventario/freeze/check | 18 distribuzioni incl. pip, file/hash, pip check exit 0 | baseline-environment.json, baseline-freeze.txt, baseline-pip-check.json |
| Cache tokenizer | 1.681.126 byte, hash atteso del pacchetto verificato; encoding cl100k_base caricabile senza fetch | tokenizer-acquisition.json, tokenizer-preparation-check.json |
| F1–F6 | Dieci input/asset più provenienza; hash/PNG/CRC/scanline controllati | tests/fixtures/run-a001-fase0-uv/, preparation-checks.json |
| Diagnostici | Cinque script stdlib visibili alla review, sintassi/help controllati | scripts/diagnostics/run-a001-fase0-uv/ |
| Gate unitari | Sette test PASS, mock/sintetici; nessuna prova R | diagnostic-unit-r003.json e log |

Le evidenze della tabella sono in
[evidence/implementation-r001](../../evidence/implementation-r001), salvo i
percorsi di repository espliciti. Le acquisizioni leggere hanno richiesto
esecuzione fuori dal sandbox per DNS; restano circoscritte alla venv/cache
identificate, senza installazione nel pyenv originale. I receipt distinguono
il primo tentativo fallito da quelli riusciti. Nessuna build o pacchetto ML.

`check_runner.py` verifica dump netlink e /proc, sole rotte locali o di rifiuto,
namespace diverso prima delle sonde IP, figlio equivalente, socketpair positivo,
daemon e socket sintetico negati, file/hash/ambienti/cache/Pandoc/temp.
`run_offline.py` richiede stage MATCH, crea solo il proprio socket di prova,
usa argv senza shell e `close_fds`, ripete R e preflight prima del comando,
registra namespace osservati e blocca il lavoro sul mismatch. I figli troppo
brevi per /proc richiedono le receipt del loro harness: non sono presunti osservati.

`make_source_manifest.py` produce S v1 canonico dopo snapshot o con scope
standalone; per la baseline distingue copia e clone host. S non è ancora
generato. Il produttore/preflight dovrà essere completato e provato per gli
input effettivi di build P2, con verify_distribution e check_python_origin.
`run_baseline.py` conserva output/diff/JSON integrali, esegue gli originali
diretti e la suite legacy esplicitamente nominata, richiede S corrente e R
del medesimo namespace prima di import/collection. È ancora da eseguire.

## Stato R/P0–P7, A1–A7 e V0–V9

| Voce | Stato corrente |
| --- | --- |
| R | S001 storica; s002 bootstrap unshare/Firejail exit2/IMPEDITA con receipt interne reali, R-baseline NON_ESEGUITA |
| P0/V0 | S002 preparata e provata solo per R bootstrap; S, baseline F1–F6/token/chunk/audit NON_ESEGUITI |
| P1–P3 | NON_ESEGUITI: nessun pyproject/lock/pin, modifica di prodotto o build |
| P4 | Sole fixture e sette test diagnostici preparatori; suite/CLI/confronti applicativi futuri |
| P5–P6 | NON_ESEGUITI: Marker/torch/Surya/native/Docker non acquisiti o collaudati |
| P7/V9 | Sola voce changelog relativa al lavoro reale; guide/help prodotto/README/AGENTS futuri |
| V1–V8 | NON_ESEGUITI; V7/V8 restano obbligatori |
| A1–A6 | Non soddisfatti dalla preparazione; nessun PASS trasferito alle prove future |
| A7/V12 | GO sul piano preesistente; review del codice e arbitrato finale futuri |
| V10/V11 | NON_ESEGUITI e non autorizzati; nessun peso/font/inferenza |

Restano da provare tutti i negativi cache/receipt D2/D3, equivalenza runner
D4, sentinelle D5/V5, catena S/B/I/E, distribuzioni/installazioni, inventari
base/dev/API, collection/conteggi/partizione, contenuti F1–F6 e asset/ordine/
formule/metadati/overlap, Docker e inventario documentale 74 fence/nove inline.
I 67 legacy/sei governance sono storia; non rieseguiti né attribuiti alla migrazione.
local_marker e server host non verificati.

## Prima consegna preparatoria — storia conservata

Primo compile: exit 2 per DNS nel sandbox, nessuna installazione. Ripetizione
autorizzata: exit 0. Il sandbox corrente nega socketpair; la prima versione
del test unitario ha mostrato due subtest ERROR per questa condizione. Il
successivo allestimento mock esercita i gate senza IPC reale, e un caso
negativo verifica il rifiuto di socketpair. Il controllo positivo reale resta
obbligatorio in R. Non è stato eseguito alcun namespace o connect ai daemon.

La richiesta contiene comandi concreti per R bootstrap, R baseline/S e V0,
path dei daemon, scope/costi e invalidazioni. Nessuna prova di questo stage
prima della risposta del supervisore con manifest/hash/worktree/prompt.
Se entrambi i runner sono negati/inadeguati, V0 resta IMPEDITA e torna al
supervisore; nessuna baseline fuori runner o modifica alla policy host.
Se F6 richiede tuning, s001 non diventa PASS finale: nuova label s002.

Consegna: [request.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
e [checkpoint proprio — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md), usando
[05-supervisor-stage-r001.md](../prompts/05-supervisor-stage-r001.md).
Processi propri attivi: nessuno. Nessun snapshot/registro comune/ADR/indice
modificato, né commit/merge/push/promozione/deploy. Changelog aggiornato solo
per il lavoro preparatorio effettivamente svolto.

Controllo finale della consegna: request schema 1 con 109 input/assenze e
58 artefatti esistenti, relativi e confinati alla run; comandi a argv espliciti,
nessun S/receipt futuro nel freeze. Hash/link/Git e perimetro ricontrollati in
[handoff-checks.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
fuori dagli input dello stage per evitare un ciclo col digest della richiesta.

## Ripresa baseline s001 — 2026-10-03

Continuazione reale con [prompt06](../prompts/06-implementation-r001-baseline-s001.md),
stesso ruolo e autore. Letti contesti aggiornati, brief, skill/protocollo,
checkpoint/report propri e risposta/reception/static-findings/transition/
checkpoint di supervisione. Piano/arbitrato già letti integralmente in questa
chat: identità riconfermata, D1–D5 riletti. Nessun subagente o review simulata.

Prima delle prove: Git atteso, indice vuoto, nuovo stage **MATCH**, 96 file e
58 artefatti. Manifest SHA-256
`ba686193b00dc068510167608a6ddb439e415210c5fb02e6cd14a6d4117a4c96`;
worktree `d2ca720b15c08bc1e65d27d488070d37d80c75a4de7971d3da08deb3bbefea3b`.
Request invariata SHA-256
`39eefbd38a2d2448210fef013d3fdb00e0cdf4abf8e1b6c62497c320568c1763`;
resume-commands SHA-256
`8414bb2fdd3877bff2dcc6ebd8b65534ebd9ff6b7d385bd6bc635171b39ea012`.
Ricalcolati sei hash d'identità, 109 input/assenze, 1.942 file distinti degli
inventari e 24 assenze complessive; directory esterne/cache conservate, profili
JSON identici alla request, sei metadata confrontati con transition.json.
Spazio disponibile prima R: 680.009.728 byte, superiore alla stima output 50 MB;
misura locale, nessuna acquisizione o inferenza di capacità V7/V8.

Sono stati eseguiti soltanto i due argv congelati **R-bootstrap**, prima unshare
e poi Firejail perché il primo era IMPEDITA. Stesso profilo di tool
`exec_command use_default`, nessuna escalation/perimetro diverso.

| Candidato | Exit wrapper / status | Osservazione | Gate interni |
| --- | --- | --- | --- |
| unshare | 2 / IMPEDITA | `PermissionError: [Errno 1] Operation not permitted` nella fase host | Nessuna invocazione registrata, inside.json e runner.json assenti |
| Firejail | 2 / IMPEDITA | Stesso errore nella fase host | Nessuna invocazione registrata, inside.json e runner.json assenti |

In ciascun tentativo il wrapper registra otto sonde AF_UNIX ai daemon esistenti,
tutte EPERM e zero connessioni riuscite. Nessuna richiesta operativa ai daemon.
La receipt non contiene traceback: la lettura del wrapper colloca il blocco
nella preparazione del socket sintetico host, senza distinguere da sola la
syscall socket()/bind(). Non è prova che i binari unshare/Firejail siano negati
o indisponibili sul Linux host: non sono stati avviati. Nessun namespace nuovo,
sonda IP, socketpair positivo, figlio R o gate di egress eseguito dal diagnostico.
I due host_netns osservati appartengono alle rispettive invocazioni del tool;
non costituiscono prova di equivalenza fra candidati.

Cleanup del proprio oggetto temporaneo registrato true in entrambe le receipt;
nessun cleanup globale o modifica ai socket/policy host. Stdout/stderr, argv,
CWD, tempi ed exit conservati. Lo stage è MATCH prima/dopo ciascun tentativo;
1.942 hash e 24 assenze identici anche fra tentativi e dopo Firejail. Nessun
processo proprio attivo. S, directory R-baseline/S/V0 e workspace v0-s001 assenti.

Evidenze:

- [Controlli/driver di ripresa — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md):
  input-check-before-unshare, execution-unshare, input-check-after-unshare-before-firejail,
  execution-firejail e input-check-after-firejail, con stdout/stderr integrali.
- [Receipt unshare — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
  e [receipt Firejail — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
- [Impedimento consegnato — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
- [Report precedente](report-r001-before-baseline-s001.md) e checkpoint precedente
  conservati prima dell'aggiornamento; handoff-checks della prima consegna è storico.

R resta **IMPEDITA**, S **NON_GENERATO**, V0 **NON_ESEGUITA**. Nessuna suite,
collection, conversione, import applicativo o fetch effettuato nella ripresa;
nessun PASS dei sette test preparatori trasferito. P1–P3 non iniziati perché
dipendono dalla baseline; nessun output di fedeltà da confrontare.

D4/prompt06 impongono il ritorno al supervisore per un perimetro diverso prima
dell'uso. Non è stata tentata esecuzione fuori sandbox, modifica ai diagnostici
congelati o attenuazione dei gate. Nessuna richiesta di riapprovazione uv.
Il supervisore deve identificare il contesto operativo compatibile e la
consegna di ripresa; eventuale retry ha nuovi output e label baseline s002,
non sovrascrive i tentativi s001. Non creare snapshot autonomamente.

Le prove e verifiche s001 si sono concluse prima della voce changelog di questa
consegna. Tale modifica rende storico il worktree s001 per **solo CHANGELOG**;
il controllo successivo deve registrare STALE spiegato, senza chiamare MATCH
il nuovo documento. Sorgenti/fixture/diagnostici/ambiente/cache/request e 58
artefatti congelati restano invariati. Nessun GO finale o integrazione Git.

## Preparazione baseline s002 nel contesto host — 2026-10-03

Ripresa reale con [prompt07](../prompts/07-implementation-r001-prepare-baseline-s002.md)
e [decisione D1/D4 del supervisore — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Letti contesti/metadata aggiornati, reception, identificazione host, argv
proposti e checkpoint del supervisore; proprie consegne/receipt s001 conservate.
Piano r003 e arbitrato invariati, nessuna r004 o nuovo GO. Le skill e i materiali
già letti rimangono applicabili; nessun subagente o ruolo aggiuntivo.

Ingresso `runner-recovery-context-r001` **MATCH**, schema 1, 96 file +96 artefatti.
Manifest SHA-256 `a899e95f0512a1728da016c423d6a34f03012f2e539632c72227128c1e074d83`;
worktree `55377e732ae49de12d6cdf050ff2875e519eb786ca342a527e77f76c43ccfdd9`.
Otto identità della response confrontate; 96 artefatti e 109 input s001
ricalcolati. Dei 1.942 file precedenti, i due output propri aggiornabili erano
gli unici diversi e coincidevano con le copie ricevute dal supervisore; 1.940
file tecnici/24 assenze invariati. Questi confronti non chiamano MATCH s001.

Eseguita una sola lettura del target nel profilo `exec_command require_escalated`,
con justification limitata a identità/policy/versioni/stat/hash degli input D4.
Review automatica mantenuta, nessuna prefix_rule; exit **0**, nessun rifiuto.
La receipt conserva il comando completo, stdout/exit/durata e parametri del tool.
Scritti solo i nuovi inventari di preparazione nel workspace; nessuna modifica
al target. Nessun socket/connect, namespace creato, runner o import applicativo.

| Inventario osservato | Esito della sola lettura |
| --- | --- |
| Host | UID/GID 1000; netns net:[4026531833], NoNewPrivs=0, Seccomp=0, CapEff=0 |
| Policy | userns/AppArmor presenti; valori letti, mai modificati |
| Binari/versioni | uv 0.10.10, Pandoc 3.1.3, git 2.43.0, unshare 2.39.3, Firejail 0.9.72; hash invariati |
| Interpreti | Bootstrap pyenv e venv baseline 3.12.3; executable/prefix/base/stdlib corrispondenti |
| Dipendenze | 18 distribuzioni baseline incl. pip, nomi/versioni/file invariati |
| File tecnici | 1.940 byte/hash invariati; nuovo stat/ownership host effettivo |
| Daemon | Stessi 11 path/alias/canonici; proprietà reali (es. Docker root:126); solo stat |
| Cache | Tokenizer 1.681.126 byte/hash atteso; 825 file uv cache con hash, bytecode escluso |
| Directory/assenze | Venv/tmp/workspace/cache conservati; 24 assenze confermate |
| Spazio | 671.911.936 byte liberi alla lettura host; output stimati 50 MB, nessuna acquisizione |

Nessuna compatibilità R dedotta dai bit di sicurezza o dal namespace osservato.
I nuovi target hanno gli stessi 1.380 path/byte/hash ciascuno e le stesse
dipendenze dei precedenti: stat/ownership e contesto ora provengono dalla lettura
host, anziché dal sandbox. Namespace storici esplicitamente non da riparare.
Cinque diagnostici, fixture, dieci originali/quattro metadata e algoritmi invariati.

[preparation-baseline-s002 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
contiene host-inventory, host-read-receipt, nuovi runner-target-bootstrap/
baseline, baseline-inputs e target-transition, oltre ai controlli preparatori.
Gli otto argv rimangono identici al JSON proposto dal supervisore, con target,
stage e output s002 e profilo require_escalated per ciascun passo. Sono dati,
`may_execute_now=false`; nessuno lanciato. Nessun futuro S/receipt/workspace V0
creato. Stime conservate: R/S circa due minuti, V0/audit circa 30 minuti;
compatibilità e tempi effettivi non provati, rete/acquisizioni previste zero.

Conservate prima degli aggiornamenti le versioni proprie
[report precedente](report-r001-before-baseline-s002.md) e
[checkpoint precedente — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
oltre alle copie ricevute dal supervisore. Request/manifest/receipt/output s001,
contesti di recupero e antecedenti non sovrascritti. Solo changelog del lavoro
reale aggiunto ai file tracciati: il contesto di recupero diventa storico per
quel documento dopo l'ingresso MATCH, senza rigenerarlo.

Consegna D1: [request s002 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[checkpoint proprio — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[prompt05 supervisore](../prompts/05-supervisor-stage-r001.md). Stato proprio
**WAITING_FOR_STAGE_SNAPSHOT**, processi propri nessuno. Freeze s002 ancora
mancante e riservato al supervisore; nessun auto-snapshot o polling.

R s002 **NON_ESEGUITA**, S **NON_GENERATO**, V0 **NON_ESEGUITA**; R s001 IMPEDITA
resta storia. Nessun nuovo test applicativo, download/install/build/sync o
P1–P3. Sette test precedenti restano preparatori; F6 runtime/audit ancora futuri.
Lettura autorizzata non autorizza le future sonde: ogni comando ha nuova review
automatica nel perimetro della decisione. V7/V8 future obbligatorie e pesanti
distinte, V10/V11 esclusi; review del codice/arbitrato finale mancanti, nessun
GO o integrazione Git/deploy.

La request s002 identifica **160 file/assenze e 109 artefatti** relativi alla
radice e confinati alla run. Include decisione/identificazione/argv e manifest
di recupero, antecedenti s001 e copie immutabili; esclude i file propri
aggiornabili e gli output futuri. Controlli della consegna in
[handoff-checks.json — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
fuori dagli input dello stage perché registra il digest esterno della request.


## Ripresa baseline s002 — R reale e nuova consegna IMPEDITA — 2026-10-03

Ripresa con [prompt08](../prompts/08-implementation-r001-baseline-s002.md),
[response s002 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
e decisione D1/D4 già ricevuta. Stesso implementatore Codex/OpenAI/famiglia
GPT-6, nessun subagente o ruolo diverso. Letti contesti e consegna aggiornati,
request/inventari/diagnostici/fixture pertinenti; piano integrale già letto in
questa chat, identità e D1–D5 riconfermati. Nessuna nuova review o GO.

Ingresso **MATCH s002**, schema1, 96 file +109 artefatti; manifest SHA-256
`c73d1547ad79ce19cd2b481b6f252ceb1cb21148a15c9fdf70768a664c0fc34f`, worktree
`7a20d1d70827e6bf152d1ee2e0b5ffe5e5daaebfa5f917e4e19da0dbda1869b2`.
Branch/HEAD/dev/base attesi, indice vuoto. Request SHA-256
`985cfb1c8c5b9430b23ac7b90f5d2ba3d41386cfc0707f2d06c69745ba07e19d` e
resume-commands `1a3a35f9d6c7c9a7f58024c5752d13ff8cd0a9dc8e1b2b6ebacd218e90769c6e`
esatti. Otto argv, CWD, timeout e prerequisiti del resume coincidono con request
congelata/proposta; flag preparatori storici non modificati.

Controlli prima, fra e dopo i due passi: dieci identità esplicite, 160 righe
request, 96 files/109 artifacts, inventari tecnici/cache/target nell'unione di
**2.874 file distinti e 24 assenze**, senza mismatch; directory esterne
conservate. Il conteggio esteso comprende anche files/artifacts dello snapshot:
non sostituisce i 2.819 input ricalcolati dal supervisore prima del freeze.
Stage MATCH in ogni controllo e nelle receipt wrapper prima/dopo. Nessun
interpreter/dipendenza/cache/fixture/diagnostico originale alterato.

Eseguiti **solo R-bootstrap unshare, poi R-bootstrap Firejail** dopo il gate
inadeguato del primo. Entrambi `exec_command require_escalated`, con la
justification del passo e senza prefix_rule. Ogni review automatica ha consentito
l'esecuzione; nessun rifiuto ricevuto o approvazione generale inferita. UID/GID
host1000 e netns `net:[4026531833]` riletti in entrambe le invocazioni. Applicazione
V0 mai avviata. Argv JSON come dati, shell=False, target/output congelati.

| Gate osservato | Unshare | Firejail |
| --- | --- | --- |
| Exit wrapper / inside / runner | 2 / IMPEDITA / IMPEDITA | 2 / IMPEDITA / IMPEDITA |
| Namespace padre e figlio, diverso host | net:[4026533872], stesso | net:[4026533878], stesso |
| IP: solo lo, nessuna rotta esterna, due sonde documentali negate | Positivo; lo DOWN | Positivo; lo UP, sole rotte loopback |
| Socketpair AF_UNIX positivo padre/figlio | true / true | true / true |
| Socket sintetico negato dentro | **false / false**, connesso | true / true, EACCES |
| Daemon su path negati dentro | **false / false**, sei path connessi | true / true, otto esistenti EACCES; tre assenti |
| Prerequisiti byte/hash/origine/cache/temp | 1.393 / 1.393 positivi | 1.393 / 1.393 positivi |
| TMPDIR uguale al target | true / true | **false / false**, assente |
| TIKTOKEN_CACHE_DIR, fd senza socket, Pandoc/git | Positivi | Positivi |
| Sintetico positivo host prima/dopo, cleanup | true / true / true | true / true / true |

Unshare funziona per il namespace IP, ma conserva il canale UNIX su path.
Padre e figlio raggiungono `/run/dbus/system_bus_socket`, `/run/docker.sock`,
`/run/snapd.socket`, `/run/user/1000/bus`, `/run/user/1000/keyring/ssh` e
`/var/run/docker.sock`, oltre al socket sintetico. Containerd è EACCES e tre
path assenti non provano il funzionamento di una blacklist. UID/GID0 dentro
la userns unshare è la mappatura richiesta, non un cambio di UID host.

Firejail avvia il runner effettivo e nega tutti i path esistenti, alias compresi,
più il socket sintetico. I tre path assenti restano inventariati come assenti.
Il gate ambiente fallisce per `TMPDIR: null` in **entrambi** i processi; il
valore richiesto è `/tmp/a001-uv-baseline-pi8cvs6x/tmp`. Le scritture sintetiche
esplicite nelle directory tmp/workspace risultano positive, ma non sostituiscono
la guardia della variabile. Il wrapper fornisce TMPDIR all'ambiente del launcher;
la receipt dimostra l'assenza all'interno senza attribuire una causa interna
non osservata. Non disabilitato il gate né alterato il wrapper congelato.
Stderr Firejail con warning di remount dei mount overlay Docker conservato
integralmente: nessuna operazione Docker o modifica a quei mount del processo host.

In ciascun wrapper otto path daemon sondeggiati fuori prima/dopo: sei connect
riuscite e due EACCES. Soltanto connect/close, **zero send/recv/API daemon**.
Dentro sono state eseguite le sole sonde R identificate; send/recv locale della
socketpair sintetica positivo. Sintetico raggiungibile fuori prima/dopo, poi
rimosso; cleanup verificato nelle receipt e path assenti. Nessun oggetto altrui
fermato e nessun processo proprio attivo: entrambe le sessioni tool terminate
exit2, Firejail registra la chiusura. PID/namespace del padre/figlio R provengono
dalle receipt dirette, non da osservazioni presunte di figli applicativi.

Spazio host prima unshare **195.943.182.336 byte**, prima Firejail
**195.941.347.328 byte**; le misure precedenti restano storiche. Nessuna nuova
acquisizione/installazione/build/sync, rete di acquisizione **0 byte**. Stima
output50MB e V0/audit30min conservata come stima, V0 non eseguita. Span delle
receipt bootstrap circa1,071s e1,119s, da timestamp iniziale a mtime finale:
misura wall-clock degli artefatti, non durata monotonic dell'intero driver o
della review. Wall_time/sessioni/chunk effettivi del tool nei file separati.

Evidenze in [resume-baseline-s002 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md):
controlli before-unshare/after-unshare-before-firejail/after-firejail, parametri
completi e risultati iniziali/completamenti dei tool, runner-analysis.json,
copie report/checkpoint precedenti preservate. Receipt e stdout/stderr originali:
[unshare — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[Firejail — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
[Impedimento r002 dello stage s002 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
identifica input/esiti/hash/limiti e consegna concreta al supervisore.

**R s002 IMPEDITA; R-baseline NON_ESEGUITA; S NON_GENERATO; V0 NON_ESEGUITA.**
Directory successive, sources.json, baseline-results e workspace/v0-s002
verificate assenti. Nessuna suite/collection/import applicativo/conversione,
nessun audit F1–F6 o tuning F6: token/multichunk/overlap rimangono futuri.
P1–P3 non iniziati. I sette test preparatori e i 67/6 storici restano distinti.
Nessuna equivalenza completa o PASS trasferibile fra candidati.

Consegna al supervisore via [prompt05](../prompts/05-supervisor-stage-r001.md):
valutare la propagazione esplicita del TMPDIR identificato nell'invocazione
Firejail, conservando tutti i gate, e disporre preparazione/request/freeze
**baseline-s003** prima di ogni retry. Non prodotta qui una request s003 o
correzione degli input congelati. Nessun fallback fuori runner, modifica policy
host, profilo persistente, privilegio nuovo, snapshot o registro comune.
Le prove e l'ultimo verify s002 MATCH precedono la voce changelog della
consegna, che rende storico il worktree s002 **solo per CHANGELOG**. I 109
artefatti e gli input tecnici rimangono invariati; il controllo finale conserva
la transizione spiegata senza chiamarla MATCH. V7/V8 obbligatori futuri, V10/V11
pesi/font/inferenza esclusi; doppia review codice/arbitrato finale mancanti.
