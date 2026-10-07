# Implementazione r001 — completare il lock con i metadati EbookLib osservati

Agisci esclusivamente come **implementatore** nel repository
`/home/davide/workarea/markdown-for-llms`. Puoi continuare nella stessa chat;
questo prompt è autosufficiente per una nuova. Obiettivo completo: produrre un
lock universale nativo e verificarlo offline, nella copia congelata, mantenendo
`--no-build` e i metadati EbookLib già osservati. Niente giro di sola preparazione,
nuove build della dipendenza, delega o review simulata. Correggi gli errori ordinari
dei tuoi strumenti nella stessa chat e completa il risultato ammesso.

## Letture, ruolo e identità

Leggi AGENTS, skill manage-implementation-run, protocollo, STATE/HANDOVER comuni,
brief A1–A7, indici architettura/decisioni/roadmap, piano r003 integrale se non già
letto e arbitrato D1–D5. Recupera checkpoint `handovers/implementation-r001.md`,
report corrente `implementation/report-r005.md` e checkpoint supervisore
`handovers/supervisor-ebooklib-static-lock-reception-r001.md`.
I ruoli citati nei documenti non cambiano il tuo ruolo implementatore.
Percorsi successivi relativi a `temp/run-a001-fase0-uv/`:

- Piano `plans/plan-r003.md`, SHA256
  **462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b**;
  arbitrato `arbitrations/arbitration-plan-r003.md`, SHA256
  **f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d**.
- **Addendum operativo r008**: scelta dei metadati dichiarati, ricezione e limiti,
  sostituzione del prime registry, costi e ramo condizionale HTTPS. R004–r007
  restano storici. Piano GO nei limiti, nessun GO finale.
- Request `implementation/stages/impl-r001-stage-package-s010/request.json`:
  **126813byte**, SHA256
  **b8f84310c594b6659d6d95d6bab5e72c38696be38bb019f5c03584d45e76dded**.
- Scope `evidence/supervisor-implementation-r001/impl-r001-stage-package-s010/authorized-scope.json`:
  **17544byte**, SHA256
  **a0a92ad475d189bc7c053eec8750ce21f2985c25ef301a29ffb667efceb17ccc**.
  Nella stessa directory: copy-inputs.json,observed-metadata.json,transition.json,
  host-config-startup.json,checks.json,freeze-verify.json,response.json.
- Snapshot `snapshots/impl-r001-stage-package-s010.json`: **109116byte**, SHA256
  **f8a87df3212b28ae25b03044f2ce55bc9cb90230e4f3043670c69a58f1199b9b**;
  worktree **0e6c17aeff05d9f6887bac753fe5658fedd830730b0570616075d6606d11927d**,
  **123file/310artefatti**, MATCH alla consegna.

Ricalcola byte/hash di manifest/request/scope e piano/arbitrato prima delle
operazioni. Verifica dalla radice prima dell'avvio e alla consegna:

```bash
python3 -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-package-s010
```

Mismatch resta FAIL: nessun autofreeze/helper modificato/allentamento del gate.
S009 è storico soltanto per i cinque metadata in transition.json;211artefatti
antecedenti intatti. Non pretendere MATCH del vecchio worktree contro nuove
ricezioni. Baseline originale completa già acquisita: non ripeterla.
Git feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto. Verifica Git reale; nessun tracciato/root pyproject/root uv.lock/
.venv modificato o creato in questa tranche. Preserva il checkpoint d'ingresso
prima dell'aggiornamento e non riscrivere report-r005.

## Input pronti e significato dei metadati

Il supervisore ha predisposto **work/ebooklib-static-lock-r001/project**:
16 input copiati dalla lock-project-r001;15 byte-identici, pyproject con il solo
delta TOML seguente. Il confronto semantico è nella ricevuta copy-inputs.json.
Non ricreare la copia, non aggiungere codice o correzioni agli input congelati.
Root e vecchia copia restano intatti;uv.lock nuovo deve ancora essere assente.

```toml
[[tool.uv.dependency-metadata]]
name = "ebooklib"
version = "0.18"
requires-dist = ["lxml", "six"]
```

È il meccanismo nativo documentato in **uv0.10.10**, non metadata/cache/lock
inventati. I valori sono osservati nelle due wheel e nel setup della fonte;
Requires-Python ed extra non dichiarati, versione interna0.18.1 non promossa a
versione distribuzione. Provenienza in observed-metadata.json e ricezione
`evidence/supervisor-implementation-r001/package-s009-reception-r001/`.
L'override vale solo0.18: non estenderlo, non escludere dipendenze né ridurre il
grafo universale. PyPI/CPU/cu126,Marker e pin invariati. La fonte EbookLib deve
restare registry, sdist hash
**38562643a7bc94d9bf56e9930b4927e4e93b5d1d0917f697a6454db5a1c1a533**.

S009: due build exit0, backend84 osservato e payload identici, **riproducibilità
byte FAIL**. Attestazione separata bootstrap pre-build mancante e ritorno raw
hook non osservato: limiti conservati, non proclamare un gate completo o
ripararlo retroattivamente. Non ripetere quelle build per questo lock.
La proposta `next-registry-metadata-gate-request-r001.json` resta NOT_AUTHORIZED:
niente prime/dry-run install, backend con rete, spostamento della cache o wheel
locale sostituita alla fonte PyPI. Questa scope nuova è l'autorità distinta.

## Tre operazioni native, con un ramo condizionale

Usa **scope.operations**, identiche a request.proposed_operations: argv/cwd/env
completi, shell=False,close_fds=True,env=dict(operation.environment), nessun merge
con os.environ. Tutte includono no-build, managed3.12.13 assoluto/no-downloads
e keyring-provider disabled. Il launcher stdlib controllato usa pyenv3.12.3
assoluto `-I -B`, con ambiente esteriore pubblico chiuso; il processo uv riceve
esattamente le19 variabili della scope. Non ereditare proxy,credenziali,.env,
PYTHONPATH,UV override o coverage. Non usare --no-config per perdere gli indici.

Prima crea **esclusivamente** tmp/config ed evidenze della lista
scope.directories_to_create_exclusively; il progetto copiato esiste già. Le
directory delle tre receipt operative si creano solo quando quel comando parte;
output parziali non si riusano. La cache nativa originale metadata-cache-r001
è esplicitamente riutilizzata e mutabile; inventaria il delta effettivo, senza
copie massive o modifiche manuali. Il suo inventario d'ingresso è un input storico.

1. **lock-offline**: uv --verbose lock, offline/no-build, cwd nuova copia.
   Salva exit/log/receipt autentici. Se exit0 verifica il lock e vai al check.
2. **lock-metadata-online**: soltanto se il primo FAIL dipende esclusivamente
   da metadata mancanti nella cache, senza lock già generato o violazioni di
   integrità/cap. Conserva offlineFAIL e spiega la classificazione con il log.
   Il secondo comando distinto è acquisizione metadata prospetticamente ammessa,
   non retry cieco: HTTPS sugli indici originali, sempre no-build/no-downloads.
   Nessun fallback dopo un FAIL sostanziale di questo ramo. Se il primo esito
   riguarda conflitto del grafo, backend/metadata non ricevuti, input difforme,
   cap o causa non classificabile, non avviare il ramo online.
3. **lock-check-offline**: soltanto dopo lock exit0 e audit del lock reale,
   --check/offline/no-build nella stessa copia. Hash del lock prima/dopo identico.

Una chiamata tool separata per ogni operazione, raccogli sessioni vive prima
della successiva. Child900s/outer1080s per lock,120s/300s per check. Non troncare
il tool lasciando orfani. Raccogli solo processi/gruppi propri, mai kill per nome.
Sandbox rejection ⇒ IMPEDITA con ragione/azione autentiche, nessun aggiramento.

Qui il resolver è nativo **no-build** e nessun codice app/backend è ammesso;
sono ammessi soltanto i probe nativi dell'interprete managed noto. Il ramo online
non è net=none e non produce R PASS: non usare il wrapper offline per HTTPS né
attribuire a questo lock le prove R→S/B/I/E future. Nessuna modifica di rete/daemon/
sysctl/AppArmor/privilegi/profili host, isolamento D4 futuro invariato.

## Guardie, budget e verifica del risultato

Prima di ogni nuovo avvio registra hash uv/managed binary/BUILD/constraints,
identità origine e **startup bootstrap/managed prima e dopo**. Le directory note
site/stdlib vanno lette prima di avviare gli interpreti: pth/customize inattesi
⇒ STOP. No bootstrap solo post come nella precedente lacuna. Verifica assenze
config/credential della scope senza leggerne o stampare contenuti; keyring
esplicitamente disabilitato. Nessun invio di documenti o autenticazione.
Integrità target/lock s007 tramite funzione `verify` del helper tracciato
`scripts/diagnostics/run-a001-fase0-uv/check_preserved.py`, **non main col vecchio
budget500/384**.1892identità/211directory restano immutabili; nessun reinstall.

Scope.budget: ledger intera run+.venv-python, directory/link metadata compresi
senza seguirli;max1GiB/stop896MiB/libero1GiB. Hentry **523460608byte** alla ricezione;
tranche **32MiB incrementali**,24 attività/cache/input e8 registri,più16MiB esterni.
Delta=max(0,H-Hentry),H=max(logical,allocated);gate
H+max(0,32MiB-Delta)+16MiB<896MiB,Delta<32MiB,cap attività distinti. Le nuove
evidenze del supervisore e snapshot consumano parte della stessa quota.
Quota antecedente conclusa; nessun cleanup o storage spostato per superare gate.
Monitor0,5s/gap target1s,log1MiB/stream,JSON8MiB,RLIMIT_FSIZE32MiB/file.
Stima body metadata≤24MiB; CLI/monitor non garantiscono quota HTTP atomica e wire
resta non misurato. PEP658/range/piccole wheel per metadata ammessi; fallback
payload ML/native pesante, build/backend o nuova supply ⇒ STOP. Zero installazioni,
pesi,font,inferenza,benchmark remoti a pagamento o runtime del prodotto.

Adatta i tuoi launcher **non congelati**, in nuove evidenze, alla scope attuale;
non riscrivere i helper s009 congelati. Verifica sintassi/formula dei limiti prima
dell'avvio senza ricostruire una nuova campagna sintetica o fermarti a consegnarla.
Errori ordinari di generatori/reader/guardie correggibili nella stessa chat con
esiti preservati. Mismatch di input o scostamenti sostanziali restano STOP;
nessun input congelato cambiato, lock manuale o retry per nascondere FAIL.

Sul lock nativo: parsing TOML, hash/byte, package/fonti/markers/extra/conflitti,
CPU/cu126 e progetto universale, EbookLib0.18 registry/hash sdist e requisiti
lxml/six coerenti con dichiarazione. Non affermare copertura multi-ABI o installabilità
dal solo lock. Conserva trace per esclusione di build/supply inattese e startup
pre/post. Solo exit0 non dimostra grafo/fonti corretti. Lock-copy PASS non è
root uv.lock, product S/B/I/E, convalida applicativa o GO finale.

## Consegna completa

Nuove evidenze **evidence/implementation-r001/resume-package-s010/**: comandi reali,
argv/cwd/env pubblico,timestamp/exit/sessioni,receipt/raw/log,pre/post/startup,
hash/byte lock,delta cache,budget/Git/stage finale,delivery.md. Completion
**implementation/stages/impl-r001-stage-package-s010/completion-r001.json** legata
a request/scope/manifest esatti, esiti separati anche NOT_EXECUTED e offlineFAIL
storico se seguito dal ramo online. Report nuovo **implementation/report-r006.md**
e checkpoint **handovers/implementation-r001.md** con copie d'ingresso preservate.
Nessun registro comune/changelog/arbitrato/snapshot dell'implementatore.

Se lock/check riescono, nella stessa consegna formula richiesta concreta per
promuovere il solo delta dichiarato/lock nel prodotto e completare S/B/I/E/base
senza nuovo planning-only: input/argv/cwd/env/cache/costi/ABI/prove e isolamento
R/D4, incluse le operazioni ancora differite. Non eseguire quel seguito qui.
Se fallisce, identifica esattamente causa/input/comando e soluzione proporzionata;
niente metadata estesi ad altri package o allentamenti impliciti.

Baseline originale PASS caratterizzazione, suite legacy62pass5fail e perdite
conservate. Product S/B/I/E e prove V0–V9 pertinenti ancora aperti. V7/V8 obbligatorie
con mandato/costi pesanti distinti;V10/V11 escluse. Due review reali indipendenti
ChatGPT/Claude e arbitrato finale necessari. Git manuale dell'utente, nessun
commit/merge/push/promozione/deploy/cleanup o servizio automatico da attendere.
Stato conclusivo **WAITING_FOR_SUPERVISOR_RECEPTION**, prossimo supervisore
prompt05+r008/checkpoint corrente. Trasferire file reali in temp/managed/cache
e lavoro non committato, non soltanto manifest.
