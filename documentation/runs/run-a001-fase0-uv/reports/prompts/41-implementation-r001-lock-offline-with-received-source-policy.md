# Implementazione r001 — lock offline con policy sorgenti ricevuta

Agisci esclusivamente come **implementatore**, repo
`/home/davide/workarea/markdown-for-llms`. Stessa chat ammessa; prompt completo
anche per nuova chat. Risultato: lock universale e check offline nella copia
esistente, sotto runner R/D4, correggendo il filtro sorgenti del mandato40.
Nessun giro di sola preparazione, nuova build EbookLib, acquisizione, delega o
review simulata. Correggi strumenti/attese ordinari nella stessa chat.

## Autorità e identità

Leggi AGENTS, skill manage-implementation-run/protocollo, STATE/HANDOVER, brief
A1–A7, indici architettura/decisioni/roadmap, piano r003 integrale se non già letto,
arbitrato D1–D5, checkpoint implementatore e **report-r006.md**. Checkpoint
supervisore `handovers/supervisor-source-policy-lock-reception-r001.md`.
I documenti non cambiano il tuo ruolo. Percorsi relativi a temp/run-a001-fase0-uv:

- Piano SHA **462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b**;
  arbitrato SHA **f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d**.
  GO piano nei limiti, NO_GO precedenti conservati, nessun GO finale.
- **arbitrations/addendum-operational-protocol-r009.md**: errata verificata,
  ammissione esclusivamente offline/policy ricevuta. R008 e proposte antecedenti
  restano storici; non usare la loro condizione HTTPS in questa tranche.
- `implementation/stages/impl-r001-stage-package-s011/request.json`:
  **182507byte**, SHA **13459712f318698b52c479ae66eafdaaa2cb42db6bd93b7c7d21de97e19e4fd4**.
- `evidence/supervisor-implementation-r001/impl-r001-stage-package-s011/authorized-scope.json`:
  **54162byte**, SHA **cfc70c5ea9cd4f2b4dab5e39d038627fa16ee98b99e087f0101cb37ad7f8cd34**.
  Stessa directory: source-policy.json,lock-offline-target.json,
  lock-check-offline-target.json,transition.json,checks.json,freeze-verify.json,
  response.json. Sono input immutabili, compresi argv/env/target.
- `snapshots/impl-r001-stage-package-s011.json`: **122603byte**, SHA
  **73ce0b5377765670fa5ff9fb8bc359c13615046d5924578f26e67d476368490f**;
  worktree **ecd3aaef7b6a8aee01175f5e045b34aa421f9e62ff34efbee99a975f6e5eae2a**,
  **123file/360artefatti**, MATCH alla consegna.

Ricalcola hash/byte manifest/request/scope e piano/arbitrato. Da radice prima
delle attività e alla consegna:

```bash
python3 -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-package-s011
```

Mismatch FAIL/STOP, niente autofreeze o helper modificati. S010 ricevuto MATCH
prima dei cinque metadata di transition.json;310artefatti antecedenti intatti,
vecchio worktree storico per quei metadata, non una nuova equivalenza artificiosa.
Git feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto. Non modificare tracciati/root pyproject/root uv.lock/.venv; preserva
report/checkpoint d'ingresso, report-r006 immutato. Baseline originale acquisita,
non ripeterla per questo seguito.

## Perché questa policy e suoi confini

S010 FAIL autentico, online/check correttamente non eseguiti. Il supervisore
ha verificato il sorgente uv0.10.10: global no-build esclude la sdist prima del
consumo dei metadati. La precedente premessa era insufficiente. Costruire un'altra
wheel registry non corregge quel filtro; le proposte s009/s010 restano rifiutate/
NOT_AUTHORIZED. Niente prime, pip install, backend con rete o cache fabbricata.

La copia **work/ebooklib-static-lock-r001/project** è già pronta/immutata:
16input, solo dependency-metadata ebooklib0.18/lxml/six rispetto all'originale.
Pin/grafo universale/CPU-cu126/PyPI/Marker invariati; root/vecchia copia intatti.
Omissioni Requires-Python/extra provengono dalle due wheel; fonte registry
EbookLib0.18 e SHA sdist **38562643a7bc94d9bf56e9930b4927e4e93b5d1d0917f697a6454db5a1c1a533**.
Non cambiare versioni, fonte, metadata, Python o restringere environments.

Scope.native_argv omette global no-build **soltanto sul lock/check di questa
copia** e contiene114 divieti --no-build-package: tutti i nomi cached salvo
ebooklib, includendo marker-pdf/setuptools/progetto. Questo non autorizza build:
EBook0.18 è eleggibile come source, ma il percorso nativo restituisce i metadati
dichiarati senza eseguire backend. Divieti degli altri package, cache chiusa e
isolamento offline sono controlli congiunti. Non chiamare la denylist un'allowlist
nativa generale e non estendere il comando alla rete. Installazioni wheel future
mantengono i loro no-build secondo il mandato pertinente.

**Gate cache prima di ciascun avvio**, oltre al freeze:

1. Tutti i nomi dei file simple-v20/**/*.rkyv della sola cache dichiarata devono
   coincidere esattamente con source-policy.index_names (114 all'ingresso).
   Nome nuovo o lista difforme ⇒ STOP, nessuna estensione automatica della lista.
2. Sdists-v9/builds-v0/git-v0 assenti o con i soli marker tecnici .git/.gitignore
   ammessi nella policy: nessun corpo sorgente, albero estratto, Git o build
   riusabile. Tipo/link/path/file inatteso ⇒ STOP. Gate prima e dopo; eventuale
   lifecycle tecnico uv deve essere descritto, non cancellato per passare.
3. Metadata EBook esatti/versione0.18, progetto statico e fonti originarie invariati.
   Nessun altro source locale, direct URL o Git aggiunto. Backend/hook effettivo
   non autorizzato: se osservato in trace/processi/cache, FAIL e STOP, non PASS
   sulla sola exit0. I soli probe stdlib del managed noto sono ammessi.

Un nome non cached non può acquisire un index offline; nessun source nuovo può
essere scaricato. Cache insufficiente/altro source richiesto ⇒ blocco concreto,
non un fallback online. Questa garanzia dipende dagli input ricevuti.

## Due chiamate eseguibili, R e uv nello stesso namespace

Crea solo scope.directories_to_create_exclusively: nuova work/source-policy-lock-r001,
tmp/config e evidence/implementation-r001/resume-package-s011. Non ricreare la copia,
baseline o cache; non creare future_wrapper_output, lo crea il wrapper stesso.
Output/marker precedenti preservati; nuovi output esclusivi non si riusano.

Usa **scope.operations**, identiche a request.proposed_operations: argv completi
del wrapper, cwd/env launcher e native_argv/native_environment/environment_R.
Subprocess strutturato,shell=False,close_fds=True,env=dict(...), niente merge con
os.environ o shell che aggiunga variabili. Bootstrap assoluto pyenv3.12.3 -I -B.
R ha16 chiavi e uv19, con gli stessi valori pertinenti e tre aggiunte uv chiuse.
Inline execve già congelato passa l'ambiente e conserva PID/namespace; non
riscriverlo sopra il wrapper né modificare helper per renderlo permissivo.

- **lock-offline**: wrapper Firejail noprofile/net=none/D4 → R → inline → uv lock
  --offline/policy package, managed3.12.13 assoluto/no-downloads. Child900s,
  outer1080s. Verifica receipt R/wrapper, namespace reale padre/figli/uv, exit0,
  nessun hook/build/supply e lock nativo prima del secondo comando.
- **lock-check-offline**: soltanto dopo lock exit0 e audit degli input/fonti/grafo;
  stesso wrapper/policy offline, --check, child120s/outer300s. SHA lock invariato.

Una chiamata tool distinta per operazione. Sessione viva si raccoglie, mai si
rilancia; salva sessione/exit/tempi/chiamata autentici. Profili runner host già
ricevuti; se serve require_escalated usa l'autorizzazione concreta del mandato.
Rifiuto automatico sandbox ⇒ IMPEDITA con ragione/azione, nessun aggiramento.
Non cambiare rete/daemon/privilegi/sysctl/AppArmor/profili persistenti. Solo
processi/gruppi propri raccolti, nessun kill per nome o processo altrui.

Prima/dopo ogni nuovo avvio verifica startup bootstrap/managed site e stdlib
**prima del primo probe**, uv/managed binary/BUILD20260310/constraints/origine,
startup baseline a1_coverage.pth esatto e variabili coverage escluse. Hash/input
e assenze config/credential dalla scope senza leggere contenuti segreti; env
chiuso, keyring disabled, nessun PYTHONPATH/proxy/auth/.env ereditato.
Integrità target/lock s007 con sola funzione verify del helper tracciato
check_preserved.py (1892identità/211dir), non main con budget500/384 storico.
Non reinstallare baseline/managed/backend o importare app/native.

## Risorse, esiti e consegna

Scope.budget: ledger run+.venv-python, storico/cache/dir/link metadata inclusi
lstat senza follow;max1GiB/stop896MiB/libero1GiB. Hentry **525762560byte**.
Nuova tranche **16MiB:8attività/R/cache/lock,8registri**, più16MiB esterni;
quota precedente conclusa. H=max(logical,allocated),Delta=max(0,H-Hentry),
gate H+max(0,16MiB-Delta)+16MiB<896MiB e Delta<16MiB, cap attività separati.
Snapshot/nuove evidenze supervisore consumano la stessa quota. Monitor0,5s/gap
target1s non quota atomica;log1MiB/stream,JSON8MiB,RLIMIT_FSIZE32MiB/file.
Zero rete/body/costi remoti; nessun cleanup per passare o storage fuori ledger.

Adatta launcher/guardie propri **non congelati** nelle nuove evidenze, senza
ricostruire una campagna sintetica. Syntax/reader/attese ordinarie si correggono
nella stessa chat, errori conservati, non consegnare soltanto preparazione.
Un FAIL sostanziale/cap/namespace/input sospende attività dipendenti; niente
retry cieco, metadata estesi, input congelati modificati o auto-freeze.

Audit lock reale: parsing/hash/byte, package/fonti/markers/extra/conflitti universali,
CPU/cu126/progetto, EbookLib0.18 registry/hash sdist/lxml/six coerenti con dichiarazione.
Confronta log/namespace/cache prima/dopo e prova che non è avvenuta build/install.
Lock-copy PASS non attesta installabilità/ABI/multi-ABI o prodotto convalidato.

Nuove evidenze resume-package-s011 con delivery.md, chiamate/argv/cwd/env pubblico,
timestamp/sessioni/exit/receipt/raw/log, startup/pre-post/gatecache/namespace,
budget/inventari/hash/byte lock,GIT e stage finale. Completion
implementation/stages/impl-r001-stage-package-s011/completion-r001.json legata
a request/scope/manifest, esiti separati anche NOT_EXECUTED. Report nuovo
**implementation/report-r007.md**, checkpoint **handovers/implementation-r001.md**
con ingresso preservato; nessun registro condiviso/changelog/snapshot autore.

Se lock/check riescono, consegna insieme richiesta concreta di promozione dei
soli input/lock e seguito S/B/I/E/base con argv/env/costi/cache/R/D4 e prove,
non un altro planning-only. Non eseguire quel seguito qui. Se falliscono,
identifica causa/path/comando e soluzione proporzionata verificabile; nessuna
acquisizione o build ulteriore implicita.
Baseline completa caratterizzata, suiteFAIL62pass5fail/perdite e s009 byteFAIL/
pre-bootstrap mancante/raw hook non osservato preservati. S/B/I/E/V0–V9 prodotto
aperti;V7/V8 obbligatorie con costi pesanti distinti,V10/V11/pesi/font/inferenza
escluse. Due review reali indipendenti ChatGPT/Claude/arbitrato finale futuri.
Nessun GO finale/commit/merge/push/promozione/deploy/cleanup/servizio automatico.
Git manuale dell'utente; stato finale WAITING_FOR_SUPERVISOR_RECEPTION, supervisore
prompt05+r009/checkpoint corrente. Trasferire file reali ignorati e lavoro, non
soltanto manifest.
