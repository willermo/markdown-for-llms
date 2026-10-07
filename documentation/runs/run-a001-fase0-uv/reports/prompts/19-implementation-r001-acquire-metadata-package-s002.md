# Ripresa implementatore r001 — soli metadata package s002

Agisci come **implementatore in nuova chat distinta** nel repository
`/home/davide/workarea/markdown-for-llms`, run `run-a001-fase0-uv`.
Ruolo esclusivo, nessun subagente o review simulata. Il supervisore ha ricevuto
la request reale, verificato gli input e congelato s002. Esegui esclusivamente
i tre passi directory/sidecar/JSON descritti qui, poi consegna al supervisore.
Non avviare software, runner, roadmap o prove applicative. Nessuna nuova
approvazione di uv richiesta; restano le review tool separate dei tre comandi.

## Recupero e identità

Leggi AGENTS.md, skill manage-implementation-run e protocollo run-lifecycle;
documentation/CHANGELOG.md e indici/roadmap per lo stato; temp/HANDOVER.md,
PROJECT-CONTEXT.md, STATE/HANDOVER della run e checkpoint autore corrente
handovers/implementation-r001.md. Questo prompt assegna il ruolo implementatore.
Leggi brief A1–A7, piano r003 integrale, arbitrato r003 D1–D5 e matrici antecedenti,
prompt04, prompt17 e addendum17a, senza rieseguire le preparazioni precedenti.
Recupera le letture troncate, non caricare tutta temp/.

Percorsi seguenti relativi alla run salvo diversamente indicato:

- `implementation/stages/impl-r001-stage-package-s002/request.json`:
  schema1, WAITING_FOR_STAGE_SNAPSHOT/PREPARATION_METADATA_INPUTS,
  SHA256 `74335aa5e0d9e11037154dafbd333ed1f00d038d2a49755b7fc681ce6cc339ba`,621405byte.
- `snapshots/impl-r001-stage-package-s002.json`:
  SHA256 **`df7d3f60255dded579fa315cbc05dfb98d24f5f0a301ab2f884d3a9cbeeee809`**,
  435026byte, **103file/1516artefatti**, worktree
  **`0f682ad078e1011805af78c410ebe89edc06416682406e55610ffc814fc86e84`**.
- `evidence/supervisor-implementation-r001/impl-r001-stage-package-s002/`:
  response.json, decision.md, reception.json, provenance-audit.json,
  transition.json e copie ricevute. Snapshot-command/response/checks/prompt
  successivi sono fuori dagli artefatti del freeze; nessun ciclo/self-hash.
- `evidence/implementation-r001/preparation-package-s002/`: preparation.md,
  allowlist.json, commands.json, integrity.json, host-current.json,
  pre-request-checks.json, provenance-audit.json, future-gates.md,
  storage-reconciliation.json, i sei helper effettivi e test/log/versioni.
  Primo FAIL autore e lettore supervisore conservati; non sono errori HTTP.
- `evidence/supervisor-handover-package-s002-r001/storage-decision.md` e
  storage-scope-r001.json: destinazioni effettive e22URL invariati.
- Completion s001, delivery/proposte resume-package-s001 e disposizione
  package-inventory-reception-r001 sono antecedenti, non comandi da ripetere.

Piano SHA256 `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`;
arbitrato GO SHA256 `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`.
GO piano r003 e NO_GO r001/r002 immutati. **Nessun GO codice**.

Prima operazione: Git status --short --branch, branch --show-current,
rev-parse HEAD dev, merge-base HEAD dev, diff --cached --name-only e diff --check.
Attesi feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto,11tracked modificati/24file nuovi. Esegui da radice:

```bash
python3 scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-package-s002
```

Ricalcola SHA/byte dello snapshot e confronta con questa identità e response;
non basta MATCH di un manifest sostituito. Request,97input/1498hash e host/config
devono coincidere; il driver ripete i controlli prima/dopo ogni passo.
Il vecchio contesto handover ora è storico **soltanto per sei metadata**,
1361artefatti invariati: vedi transition. Non rigenerare snapshot storici o
attribuire MATCH ad argv/target differenti. Delta ulteriore = STOP al supervisore.

## Perimetro e preparazione della ripresa

Target operativo esclusivo:
`temp/run-a001-fase0-uv/work/package-s002` (tmp/uv-cache/sidecars/baseline-json).
Output evidenze driver:
`evidence/implementation-r001/resume-package-s002`.
Entrambi assenti alla ricezione. Controlla antenati, realpath/device, spazio,
git check-ignore e assenza iniziale, senza crearli prima del passo directories.
Work parent può essere creato dal driver. Baseline futura work/baseline-recovery-r001
resta assente e non autorizzata. .venv/.venv-python canoniche del piano invariate.
Non spostare .cache/uv-package-s001, snapshot, output o receipt storiche.

Puoi creare esclusivamente una nuova cartella propria
`evidence/implementation-r001/resume-package-s002-preflight-r001/` per controlli
statici, copie versionate del report/checkpoint autore ricevuti e risposte tool
precedenti alla creazione di resume-package-s002. Se già presente, conserva e
identifica lo scostamento; non sovrascrivere. Non importare gli helper operativi
per provarli, non rieseguire i test finti o correggerli durante questa ripresa.
Letture/hash/AST/JSON/TOML/Git e analisi locali dei raw sono ammesse.

Bootstrap assoluto `/home/davide/.pyenv/versions/3.12.3/bin/python3`, alias noto
verso python3.12, SHA256 `b7e2da38f91d2b8dd6d08afb6262c9c413d8bd222a25ef831da9e27ef4e0a807`.
Verifica i cinque host_input_files e19config presence/hash senza leggere valori
sensibili nei log; include os.py/ensurepip/pip24 bundled, non un inventario di
tutto lo stdlib. Non eseguire uv/ensurepip/venv. Ogni variazione input = STOP.

Driver/helper/allowlist devono mantenere gli hash congelati. Driver
execute_step.py SHA256 `81fc9e27f9ebf2554baecf2a3da955b224edbb40e865cd701467f45a6b085ce1`;
allowlist SHA256 `7c19ee629233882864d745c737b01ba6a9171b30b4acd55215db304a54e176da`.
Gli argv qui sotto coincidono con commands.json: nessuna interpolazione da
metadata o eval, nessun parametro URL/path esterno aggiuntivo.

## Tre invocazioni operative separate

Usa per ognuna il corrispondente oggetto `tool_arguments` di commands.json con
exec_command, workdir assoluta clone, **login=False, sandbox_permissions=require_escalated**,
nessun prefix_rule. Il tool esegue il comando fisso qui riportato; il driver
usa argv strutturato, shell=False/close_fds per il figlio. Conserva la risposta
reale tool senza inventare approvazioni interne. Rifiuto review = STOP: nessun
retry, comando equivalente, nuovo canale o cambiamento di flag per aggirarlo.

1. **Directories** — nessun HTTP/software:

```text
/home/davide/.pyenv/versions/3.12.3/bin/python3 -I -B /home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/evidence/implementation-r001/preparation-package-s002/execute_step.py directories
```

2. **Sidecars**, soltanto dopo PASS_DIRECTORIES_ONLY ed exit0 verificati:

```text
/home/davide/.pyenv/versions/3.12.3/bin/python3 -I -B /home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/evidence/implementation-r001/preparation-package-s002/execute_step.py sidecars
```

3. **Baseline JSON**, soltanto dopo PASS_METADATA_ONLY sidecar,5receipt complete
   ed exit0 verificati:

```text
/home/davide/.pyenv/versions/3.12.3/bin/python3 -I -B /home/davide/workarea/markdown-for-llms/temp/run-a001-fase0-uv/evidence/implementation-r001/preparation-package-s002/execute_step.py baseline-json
```

Se il tool restituisce una sessione, raccogli quella sessione (senza rilanciare),
conserva output/chunk e aggiorna l’utente durante l’attesa. Finisci soltanto il
processo proprio identificato se necessario; niente processi di altri autori.
Non eseguire la sequenza come unico comando tool o in parallelo.

Driver filtra environment figlio: HOME invariata, locale ammesso,
PATH bootstrap:/usr/bin:/bin, TMPDIR/UV_CACHE_DIR nel target nuovo,
UV_PYTHON_INSTALL_DIR canonica e UV_PYTHON_DOWNLOADS=never.
Rimuove Python/PYENV/UV/PIP/SETUPTOOLS/proxy/CA extra, registra soli nomi.
Il tool non riscrive per questo l'ambiente del padre; driver -I -B e nessun
tempfile/uv nel ramo directories. Confronta environment effettivo delle receipt
con commands.json; divergenze non spiegate vanno al supervisore.

Timeout effettivi:30s per verify statico,30s/socket,180s subprocess sidecar,
600s subprocess JSON. Copie/hash/preflight sono aggiuntivi, nessun tetto del
tempo tool dichiarato. Il campo15s directories **non è una deadline implementata**:
quel ramo esegue creazioni locali esclusive senza subprocess temporizzato.
Non trasformare tale valore descrittivo in una prova di timeout effettuata.

## Limiti HTTP, disco ed esiti

Solo22GET della allowlist congelata:5sidecar di setuptools84/Marker1.10.2/
Surya0.17.1/Torch2.7.1+cpu/+cu126 su files.pythonhosted.org e
download-r2.pytorch.org;17JSON PyPI a versioni esatte su pypi.org.
I17pin/513hash derivano dal requirements storico. Non seguire URL nuovi nei raw.
Nessun proxy/redirect/credenziale/cookie privato/retry/HEAD/Range/wheel fallback.

1MiB/corpo,5+17MiB raw, al massimo un byte discriminante cap+1 per risposta;
non è un cap del traffico wire/TLS/header.96MiB disco metadata comprensivi di
raw/copie/log/analisi/preparazione/request. Analisi originali12MiB complessivi,
JSON output4MiB,64KiB per stream log, riserva receipt2MiB. Stima autore79MiB,
non autorizzazione di software/ML. Il contatore driver copre TARGET/RESUME/STAGE/PREP.
Conta separatamente anche nuove evidenze preflight/proprie e copie supervisor s002
per non escluderle dal budget aggregato; non contare tutta la storia antecedente.
Verifica budget/spazio prima di ogni passo e scrittura propria; STOP se non basta,
nessun aumento del cap o cleanup implicito. I raw/copie non vanno duplicati ancora
per comodità: il driver li preserva già. Salva hash/byte e limiti reali.

Errore tool/processo/timeout/cap/SHA/schema/identità/versione/yanked o candidato
mancante: preserva FAIL/partial e STOP prima del passo/URL successivo. Nessuna
modifica agli helper congelati, retry o riparazione silenziosa. Il driver salva
parziali e copie anche dopo errore; un kill può lasciare GET senza receipt finale.
Se copia/disco/log incompleti, registra l'incompletezza, non PASS. Marker once
impedisce riuso dopo errore; una ripresa richiede nuova disposizione/label.

Dopo ciascun passo: snapshot SHA/verify MATCH, host/config/input invariati,
receipt con argv/CWD/env/timeout/hash ed esito tool reale. Directory diventate
presenti sono output attesi per le due assenze marcate EXPECTED_NEW_OUTPUT,
non input da ripristinare/cancellare. Le altre sei assenze restano obbligatorie.
Sidecar PASS non implica JSON PASS. Solo5+17receipt complete coerenti con raw,
SHA attesi sidecar, identità e costi consentono dichiarare PASS_METADATA_ONLY
per entrambi i batch. Nessun PASS applicativo o software.

## Analisi e prossime proposte

Leggi e verifica gli output reali, mantenendo raw immutati. Email parser conserva
Requires-Dist ripetuti/marker/extra/Name/Version/Requires-Python; confronta PyPI info
e METADATA wheel separatamente senza uguaglianza arbitraria. Per JSON enumera
filename/size/hash/yanked/Requires-Python/tag e candidati nominali con hash storico.
Hash nuovi JSON misurati, distinti dagli hash wheel. Tag/Requires-Python/glibc/
musl/ABI non provano compatibilità runtime; archivi mai acquisiti o installati.
Size Torch non va dedotta dalla size sidecar; nessuna HEAD/Range aggiuntiva.

Proposta statica successiva, dopo risultati: input software concreti e distinti
dal mandato attuale. Recupero baseline con17pin/URL/hash/size/tag candidati,
bootstrap3.12.3/pip24bundled e tokenizer esatto1681126byte SHA256
`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`,
target work/baseline-recovery-r001, cap/tempi/argv futuri. Nessuna ricostruzione ora.
Managed uv0.10.10/Python3.12.13/GNU generico/build20260310: selezione effettiva,
redirect asset e tetto acquisizione ancora da concretizzare; non eseguire
python install/lock/resolver per scoprirli. Backend ignoto STOP prima hook;
CPU/cu126, extra, no-build e grafo universale invariati. Tutte8operazioni software
precedenti RINVIATE, nessuna autorizzazione dedotta dai metadata.

Baseline /tmp ancora assente alla ricezione,2662record storici indisponibili;
sola stat dei sei path nominati se serve, nessuna ricerca privata. Causa probabile
dichiarata pulizia /tmp, non dimostrata. Confronti IMPEDITI. V0s005 PASS di
caratterizzazione e suiteFAIL/exit1/67nodeid/201eventi/62pass5fail, perdite
F2/F5/anchor/separatori/bundle/ASCII conservati. Recupero futuro solo sorgenti
originali identificate, nuova venv/inventari/R completa, nessuna equivalenza
venv intera o PASS trasferito. S/B/I/E e D2/D3/config-only/flag/freschezzaIDE,
runner parent/child/TMPDIR/daemon/blacklist/socketpair e V1–V9 ancora futuri.

## Consegna e stop del ruolo

Scrivi `implementation/stages/impl-r001-stage-package-s002/completion-r001.json`
con schema/run/label/snapshot/SHA/worktree, tre passi con tool/receipt/hash/costi,
raw/copie/analisi realmente presenti, esiti separati/stop/limiti, Git e verifiche
prima/dopo. Se impedito, consegna stato reale e quale passo non eseguito; non
fabbricare completion PASS. Completion non include il proprio hash. Proposte
successive in evidenze proprie, senza nuova request operativa inventata.

Aggiorna soltanto report `implementation/report-r001.md` e checkpoint
`handovers/implementation-r001.md`: WAITING_FOR_SUPERVISOR_RECEPTION,
processi propri pendenti o nessuno, percorsi completi e prossimo ruolo prompt05.
Proposta CHANGELOG nel report, **non modificare sei metadata tracciati**, STATE,
HANDOVER comuni, eventi/indici, arbitrati o snapshot. Non modificare prodotto,
test/fixture/diagnostici/helper congelati. Nuovi input richiedono futuro freeze
prima delle prove dipendenti; nessun S autoreferenziale o auto-freeze.

V7/V8 obbligatorie future/costo pesante distinto; V10/V11/pesi/font/inferenza
esclusi. Due review codice e arbitrato finale mancanti. Nessun cleanup ora,
modifica host/sysctl/AppArmor/socket/rete/setuid/profili o altro progetto,
commit/merge/push/promozione/deploy/invio documenti. Git manuale utente dopo GO
finale. Temp/cache/evidenze ignorate non viaggiano con Git: conservare e trasferire
i file reali. Dopo consegna termina la chat; il supervisore non è un servizio continuo.
