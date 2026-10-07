# Review — implementazione — run-a001-fase0-uv — r003 — Claude

- Autore/provider/modello e chat: **Anthropic, Claude Opus 5.5**
  (`claude-opus-5-5[1m]`), Claude Code nell'estensione VSCode. Nuova chat
  dedicata solo a questa review; ID chat non esposto. Nessun subagente, workflow
  o delega.
- Prompt di origine: [prompt61](../prompts/61-review-implementation-r003-claude.md)
  e [criteri comuni r003](../prompts/review-implementation-r003-common.md).
- Oggetto: fix di prompt59 (report-r020), cinque path dell'autore, chiusure
  dell'arbitrato r002, regressioni di launcher e cleanup, guide e contabilità.
- Snapshot: `impl-r001-stage-final-s003`, manifest SHA-256
  `6295835b1407ea44f6c49780d38061c0fd5d6389a2ef1e096da9dd260f778962`
  (1204624 byte), worktree `ee433d5b…a2af2`, 133 file e 4245 artefatti, uguali a
  [freeze-identity — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
  `run_context.py verify` con il managed 3.12.13: **MATCH prima** (14:04:19) e
  **MATCH dopo** (vedi `verify-before.txt` e `verify-after.txt`).
  Branch `feature/run-a001-uv`, HEAD/dev `66ba822…`, indice vuoto.
- Indipendenza: **non ho letto la review r003 di ChatGPT**, né il suo
  checkpoint o le sue evidenze. Ho letto come storia comune le due review r002,
  l'arbitrato r002 e R021/R022.
- Giro: **fix mirato (terzo)**. Base `66ba822` più worktree autorizzato. Delta
  final-s002→final-s003: i cinque path dell'autore e cinque file `documentation/`
  del supervisore ([delta — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)).
  Gli artefatti di final-s002 sono tutti invariati; ne sono stati aggiunti 195.
- Esito: **GO**, con CLA-I009 non bloccante, da registrare nell'arbitrato.

## Ambito e prove

### Letture

- **Governance:** AGENTS, template review, HANDOVER/STATE, arbitrato
  implementazione r002, entrambe le review r002 e il mio checkpoint r002,
  prompt59, R021, R022, report-r020, la copia del checkpoint ricevuto, la
  reception, i delta s002→ricevuto, il delta documentale del supervisore,
  freeze-identity e final-checks della ricezione.
- **Codice, integrale:**
  - `run_offline.py`, confrontato con `diff -u` con la versione precedente
    `2946aa16…`, conservata nei cloni r002;
  - `tests/unit/test_core_diagnostics.py`, confrontato con `diff -u` con
    `9edb4450…`;
  - `snapshot_check` del produttore S (`make_source_manifest.py`);
  - `run_context.py`: `RUN_ID`/`LABEL`, `repository_root`, `verify`;
  - CLI di `verify_distribution.py`, `check_preflight.py` e
    `make_input_inventory.py`;
  - `tests/docker/test_marker_contract.py` e la selezione della modalità nel
    conftest.
- **Guide:** le sezioni modificate di `ambiente-uv.md`,
  `marker-legacy-docker.md` e `toolchain-legacy.md`. Del delta documentale del
  supervisore ho letto la voce del CHANGELOG e le righe di stato.
- **Prove ricevute in `fix-review-r003/`:**
  - `entry`, `admission`, `execute.py`, le 16 voci di `calls.json` (incluso
    l'output unittest completo) ed `execution-result`;
  - R-a001/R-b002: S, I, target effettivo, preflight, invocation, receipt
    outer/inside/runner, stdout/stderr, process-collection, result e monitor;
  - clone b002: copie, `.git` e snapshot;
  - `static_and_reuse.py`, static e reuse, `finalize.py`, matrix,
    final-checks, delivery, manifest e manifest-closing;
  - `close_cost.py`, finalize-failure e closing-accounting.

### Sonde proprie

Tutte stdlib, in sola lettura, con il managed `-I -B` e `timeout` finito.
Output in `evidence/review-implementation-r003-claude/`.

| Sonda | Esito | Wall |
| --- | --- | --- |
| `reuse_and_timing_check.py` | Contesto Docker r007 (24 oggetti), C-docker (5 file) e B56 uguali al worktree corrente. Nessuno dei cinque path del delta è un input consumato da build o contesto. Campi build di S-a001 e S-b002 uguali a S60. Ricostruisce il cumulativo per mtime (vedi CLA-I009) | 0,08 s |
| `doc_links_check.py` | 59 link locali validi nelle tre guide e nei cinque file del supervisore. Nessuno spazio finale o tab, newline finale presente | 0,07 s |
| `space_ledger_check.py` | Radici ed esclusioni R021. La misura è approssimata e **meno conservativa** del ledger ufficiale (H 868,6 MB contro 913,9 MB), perché tratta i link diversamente: non la uso come sostituto | 0,26 s |

Altre letture: `diff -u`, `sha256sum`, `stat` e `find` sugli mtime (repository,
`.git` e prefix `/tmp`), `git diff --check` (exit 0) e `git rev-parse`.

**Costo:** circa 0,4 s di sonde, entro i 30 s di R022. I `verify` di identità
(circa 1,2 s ciascuno) sono il controllo richiesto e li conto a parte. Output
propri: circa 12 KB di evidenze, più report e checkpoint, sotto 1 MiB. Nessuna
rete, Docker, build, installer, suite, R o ML.

### Non eseguito, con motivo

- **Nessun R, suite o C-fast:** vietati da R022. Le due R reali ricevute sono
  leggibili e legate agli input.
- **I 12 test non sono stati rieseguiti:** ho letto l'output completo in
  `calls.json` e il codice. Gli mtime mostrano i file scritti prima
  dell'esecuzione e non più modificati.
- **Nessun `run_context verify` nel clone b002:** git nel clone potrebbe
  aggiornarne l'index, che fa parte delle evidenze congelate. Ho confrontato
  snapshot e copie per hash.
- **Nessuna dimostrazione del limite in Backlog 1:** è un'osservazione dalla
  sola lettura del codice.

## Chiusure

| ID | Esito | Motivo e prova | Blocco |
| --- | --- | --- | --- |
| **GPT-I002 / residuo CLA-I001** | **Risolto** | Vedi nota GPT-I002 | no |
| **GPT-I001**, regressione | **Risolto, nessuna regressione** | Vedi nota GPT-I001 | no |
| **GPT-I003 / CLA-I007 / residuo CLA-I004** | **Risolto** | Vedi nota GPT-I003 | no |
| **CLA-I008** | **Risolto, alternativa documentale** | Vedi nota CLA-I008 | no |
| **CLA-I006** | **Risolto nei registri** | Vedi nota CLA-I006 | no |
| Nuova chiusura r020 | **Quota prospettica NON PASS confermata.** Nessun workload prodotto oltre il cap | Vedi CLA-I009 | no (deviazione già dichiarata) |
| CLA-I002, CLA-I003, CLA-I005 | Risolti in r002, non toccati | Contesto Docker, C-docker e README/B56 sono byte-identici (sonda) | no |

### GPT-I002 / residuo CLA-I001 — launcher generalizzato

**Codice.** La diff riguarda solo `snapshot_check`, più `import re` e la
docstring. I controlli avvengono in quest'ordine, prima del verificatore:

1. repo e snapshot assoluti, senza `..`, e nessun symlink sul file, sugli
   antenati e sulla radice;
2. dizionario con le otto chiavi del produttore S;
3. `type(schema) is int` e `== 1`, che chiude il caso `True`, accettato dal
   vecchio `data.get("schema") != 1`;
4. run_id e label con le regex identiche a `run_context.py` e al produttore S;
5. path esatto `repo/temp/<run_id>/snapshots/<label>.json`.

Poi chiama il `run_context.py verify` reale del repo dichiarato, con l'interprete
target. Il ramo bundle, `binding_before`/`binding_after`, l'ordine in
`outside()`, D4, `inside()` e il cleanup sono byte-invariati. Non c'è fallback,
né accettazione di JSON arbitrario.

**Test (mock, 12 PASS ricevuti):**

- `LauncherSnapshots` è una classe nuova e additiva:
  - positivi a001/b002 con l'argv esatto del verificatore;
  - schema `True`/`2`/`1.0`, run_id `../escape`/`run-x`, label
    `../escape`/`/absolute`/`UPPER` e path difforme: tutti fermati prima del
    verificatore;
  - symlink sul file e sull'antenato;
  - STALE con receipt FAIL, senza `mkdtemp` (socket) e senza `invocation`.
- I test esistenti sono invariati.

**R reale (letto e ricontrollato):**

- Wrapper `ea1d85b3…`, uguale al file corrente, runner Firejail
  `--noprofile --net=none` e blacklist dei daemon e del socket sintetico.
- Netns: host `4026531833`, interno `4026533887`. Padre, figlio e comando
  sono nello stesso netns interno.
- `realpath` del padre e del figlio: managed 3.12.13. Prefix
  `/tmp/run-a001-product-wheel-env-r001`, non modificato dopo prompt59.
- Preflight interno `PASS_CURRENT_I`, comando sintetico exit 0.
- `binding_before`/`binding_after` MATCH, con lo stesso SHA dello snapshot:
  a001 `a351ef30…`, b002 `cbee4243…`.
- `inputs_unchanged`, socket raccolto, nessun superstite.
- Diagnostico `check_runner.py` e target effettivi: gli SHA corrispondono.
- S-b002 è run-stage e legato allo snapshot del clone. I-b002 deriva dal
  prefix B56.

**Clone b002:**

- Le 40 copie (moduli, input di build, `scripts/diagnostics/*.py`,
  `run_context.py`, `.gitignore`) sono byte-identiche al repository corrente.
- `.git` del clone: 0 oggetti propri, `alternates` verso
  `.git/objects` principale (sola lettura per Git) e due ref al base `66ba822`.

**Repository:** nessuna scrittura del clone. Le ref di branch sono invariate.
In `.git` principale compaiono oggetti dopo l'ingresso, ma provengono da
`refs/codex/turn-diffs/checkpoints|captures`, il checkpoint per turno
dell'harness Codex. È un comportamento preesistente (38 ref dal 1790950780)
e non è attribuibile ai comandi della run.

**Attribuzione:** s076 differisce da final-s003 solo per i cinque file
`documentation/` del supervisore, che non sono applicativi. La prova a001 vale
quindi per un codice identico all'oggetto.

### GPT-I001 — cleanup, regressione

- Il blocco `finally` e la gerarchia `error`/`cleanup_error` sono identici alla
  versione `2946aa16…`.
- I tre negativi (EACCES, timeout, cleanup EACCES) sono stati ripetuti dopo
  l'edit: PASS.
- Lo STALE è validato prima del socket e del workload.
- Le due R reali hanno `temporary_socket_cleaned:true`.
- Nessuna guardia rimossa, nessun falso PASS.

### GPT-I003 / CLA-I007 / residuo CLA-I004 — guide

- `ambiente-uv.md:132-133`: I si crea con `verify_distribution.py` e deriva lo
  scope da S, senza flag. L'help reale non ha `--standalone`.
- `toolchain-legacy.md:109` cita `RUN_IMAGE_CONTEXT_RECEIPT`, come richiesto da
  `test_marker_contract.py:8`.
- Le opzioni citate di `make_input_inventory`, `make_source_manifest` e
  `check_preflight` coincidono con argparse.
- Link e whitespace validi.
- Le 74+9 disposizioni non sono state riaperte.

### CLA-I008 — selezione C-docker

- Il test passa `--mode` da `RUN_TEST_MODE` (default `official`).
- Il conftest rifiuta valori invalidi e scelte discordanti. Il solo
  `--test-mode standalone` porta a un rifiuto del driver, che è fail-closed.
- Le tre guide lo dicono esplicitamente (`ambiente-uv.md:147-150`,
  `toolchain-legacy.md:117-119`, `marker-legacy-docker.md:91-94`), senza
  promettere equivalenza del flag.
- Test e driver Docker invariati (SHA esatti).

### CLA-I006 — ledger storico

- Report-r020 e i registri riportano un minimo di 8321,418589 s e uno
  sforamento minimo di 41,418589 s, con limite superiore ignoto e NON PASS.
- Report-r019, final-checks e delivery r002 e il manifest final-s002 hanno gli
  stessi hash di final-s002.

## Rilievi

### CLA-I009 — chiusura temporale r020: addebito non superiore e FAIL ricostruito

| Campo | Contenuto |
| --- | --- |
| Severità e blocco | **Bassa — blocco: no** |
| Posizione | `evidence/implementation-r001/fix-review-r003/closing-accounting.json` (`prospective_charged_upper`), `finalize-failure.json`, `close_cost.py:7-12`, `finalize.py:12,21,25`; report-r020 (ultimo paragrafo); R022 e la ricezione che li recepiscono |
| Raggiungibilità e input | Ledger temporale della consegna r020, letto da supervisore e arbitrato |
| Attribuzione e requisito | Run, consegna dell'implementatore. R021: ogni riscrittura finale ha il comando conservato e una misura comprensiva, senza dati esatti fittizi. Criteri r003: cercare attività ulteriori o escluse |

**Problema.** La formula è corretta: 8380 + 60 + Δmonotonic = 8946,997655 s,
coerente anche col wall. Tre aspetti del registro sono però più favorevoli dei
fatti osservabili.

1. **Il cap è stato superato prima di `finalize.py`.** Il cumulativo arriva a
   8880 al wall 1791373781,53. `finalize.py` ha mtime corrispondente a
   8902,32 s ed è stato eseguito a circa 8903 s. Il gate `assert charged<8880`
   è l'ultima istruzione e segue:
   - due `run_context verify` reali;
   - il ledger dello spazio;
   - le scritture di report, checkpoint, matrix, final-checks, delivery e
     manifest.

   Il gate non ha «intercettato» nulla, contrariamente a quanto dice il
   report-r020: ha solo impedito un `closing-accounting` PASS. I comandi oltre
   il cap sono verifiche di governance già registrate in final-checks, non
   workload prodotto.
2. **`finalize-failure.json` è una ricostruzione.** Lo ha scritto
   `close_cost.py` (riga 7) circa 44 s dopo. Exit, testo dell'errore e valore
   osservato del run fallito non sono conservati come output originale.
   L'mtime del manifest (8903,36 s) conferma comunque che il gate doveva
   fallire.
3. **L'addebito di 8947,997655 s non è un limite superiore,** benché il campo
   si chiami `prospective_charged_upper`. Dopo la misura restano almeno la
   stampa finale e la risposta in chat. Il checkpoint di fine turno dell'harness
   Codex (`refs/codex/turn-diffs/checkpoints/ef7680e5…/1791373867769`) cade
   19,24 s dopo `closing-accounting`, cioè a un cumulativo di **8966,24 s**.
   Il collegamento al turno dell'implementatore è un'**inferenza**: è lo stesso
   scarto di circa 20 s osservato dopo l'ultima scrittura della review r002
   ChatGPT (+21 s) e della ricezione del supervisore (+20,6 s). In tal caso lo
   sforamento plausibile è **almeno 86 s**, non 67,998 s. Il limite superiore
   resta ignoto.

**Nessuna attività nascosta trovata:**

- tra prompt59 (1791372618) e l'ingresso non ci sono scritture;
- tra la chiusura e la ricezione del supervisore non ci sono scritture
  dell'implementatore;
- il prefix target non è cambiato;
- l'ultimo workload reale (R-b002) è a 8671,266 s, sotto il cap secondo la
  convenzione R021 (ingresso prudenziale 8380 + 60).

**Impatto.** Non tocca prodotto, fix o validità delle R. La quota è comunque
NON PASS e la deviazione è già dichiarata. Il registro corrente però presenta
come limite superiore, e come gate che ha intercettato, ciò che non lo è.

**Evidenza.** `reuse_and_timing_check.out.json` (`cumulative_at_mtime`,
`cap_crossed_wall`), letture di `finalize.py` e `close_cost.py`, `find .git
-newermt` e la lista delle ref Codex.

**Criterio di risoluzione.** Il supervisore registra nell'arbitrato:

- 8947,998 s come addebito e limite inferiore, con il probabile ≥8966,24 s
  come inferenza e il limite superiore ignoto;
- il superamento del cap durante la stesura di `finalize.py`, con verify e
  ledger eseguiti oltre il cap (non workload prodotto);
- `finalize-failure.json` come ricostruzione di `close_cost.py`.

Nessuna nuova prova, sanatoria o riscrittura delle receipt. Per le run future:
gate all'inizio degli script di chiusura e cattura dell'output originale dei
fallimenti.

## Backlog, non requisiti

1. **Root del verificatore.** `snapshot_check` (come `make_source_manifest`)
   confina il path del file letto. `run_context.py verify` però ricava la root
   da `git rev-parse --show-toplevel` della cwd. Se `--repo` è una
   sottodirectory non-Git di un altro repository, il file letto e quello
   verificato possono differire. Il caso richiede uno snapshot omonimo e MATCH
   nel repository esterno, quindi è artificioso e non riguarda b002, che ha il
   proprio `.git`. Irrobustimento possibile: confrontare la toplevel con
   `repo`. Rilevato solo dalla lettura del codice, non dimostrato.
2. **`RUN_DOCKER_TIMEOUT`.** Variabile opzionale (default 900) letta dal test
   C-docker e non documentata. Codice invariato, nessuna regressione.

## Motivazione dell'esito e limiti

**GO** per l'oggetto congelato `impl-r001-stage-final-s003`.

**Il bloccante GPT-I002/CLA-I001 è chiuso:**

- il launcher permanente accetta qualunque run_id/label validi con lo stesso
  contratto di S;
- rifiuta prima del verificatore schema bool o invalido, ID, label e path
  evasivi e symlink;
- ferma uno snapshot stantio prima di socket e workload;
- conserva binding prima/dopo, D4, R e cleanup.

Due R reali sul launcher corretto, a001 e b002 nel clone confinato, lo
dimostrano con il target installato e lo stesso managed nei figli. I 12 test
mock coprono i rami negativi e non sono presentati come R. Il diff applicativo
è minimo. Le guide chiudono GPT-I003/CLA-I007/CLA-I008 con l'alternativa
arbitrata. I riusi B56, Docker r007 e C-docker sono esatti; le prove host, V1,
CLI e fedeltà restano ai loro stage, poiché nessuno dei cinque path è un loro
input consumato. La vecchia C-fast non è attribuita al nuovo harness.

**La quota temporale prospettica NON PASS** è una deviazione di processo già
dichiarata da R022 e da arbitrare. Non nasconde workload oltre il cap e non
invalida le prove. CLA-I009 ne corregge la presentazione e non blocca.

### Limiti

- Nessuna R, suite o test rieseguiti in review.
- L'inferenza sul checkpoint Codex non è una misura.
- La misura di spazio propria è approssimata. Vale il ledger del supervisore
  (8,7 MB residui alla ricezione); i miei output sono circa 31 KB.
- Inferenza, GPU, Windows e V10/V11 sono fuori scope.
- La review vale solo per final-s003, MATCH prima e dopo.

Nessuna modifica a prodotto, prove, snapshot o registri comuni. Nessuna
operazione Git di scrittura. Consegna al **supervisore** per l'arbitrato r003,
insieme al report ChatGPT.
