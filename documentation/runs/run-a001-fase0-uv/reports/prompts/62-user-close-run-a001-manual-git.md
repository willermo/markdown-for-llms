# Chiusura run a001 — comandi Git manuali dell'utente

Arbitrato implementazione r003 **GO**, due review r003 GO sullo stesso
final-s003 SHA `6295835b1407ea44f6c49780d38061c0fd5d6389a2ef1e096da9dd260f778962`.
Stato READY_FOR_MANUAL_INTEGRATION; quote temporali **NON PASS** conservate.
Questo file contiene istruzioni all'utente, non un nuovo mandato implementativo.

Repository `/home/davide/workarea/markdown-for-llms`, branch `feature/run-a001-uv`;
HEAD e dev verificati a `66ba82200e5def5a4db76f9bafccb0731b506091`, indice vuoto.
Il supervisore ha preparato archivio e lista esatta dei path da includere,
senza eseguire operazioni Git di scrittura. Non usare lo snapshot storico come
richiesta di nuove review dopo il solo commit/cambio branch.

## Commit della feature

Da una shell nella radice del repository, controlla prima i contenuti. Il
verificatore legge soltanto file/Git: nessun prodotto, test, download o modifica.

```bash
cd /home/davide/workarea/markdown-for-llms
.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -I -B temp/run-a001-fase0-uv/evidence/supervisor-arbitration-implementation-r003/verify-delivery.py worktree
git status --short
git add --pathspec-from-file=temp/run-a001-fase0-uv/evidence/supervisor-arbitration-implementation-r003/commit-paths.txt
git diff --cached --check
git diff --cached --stat
.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -I -B temp/run-a001-fase0-uv/evidence/supervisor-arbitration-implementation-r003/verify-delivery.py index
git commit -m "build: migrate Python toolchain to uv and archive run a001"
git status --short
git rev-parse HEAD
```

La lista include modifiche/eliminazioni e file nuovi del risultato approvato,
più archivio e registri finali documentali. Non include temp, venv/cache,
.env o configurazione IDE personale. Il check index impedisce di attribuire
il GO a un diverso contenuto staged; in caso di differenza conserva il lavoro
e verifica il delta, senza reset/clean/stash.

## Squash merge in dev

Procedi dopo il commit della feature e con working tree pulito. Se dev non è
più sulla base indicata, o se il merge segnala conflitti, fermati prima del
commit di integrazione e passa il delta al supervisore: una base/contenuto
diverso richiede valutazione, non un GO presunto.

```bash
git switch dev
git rev-parse HEAD
git merge --squash feature/run-a001-uv
git diff --cached --check
.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -I -B temp/run-a001-fase0-uv/evidence/supervisor-arbitration-implementation-r003/verify-delivery.py index
git diff --cached --stat
git commit -m "build: adopt uv toolchain and packaging (run a001)"
.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -I -B temp/run-a001-fase0-uv/evidence/supervisor-arbitration-implementation-r003/verify-delivery.py ref --ref dev
git diff --exit-code feature/run-a001-uv dev -- .
git rev-parse feature/run-a001-uv dev
git status --short
```

I commit avranno SHA diversi per lo squash; i contenuti devono coincidere.
Comunica al supervisore i due SHA e il risultato dei controlli: registrerà
commit e integrazione effettivi senza inventarli prima dell'esecuzione.

## Push facoltativo

Dopo l'integrazione verificata, se vuoi pubblicare dev:

```bash
git push origin dev
```

Remote configurato: `git@github.com:willermo/markdown-for-llms.git`. Se il push
non è fast-forward, non usare force; riconciliare lo stato remoto prima di
pubblicare. Nessuna promozione a main o deploy inclusa in questa chiusura.
Nessuna nuova fase avviata automaticamente.

## Archivio e materiale locale

[Archivio permanente](../../manifest.md)
conserva decisioni, review/esiti, snapshot e ricevute selezionate con hash;
i collegamenti degli estratti sono adattati e la selezione è esplicita.
Payload pubblici, ambienti, cache, immagini e prove non selezionate restano
locali: non eliminare temp o fare prune/clean durante questi comandi.
La pulizia selettiva sarà valutata solo dopo integrazione registrata e verifica
delle dipendenze; nessuna pulizia è stata eseguita dal supervisore.
