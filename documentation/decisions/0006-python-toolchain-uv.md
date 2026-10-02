# ADR 0006 — Gestione del progetto Python con uv

- Data: 2026-10-02
- Stato decisione: accettata
- Stato implementazione: autorizzata, da pianificare nella run a001
- Origine: approvazione esplicita dell'utente dopo la valutazione del setup Python

## Decisione

Adottare `uv` per ambienti, dipendenze e comandi Python, con `pyproject.toml`,
`uv.lock` versionato, `.python-version` e ambiente `.venv` locale ricreabile.
Separare runtime, sviluppo ed extra dei motori. Correggere il packaging legacy e
gli entry point durante una migrazione circoscritta, prima dei benchmark.

Gestire progressivamente anche l'interprete con uv, senza rimuovere pyenv dalla
macchina o modificarne gli altri progetti. Fissare una versione compatibile con il
codice e verificarla; la versione definitiva per i motori dipende dai benchmark.
La configurazione applicativa rimane in `.env`.

Allineare Docker al lockfile e a versioni riproducibili, distinguendo librerie Python,
dipendenze native e pesi dei modelli. Configurare le varianti CPU/GPU esplicitamente;
valutare ambienti separati se i motori hanno requisiti incompatibili. Evitare di
trasformare l'adozione di uv in una riscrittura anticipata della pipeline.

## Contesto e verifica

La ricognizione ha rilevato Python 3.12.3 da pyenv nella shell disponibile, nessuna
versione fissata nel repository o ambiente locale, dipendenze senza lock, strumenti
di sviluppo installati come runtime, packaging incompleto, commenti inline passati
a `install_requires` e Marker installato da `master` nel Dockerfile.

Verificare installazione pulita, import/entry point fuori dalla directory del codice,
suite legacy selezionata e comandi documentati. Non dichiarare verificato un motore
AI se sono stati usati soltanto mock. Il codice applicativo non è ancora migrato.

Riferimenti: [progetti uv](https://docs.astral.sh/uv/guides/projects/),
[Python](https://docs.astral.sh/uv/concepts/python-versions/),
[Docker](https://docs.astral.sh/uv/guides/integration/docker/),
[roadmap](../roadmap.md), [ciclo delle run](../development/run-lifecycle.md).
