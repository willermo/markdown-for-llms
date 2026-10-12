# Integrazione manuale della consegna sperimentale a002

**Sequenza applicativa già eseguita:** commit/fast-forward ricevuti e verificati
a `7caff336119b6d7d78978204142cfdc16d620fff`. Non ripetere i comandi storici sotto.

## Completare soltanto registri ed evidenze

I24logarchiviati sono stati esclusi dal primo commit dal Gitignore storico.
Eccezione limitata a quell’archivio e registri aggiornati sono già predisposti,
nessuna modifica al parser. Sul branchdev corrente:

```bash
git add -- .gitignore documentation/
git commit -m "Registra integrazione a002 e completa archivio"
git status --short --branch
```

Questo commit amministrativo può avanzaredev rispetto alla feature a002:
non richiede di ripetere merge o review. La nuova feature a003 partirà da
dev aggiornata. Nessun push/main automatico; l’integrazione applicativa resta
identificata dal commit7caff33, distinto dai registri successivi.

## Istruzioni precedenti conservate

Preparata il2026-10-12; nessun commit o merge eseguito dal supervisore.
La consegna è accettata esplicitamente con debito, secondo il manifest;
non dichiarare GO tecnico r014 o soluzione del parser.

Prima della nuova feature: commit di a002, integrazione in dev, ricezione
dei nuovi SHA e confronto dei contenuti. Stato osservato: feature/run-a002-corpus,
HEAD=dev=origin/dev locale80c31da46e5caab990d3596c79c44e524ec1dc95, indice vuoto.
Nessuna query remota effettuata. Il delta comprende anche aggiornamenti dei
registri a001 già ricevuti e il wrapper R/test di binding usati nelle prove a002.

Dalla radice del repository, seleziona la consegna e controlla l'indice:

```bash
git add -- .gitignore docs/ documentation/ scripts/benchmarks/ \
  scripts/diagnostics/run-a001-fase0-uv/run_offline.py \
  tests/unit/test_core_diagnostics.py 'tests/unit/test_conversion*.py' \
  tests/integration/test_conversion_writer.py 'tests/fixtures/conversion*'
git diff --cached --stat=100,60,20
git status --short --branch
```

Verifica che l'indice contenga soltanto la consegna prevista. Le fixture,
trascrizioni e copie originali dei report conservano i byte storici: non
normalizzarli per ragioni estetiche. temp/ rimane ignorata e conservata;
l'archivio permanente contiene la selezione verificata e le dipendenze residue.

Quindi registra il commit e integra nella base dev:

```bash
git commit -m "Consegna sperimentale a002 con debito documentato"
git switch dev
git merge --ff-only feature/run-a002-corpus
git status --short --branch
git rev-parse HEAD dev feature/run-a002-corpus
```

Il merge ff-only conserva il commit e rifiuta basi divergenti: se fallisce,
conservare output e far valutare il delta, senza reset o merge forzato.
Nessun push o promozione a main impliciti. Trasmettere output/SHA al supervisore
per registrare l'integrazione reale; non attribuire il successo prima che avvenga.
La nuova chat può poi preparare a003 dalla base dev effettivamente integrata.
