# A004 — Integrazione manuale del primo prodotto

17 ottobre 2026. **GO I1 r004004**, [esito e limiti](manifest.md).
Integrazione e push eseguiti dall'utente e verificati: feature `323f172`,
dev `f8bf44c`, alberi identici e731 file selezionati invariati. Le sezioni
di commit/merge sotto conservano i comandi storici: non ripeterli. Nessun deploy
pubblico o promozione a main è incluso. P14 resta una lettura umana pendente,
non una certificazione sostituita dai test.

## Prima del commit

Dalla radice del repository verificare:

```bash
git branch --show-current
git rev-parse HEAD dev
git status --short
git diff --cached --stat
```

Attesi: branch `feature/run-a004-usable-product`, HEAD
`f0260568aa544b36b15c351d366583ddd9f8973f`, dev
`9f702a2d368cb8f4a3a6bfb7b9146c6757ecd9df`, index inizialmente vuoto.
Se base o contenuti cambiano, confrontarli col contesto approvato prima del
merge; non forzare né ripristinare vecchi file per ottenere MATCH.

La lista locale `temp/maintenance-cleanup-2026-10-17/staging-paths.txt`
elenca file verificati e cancellazioni documentali da includere dopo la pulizia, senza temp, cache, modelli,
DB o segreti. Usarla solo con i contenuti della consegna invariati:

```bash
git add --pathspec-from-file=temp/maintenance-cleanup-2026-10-17/staging-paths.txt
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
git commit -m "Add experimental local PDF converter with CLI and web UI"
```

Controllare elenco e dimensioni prima del commit: sono sorgenti, test,
configurazione d'esempio, lock, guide e sintesi. Le prove restano locali;
non aggiungere tutta temp o bundle diagnostici. Il file di path non è un
backup: se non è disponibile, ricostruire la selezione dal contesto locale
prima di eseguire git add.

## Integrazione in dev

Dopo il commit e con worktree pulito:

```bash
git switch dev
git merge --no-ff feature/run-a004-usable-product -m "Merge experimental usable converter"
git diff --exit-code feature/run-a004-usable-product dev
git log -3 --oneline
git status --short
```

La base dev verificata è antenata della feature: non sono attesi conflitti.
Se compaiono cambiamenti concorrenti o conflitti, farne valutare il delta
prima di promuovere il risultato. Il solo cambiamento di branch/HEAD rende
storico il verify rigoroso; albero identico e attribuzione al nuovo commit
consentono di registrare l'integrazione senza ripetere le review.

Push, se desiderato, solo dopo tali controlli:

```bash
git push origin feature/run-a004-usable-product dev
```

Riportare commit feature/merge e risultato dei controlli al supervisore per
aggiornare i registri con fatti osservati. Per iniziare l'uso locale seguire
il [tutorial](../../../docs/tutorials/primo-pdf-locale.md) e confrontare un
proprio PDF con Markdown/asset, annotando utilità e perdite per P14.
Nessun altro incremento o nuovo giro di fix parte automaticamente.

## Registrazione della verifica — 17 ottobre 2026

Verificati i commit completi e la topologia nel [manifest](manifest.md).
Origin interrogato direttamente: dev/feature coincidono coi commit locali,
main invariato. Working tree pulito prima di aggiornare i registri; questi
aggiornamenti editoriali successivi sono ancora locali su dev.

Per salvarli dopo averli letti, usando la selezione della ricezione:

```bash
git add --pathspec-from-file=temp/run-a004-fase2-prodotto-utilizzabile/evidence/supervisor-integration-reception-r001/record-paths.txt
git diff --cached --check
git diff --cached --stat
git commit -m "docs: record verified integration of usable converter"
git push origin dev
```

Questi comandi non sono stati eseguiti dal supervisore. P14 resta pendente;
il prossimo passo applicativo è il primo uso locale con feedback dell'utente.
