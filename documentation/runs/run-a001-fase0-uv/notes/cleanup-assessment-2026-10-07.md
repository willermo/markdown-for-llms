# Valutazione della pulizia dopo a001 — 2026-10-07

Nota17ott2026: riferimenti operativi eliminati o già indisponibili sono
resi testo; esiti storici invariati. [Disponibilità delle prove](../../../development/cleanup-2026-10-17.md).

**Esito:** integrazione e push verificati; nessuna cancellazione eseguita.
La richiesta era verificare ciò che si può eliminare e preparare il passaggio.
La prossima run può partire senza questa pulizia: circa 117 GiB liberi sul filesystem
alla sonda. I residui non sono una nuova implementazione da chiudere.

## Verifiche svolte

- Dev e remoto `refs/heads/dev` coincidono a `80c31da46e5caab990d3596c79c44e524ec1dc95`.
  La receipt (riferimento storico non disponibile) separa squash e registri.
- Verificati i 266 record di `archive-files.json`: tutte le copie permanenti MATCH.
  Questo non rende l'archivio un backup completo: gli oggetti esclusi restano esclusi.
- Inventario senza seguire symlink: vecchia run circa 1.43 GiB allocati, di cui
  work circa 1.20 GiB, evidence circa 223 MiB, snapshot circa 30 MiB. La somma
  dei blocchi per voce è un limite superiore se esistono hardlink o dati condivisi.
- Nessun symlink trovato in `.venv`, `.venv-python`, `.cache` verso la vecchia run.
  I file candidati non sono destinazioni dei symlink inventariati nella run stessa.
  Nell'accessibile `/proc` non sono stati trovati cwd o file aperti nella run;
  menzioni dei path nei comandi Codex/shell non identificano workload residui.
  La sonda non certifica processi o filesystem invisibili al contesto corrente.
- Le tre immagini Docker superate hanno un solo tag ciascuna e nessun container
  associato secondo `docker ps -a --filter ancestor=...`. Nessuna rimozione Docker.

## Eliminabili selettivamente

| Elementi esatti | Motivo e condizione | Spazio |
| --- | --- | --- |
| 193 wheel sotto `work/docker-cpu-r001/fix-supply-r001/wheels`, `fix-supply-r002/wheels`, `fix-supply-r003/wheels`, identificate nel JSON | Copie byte per byte di payload pubblici conservati in `work/docker-cpu-r001/wheels`; hash e keeper ricontrollati. Eliminare solo i singoli file elencati, conservando receipt e gli altri file | 319.40 MiB logici; blocchi circa 319.75 MiB, recupero reale da misurare |
| `run-a001-marker-cpu:r004`, ID `2ffeb7d1a5c0…` | Immagine precedente, nessun container associato; tenere r007 | Dimensione immagine 1.42 GB, non tutta esclusiva |
| `run-a001-marker-cpu:r005`, ID `27b5f840f990…` | Come sopra | 1.40 GB, layer condivisi |
| `run-a001-marker-cpu:r006`, ID `b7331b907c97…` | Come sopra | 1.40 GB, layer condivisi |

La lista macchina esatta (riferimento storico non disponibile) contiene path,
SHA-256, byte, copia conservata e la selezione motivata; non autorizza a cancellare
le tre directory intere. Rimane valida solo se hash e keeper restano quelli osservati.
I vecchi path eliminati non sarebbero più direttamente rieseguibili: i riferimenti
storici restano provenienza e la mappa identifica dove sopravvivono gli stessi byte.
Non attribuire un nuovo PASS a una receipt dopo la pulizia.

Se l'utente decide di eseguire la parte Docker, ricontrollare i container prima;
la sola rimozione mirata prevista è:

```bash
docker image rm run-a001-marker-cpu:r004 run-a001-marker-cpu:r005 run-a001-marker-cpu:r006
```

Nessun force o prune è necessario. Se l'oggetto è ora usato, conservarlo e spiegare
la nuova dipendenza. Non sommare le dimensioni nominali delle immagini come recupero.

## Conservare per la prossima run

- `temp/README.md`, `PROJECT-CONTEXT.md`, `HANDOVER.md`, `RUNS.md`;
  `temp/run-a002-fase1-corpus/` e `temp/maintenance-a001-to-a002/`.
- `.venv-python/` (managed corrente) e `.venv/` canonica. Preparare nuovi S/B/I/R
  per input nuovi; la presenza degli ambienti non attesta installazione attuale.
- `work/core-profiles-r001/dev` e `api`, backend, tokenizer/Pandoc e cache di
  preparazione sotto `core-completion-r001` e `baseline-recovery-*`. La scelta
  di riusarli e i relativi input vanno esplicitati dal nuovo piano.
- Immagine corrente `run-a001-marker-cpu:r007`, ID `b3db6dc22d92…`, supply finale
  `work/docker-cpu-r001/fix-supply-r004/`, wheel/APT originali e receipt.
  I quattro payload duplicati del supply finale non sono candidati: conservati
  nei path correnti. Anche il suo `apt-repository/snapshot.txt` è unico e conservato.
- Report, review, arbitrati, snapshot, FAIL storici e raw evidence ancora unici.
  Non cancellare `evidence/` o l'intera a001 basandosi sul solo archivio selettivo.

## Residui da trattare soltanto dopo conservazione ulteriore

- `/tmp/run-a001-product-wheel-env-r001`: circa 15 MiB; contiene non solo una venv
  ma output grezzi di fedeltà e supplementary proof. Conservare o archiviare gli
  output e verificarli prima di rimuovere il prefix. Non affidare l'unico esemplare
  futuro a `/tmp`, che può essere svuotata dal sistema.
- I file `/tmp/a001-supervisor-*.py` sono candidati di manutenzione, ma la sonda
  non ha dimostrato per ciascuno una copia permanente e proprietà/utilità certe:
  conservarli per ora. Gli altri `/tmp/markdown-*` non sono attribuiti interamente
  ad a001, quindi non includerli nella pulizia.
- Venv sperimentali, cache-discriminant, contesti Docker precedenti e cloni di
  lavoro sono ritirabili dopo aver salvato gli eventuali input, receipt e output
  unici. Non fanno parte della lista già verificata delle 193 copie duplicate.
- Build cache Docker osservata: 19.23 GB globali, condivisi con altri progetti e
  builder. Non è dimostrata l'attribuzione dei record alle sole prove a001,
  comprese le due sonde del reviewer: nessun `docker builder prune`/`buildx prune`
  o `docker system prune`. Volumi e container degli altri progetti restano intatti.
- Nessuna modifica a `.git`, checkpoint dell'harness, branch storici, cache globali
  uv/pip, modelli, font, configurazione utente o IDE.

Non esiste una pulizia obbligatoria aggiuntiva per entrare in a002. I dati non
eliminati sono residui conservati esplicitamente, non FAIL nascosti o quote azzerate.
L'archivio originale conserva i propri record di hash; questa valutazione e la
receipt del push sono integrazioni successive, separate dalla consegna GO.
