# ADR 0006 — Gestione del progetto Python con uv

- Data: 2026-10-02
- Stato decisione: accettata
- Stato implementazione: GO piano e implementazione r003 dopo doppie review; integrata manualmente in dev a `b90c6d4` il 2026-10-07, albero identico alla feature `8479f97`; archivio verificato, deviazioni temporali NON PASS conservate, push non verificato
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

## Aggiornamento operativo del 2026-10-06 — metadati EbookLib per il resolver

Adottato, nel perimetro della migrazione autorizzata, il meccanismo ufficiale
`[[tool.uv.dependency-metadata]]` di uv0.10.10 per **ebooklib==0.18**:
`requires-dist = ["lxml", "six"]`. I campi provengono dal setup auditato e da
due wheel native offline con METADATA identici; Requires-Python ed extra assenti.
La dichiarazione resta circoscritta alla versione osservata e conserva fonte PyPI,
hash sdist, pin, indici e grafo universale. Evita di eseguire il backend durante
il lock mantenendo `--no-build`. Non è un override dei requisiti per aggirare
un conflitto né una cache o un lock compilato a mano.

Alternativa valutata: prime registry con backend in rete e cache da riusare;
esclusa perché aggiunge un'eccezione d'isolamento e non garantisce il riuso.
Le build esistenti differiscono nei timestamp ZIP: riproducibilità byte FAIL,
contenuti identici. Attestazione bootstrap pre-build incompleta dichiarata;
nessun PASS completo trasferito al prodotto. Non serve ripetere le build per
utilizzare metadati coerenti con la fonte statica, con questi limiti espliciti.

**Stato implementazione:** dichiarazione congelata nella sola nuova copia della
run; root pyproject immutato e lock ancora da eseguire. Dopo ricezione del lock,
la promozione degli input nel prodotto richiederà nuova identità/snapshot e
confronto dei metadati installati con quelli dichiarati. Una variazione di fonte,
versione o requisiti invalida questa dichiarazione e le prove pertinenti.
Non autorizza build/installazioni runtime o modelli; V7/V8 restano distinte.

Fonti: [documentazione pinned](https://github.com/astral-sh/uv/blob/0.10.10/docs/concepts/resolution.md#dependency-metadata),
[schema pinned](https://github.com/astral-sh/uv/blob/0.10.10/uv.schema.json).
Provenienza dettagliata e ricevute nel package-s010 della run, da archiviare
in documentation/runs alla chiusura.

## Errata operativa del 2026-10-06 — eleggibilità sdist nel lock

Il tentativo s010 è FAIL: uv0.10.10 filtra le sdist con no-build prima di consumare
dependency-metadata. La precedente scelta dei metadati resta valida/proveniente,
ma il claim che bastasse con quel flag era incompleto. Non si corregge costruendo
un'altra wheel nella cache: il ramo di selezione non la consulta.

Adottata nella sola copia una ripresa **offline** R/D4: nessun filtro globale
no-build sul lock/check, divieto build per ogni altro nome degli index-cache
ricevuti, metadati osservati solo ebooklib0.18. Lista nomi chiusa e source/build
cache vuota sono gate verificati pre ogni avvio; zero rete/backend/installazioni.
Non è un'allowlist nativa generale e non autorizza esecuzioni online del comando.
P1 ammette sdist identificate/backend vincolati; la policy globale era aggiunta
operativa, ora corretta esplicitamente senza cambiare pin/grafo/fonti/prodotto.
I comandi di installazione wheel conservano i loro no-build e costi futuri distinti.

**Stato:** mandato41 pronto/input congelati; lock ancora da ottenere, nessuna
convalida anticipata. Cache incompleta/nuovo nome/altro source richiede disposizione
concreta; nessun estensione automatica. ByteFAIL/lacuna bootstrap s009 conservati.
Fonte: [selezione uv0.10.10](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-resolver/src/version_map.rs#L521),
[consumo metadata](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-distribution/src/distribution_database.rs#L543).

## Recupero metadata del 2026-10-06 — mandato43

Il lock del mandato42 ha consumato i metadata EbookLib dichiarati, ma è FAIL
per dati di narwhals/scikit-learn e altri index assenti dalla cache universale.
R/D4 PASS resta legato a quella chiamata; non dimostra riuscita del lock.

Adottato priming diagnostico nativo uv pip compile con risoluzione transitive,
no-build/only-binary e configurazione separata, seguito nella stessa chat da
lock/check originali offline. --no-deps non acquisisce i metadata delle wheel:
il resolver pinned richiede metadata solo in modalità transitive e interrompe
la visita delle dipendenze in modalità direct. La chiusura diagnostica e i suoi
pin non sono il grafo o i pin del prodotto. I nomi acquisiti nei limiti ricevuti
sono sigillati e tutti vietati alla build nel successivo lock salvo l'eccezione
EbookLib già osservata. Nessuna cache forgiata o ulteriore metadata dichiarata.

La precedente policy di nomi statici/offline resta storica per il suo stage.
R011 riceve acquisizione pubblica delimitata e nomi dipendenti senza un passaggio
di supervisione per ogni miss; il lock prodotto resta offline/R/D4. Eventuale
fallback nativo alla wheel per leggere metadata conta come acquisizione nei caps,
mai installazione o prova ABI. Nessun peso/font/modello o nuovo backend.

Fonti pinned: [richiesta metadata transitive](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-resolver/src/resolver/mod.rs#L1669),
[modalità direct](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-resolver/src/resolver/mod.rs#L1808),
[fallback wheel](https://github.com/astral-sh/uv/blob/0.10.10/crates/uv-distribution/src/distribution_database.rs#L520).
Copie e hash delle fonti già ricevute nella run; acquisizione e lock riuscito
ancora da osservare. Stato e limiti non equivalgono al GO finale della migrazione.

## Ricezione prodotto del 2026-10-06 — mandato46

I precedenti stati della copia restano storici. Lock universale e check offline
del mandato45 ricevuti PASS; mandato46 ha promosso nel root gli stessi byte
del lock e la sola tabella metadata EbookLib. La ricezione47 riconferma hash,
delta TOML, nove wheel runtime base e inventory reale per product-s016.
La dichiarazione dei metadata è ora nel prodotto, senza modifica del grafo.

S/B/I/E base sono pronti alla ripresa dopo freeze supervisore: build nativa uv
offline con backend84 esistente/no-build-isolation, startup e import/get_requires
prima della build, poi stessa canonica nei runtime. È una scelta operativa
ricevuta entro r013/r014, da valutare nelle due review; attestazioni ancora
future. Sync del solo runtime non prova il rebuild config-only/flag di V1.
Nessun PASS completo, installabilità multiABI o GO finale; costi112MiB cumulativi
e perimetri pesanti distinti invariati. Dettagli nel [changelog](../CHANGELOG.md).
