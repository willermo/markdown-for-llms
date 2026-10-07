# Report implementazione — run-a001-fase0-uv — r002

2026-10-05, Europe/Rome. Autore Codex/OpenAI, famiglia GPT-6, chat
implementatrice del prompt36; nessuna delega, supervisione o review.
**WAITING_FOR_SUPERVISOR_RECEPTION**. Il codice e gli input preparabili senza
acquisizioni sono consegnati; la convalida completa della migrazione resta aperta.

Autorità: [prompt36](../prompts/36-implementation-r001-complete-uv-core.md),
[piano r003](../plans/plan-r003.md), [arbitrato r003](../arbitrations/arbitration-plan-r003.md)
e [addendum r004](../arbitrations/addendum-operational-protocol-r004.md).
Skill applicate: manage-implementation-run, write-diataxis-docs e
verify-conversion-fidelity. Branch `feature/run-a001-uv`, HEAD/dev/base
`66ba82200e5def5a4db76f9bafccb0731b506091`, indice vuoto. Nessun commit, merge,
push o deploy. Nessuno snapshot prodotto dall'implementatore.

Il contesto `process-adaptation-context-r001` era MATCH all'ingresso; ora è
storico rispetto al delta autorizzato. SHA del piano e arbitrato verificati
contro il mandato. Report r001 e checkpoint d'ingresso preservati byte per byte
in [before — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Il delta governance già presente all'ingresso resta del supervisore: STATE,
HANDOVER comune, changelog, ADR e protocollo non aggiornati in questa chat.

## Modifiche e requisiti coperti

| Piano | Risultato consegnato | Convalida ancora necessaria |
| --- | --- | --- |
| P1 | `pyproject.toml`, pin Python/uv/setuptools, runtime separato da dev, extra CPU/cu126 esclusivi, indice torch esplicito e Marker wheel-only; guida managed e IDE | Acquisizione costata, origine managed reale, lock universale, grafo e backend, matrice V1/sync/probe cache |
| P2 | Dieci moduli flat/cinque entry point, MANIFEST e diagnostici S/B/I/E; preflight prima della collection, nessuna build/installazione dai test | Build standard sdist→wheel e installazioni canoniche identificate; prove reali V2/V3 |
| P3 | Workspace/dotenv/config-only, `--show-config`, fasi installate con `sys.executable -I -B -m`, preflight e ambiente figlio controllato; help aggiornato | CLI/override/origini/transitive e sentinelle eseguite negli ambienti installati V4/V5 |
| P4 | Strati fast/API/packaging/Docker, discovery circoscritto, integrazione senza output preseminati, guardia rete prima della collection; originali/fixture e runner pronti | R ufficiale/V0, E per ogni strato, conteggi e confronto di contenuti effettivi V5/V6 |
| P5 | Pin Marker1.10.2/Surya0.17.1/torch2.7.1, fonti primarie e diagnostica del contratto offline con patch esplicite | Artefatti reali/hash/confronto tag-wheel e contratti installati; nessuna equivalenza provata dal solo sorgente web |
| P6 | Docker multistage, lock comune e wheel canonica, digest/apt obbligatori, allowlist contesto, due Compose CPU e override GPU; cache/tmp/font persistenti | Manifest registry/apt/dpkg/contesto e build CPU V7, probe offline V8, mandato pesante distinto |
| P7 | README italiano, guide/how-to/reference, AGENTS operativo, `.env.example` e esempio IDE; setup.py/requirements.txt rimossi dopo riallineamento consumatori | Comandi operativi e help reali V9 dopo gli ambienti; frammenti nuovi esplicitamente non collaudati |

I file di prodotto sono nel repository. Scratch, copie e ricevute sono in
`work/core-completion-r001/` ed `evidence/implementation-r001/core-completion-r001/`.
La rimozione degli elenchi legacy non rimuove le loro copie originali di confronto.
Nessuna modifica semantica a cleaning, validation o chunking; i difetti di
contenuto legacy restano visibili e vengono confrontati dai nuovi test.

Il driver `run_tests.py` lega E all'installazione I, alla sorgente S e a R nella
netns corrente, registra collection/report/warning e rifiuta skip essenziali.
`tests/conftest.py` richiede un'installazione corrente prima della collection;
non prepara ambienti. Il driver Docker richiede immagine locale identificata,
`--pull=never --network none` e diagnostica su stdin: nessun build/pull/up dai test.

## Verifiche ed evidenze

Percorsi della tabella relativi a
`evidence/implementation-r001/core-completion-r001/`.

| Controllo | Ambiente e risultato osservato | Evidenza |
| --- | --- | --- |
| `python3 -I -B -m unittest discover -s tests/unit -p 'test_*diagnostics.py' -v` | 53 test stdlib PASS; confini runner/binding/archivi/suite/contenuto sintetico. Non è C-fast della distribuzione installata | `diagnostics-final.stdout`, `diagnostics-final.stderr` |
| AST e link locali | Nessun errore AST o link mancante nei documenti controllati | `static-checks-r001.json` |
| `git diff --check` | Exit0 | `diff-check-r001.json`, `delivery-checks-r001.json` |
| Inventario README originario | 74 fence e 9 comandi inline riconciliati individualmente; 11 blocchi non operativi illustrativi; nuovi comandi etichettati non eseguiti | `docs-inventory-r002.json`, `docs-reconciliation.json` |
| Tre `docker compose ... config` | Exit0 CPU/build/GPU, ambiente chiuso e copia senza `.env`; digest all-zero e timestamp solo placeholder sintetici | `compose-static-checks.json`, raw `compose-*.stdout/stderr` |
| `uv --version`, help lock/catalogo Python | uv0.10.10 osservato; letture native senza download | `uv-help-receipts.json`, `python-catalog-r001.json` |
| Preservazione s007 | 1892 identità file incluso lock, 211 directory, nessun link; pre/post sonde e consegna PASS | `preservation-pre.json`, `preservation-post-probes.json`, `preservation-final.json` |
| R preliminare | Due gruppi/cinque chiamate. Primo gruppo impedito prima del runner; secondo unshare IMPEDITA, Firejail bootstrap PASS e Firejail s007 PASS | `preliminary-results.json`, `r-*/receipt.json`, raw/runner/inside del gruppo2 |

R preliminare esercita listener host positivo prima/dopo, parent/child stessa
netns isolata, assenza egress, connect sintetico/daemon negati e socketpair
positivo. Unshare isola IP ma lascia raggiungibili socket locali: nessun PASS D4.
I PASS Firejail valgono soltanto per i bundle/versioni indicati nelle ricevute.
Il wrapper corrente ha ulteriori controlli startup prima del lancio e ambiente
workload esplicito: **R deve essere ripetuta sul nuovo snapshot**, nessun trasferimento
automatico del PASS. Versione realmente sondata conservata in `run_offline-r002.py`.
Tutte le sessioni avviate sono state raccolte; nessun servizio da attendere.

Non eseguiti: R/S/V0 ufficiali nuovi, managed download, lock/build/sync/install
del prodotto, C-fast/API/package/governance/C-all installati, V1–V8 completi,
help/output CLI nuovi in ambiente installato, registry/apt/build/startup Docker,
inferenza/pesi/font/GPU. V7/V8 restano obbligatorie; V10/V11 escluse dal mandato.
`local_marker` e server host `.venv-marker` non verificati dopo la migrazione.

## Correzioni locali

Gli esiti invalidati restano preservati, dettaglio in `corrections.json`.

- Conteggio del verificatore di preservazione ometteva la directory radice;
  corretto senza alterare target o manifest originale.
- AF_UNIX path troppo lungo nel primo gruppo R: due ricevute IMPEDITA prima
  del lancio, nuovo socket_root corto e nuovi bundle/output nel secondo gruppo.
- Scanner README contava due nomi di file come comandi: inventario r001
  conservato, r002 corretto a 74/9 e riconciliazione esplicita.
- Generatore request tentava di leggere una directory `__pycache__` negli
  originali: IsADirectoryError prima di scrivere request; lista delle 14 copie
  da manifest e generatore r002, nessuna modifica alla baseline.
- Opzione documentale non disponibile e whitespace Docker corretti prima delle
  verifiche finali; nessun comando operativo eseguito sulla base di quel frammento.

## Gate e limiti

[Request baseline-s006 — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md):
`WAITING_FOR_STAGE_SNAPSHOT`, owner supervisor, input reali e hash/argv/cwd/env
identificati. Originali10+4 da copie preservate, cinque file suite legacy da HEAD
base e pytest ini vuoto dedicato; fixture F1–F6 senza ritocchi per uguagliare output.
S/R/receipt/output futuri sono output, non artefatti già esistenti da congelare.
La request propone R→S→V0 senza rete/acquisizioni, con preservazione s007 prima/dopo.
Il numero storico 62pass5fail è un termine di confronto, non l'esito del nuovo V0.

[Richiesta acquisizioni — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md):
`WAITING_FOR_SUPERVISOR_RECEPTION`, candidato package-s008 disponibile,
**ready_for_execution=false**. Contiene uv/pin/fonti/checksum Python, target/cache,
ambiente chiuso e comandi standard managed/lock/check/backend wheel-only.
Managed e lock sono assenti; non rappresentati come input consumabili.
Stima conservativa incremento di picco **304MiB** (Python220, metadata64,
backend20), rete massima72MiB, senza pacchetti ML/native pesanti. Storico della
run più riserve80MiB più acquisizione non entra nei limiti attuali: occorre
una disposizione esplicita su costi/risorse/ledger. Nessun incremento deciso
dall'implementatore, nessun cleanup o cache nascosta per superare il limite.
Checksum backend reale e inventari saranno acquisiti e ricevuti prima di build.
Resolver `--no-build`: metadata dinamici/backend ignoti o sdist non ammessa ⇒ STOP.

Il gruppo tests candidato `impl-r001-stage-tests-s001` è disponibile ma **non pronto**:
mancano uv.lock, managed/inventari runtime/backend, B/I canonici e V0 ufficiale.
Non è stata creata una request tests incompleta. Dopo il lock il supervisore
potrà raggruppare package/tests compatibili; non si congela oggi un output futuro.

Budget corrente misurato in `preservation-final.json` e `delivery-checks-r001.json`:
storico incluso, run500/stop384MiB, scratch/report16MiB, riserve64+16MiB,
log1MiB/stream, JSON8MiB, file32MiB, free516MiB. Le guardie monitorate
non costituiscono quota atomica. Nessuna acquisizione pesante o documento inviato.

## Handover al supervisore

Ricevere [delivery — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
questo report e [checkpoint — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md) con **prompt05 e r004**.
Aggiornare registri comuni/changelog, verificare input e budget, produrre snapshot
baseline-s006 prima di R/S/V0; valutare la richiesta concreta dei costi managed/lock.
Proseguire il medesimo obiettivo dopo i gate senza riaprire pianificazione per
fix locali. Nessun GO finale: A1–A6 restano aperti fino alle prove previste,
A7 richiede due review reali indipendenti dello stesso snapshot e arbitrato.

Conservare e trasferire anche file reali ignorati: baseline s007/lock immutabili,
originali, fixture/provenienza, nuovi work/evidence, request e copie d'ingresso.
Non trasferire soltanto i manifest. FAIL/V0/perdite storiche e sei path/tmp
mancanti rimangono distinti; nessuna identità o equivalenza ricostruita.
