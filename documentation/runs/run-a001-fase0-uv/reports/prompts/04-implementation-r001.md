# Implementazione r001 — piano uv r003 approvato — run-a001-fase0-uv

Agisci come **implementatore in una nuova chat distinta** da pianificatore,
supervisore e revisori. Repository `/home/davide/workarea/markdown-for-llms`, fase
**0.1**, branch esistente `feature/run-a001-uv`. Il ruolo di questo prompt prevale
sull'handover globale. Esegui il piano r003 identificato dal **GO del supervisore**
e le disposizioni dell'arbitrato r003. Non simulare review, creare subagenti o
avviare nuova app/benchmark dalla roadmap. Adozione uv e bootstrap già approvati;
non ripetere branch/squash/push o chiedere nuovamente autorizzazione uv.

## Input e identità

Leggi in ordine, recuperando le porzioni eventualmente troncate:

1. AGENTS.md; `documentation/CHANGELOG.md`; `temp/PROJECT-CONTEXT.md`,
   `temp/HANDOVER.md`; STATE e HANDOVER della run. Questo prompt e l'eventuale
   checkpoint proprio `handovers/implementation-r001.md` hanno precedenza sul ruolo.
2. `.agents/skills/manage-implementation-run/SKILL.md`,
   `documentation/development/run-lifecycle.md`, template implementation-report/handover;
   indici documentation/ADR, ADR 0001/0006/0007 e fase 0.1 della roadmap.
3. `brief.md` (A1–A7); **integralmente** `plans/plan-r003.md` e
   **`arbitrations/arbitration-plan-r003.md`**, specifica vincolante D1–D5/S1–S4;
   arbitrati r001/r002 per le 24 disposizioni conservate; checkpoint/findings/checks
   planning r003. Non correggere retroattivamente piano o evidenze.
4. Entrambi i report `reviews/review-plan-r003-chatgpt.md` e
   `reviews/review-plan-r003-claude.md` e loro checkpoint per motivazioni/limiti.
   L'implementatore può leggere queste review antecedenti; non è un revisore.
5. Evidenze `evidence/supervisor-arbitration-plan-r003/`: receipt, sources,
   transition e checks finali; manifest `snapshots/plan-r003.json`,
   `snapshots/arbitration-context-r003.json` e **implementation-context-r001.json**.
6. Sorgenti/test/packaging/Docker/Compose/README/help e antecedenti pertinenti
   al passo che stai eseguendo, con help uv0.10.10 congelati e fonti primarie
   verificate quando necessario. Non caricare tutta temp/ o leggere segreti.

I percorsi relativi dei punti 3–6 sono in `temp/run-a001-fase0-uv/`.
Piano SHA-256 `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`.
Arbitrato SHA-256 `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`.
Manifest comune review plan-r003 SHA-256
`99f33ba4c9ab452fd0015a654ce59251b68d1467fa97763b2c5c3de921fe4046`.
GO sul piano, **nessun GO sul codice**, migrazione non iniziata a questa consegna.

Alla prima esecuzione, prima di modificare file:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
git diff --cached --name-only
python3 scripts/run_context.py verify run-a001-fase0-uv --label implementation-context-r001
```

Attesi branch feature/run-a001-uv, HEAD/dev/merge-base
`66ba82200e5def5a4db76f9bafccb0731b506091`, indice vuoto, soltanto sei modifiche
documentali di supervisione già identificate e **MATCH** sul nuovo contesto.
Digest/impronta del manifest d'ingresso sono nei checks e nell'handover della
consegna: calcolarli e confrontarli. Il prompt è artefatto di quello snapshot,
quindi non contiene l'hash del manifest che lo congela per evitare un ciclo.

Plan-r003/arbitration-context-r003 sono storici dopo soli sei metadata di
supervisione: transition registra before/after, tutti gli artefatti congelati
invariati. Non rigenerare/sovrascrivere snapshot. Dopo modifiche previste del
GO, il contesto iniziale diventerà storico; per le prove usare gli stage D1.
Una divergenza ulteriore in ingresso va al supervisore, non diventa MATCH.
Se mancano file, recuperare contesto/lavoro: Git non trasferisce temp/ né le
modifiche non committate, e i manifest non contengono i sorgenti.

## Mandato e ordine operativo

Resta la pipeline legacy; preserva algoritmi, JSON, output, numeri/formule,
ordine/riferimenti e perdite già note. Nessuna riscrittura del dominio/parser,
scelta OCR definitiva, revisione AI o Web UI. Candidati invariati: uv0.10.10,
CPython3.12.13, setuptools84.0.0, Marker wheel PyPI1.10.2, Surya0.17.1,
torch2.7.1 CPU/cu126 distinti. Disponibilità/grafo sono da provare, nessun
upgrade/fallback mobile/indice implicito per ottenere PASS. Scostamenti
sostanziali tornano al supervisore con evidenze prima di sostituire il disegno.

1. **Preparazione di §R/baseline.** Inventaria interpreter diretto pyenv3.12.3,
   Pandoc, uv, runner/binari/policy/cache/TMPDIR senza modifiche globali. Prepara
   diagnostici R e wrapper non interattivo nel repository e input F1–F6 con
   provenienza sintetica. Conserva copia/hash dei dieci legacy originali.
   Baseline in venv nuova esterna via pyenv assoluto `-I -m venv`, poi uv pip
   con --python assoluto e --no-build oppure backend/constraints identificati
   prima di qualsiasi sdist. Installa nella venv, mai nel pyenv originale;
   registra versioni/hash/freeze/tiktoken/cache. Confronti incrociati stessa regola.
2. **Freeze di stage baseline.** Segui D1, consegna richiesta e checkpoint al
   supervisore e attendi il manifest. Il supervisore ha già identificato il target
   R: host Linux del clone, UID corrente, binari esistenti, nessun cambiamento
   sysctl/AppArmor/rete/socket/profilo persistente. Nuovi privilegi/perimetri
   operativi non coperti lasciano IMPEDITA e tornano al supervisore.
   Nella ripresa R deve essere PASS prima di eseguire P0/V0: namespace senza
   egress/rotte, figlio equivalente, socketpair positivo, daemon noti inaccessibili,
   cache/Pandoc/temp disponibili. Primario unshare solo se soddisfa anche i path
   daemon; altrimenti Firejail esistente con blacklist dei path/alias/canonici D4.
   Sola connect/close per sonde daemon, nessuna richiesta operativa. Se entrambi
   negati/inadeguati, nessuna baseline fuori runner. Resta IMPEDITA.
3. **P0/V0.** Baseline originale su F1–F6 con gli script diretti e i difetti
   documentati; per F6 validated/custom1000/100, almeno due chunk e complemento
   sliding con overlap >0. Tuning prima del freeze definitivo; fixture cambiata
   richiede nuova label baseline prima delle prove finali. Nessuna normalizzazione
   Markdown/LaTeX; asset mancanti restano perdite inventariate. Applica la skill
   verify-conversion-fidelity per verificare contenuti e confronti.
4. **P1–P3.** Implementa pyproject/lock/pin/constraints e packaging dieci moduli
   flat/cinque script. Managed solo `.venv-python`, nessun pyenv/PATH globale.
   Runtime/dev/API/motore separati, default-groups vuoto, indici torch espliciti,
   extra CPU/cu126 in conflitto. Rimuovi setup/requirements solo dopo allineamento
   dei consumatori. Config.main e helper dotenv nel workspace prima override;
   figli `[sys.executable, '-I', '-m', modulo]`, ambiente ripulito e preflight
   nel subprocess, CWD dati. Conserva CLI/override/exit legacy.
5. **P2/V1–V3, stage package.** Completa diagnostici stdlib visibili alla review:
   check_runner.py, check_python_origin.py, verify_distribution.py,
   **make_source_manifest.py** e **run_offline.py**, in
   `scripts/diagnostics/run-a001-fase0-uv/`. Congela input tramite supervisore,
   poi genera S v1 D2 (`--repo --snapshot --output`); snapshot → S → sdist →
   wheel dalla sdist → installazioni → esiti. Dieci byte/hash, archivi/RECORD,
   backend/dynamic requirements/constraints e input metadata realmente consumati.
   S/B/I del base leggono server senza import FastAPI/Marker. Rebuild mirato
   persistente e flag; wheel canonica per ultima in ciascun ambiente host di prova
   e nell'uso normale. No editable/conftest/copia manuale come prova della wheel.
   Variante utente standalone D2 senza dipendenza da temp/run; scope distinto.
6. **Probe cache/config.** Copia/cache isolate già popolate, metadata/lock
   invariati, sola modifica innocua exceptions.py. Negativo del modulo stantio
   prima sync; config-only senza flag reinstall/refresh, hash/log prima della
   wheel canonica; secondo caso con flag e nuova modifica dopo riallineamento.
   Negativi obbligatori S vecchio/clone nuovo, receipt vecchia/S nuovo e hash
   incompatibile, rifiutati prima collection/import, poi positivo e ripristino.
   Nessun fetch nel probe offline. Copia-probe identificata e diversa dal clone
   finale: stage/S specifici con supervisore, mai attribuirle il PASS del clone.
   Le istruzioni operative/IDE conservano flag/preflight espliciti.
7. **V1 origine/config.** Matrice prima/dopo pin con/senza PYENV_VERSION,
   shim prioritario e bin diretto nei soli processi; sys.executable/realpath/
   prefixes/stdlib. Guardie manual/origine distinte. Caso senza UV_*, VIRTUAL_ENV,
   PYENV_VERSION dentro R con/senza venv: niente flag offline/no-download che
   mascherino manual, errore o solo origine attesa; tentativi di download FAIL.
   Presenza/path/hash config utente/sistema/XDG/progetto e file scoperti nei log,
   senza contenuti/segreti. Nessun uv.toml locale può oscurare la configurazione
   del pyproject senza rilevamento. Lock --check separato, hash prima/dopo.
8. **P4/V4–V6, stage tests.** Ambienti dev/api/wheel pronti e installati con
   catena verificata prima dei C. Test/fixture/harness nel repository, mai codice
   di prodotto nascosto in evidence. Runner non interattivo: R, preflight e test
   stessa invocazione/namespace, receipt padre/figli; guardia pytest IP e path
   UNIX, socketpair positivo. Fixture integrazione/avvio confrontano clone/copia
   installata in sys.executable -I; controllo locale IDE non sostituisce prove
   ufficiali R/S/B/I/E. Nessuna build/sync/fetch in collection/test. V5 cinque
   CLI e contenuti effettivi, fasi reali, dotenv1200/80 e shell1600/120, quattordici
   sentinelle padre/figli e negativo senza input; tutti gli output confrontati.
9. **P5/P6/V7/V8, stage image e passo pesante distinto.** Prima di acquisire
   pacchetti ML/native GB prepara stima rete/disco/tempo/target/cache e comandi,
   e consegnala al supervisore quando il perimetro non è già coperto. Nessun
   download pesi/font/inferenza. V7/V8 restano obbligatorie, IMPEDITA non è PASS.
   Fonti/hash/metadata/contratto della wheel Marker sul codice installato;
   digest nativi Python/uv, apt snapshot disponibile/firme/dpkg, native locked.
   Entrambi i Compose CPU e override GPU config; niente startup normale.
   Sentinelle D5 in copia usa e getta del solo contesto ammesso; nuovi file con
   apertura esclusiva, hash/cleanup/confronto clone, mai .env/config/dati reali.
   Builder sdist/wheel e /opt/venv vs S/B, immagine locale identificata. V8 solo
   entrypoint Python -I stdin, pull never/network none, modelli/font patchati
   prima, import/firme/provider/MarkdownOutput/wrapper/native WeasyPrint reali.
10. **P7/V9/consolidamento.** Guide italiane con skill write-diataxis-docs,
    README/help/AGENTS operativo mirati. Riconcilia tutti i 74 fence e nove
    inline pre (incluse cinque indentate e undici non operativi) col post,
    delta motivati e prove/limiti illustrativi. Correggi comandi assenti, epilogo,
    opzione inesistente, output/config, cache/mount/health e pulizie globali.
    Sostituisci soltanto istruzioni operative AGENTS autorizzate, niente governance.
    README/licenza consumati dal backend cambiati invalidano build/immagine:
    nuova richiesta/stage e ripetizioni necessarie prima del report finale.

## Stage: consegna e ripresa senza auto-snapshot

D1 dell'arbitrato è vincolante. Usa label progressive per baseline/package/tests/
image/final, s001 poi s002 ecc. per ogni invalidazione, niente sovrascritture.
Prepara `implementation/stages/<label>/request.json` con input/hash/assenze,
artefatti/comandi/perimetro/invalidazioni; aggiorna **soltanto** checkpoint proprio
con `WAITING_FOR_STAGE_SNAPSHOT` e percorso esatto. Consegna al supervisore:
`prompts/05-supervisor-stage-r001.md`. Non chiedere una riapprovazione del piano;
serve il freeze di proprietà del supervisore già previsto da protocollo e D1.

Attendi risposta del supervisore con manifest/hash/worktree e prompt di ripresa.
Poi verifica lo stage e genera S; dopo la prova E senza alterare gli input.
Receipt vecchie/mismatch fermano il lavoro. Le label iniziali sono
`impl-r001-stage-baseline-s001`, `impl-r001-stage-package-s001`,
`impl-r001-stage-tests-s001`, `impl-r001-stage-image-s001`,
`impl-r001-stage-final-s001`. Se una copia-probe richiede uno stage separato,
consegna richiesta con label progressiva esplicita della stessa famiglia package;
S identifica anche il repo/copia reale e manifest dei suoi input nella run.

Non modificare snapshots/ o lo strumento run_context per evitare l'attesa.
Alla ripresa non pretendere MATCH del contesto iniziale dopo modifiche autorizzate:
confronta transition/checkpoint e usa lo stage corrente. Il finale congela report
ed evidenze già esistenti, senza creare un S autoreferenziale; se cambiano
sorgenti/build input prima, rifai stage package/tests/image pertinenti.

## Comandi/test e criteri conservati

Usa i comandi esatti di V1–V6 del piano, con variabili RUN_* locali e argv fidati,
preflight e ambienti reali (mai HOME/CODEX_HOME riutilizzati). Prima di ogni pip
che può costruire: --build-constraints identificato, oppure --no-build/sole wheel.
Due base/dev nuovi stessi inventari e lock; canonical install --no-deps --no-build.

C-fast è `"$RUN_DEV_PY" -I -m pytest tests/ --ignore=tests/api
--ignore=tests/packaging --ignore=tests/docker -q`; C-api richiede api-env con
marker-server; C-package driver dev e wheel-env runtime separata; C-docker driver
dev/immagine distinta. C-governance e tutte le collection C-discovery/C-all con
nodeid/conteggi/partizione/esclusioni, nessuno skip essenziale o conflitto smoke.
I 67 legacy/6 governance sono storia, non risultati della migrazione.

F1–F6 sorgente/baseline/output nuovi/hash/config separati, confronto integrale
contenuto/formule/numeri/riferimenti/asset/ordine/metadati/overlap. Incroci solo
su divergenze con codice/interprete/deps/tiktoken controllati; nessun nuovo
algoritmo o normalizzazione per fare passare. Perdita nuova o causa incerta FAIL
al supervisore. Exit0/help/health/file presente non sostituiscono contenuto.

## Output, responsabilità e limiti

Output esatti del ruolo:

- Codice/lock/constraints/test/diagnostici/docs previsti e soltanto fix necessari
  al packaging/avvio, nel repository visibile alla review.
- `temp/run-a001-fase0-uv/implementation/report-r001.md`: autore/provider/modello
  effettivi, prompt/piano/arbitrato/snapshot, §R/P0–P7/A1–A7, V0–V9 per stage,
  conteggi, contenuti, invalidazioni/ripetizioni/equivalenze e limiti. Nessun GO.
- `implementation/stages/` con richieste, manifest di input/S e receipt S/B/I/E
  (S generato dopo il freeze); `evidence/implementation-r001/` per runner,
  backend/build/install/probe negativo/positivo/ripristino, baseline/F1–F6,
  output integrali/log/confronti/inventari/config/contesti/image/doc-checks.
  Hash e ID univoci, niente segreti, file finali senza self-hash.
- `handovers/implementation-r001.md`: checkpoint prima di ogni passaggio o cambio
  chat, stage/next request/processi propri (o nessuno), output, ciò che è provato
  e ciò che resta da fare. Nessun processo di un altro autore va fermato.
- Changelog: soltanto lavoro effettivamente realizzato/verificato con limiti;
  gli stati condivisi/ADR/indici/eventi/snapshot appartengono al supervisore.

Non sovrascrivere piani/review/arbitrati/output originali né completare report
altrui. **V7/V8 obbligatorie future; V10/V11, pesi/font/inferenza non autorizzati.**
local_marker e server host non verificati senza prova pertinente. Se emerge
incompatibilità che rende V10 essenziale, segnala il bisogno senza eseguirla o
passare quel criterio. Nessun invio remoto implicito o benchmark a pagamento.

Nessun commit, merge, push, promozione o deploy. Non generare comandi di
integrazione prima del GO finale. Niente reset/stash implicito/git clean/prune/
down-v o cleanup globale; fermare soltanto oggetti di prova identificati.

Consegna al **supervisore**, inizialmente per gli stage e infine per il report.
Dopo il report servono snapshot finale e **due nuove chat reali indipendenti**
ChatGPT/Claude di review del codice, poi arbitrato finale. Test passati e GO del
piano non sostituiscono questo passaggio. Non iniziare la fase1/2 alla consegna.
