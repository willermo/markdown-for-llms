# Review implementazione r001 — Oggetto e criteri comuni

Prima review del **delta completo della run a001**, migrazione uv della pipeline
legacy, rispetto a `66ba82200e5def5a4db76f9bafccb0731b506091`. Branch
`feature/run-a001-uv`, HEAD/dev/base invariati alla consegna, indice vuoto.
Il codice è nel worktree modificato e nei nuovi file non ignorati: il solo
`git diff dev...HEAD` non mostra il risultato da revisionare.

## Oggetto immutabile e indipendenza

Snapshot comune **`impl-r001-stage-final-s001`**, prodotto dal supervisore:
`snapshots/impl-r001-stage-final-s001.json`. Identità esatta in
`evidence/supervisor-implementation-r001/final-review-reception-r001/freeze-identity.json`.
Calcola il SHA dello snapshot e confrontalo a quella ricevuta; verifica prima
e dopo la review con l'interprete managed locale ricevuto:

```bash
/home/davide/workarea/markdown-for-llms/.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -I -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-final-s001
```

I percorsi in questo documento sono relativi a `temp/run-a001-fase0-uv/` salvo
quelli del repository esplicitamente indicati. Il freeze comprende codice,
documentazione e prove selezionate; le review future, i loro output e i checkpoint
vivi non sono input del medesimo snapshot. Non aggiornarlo o correggere il prodotto.
Se è STALE, identifica il delta e riferiscilo: nessun GO per un oggetto diverso.

Agisci solo come revisore nella **nuova chat indipendente** assegnata. Non leggere
il report corrente dell'altro reviewer e non impersonare gli altri ruoli, avviare
agenti, emettere arbitrati o fare Git del repository. Le review storiche del piano
sono fonti approvate, non la review concorrente dell'implementazione.

## Letture e fonti

1. AGENTS, skill manage-implementation-run e protocollo del repository; indici
   documentation/README, decisions/README, roadmap e ADR0006/0007 per contesto.
2. `brief.md`, `plans/plan-r003.md` completo e `arbitrations/arbitration-plan-r003.md`.
   Piano SHA462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b,
   arbitrato SHAf14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d.
   R016/R017/R018 superano attese di freeze e scope precedenti nei punti disposti;
   R013–R015 precisano fonti, costi e approvazione strumenti. Non concatenare52prompt.
3. `implementation/report-r017.md`, poi in
   `evidence/implementation-r001/completion-r003/`: `delivery-r017.json`,
   `final-checks-r017.json`, `matrix-r017.json`, `equivalence-final.json`,
   `V1-native-hook-binding.json`, S55/sei I55 e receipt dei12 workload reali.
4. Ricezione e inventario oggetto in
   `evidence/supervisor-implementation-r001/final-review-reception-r001/`:
   `reception.json`, `git-delta.json`, `freeze-inputs.json` e identità del freeze.
5. Prove riusate indicate dall'equivalenza: API/discovery e CLI in completion-r002,
   fedeltà installata in completion-r001, baseline ufficiale, Docker in
   `evidence/implementation-r001/docker-cpu-r001/`. Report-r015/r016 sono contesto
   dei riusi/errori; leggere solo i relativi input quando necessario.

Verifica i binding pertinenti e leggi il contenuto delle prove: un hash o un
riepilogo PASS non dimostra da solo il requisito. Il report dell'autore e questa
ricezione non sostituiscono il tuo giudizio indipendente.

## Criteri tecnici comuni

| Ambito | Verifica richiesta |
| --- | --- |
| A1/V1 toolchain | Python managed3.12.13 e uv ricevuti, pin/origini/configurazioni, lock coerente; config-only distinta dal flag, negativi stale prima del sync, hash prima della canonica, restore reale e outer R/postcheck PASS. Seed B46 prima della cattura target, nessuna esenzione METADATA. |
| A2/V2 packaging | Pyproject/lock/runtime/dev/extra, moduli e cinque entry point completi; sdist/wheel dalla sdist, backend84/get_requires e hook nativi, audit B prima install, S/B/I/E correnti nei sei prefix. RECORD: extra uv confinati e verificati, direct_url senza digest solo con URI/B/payload esatti, digest errato respinto, symlink/escape/duplicati rifiutati. |
| A3/V3–V5 avvio e fedeltà | Import e CLI fuori dai sorgenti, sys.executable/-I/-B e ambiente figli, origini/transitive/sentinelle/PYTHONPATH, workspace e override dotenv/shell/config-only; F1–F6, contenuti/numeri/formule/chunk/report/riferimenti confrontati senza normalizzazioni. Valuta i fix del confronto per filename univoco e i relativi negativi. |
| A4/V6 test | Fast134+300subtest e packaging11 correnti senza skip/collection errors, governance6; API7 mock e discovery134/153/153 riusati solo con equivalenza sufficiente di test, profili, sorgenti e prerequisiti. Postcheck pytest in figlio isolato fresco, stesso interprete/namespace, senza nascondere regressioni. |
| R/D4 isolamento | Confinamento prima del workload e nei figli, AF_UNIX sintetico positivo, namespace/route/daemon probe e raccolta processi; escalation approvata distinta da PASS R. Startup/backend prima degli interpreti reali, niente cambi host o dati privati. |
| A6/V7–V8 Docker | Input OCI/apt firmati/29deb/backend EbookLib e supply locked, export/sync offline e constraints equivalenti al piano, contesto filtrato e sentinelle; B/I full builder/runtime prima import, grafo CPU reale106distribuzioni, native/WeasyPrint/contratto wrapper/constructor mock/9sottocasi. Runtime e gruppi conformi al lock, nessun compiler/cache/supply non necessario; Compose CPU/alternativa e GPU solo config. |
| Riusi finali | B46: confronto esatto degli input build consumati. Docker r005:25input filtrati/stdin/probe/Compose/supply equivalenti a oggetti finali, ID storico S53 conservato. API/CLI/fedeltà: README body e diagnostici variati non consumati, dipendenze/versioni pertinenti identiche, nuovi I/E55. Verifica sufficienza della dimostrazione, non solo il numero dei file. |
| A5/V9 documenti | Comandi e help veri, inventario pre/post e link, ambiente riproducibile, guide coerenti con isolamento/installazione e limiti. Applicazione nuova Web/DB/worker ancora pianificata, non dichiarata disponibile. |
| Risorse e scostamenti | Quote cumulative core304MiB/Hentry553541632/pool1GiB/stop896/riserva16 e7200s; Docker16GiB/7200s/rete1GiB distinti. Costi conservativi dichiarati, monitor non atomico, correzione omissione immagine r004, networkupper distinto da wire misurati, nessun reset/cleanup o spesa nascosta. Scelte operative emerse da valutare sul requisito, senza richiedere un permesso per ogni fix. |

Il primo giro riguarda anche codice/test/diagnostici nuovi, Docker/Compose,
configurazione e documentazione/governance modificati nel worktree. Confronta la
base iniziale, leggi i file nuovi e verifica le parti raggiungibili del codice;
non limitarti al report o alla sola ultima correzione V1.

Baseline62PASS/5FAIL e perdite legacy sono caratterizzazione conservata, non PASS
del codice nuovo. Difetti preesistenti possono bloccare un requisito esplicito;
migliorie estranee sono backlog motivato. V10/V11, inferenza/pesi/font remoti,
GPU, Windows e installabilità universale non sono collaudi inclusi. A7/V12 attende
le due review e l'arbitrato: questo passaggio amministrativo non è da solo NO_GO.

## Prove proporzionate e output

Parti da letture, diff, hash e receipt; non rieseguire installer/baseline o tutte
le campagne senza un dubbio concreto. Sono ammesse prove mirate reversibili con
risorse già disponibili, R/preflight pertinenti e output esclusivi nella tua
directory. Usa i costi finali r017 come ingresso, non una quota azzerata; niente
nuovi download/installazioni/modelli/build pesanti o mutazioni dei target di
confronto. Non lanciare automaticamente driver storici legati a snapshot superati.
Un test mirato sul codice corrente può usare il manifest dello snapshot comune
e un output proprio, conservando tutti i file congelati. Se non eseguibile,
riporta il limite e valuta cosa resta dimostrabile, senza inventare un PASS.

Scrivi nei percorsi assegnati dal tuo prompt report, evidenze e checkpoint propri,
seguendo `documentation/development/templates/review.md`. Dichiarare modello/chat
effettivi, indipendenza, snapshot/identità prima e dopo, file letti, prove reali
e non eseguite, limiti, esito **GO oppure NO_GO** motivato.

Per ogni rilievo: ID progressivo del revisore, severità/blocco, file e posizione,
input ammesso che lo raggiunge, requisito/effetto, attribuzione alla base/run/
mandato, evidenza e criterio di risoluzione. I sintetici e i casi rari sono validi
per fedeltà/confini/sicurezza. Non inventare rilievi per riempire una tabella;
non scartare un blocco per la sola assenza nel corpus. Distingui suggerimenti da
requisiti essenziali non dimostrati. Consegna al supervisore senza correggere il
prodotto; nessun commit/merge/push/deploy o GO di arbitrato.
