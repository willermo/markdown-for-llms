# Review implementazione r002 — Fix e regressioni: criteri comuni

Agisci esclusivamente come reviewer nella nuova chat assegnata. Questo è il
**secondo giro**, sui fix di CLA-I001–I005 e GPT-I001 arbitrati nella run a001.
Il perimetro comprende delta e dipendenze toccate, regressioni e prove di chiusura;
base iniziale66ba82200e5def5a4db76f9bafccb0731b506091, branchfeature/run-a001-uv,
HEAD/dev invariati e indice vuoto alla ricezione. Il lavoro è nel worktree/file
nuovi: git diff base...HEAD da solo non identifica il prodotto da revisionare.

## Oggetto e indipendenza

Snapshot comune **impl-r001-stage-final-s002**, file
`snapshots/impl-r001-stage-final-s002.json`. SHA/bytes/worktree in
`evidence/supervisor-implementation-r002/reception-r001/freeze-identity.json`.
Verifica digest e MATCH prima e dopo con il managed locale ricevuto:

```bash
/home/davide/workarea/markdown-for-llms/.venv-python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 -I -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-final-s002
```

I percorsi sotto sono relativi a temp/run-a001-fase0-uv, salvo file del repository.
Se STALE, identifica il delta e non dare GO per un oggetto diverso. Non creare
snapshot sostitutivi o correggere prodotto/prove. Checkpoint vivi e output r002
dei reviewer sono esclusi dal freeze. Non leggere il report corrente dell'altro
reviewer, non avviare agenti, impersonare altri ruoli o aggiornare registri comuni.
Review r001 e arbitrato sono storia condivisa da leggere per i fix.

## Ingressi essenziali

1. AGENTS, skill manage-implementation-run, protocollo/template review; indici
   documentation/README, decisions/README, roadmap e ADR0006/0007 per contesto.
2. `arbitrations/arbitration-implementation-r001.md`, review ChatGPT/Claude r001
   e prove dei rilievi pertinenti; `prompts/55-implementation-r001-fix-review-findings.md`,
   prompt56, R019/R020. Piano r003/arbitrato piano r003 per D2/P6/P7 e criteri
   essenziali toccati, senza concatenare i58prompt storici.
3. Report-r018/r019; in `evidence/implementation-r001/fix-review-r002/`:
   delivery/final-checks/matrix-final/equivalence-final/manifest.json,
   resource-authorization-r002.json, S60/sei I60, proof finali e FAIL conservati.
4. In `evidence/implementation-r001/fix-review-r001/`: B56 audit/hook raw,
   matrice/documenti74+9/cleanup mirati. In `docker-cpu-r001/fix-review-r001/`:
   Compose r004/history/inspect e C-docker-r002/contract. Supply receipt pubbliche
   e strumenti permanenti realmente usati, non soltanto il riepilogo.
5. `evidence/supervisor-implementation-r002/reception-r001/`: reception,
   delta-s001-to-received, freeze-inputs e identità. Confronta i file final-s001
   con quelli attuali: questo delta identifica i fix/file nuovi, non una diff
   basata su commit inesistenti. Le prove storiche riusate si leggono quando
   necessarie per valutarne la sufficienza.

Report/matrice dell'autore e hash di ricezione non sono il tuo giudizio: controlla
code path, contenuto delle receipt e input consumati. Per il source S60 i soli
delta documentation del supervisore successivi sono inventariati nel freeze;
non fingere che lo snapshot di esecuzione storico sia MATCH sul nuovo contesto.

## Chiusure richieste, uguali per entrambi

| ID originario | Valutazione e prova di chiusura |
| --- | --- |
| CLA-I001, bloccante | Conftest/test-mode official/standalone esplicito, conflitti/valori invalidi respinti senza fallback; gate permanente non cablato ad a001, run_id/schema/label/path/SHA confinati. C-fast reale senza temp/snapshot locale in copia clone e S/B/I standalone, negativi S/I stantii prima di collection/import senza sync, official respinge standalone; run_id diverso reale valido e confini invalidi respinti. Verifica che la prova eserciti il codice permanente, non un surrogato/mock. Origine installata/-I/-B/startup/rete restano guardie effettive. |
| CLA-I002, bloccante | Due Compose di build funzionanti: network none, contesto identificato creato da prepare_docker_inputs/make_input_inventory permanenti e supply locale verificata. Nessuna dipendenza da temp/HOME personale/stage vecchio o input opaco per riprodurre il percorso. Pin OCI/APT firmato/ABI/hash/constraints84/EbookLib/lock/backend e allowlist effettiva; ricostruzione supply esplicita documentata. Compose CPU r007 costruita veramente con history Completed27/27, secondo Compose equivalente su tutti gli input build; GPU solo config. Leggi anche campo storico native_completion:false e perché history/inspect dimostrano il completamento. |
| CLA-I003 | Test C-docker versionato realmente eseguito con un nodeid/nove sottocasi, driver permanente completo. Container per ID/pull never/netnone/nomount/cap-drop ALL/no-new-privileges/nohealth/mem-pids; startup a container fermo e full I prima degli import; stdin fidato, constructor/mock/native senza pesi/font/inferenza, raccolta propri e receipt anche sul FAIL. Nessuna semplice riduzione della guida. |
| CLA-I004 |74fence+9inline riconciliati individualmente con identità originale e destinazione/motivo. Distinguere hash normalizzati storici r003 e hash dei byte reali delle fence; documentazione JSON/backends/polling/multi-formato/scenari recuperata e confrontata al codice, niente comandi inesistenti o promesse ML. README compatto non è da solo un difetto. |
| CLA-I005 | README/guide durevoli, stato della run nei registri di sviluppo; AGENTS corretto. README input B finale è precedente a B56; ogni edit di input build gestito con invalidazioni corrette. Applicazione nuova ancora pianificata, GO/inferenza non dichiarati. |
| GPT-I001 | Cleanup socket s reale, errore primario preservato/secondario distinto e receipt anche con cleanup FAIL. Tre test mirati EACCES/timeout/cleanup verificabili e nuovi R reali col wrapper finale; nessun falso PASS o indebolimento isolamento. |

Controlla le regressioni delle parti toccate: conftest/manifest/preflight/run
identity, launcher/R/figli, preparazione supply/context/Compose/contratto Docker,
metadata/package/documentazione. Nuove guide e tool sono parte del fix, non
esenti dalla review perché la prima review li precedeva.

Host finale: sei I/E60, fast137+300subtest, packaging11/API7, discovery137/156/156,
zero skip/collection nascosti/postcheck. V1 quattro fasi/51chiamate, seed B56
prima R, config-only distinto dal flag, negativi stale/rebuild/hash/restore e
outer postcheck. Discovery-tests-r001 fallita con dev senza FastAPI è preservata,
r002 usa API già verificata senza installare; i tentativi non sono un PASS nuovo.

Riusi: B56 input modules/build_inputs/backend/toolchain esatti; Docker r007
contesto/Compose/supply/driver/test/probe equivalenti; payload/header METADATA,
entrypoint/runtime/cache/Pandoc/fixture e output effettivi per CLI/fedeltà/baseline.
Solo README body non consumato cambia: valuta sufficienza dell'equivalenza,
senza trasferire vecchi R/I ai nuovi input né chiedere una campagna per sola label.
Baseline62PASS/5FAIL/perdite resta caratterizzazione storica. A7/review/arbitrato
è amministrativo e non è da solo NO_GO; V10/V11/inferenza/GPU/Windows escluse.

## Deviazioni, costi e prove proprie

R020 registra l'autorizzazione utente+180s: cap implementazione8280, ultimo
workload8251.006317519117, conto finale8285.28744326299. Lo sforamento
**5.287443262990564s resta quota non PASS**, dichiarato come chiusura documentale.
Verifica origine della misura, interval accounting e reale confine dei workload;
motiva effetto sull'esito, senza sanatoria/falso PASS o una prova per cancellare
il costo storico. R020 ammette la consegna alle review, non risolve il rilievo
con un GO di arbitrato. Storage352MiB/pool1GiB/stop896/riserva16 e Docker16GiB/
7200s/rete1GiB conservati; Shared nativo/immagini intere/floor storico/reviewer
cache-only inclusi, networkupper diverso da wire. Valuta confini e costi nascosti.

Parti da letture/diff/hash/AST. Niente nuovi workload prodotto, campagne/suite
sotto R, installer/build/Docker/rete/ML: tempo implementativo esaurito, prove
ricevute disponibili. R020 autorizza, solo per dubbio concreto, proprie sonde
stdlib/mock reversibili <=30s cumulativi per reviewer, timeout finiti, output
totale<=2MiB entro spazio disponibile; registra costi/limiti. La quota è specifica
della review, non reset o trasferimento delle campagne core. Una sonda mock non
è un nuovo PASS R/V1/V8. Se una prova prodotto è indispensabile, registra motivo
e limite e continua le letture utili; non inventare un PASS o saltare un requisito.

## Report e uscita

Scrivi report/evidenze/checkpoint nei percorsi del wrapper assegnato, secondo
template review. Dichiarare provider/modello/chat effettivi, indipendenza,
manifest/SHA/MATCH prima e dopo, file letti, prove eseguite/non eseguite e limiti.
Tabella dei sei ID originari: **risolto oppure ancora aperto**, motivo e prova;
se bloccante ancora aperto, NO_GO. Nuovi rilievi con ID progressivi assegnati,
severità/blocco, file/posizione, input ammesso/raggiungibilità, effetto/requisito,
attribuzione base/run/mandato, evidenza e criterio di risoluzione. Rarità o
assenza dal corpus non rendono innocuo un difetto; miglioramenti estranei backlog.

Esito **GO oppure NO_GO** motivato per l'oggetto congelato; non arbitrato finale,
fix o Git. Dopo entrambi i report il supervisore arbitra chiusure e nuovi rilievi.
