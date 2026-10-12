# Implementazione a002 r010 — confini HTML/Markdown nelle foglie inline

Agisci esclusivamente come **implementatore**, proseguendo nella chat autrice
r009. Stessa run `run-a002-fase1-corpus`, incremento e risorse. Completa F43–F45
e regressioni nella stessa chat; nessuna nuova pianificazione, delega o review
impersonata. Mandato completo per risultato: scelte reversibili locali ammesse,
non consegne parziali per errori ordinari, gate, argv o freeze delle prove.
Registra provider/modello/effort effettivi se esposti, senza inventarli.

## Ingresso e identità

Repo `/home/davide/workarea/markdown-for-llms`, feature `feature/run-a002-corpus`;
HEAD/dev/origin-dev locale/merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`,
indice vuoto alla ricezione. Delta locale esistente autorizzato, niente reset,
checkout o query remota implicita. Commit/merge/push restano manuali dopo GO.

Leggi nell'ordine, senza ricaricare tutta la storia:

1. AGENTS, temp/HANDOVER e sola prima parte corrente PROJECT-CONTEXT,
   STATE/HANDOVER/brief della run, questo prompt; skill manage-implementation-run,
   verify-conversion-fidelity e run-lifecycle.
2. `arbitrations/arbitration-implementation-r009.md` integrale, entrambi i
   report/checkpoint review r009 integrali. Claude `findings-detail.md`,
   `probe_cases.py`, attesi `expectations-frozen.json`, gruppi A/B e relativi
   risultati `out-{v9-002,v8-001,r007-001}/summary.json`. ChatGPT
   `inert-tail-cases-002.json`, collector e `inert-tail-results-002/summary.json`,
   receipt R-probes-inert-tail-002; casi e input originali sono le prove,
   il messaggio di uscita non sostituisce i report.
3. Report/CP tuoi r009 finali, PROTOCOL e due guide, report/dati permanenti
   r009. Prompt30 e arbitrato r008 per invarianti F39–F42/F36 da preservare,
   non per ripetere la consegna conclusa. Piano/errata e ADR0001 per contratti.
4. Reader integrale, rami audit/observe_bundle/score/safety/PassiveHTML,
   scanner/producer/consumer e test r009 pertinenti; guida uv e toolchain.
   Addendum-resources-r008-r001 e nota originale estensione24h.

Comune revisionato `snapshots/implementation-r009.json`, SHA256
`bfc26ebc0db5946f30a0a47b6aa3a558674a167eb1c022dce1caa39fb6a03b9f`,
impronta `1c5d94c7db9f4a8508143a3bd1e7b72f559dfcaa6bb9909602ac12ebee20eb66`,
328file/91038artefatti: MATCH reviewer e supervisore prima dei registri.
Ora storico per soli quattro documenti amministrativi, non rigenerarlo.
Reader v9 ricevuto SHA256
`a8ffa9eb44cf1695ca2e7e611c91ef8c7594605fe38a3bb194811f056cdbc737`.
Conserva questi byte per confronto prima di modificarli; v8/r007 originali
restano intatti. Nuovi snapshot tecnici r010 autonomi, output vivi esclusi,
nessun passaggio anticipato al supervisore per una prova ordinaria.

Ingresso attuale `snapshots/implementation-fix-context-r009.json`, identità
in `evidence/supervisor-implementation-arbitration-r009/snapshot-receipt.json`
e checkpoint supervisore. Inventario non elenco di letture obbligatorie.
Verifica all'ingresso, registra argv/cwd/output/exit e clock reale:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r009
```

Ricontrolla branch/HEAD/indice/delta anche non committato e budget prima
dei workload; nessun import managed prima dei gate effettivi.

## Risultato richiesto e nota tecnica iniziale

Due cause bloccanti ereditate ricadono nel requisito esplicito F39/F42:
aperture reali devono passare la guardia, i letterali chiusi devono restare
inerti. Correggile insieme: feed HTML indiscriminato sulla coda per chiudere
F43 farebbe riemergere F44; limitare la validazione al primo tag per chiudere
F44 rischierebbe di nascondere unsafe reali successivi (F39).

Prima del codice scrivi nel report r010 una nota tecnica breve: proprietario
e confine della foglia HTML inline, span inerti del dominio Markdown, fallback
code/math/link unsupported, punto di guardia e mapping al sorgente; quali
interpretazioni duplicate vincolare/eliminare e quali prove invalidare.
Nota nella stessa chat, non nuovo piano o approvazione intermedia.
Scelta tecnica locale dell'autore, incluso refactoring reversibile necessario;
nessuna nuova libreria/download o parser HTML/Markdown universale.

### F43 — apertura reale nel fallback Markdown

CLA-R009-I001, media/bloccante, stesso risultato v9/v8/r007. Esempi ammessi:

````markdown
See `x <script>alert(1)</script>

See $x <img src="../outside.png"> tail.

See [a](<../../../../../temp/run-a002-fase1-corpus/prompts/b c>) <img src="../outside.png"> tail.
````

Consumer inline code/math/link inghiotte fino a leaf_end in unknown; node
non valida unknown già munito di lista. Audit oggi COMPLETE senza diagnosi
nei casi code/math, pur con score FAIL; link dà UNSUPPORTED. Scanner riconosce
il tag reale. Non reinterpretare il delimitatore non chiuso come un corpo
inerte che nasconde tutte le aperture riconosciute dal lexer finito.

Dopo: A1/A2/A4/A5/A6/A10 e A3 Claude danno INCOMPLETE/UNSAFE_REFERENCE
con detail, senza file evasivi/fetch. A9 `p.png` sicuro deve conservare e
auditarne il riferimento, oppure almeno INCOMPLETE con diagnosi di risorsa
non rappresentabile, mai COMPLETE con risorsa reale silenziosamente perduta.
Controlli A7/A8 invariati, code/math chiusi e corpi opachi non validati come
HTML. Guardia indipendente dal successo della proiezione, prima unsafe
reale in ordine sorgente; non inventare aperture da token/attributi/code.

### F44 — coda Markdown dopo HTML inline

GPT-R009-I001, alta/bloccante, ereditato v8 e confermato v9. Input:

````markdown
Known <span>words.</span> `<img src="../outside.png"/>`

# Visible heading

After words.
````

HTMLTree valida tutta `raw[i:]` prima di scegliere il primo subtree e
trova img nei backtick; fallback validate_html riceve ancora tutta la coda.
Falso UNSAFE/INCOMPLETE, un unknown con documento intero e FAIL/score3.
Delimita l'intervallo HTML reale prima della proiezione/validazione usando
dominio e span coerenti del producer. Non eliminare la guardia sulle vere
aperture di subtree o sulle aperture successive fuori code/opaque.

Dopo: tre blocchi (paragrafo `Known words. <img src="../outside.png"/>`,
heading `Visible heading`, paragrafo `After words.`), zero refs/assets,
audit COMPLETE, automatic PASS, gold PROVISIONAL/qualità INDETERMINATE,
score5. Stessi criteri per script/onclick letterali e variante quote del
reviewer. Code solo o prima dello span resta conforme; unsafe reale fuori
code resta UNSAFE/detail e run/evaluate2. Preserva raw e posizioni char/UTF8,
testo, numeri, ordine e blocchi esterni.

### F45 — prove delle interazioni e limite F36 esplicito

Integra matrice finita e test delle due cause, attesi indipendenti fissati
prima (il prodotto non genera i propri attesi). Trasformazioni pertinenti:
ordine HTML/code, chiusura/unclosed e uno/due backtick, math chiusa/aperta,
link fuori profilo con code chiuso interno, unsafe/safe/script/onerror,
HTML bilanciato/errore strutturale, paragrafo/quote/list/note/cella e Unicode
nelle classi toccate. Non prodotto cartesiano arbitrario/fuzzing illimitato:
registra dimensioni/seed se generi casi e motiva copertura e limiti.
Asserisci audit→observe→score e aperture asset, non sola assenza di eccezioni.

CLA-R009-I002 è **non bloccante, limite documentale accolto**: conservare F36.
Nei producer HTML blocco/raw, un prefisso prima di opener inerte viene
separato e il corpo resta opaco anche quando il tokenizer CPython puro
riconoscerebbe altre aperture (B1–B4). Confronta e dichiara la divergenza,
INCOMPLETE/UNSUPPORTED/exit2 e nessuna apertura: non introdurre obbligatoriamente
un secondo feed puro né cambiare attesi prefix per ottenere UNSAFE.
Esempio `<div <!-- -- ><img src="../outside.png">` resta nel profilo F36.
La sola estensione documentata non giustifica nascondere tag realmente
riconosciuti dal lexer scelto nei fallback Markdown (F43).

Rettifica PROTOCOL/due guide e nuovi report/dati permanenti r010, storia
r009 e reviewer conservata; readerID finale nuova e input reali. Descrivi
con precisione domini dove guardia precede unsupported, limite F36 e
letterali Markdown chiusi, senza rivendicare HTML5/renderer universale.
F40/F41 chiusi nei domini revisionati: preservarli e verificare regressioni
dipendenti; F39/F42 parziali da completare, non azzerare risultati r009.

## Prove finali sui byte consegnati

- Gate origine managedCPython3.12.13/startup/package prima del primo import.
  Dopo codice/input nuovi ricostruisci S/B/I, sdist/wheel canonica/reinstalla;
  dieci hash source/wheel/installato e interprete dei figli prima di --no-sync.
  Helpers fuori wheel con manifest/hash/copie prima-dopo; S006 integrale
  storico STALE per registri non è PASS. Nuovi gate/input autonomi finali.
- Prima/dopo sui byte v9 conservati, attesi uguali; regressioni r003–r009
  dei rami toccati e matrice308 preservata nei contratti ammessi. F35/F36/
  F40/F41/F32/F27/F30/F28 locale/F26/F16, token opachi interi/progressione
  terminante e domini non presi da producer successivi.
- Suite veloce con preflight/R attuali: `"$RUN_DEV_PY" -I -B -m pytest tests/
  --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q`;
  smoke radice esclusi; non campagne API/Docker/ML o nuove acquisizioni.
- CLI reali run/evaluate su discriminanti delle due cause, exit OS attesi
  fissati prima e output/run/final/errori conservati; gold sintetici
  PROVISIONAL e writer dichiarato. Contratti dipendenti dai cambiamenti,
  H12 FAIL deliberati conservati, non tutti i contratti indiscriminatamente.
- Rescore/invarianza64raw con reader finale e checker indipendente su
  catena/aritmetica/domini/bound, senza motori. Non forzare4PASS/28FAIL:
  misura risultati reali. Counts condizionati/globalbound/full distinti,
  TRIVIAL_ZERO non fullPASS, checker distanza non allineamento indipendente.
  Corpus/gold/raw/V13/A03 chiusi protetti, nuovaV generale0; L04output
  UNAVAILABLE distinta, L01/L02 NOT_MEASURED e L05lower/sampled conservati.

Errori ordinari di fixture/checker/argv/log/timeout/launcher e permessi si
diagnosticano e correggono nella chat, preservando fallimenti. EPERM/EACCES
non rifiuto approvazione: escalation tool autorizzata se necessaria con
isolamento R/Firejail/no-network/registry/shell=False invariato; fermarsi
a un rifiuto effettivo o nuovo confine, nessun privilegio persistente.
Snapshot attribuisce prove, non approvazioni; nessun freeze ogni comando.

## Risorse senza reset e consegna

Disposizione24h osservata2026-10-10T09:37:31Z, entry originale
2026-10-08T07:06:54.369068Z. **Stop nuovi workload2026-10-11T06:37:31Z
=08:37:31 Europe/Rome; deadline09:37:31Z =11:37:31**, riserva3h.
CPU21600/stop19800s, disco6/stop5GiB, RAM8/stop sampled6GiB,
riserva1800CPU-s/1GiB; R≤900s/stream1MiB/file32MiB, rete/acquisizioni/
invii/servizi benchmark0/€0. Stop al primo limite reale; preparazione,
attese, future review e registri inclusi. Nessun modello motore/download.

Base visibile12864,569662CPU-s, più reviewer r009185,908129GPT e106,65Claude
lower; propria supervisione nella reception.json/budget-final.json corrente.
45sClaude r007 storici approssimati distinti, nessun doppio conteggio o upper
certificato; amministrazione non tutta misurata. Misura retained attuale
circoscritto, non sommarlo a storico stesso scope; RSS749760512B storico
sampled, non picco garantito. Leggi clock/budget e programma riserva prima
dei workload. Al limite sostanziale reale checkpoint e disposizione necessaria;
prosegui lavoro indipendente consentito, nessuna estensione implicita.

Scrivi tuoi output `reports/implementation-r010.md` (nota tecnica e consegna
completa), `evidence/implementation-r010/`, `handovers/implementation-r010.md`,
nuovi snapshot `implementation-r010-*`, nuovi report/dati datati all'esecuzione
in documentation/benchmarks. Non modificare stato/indici/changelog/arbitrati/
review/checkpoint supervisore. Conserva input/failures/costi/processi pendenti.
Nessun GO autore. Prossimo ruolo alla consegna completa: supervisore per nuovo
comune e due NUOVE chat indipendenti ChatGPT/Claude, poi arbitrato reale.
Nessuna promessa di ultimo giro/Git automatico/cleanup/fase2/motore/nuovaV.
