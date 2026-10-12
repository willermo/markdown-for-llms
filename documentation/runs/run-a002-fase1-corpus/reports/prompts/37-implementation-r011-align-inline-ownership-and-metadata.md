# Implementazione a002 r011 — ownership e metadati delle foglie inline

Agisci esclusivamente come **implementatore**, proseguendo nella chat autrice
r009/r010. Stessa run `run-a002-fase1-corpus`, incremento e risorse. Completa
F46–F49 con regressioni nella stessa chat. Mandato per risultato, refactoring
locale reversibile ammesso; nessuna nuova pianificazione/delega/review
impersonata o consegna parziale per errori ordinari, gate, argv o freeze.
Registra provider/modello/effort effettivi solo se esposti.

## Ingresso mirato

Repo `/home/davide/workarea/markdown-for-llms`, feature `feature/run-a002-corpus`;
HEAD/dev/origin-dev locale/merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`,
indice vuoto alla ricezione. Delta locale autorizzato, niente reset/checkout
o query remota implicita. Git manuale utente solo dopo GO finale.

Leggi nell'ordine:

1. AGENTS, temp/HANDOVER, sola prima parte corrente PROJECT-CONTEXT,
   STATE/HANDOVER/brief della run; skill manage-implementation-run e
   verify-conversion-fidelity, protocollo run-lifecycle, questo mandato.
2. Arbitrato r010 e entrambi i report/checkpoint reviewer r010 **integrali**;
   Claude `evidence/review-implementation-r010-claude/findings-detail.md`,
   attesi congelati/probe_cases e summary v10/v9/cause/CLI; ChatGPT
   `evidence/review-implementation-r010-chatgpt/cases-003.json`, `probe.py`,
   summary `out-decisive-{v10,v9}-001/` e relative receipt R. Il messaggio di
   uscita non sostituisce i record originali. Conserva errori/gold errati
   storici senza includerli fra i discriminanti accolti.
3. Report/CP tuoi r010 finali, nota tecnica, matrice303 e regressioni308,
   PROTOCOL/due guide/report e dati permanenti; prompt34/arbitrato r009 per
   F43/F44/F45/F36 da preservare. Piano/errata/ADR0001 per contratti, non
   ricaricare tutti i log dei giri precedenti.
4. Reader integrale e rami dipendenti scanner/producer/consumer/audit/
   PassiveHTML/observe_bundle/score/safety, test pertinenti; guida uv e
   toolchain, addendum-resources-r008-r001 e nota originale estensione24h.

Comune r010 SHA256
`cfc1e56f66a2f38c4159b0e4624271c70bb3d8d3092c94149762c0a89b0f3945`,
impronta `b36a7ec224c2cf1777c0ccda50955eda7f2573702ea2d5a8fc60346ba2ff9bf4`,
331file/107203artefatti: MATCH reviewer/supervisore prima dei registri.
Ora storico per soli quattro documenti amministrativi, non rigenerarlo.
Reader v10 `382d6be9ad2c7d376bcfdb44ca2c6ba7acaa12132c2545960692ed2a7c8394b5`:
conserva byte fisici prima di modificarli per confronto. V9 originale
`work/implementation-r010/r009-observations.py`, SHA256
`a8ffa9eb44cf1695ca2e7e611c91ef8c7594605fe38a3bb194811f056cdbc737`,
e v8/r007/prove storiche intatti.

Ingresso attuale `snapshots/implementation-fix-context-r010.json`, identità
in `evidence/supervisor-implementation-arbitration-r010/snapshot-receipt.json`
e CP supervisore. Inventario non lista di letture obbligatorie. Leggi clock
reale e verifica all'ingresso, registrando argv/cwd/output/exit:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r010
```

Ricontrolla branch/HEAD/indice/delta e budget prima dei workload. Nessun
import managed prima dei gate reali. Snapshot tecnici autonomi identificano
input/prove, non sono approvazioni; non un nuovo freeze per ogni comando.

## Risultato e nota tecnica prima dei cambiamenti

Tre cause accolte dipendono da interpretazioni diverse dello stesso dominio:
metadata di blocco riapplicato a foglie già prodotte, bound della label
diverso per scanner e consumer, ledger code che ignora il proprietario HTML.
Nel report r011 scrivi una nota breve con ownership e confine effettivi di
blocco/foglia/label/HTML, atomi chiusi inerti, fallback non chiusi e mapping
al sorgente; spiega quali interpretazioni vincolare o eliminare, quali prove
invalidare e come verificare l'ordine delle aperture. Nota nella stessa chat,
non nuovo piano o approvazione intermedia. La tecnica è scelta dell'autore,
incluso refactoring locale necessario; nessuna libreria/download o parser
HTML5/CommonMark universale.

### F46 — metadata soltanto nel dominio che li possiede

CLA-R010-I001 media/bloccante. Regola heading 215–217 di `markdown_lexemes`
applicata anche inline: dopo stripping del producer, `# x {…}` è testo
della foglia. Lo skip metadata nasconde eventi reali, e fallback code/math
non chiusi danno COMPLETE senza diagnosi. Esempi originali:

````markdown
- # x {`<img src="../outside.png">}

> # x {`<img src="../outside.png">}

- # x {$y <img src="../outside.png">}

- # x {<img src="../outside.png">}
````

H1/H4–H11 e H3 safe ereditati v9; H2 senza delimitatore regressione r010:
UNSAFE v9→UNSUPPORTED v10. Correggi una sola causa di dominio coerentemente
fra producer, guardia e consumer. H1/H4–H11 unsafe devono essere
INCOMPLETE/UNSAFE_REFERENCE con detail corretto, ogni apertura reale visitata;
H2 recupera UNSAFE. H3 `p.png`: proiezione/riferimento/asset auditati oppure
almeno INCOMPLETE/UNSUPPORTED per risorsa non rappresentabile, mai COMPLETE
con risorsa persa. Nessuna apertura evasiva/rete.

HC1 senza `#` resta UNSAFE. **HC2 vero heading** `# x {\`<img src="../outside.png">}`
mantiene metadata effettivo nel profilo ricevuto: non riattivarlo globalmente.
H7 ha testo di heading `# x {…}` dopo stripping del vero `##` e propri
attributi `{.c}`; distinguere i due livelli. List/quote/note/cella/caption/
heading e continuazioni pertinenti restano foglie del producer.
CLI originale H1/H6 run0/evaluate5 non è qualità PASS (INDETERMINATE);
dopo fix unsafe atteso run2/evaluate2. H3 atteso coerente con proiezione
o alternativa unsupported dichiarata, senza forzare uno score numerico.

### F47 — math chiusa nella label mantiene il corpo opaco

GPT-R010-I001 alta/bloccante, otto regressioni rispetto v9:

````markdown
È See [Known $x `first
second` <img src="../outside.png"> y$ words](https://example.invalid) end.

# Visible heading

After words.
````

Il code attraversa LF e rende la label realmente accettata; non usare come
oracolo una label con LF libera fuori profilo. La guardia limita il bound
alla prima LF nella label, il consumer ricorsivo `math_end` usa tutta la
foglia: `$…$` chiusa diventa falso img/script reale prima della ricorsione.
Dopo: nessun falso UNSAFE, formula raw integra, link esterno conservato,
testo e tre blocchi tipizzati esterni, posizioni char/UTF8 coerenti, nessuna
apertura asset dai letterali. La struttura math può essere unsupported:
il corpo lessicale chiuso resta opaco. Gold reviewer volutamente privo di
formula/score3 non è qualitàPASS; costruisci gold indipendente adatto al
contratto senza usare lo scorer per generare gli attesi.

Stessi attesi per otto casi `math-code-label-*` di cases003 (paragrafo,
quote/list/note-tab × img/script), più controlli fuori label/chiusura.
Math/code non chiusi continuano a mostrare aperture reali alla guardia,
F43/F46; bound/metadata/HTML owner devono concordare, non solo spostare LF.

### F48 — HTML owner prevale sui ledger Markdown della label

GPT-R010-I002 media/bloccante, otto regressioni rispetto v9:

````markdown
È See [<span>`<script>x</script>`</span>](https://example.invalid) <img src="../second.png">
````

Lo span possiede il contenuto HTML nel profilo ricevuto: i backtick al suo
interno sono testo HTML, non code Markdown inerte. Il salto incondizionato
`label_codes` prima di `html_stack` nasconde script interno; la guardia
emette soltanto span e img successivo e detail passa alla seconda violazione.
Payload invertito: `../first.png` interno seguito da script esterno dà
detail script invece della prima img. Entrambi restano UNSAFE oggi, ma
prima diagnosi in ordine sorgente e visita di tutte le aperture sono requisiti.

Dopo: tutti gli start reali, prima unsafe globale stabile, visite proseguono
anche dopo prima unsafe/errore di proiezione; nessun fetch/file evasivo.
Otto `HTML-owner-label-first-*` con quattro trasporti/due ordini. Controlli
senza seconda violazione e code dopo **chiusura** span: quest'ultimo resta
inerte e conserva F44. Non disattivare tutti i code nella label né cambiare
il profilo owner HTML per ottenere gli attesi; vincola ledger e dominio.

### F49 — invarianti finite, interazioni e documenti

F43/F44 chiusi sui casi originali ma complessivamente parziali via F46–F48;
F45 parziale. Preserva quei risultati e F40/F41/F36. Matrice finita per
classi dei rami modificati, non solo tre payload: inline vs metadata reale,
prefix logico e continuazione, label con LF in code accettata vs LF libera
unsupported, math/code chiusi/non chiusi, owner HTML aperto/chiuso e atomi
interni/esterni, ordine di due unsafe e safe/unsafe, backtick uno/due e
Unicode/posizioni, trasporti pertinenti anche heading/caption. Usa attesi
indipendenti fissati prima, trasformazioni documentate e dimensioni/seed
se generati; niente prodotto cartesiano arbitrario o fuzzing illimitato.
Asserisci guardia/diagnosi/visite e audit→observe→score/output/asset/CLI,
non sola assenza di eccezioni. Non assumere indipendente un oracolo che
riusa la stessa scansione appena cambiata.

Conserva F36 B1–B4 INCOMPLETE/UNSUPPORTED, differenza deliberata dal tokenizer
puro dopo prefissi opachi; nessun pure feed extra obbligatorio. Osservazioni
Claude non bloccanti arbitrate: backtick-run ereditato diverso da CommonMark,
raw `{=html5}`/`{=html }` fuori profilo esatto (Pandoc attivo non confermato),
4000 commenti v10~7,3s/v9~1,4s debito di tempo senza soglia violata, ID fallback
`observations-v1` ereditato fuori delta. Preserva prove/limiti nei report,
attribuisci il reader reale con byte/gate/R/snapshot; nessuna ottimizzazione,
ampliamento formato o modifica benchmark/scorer obbligatoria. Se tocchi un
ramo dipendente, verifica regressioni proporzionate, senza espandere il mandato.

Aggiorna PROTOCOL/due guide e nuovi report/dati permanenti r011: ownership
effettiva, regole di metadata e inertness, prima unsafe e visite, alternative
unsupported, F36 e limiti esatti. Nuovo readerID finale, mai dichiarare
fallback observations-v1 uguale alla versione eseguita. Storia r010 intatta.

## Prove sui byte finali

- Gate origine managedCPython3.12.13/startup/package prima del primo import.
  Dopo nuovi input/codice ricostruisci S/B/I e sdist/wheel canonica/reinstalla,
  confronta dieci hash source=wheel=installato/interprete figli prima di
  --no-sync; helpers fuori wheel con hash/copie prima-dopo. S009 integrale
  storico STALE per registri non è PASS; gate finali autonomi effettivi.
- Prima/dopo v10 fisico con stessi raw/attesi, v9 storico per attribuzione
  dove utile; matrice303 r010 e308 r009 preservate nei contratti ammessi,
  regressioni dei rami r003–r010 toccati. F35/F36/F40/F41/F32/F27/F30/F28
  locale/F26/F16, progressione terminante, token opachi e domini successivi.
- Suite veloce con preflight/R attuali: `"$RUN_DEV_PY" -I -B -m pytest tests/
  --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q`;
  smoke radice esclusi, nessuna campagna API/Docker/ML/acquisizione.
- CLI reali run/evaluate su discriminanti delle **tre** cause e controlli,
  exit OS attesi fissati prima, gold PROVISIONAL/writer sintetico dichiarato,
  output/final/errori conservati. H1/H6 unsafe2/2, F47 senza falso run2,
  F48 prima diagnosi oltre all'exit2/2; H12 FAIL deliberati conservati.
- Rescore64raw con reader finale e checker indipendente catena/aritmetica/
  domini/bound, senza motori. Non forzare4PASS/28FAIL: misura esiti reali.
  Conversion_final originale r001 della catena r010 è distinto da rescore
  r002 r009, mantieni provenienza verificabile. Condizionato/globalbound/full
  separati, TRIVIAL_ZERO non fullPASS, distanza non allineamento indipendente.
  Corpus/gold/raw/V13/A03 chiusi protetti, nuovaV generale0,L04outputUNAVAILABLE
  distinta,L01/L02NOT_MEASURED,L05lower/sampled e NONPASS storici conservati.

Errori ordinari di fixture/checker/argv/log/timeout/launcher/permessi si
diagnosticano e correggono nella chat, fallimenti conservati. EPERM/EACCES
non è rifiuto approvazione: escalation tool disponibile per azione già
autorizzata se necessaria, mantenendo R/Firejail/no-network/registry/shell=False;
fermati a rifiuto effettivo o nuovo confine, nessun privilegio persistente.
Non ripassare al supervisore ogni snapshot o prova invalidata ordinaria.

## Risorse e consegna completa

Entry2026-10-08T07:06:54.369068Z e disposizione24h osservata
2026-10-10T09:37:31Z senza reset. **Stop nuovi workload2026-10-11T06:37:31Z
=08:37:31 Europe/Rome; deadline09:37:31Z =11:37:31**, riserva3h.
CPU21600/stop19800s,disco6/stop5GiB,RAM8/stop sampled6GiB,
riserva1800CPU-s/1GiB,R≤900s/stream1MiB/file32MiB,rete/acquisizioni/invii0/€0
benchmark. Stop al primo limite reale, preparazione/attese/review/registri
incluse. Leggi clock e contabilità prima dei workload, preserva riserva.

Base supervisore14002,663555CPU-s + GPT163,210780 + Claude143,93 =
14309,804335 lower prima della ricezione propria. CPU finale e retained
attuale deduplicato in `evidence/supervisor-implementation-arbitration-r010/budget-final.json`;
45sClaude r007approx distinti non risommati, niente R annidata/doppia base
o somma del retained storico stesso scope. Admin parzialmente non misurata,
RSS storico749760512B sampled, nessun upper certificato. Al limite sostanziale
reale conserva CP e richiedi disposizione necessaria, continuando lavoro
indipendente consentito; nessuna nuova finestra o estensione implicita.

Output tuoi `reports/implementation-r011.md` (nota tecnica e consegna finale),
`evidence/implementation-r011/`, `handovers/implementation-r011.md`, snapshot
`implementation-r011-*`, report/dati datati all'esecuzione in
documentation/benchmarks. Non modificare stato/indici/changelog/arbitrati/
review/checkpoint supervisore. Conserva input/errori/costi/processi pendenti.
Nessun GO autore. Alla consegna completa supervisore riceve delta/prove,
crea nuovo comune e due NUOVE chat indipendenti ChatGPT/Claude, poi arbitrato.
Nessun GO futuro/ultimo giro promesso/Git automatico/cleanup/fase2/motore/nuovaV.
