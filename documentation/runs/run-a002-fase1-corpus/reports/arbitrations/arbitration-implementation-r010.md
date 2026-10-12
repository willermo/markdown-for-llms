# Arbitrato supervisore — implementazione a002 r010

2026-10-10 Europe/Rome. **NO_GO reale** sul comune `implementation-r010`.
Entrambi i report e checkpoint indipendenti sono ricevuti e letti integralmente:
[ChatGPT](../reviews/review-implementation-r010-chatgpt.md),
[Claude](../reviews/review-implementation-r010-claude.md). Nessuna review
simulata/delegata, nessun runtime del prodotto eseguito dal supervisore.

## Oggetto e prove ricevute

Comune SHA256 `cfc1e56f66a2f38c4159b0e4624271c70bb3d8d3092c94149762c0a89b0f3945`,
impronta `b36a7ec224c2cf1777c0ccda50955eda7f2573702ea2d5a8fc60346ba2ff9bf4`,
331 file/107203 artefatti/29434812 byte, 20 PDF generali. MATCH prima/dopo
dei reviewer e alla ricezione del supervisore prima dei registri: argv,
cwd, output, exit e costo in
[verify](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-implementation-arbitration-r010/verify-common-before.json).
Reader v10 SHA256 `382d6be9ad2c7d376bcfdb44ca2c6ba7acaa12132c2545960692ed2a7c8394b5`.
Feature/HEAD/dev/origin-dev locale/merge-base e indice vuoto confermati nella
[ricezione](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-implementation-arbitration-r010/reception.json).
Nessun reset/checkout/query remota o modifica di prodotto.

ChatGPT ha una matrice finale di 21 casi, stessi attesi e byte sui due reader:
v10 5/21, v9 20/21. Sedici discriminanti, otto per ciascuna causa, conformi
su v9 e difettosi su v10; cinque controlli distinguono chiusure e owner.
Il gold math omette intenzionalmente formule: score3 storico non confuta
la preservazione richiesta e non prova qualità PASS. Link accettato e
posizioni UTF8 sono osservati, non desunti da una label fuori profilo.
I precedenti otto attesi errati con LF libera nella label sono conservati
e corretti, esclusi dai sedici discriminanti accolti.

Claude ha 35 casi v10/v9, cinque sonde della causa, quattro fixture CLI
con otto exit OS reali e misure finite di tempo. Otto receipt finali
selezionate (due ChatGPT, sei Claude) PASS, input invariati, binding
prima/dopo MATCH, snapshot identificato, rete nulla e socket ripulito:
[receipt](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-implementation-arbitration-r010/reviewer-receipts.json).
Inventario per hash e record originali nella stessa cartella. Snapshot
tecnici storici possono legare launcher/helper precedenti conservati;
il MATCH storico non si trasferisce a una versione successiva.
S009 storico STALE solo per quattro registri amministrativi, conservato;
le sonde hanno gate circoscritti effettivi, non una sanatoria del preflight.

## Decisioni sui tre rilievi bloccanti

| Rilievo | Severità e decisione | Attribuzione | Fix |
| --- | --- | --- | --- |
| CLA-R010-I001 | Media, bloccante, accolto | Buco ereditato v9; variante H2 regredisce in r010 nella diagnosi | F46 |
| GPT-R010-I001 | Alta, bloccante, accolto | Regressione r010 su otto casi conformi v9 | F47 |
| GPT-R010-I002 | Media, bloccante, accolto | Regressione r010 su otto casi conformi v9 | F48 |

### F46 — heading metadata ricostruiti dentro una foglia inline

`markdown_lexemes` applica la regola heading/attributi alle righe logiche
anche con `block_context=False` (215–217). Dopo stripping del producer,
`- # x {\`<img src="../outside.png">}` contiene testo inline, non un nuovo
heading di blocco. Lo skip metadata precede la guardia, mentre il consumer
entra in unknown per code non chiuso; `lost_html` vuoto non segnala risorse.
H1/H4–H11, H3 safe incluso, risultano COMPLETE senza diagnosi di risorsa.
H8 ha score3 per gold negativo: resta COMPLETE e non viene escluso dal buco.

CLI reale H1 list/H6 cella: run0/evaluate5 contro atteso2/2 unsafe.
H3 safe oggi run0/evaluate5: qualità e automatic INDETERMINATE, **non PASS**.
H2 senza delimitatore: v9 UNSAFE, v10 INCOMPLETE/UNSUPPORTED, sempre bloccato
ma prima diagnosi unsafe perduta; medesima causa, non un nuovo finding.
Non si osservano aperture evasive/fetch. Viola F43/F45/C05 e le affermazioni
del PROTOCOL su fallback e producer. Raggiungibile via list/quote/note/cella,
caption e testo di heading; non estensione a renderer universale.

Accolto fix locale per applicare metadata al dominio effettivo del producer.
Il vero heading HC2 `# x {\`<img src="../outside.png">}` mantiene i propri
attributi di heading: scanner e consumer lì concordano. Riattivare tutto
il metadata del sorgente romperebbe quel controllo e il profilo ricevuto.
H3 deve proiettare/auditare la risorsa safe oppure essere almeno
INCOMPLETE/UNSUPPORTED, mai COMPLETE con risorsa persa.

### F47 — bound divergenti per math chiusa nella label di un link

Caso discriminante reale:

````markdown
È See [Known $x `first
second` <img src="../outside.png"> y$ words](https://example.invalid) end.

# Visible heading

After words.
````

La label è accettata nel profilo finito perché il code span attraversa LF.
Il consumer ricorsivo ha tutta la foglia e riconosce math chiusa. Il walk
esterno limita invece `bound` alla prima LF nella label, non trova `$`
finale, salta il code registrato ed emette un falso evento img/script.
La nuova guardia solleva UNSAFE prima della ricorsione. V9 conserva formula
raw/link/tre blocchi; v10 perde formule/link e proiezione tipizzata in un
unknown di documento, pur conservando i byte raw. Otto trasporti/payload
confrontati, nessuna evasione osservata. Viola inertness dei corpi chiusi,
F44/F45 e preservazione. Non rendere HTML attivo una formula solo perché
la struttura math interna è fuori subset: il corpo lessicale chiuso resta
opaco; scoring strutturale distinto. Accolto allineamento dei bound/atomi
effettivi fra guardia, producer e consumer, con mapping char/byte conservato.

### F48 — code della label scavalca il proprietario HTML

````markdown
È See [<span>`<script>x</script>`</span>](https://example.invalid) <img src="../second.png">
````

Nel profilo ricevuto lo span possiede il contenuto HTML: i backtick al suo
interno sono testo HTML, non uno span code Markdown inerte. `label_codes`
viene saltato prima del ramo `html_stack`: v10 omette script interno ed
emette span/img successivo. La guardia cattura la seconda violazione e
solleva prima della ricorsione della label, che altrimenti troverebbe la prima.
V9 dettaglia script; v10 `../second.png`. Payload invertito: img interno
`../first.png` viene sostituito nel detail da script esterno. Otto casi,
quattro trasporti e due ordini. Il documento resta UNSAFE/FAIL senza file
evasivi aperti, ma ogni apertura reale e prima diagnosi in ordine sorgente
sono requisiti espliciti F39/F43/F45: severità media bloccante conservata.
Controllo senza seconda violazione rileva la prima anche su v10; non smentisce
il difetto. Controllo code **dopo** chiusura span resta letterale (F44).
Accolto fix dell'ownership dei ledger label e validazione di tutte le
aperture reali, prosecuzione dopo errori e prima unsafe globale stabile.

## Chiusure e osservazioni non bloccanti

F43 chiude i sei producer originali e trasporti aggiuntivi T1/T2/T4/T5
Claude; T3 figura corretto sul v10, confronto v9 non attestato per errore
del gold del reviewer. F44 chiude i quattro originali e varianti di code
dopo HTML chiuso. Queste chiusure circoscritte restano; invarianti F43/F44
complessive e F45 sono **parziali** per F46–F48. F40/F41 restano chiusi nei
domini già revisionati, nessuna riapertura senza prova. F36 limite deliberate
B1–B4 resta INCOMPLETE/UNSUPPORTED e nessuna apertura; non richiede pure feed.
F49 consolida prove finite e documenti sulle tre interazioni, nello stesso fix.

Le quattro osservazioni Claude ricevono decisione esplicita:

- Run di backtick lungo che chiude un code più corto: comportamento ereditato
  e condiviso scanner/consumer, differenza da CommonMark; limite/debito di
  profilo conservato. Nessun nuovo contratto CommonMark completo o fix
  bloccante. Se il ramo viene modificato, verificarne le dipendenze.
- Raw `{=html5}`/`{=html }`: supporto attivo Pandoc non confermato da prove,
  mandato profilo esatto `{=html}`. Questione/limite conservato, non nuovo
  bypass dimostrato; nessun Pandoc o ampliamento formati implicito.
- 4000 commenti (~59KB), circa 7,3s v10 contro1,4s v9: regressione temporale
  osservata finita, senza soglia quantitativa violata. Debito non bloccante,
  niente nuova campagna/ottimizzazione obbligatoria; R/timeout/reap restano.
- Unsafe fallback `observations-v1` fisso in `benchmark.py`, fuori delta:
  imprecisione di ID ereditata non bloccante. Byte reader/gate/R/snapshot
  attribuiscono l'esecuzione. Dichiararla nei dati/report, senza presentare
  l'ID di fallback come versione reale né imporre modifica dello scorer.

64 misure/catene coerenti, 4PASS/28FAIL per set32, 56EXACT_GLOBAL_DISTANCE e
8TRIVIAL_ZERO; numeri osservati, non attesi da forzare nel fix. ChatGPT controlla
19 campi e componenti delle distanze; allineamento completo non indipendente.
`conversion_final` r010 rimanda all'originale r001 finale anziché al rescore
r002 come r009: provenienza coerente, non differenza numerica/gate da occultare.
V13/A03/corpus/gold/raw chiusi, L04outputUNAVAILABLE distinta, L01/L02/L05 e
ChromeNONPASS/helper12gap/H15/E9/errori intermedi restano conservati.

## Mandato e risorse

[Prompt37 r011](../prompts/37-implementation-r011-align-inline-ownership-and-metadata.md)
è completo per la **chat autrice r009/r010**, stesso incremento. Nota tecnica
breve nella chat prima dei cambiamenti, refactoring locale reversibile ammesso,
oracoli indipendenti/trasformazioni finite e prove end-to-end, non nuovo piano.
Errori ordinari propri dei reviewer e supervisore conservati; nessun FAIL
strumentale è attribuito al prodotto. Nuovo contesto fix-r010 dopo soli
quattro documenti generali amministrativi; comune originale non rigenerato.

Base visibile14002,663555CPU-s + GPT163,210780 + Claude143,93 =14309,804335
lower prima della ricezione propria. Claude112,23 + verify finale31,70,
senza risomma R annidate;45sClaude r007 approssimati distinti non risommati.
Risorse proprie/finali e retained scoped deduplicato nelle receipt correnti;
CPU/admin parzialmente misurati, nessun upper certificato. Entry originale
2026-10-08T07:06:54.369068Z e disposizione24h2026-10-10T09:37:31Z invariati.
Stop nuovi workload **2026-10-11T06:37:31Z =08:37:31Europe/Rome**,
deadline09:37:31Z =11:37:31,riserva3h;CPU21600/stop19800s,disco6/stop5GiB,
RAM8/stop sampled6GiB,riserva1800CPU-s/1GiB,R≤900s/stream1MiB/file32MiB,
rete/acquisizioni/invii0/€0benchmark. Tutte le attese/review/admin incluse.
Stop al primo limite reale; nessuna nuova finestra, modello o sanatoria.

Alla consegna completa r011: ricezione del supervisore, nuovo comune e due
NUOVE chat indipendenti ChatGPT/Claude, poi arbitrato reale. Nessun GO futuro,
promessa di ultimo giro, commit/merge/push automatici, nuovaV generale, motore,
fase2, download o cleanup. Git manuale soltanto dopo GO finale effettivo.
