# A003 — Confronto motori, riesame del metodo

## Disponibilità aggiornata — 2026-10-17

La pulizia autorizzata rimuove archivi operativi, duplicati, build e cache
superati; conserva sintesi/esiti, casi decisivi del debito e dipendenze del
prodotto. Le descrizioni di conservazione nelle sezioni storiche sotto si
riferiscono alla data originaria, non alla disponibilità attuale.

[Inventario selettivo e limiti](../../development/cleanup-2026-10-17.md).

Aggiornata al 2026-10-16. Fase1 aperta, branch
`feature/run-a003-engine-comparison`, base/dev `9f702a2`.
Lavoro nel working tree, nessun commit o integrazione della run.

## Risultati e stato

13 conversioni ufficiali e quattro piloti su sette casi TXT/DOCX/EPUB/PDF.
Nel campione TXT e DOCX preservano i contenuti verificati; EPUB perde
nota/backlink e nessun PDF intero supera tutti i requisiti. Giudizi umani
PENDING; nessuna scelta definitiva del motore.
[Consegna r007](../../benchmarks/a003-engine-comparison-r007.md).

Stato corrente **FIX_PLANNING / VALIDATION_DEFERRED_BY_USER**: il16ott2026
l'utente dà priorità al prodotto utilizzabile con limiti dichiarati, rinviando
parser/validazione avanzata e risoluzione del debito. Trace r005 consegnato e
identificato, non revisionato; nessun nuovo ciclo automatico di review/fix ora.
R007 e r004 NO_GO storici invariati, nucleo r002 GO circoscritto, D-A002 aperti.
23proiezioni/17assessment restano risultati del campione, non qualifica generale.
[Disposizione vigente](../../decisions/0008-maintained-parser-and-provisional-benchmark.md).

Suite autoriali ricevute: 82 test + 13455 subtest di regressione e 3238 test +
13765 subtest della selezione veloce PASS, zero skip. API/packaging/Docker/smoke
esclusi, non collaudo completo; non rieseguite nel riesame del supervisore.
Controprove sintattiche mirate confermate, senza nuove inferenze o conversioni.

## Prossimo passaggio

Preparazione prodotto a004 in temp: pianificare un percorso reale PDF→Markdown
con asset, CLI/Web UI e fondazioni necessarie; poi incrementi della roadmap.
Controlli operativi e stato della qualità distinti, nessuna certificazione
implicita o dipendenza dal fork sperimentale. Branch prodotto da predisporre
manualmente, nessun prodotto o integrazione realizzati da questa decisione.
Prove e output a003 preservati in temp senza altri bundle permanenti; eventuale
ripresa del debito avrà mandato specifico e prove/review sulle proprietà richieste.

I paragrafi seguenti conservano le precedenti decisioni della run.

Errata E01–E06 del prototipo verificate e accolte il2026-10-16: wrapper AST
e testimone di soli valori scartati per il contratto pieno.16controchiamate
inerti del supervisore riproducono le collisioni AST su6contrasti e le risorse
overflow omesse. Il conteggio URL ricevuto26è corretto a24segmenti/23unici,
senza riscrivere gli originali.1257input/211artefatti/16input esecuzione
conformi;42autocontrolli restano prove dell'autore ricevute.

Mandato corrente prompt29 nella stessa chat implementatrice: esperimento
isolato di trace interno Pandoc3.1.3. Ora consentiti toolchain/build nella
sola temp; confronto AST base/variante nella stessa build e contabilità raw
sul piccolo nucleo prima di ampliare. Nessun fork permanente, nuovo motore
o integrazione dei due adattatori approvati. È una prova di fattibilità
delimitata; manutenzione e adozione restano da valutare dopo il risultato.
La consegna attribuisce il runtime solo alla famiglia GPT-6: Astra/xhigh
richiesti non sono certificabili dalla sessione, nessun confronto fra modelli.

I paragrafi seguenti conservano la ricezione del piano e il suo arbitrato.

Ricevuto il piano r002 e il prototipo: **NOT_VIABLE preliminare per a003 e
a002**, limitatamente alle interfacce Pandoc markdown/JSON, commonmark_x con
sourcepos e micromark con le estensioni provate. È un riscontro negativo da
revisionare, non impossibilità generale del parser mantenuto. Coordinate e
dialetto non risultano insieme sufficienti; la proposta di strumentare Pandoc
e le altre alternative non hanno ancora prova di fattibilità. Nessun parser,
fork o integrazione approvati. Le 362 sonde sintetiche, 23 mappe/17 assessment
e 37 microcasi a002 sono prove autoriali di pianificazione, non nuovi score.

Review completate: ChatGPT GO limitato alla diagnosi/esplorazione, Claude
NO_GO al piano operativo. Arbitrato 2026-10-16: diagnosi circoscritta accolta,
NO_GO alla prosecuzione non delimitata, errata locale E01–E06. Prossimo passo
prompt28 in nuova chat implementatrice del prototipo con Astra, per disposizione utente: requisiti necessari per ciascuna
T e consumatore, URL nudi/dialetto, diagnosi delle coordinate, gate statico
senza build della provenienza Pandoc, testimone strutturale e risorse nelle
celle eccedenti. La perdita di risorse dall'AST P/C è controprovata, non un
falso COMPLETE già dimostrato in un nuovo adattatore. Nessun fork approvato;
coordinate motivate dove necessarie, alternativa strutturale da verificare.
Le chiusure a002 restano separate. Correzione locale del piano, nessun nuovo
piano da zero o doppia review completa automatica dello stesso oggetto. Resta necessaria la
**prova di fattibilità del nucleo prima dell'integrazione**:
parser mantenuto per sintassi, gold indipendenti per fedeltà, coordinate sul raw
e copertura effettiva da dimostrare. Poi due review del piano modificato e arbitrato.
Nessun ottavo giro automatico di patch, terza review dello stesso codice o
accettazione implicita del debito. Parser/adattatore mantenuto non implementato;
debito a002 ancora aperto. Per disposizione utente la sua ripresa viene
anticipata nel medesimo piano: nucleo condiviso, due adattatori e chiusura
D-A002-01–05 da provare separatamente. Nessuna attesa obbligatoria della fase3.
[ADR0008](../../decisions/0008-maintained-parser-and-provisional-benchmark.md),
[mandato corrente](../../development/next-run-a003-engine-comparison.md).

## Disponibilità delle prove

**Aggiornamento vigente16ott2026:** pulizia selettiva autorizzata dopo il rinvio
riconosciuto della validazione. Rimossi toolchain/unpacked/store Haskell e
file compilati/cache ricostruibili:8.777.867.264byte allocati liberati in temp.
Conservati sorgenti/patch/schema, report/review/attesi, stdout e certificati,
output di conversione/asset/raw/gold, archivi di acquisizione e sei binari
finali con hash verificati. Modelli e ambienti ML non rimossi. Nessuna prova
salvata o report è stato trasformato in GO; trace r005 non revisionato.

I vecchi snapshot di ambiente/build/consegna che includono i file rimossi
restano documenti storici: verifica integrale MATCH non più ottenibile senza
ricostruzione. Non ripristinare silenziosamente né attribuire nuove prove ai
vecchi hash. La ripresa richiede reinstallare toolchain e ricostruire dipendenze
con origine verificata; gli output salvati restano consultabili. Elenco esatto
delle directory/file e ricevuta in temp/run-a004-fase2-prodotto-utilizzabile/
evidence/supervisor-priority-r001/cleanup-plan.json e cleanup-result.json.
Il [debito riconosciuto](../../development/technical-debt.md) resta da consolidare
appena tempo/risorse lo consentiranno. La storia della migrazione sotto non
descrive la disponibilità dell'ambiente dopo questa pulizia.


In questa cartella resta soltanto il manifest editoriale. I 8973 file operativi
a003 non tracciati, 721.430.863 byte, sono stati spostati senza cambiare i byte in
`temp/run-a003-fase1-engine-comparison/legacy-documentation/documentation/runs/run-a003-fase1-engine-comparison/`.
Zero differenze SHA256, nessuna prova cancellata; altre run e archivi tracciati
non spostati. Alla migrazione i nuovi file di documentation candidati al commit erano 11:
sintesi, brief e report selezionati; i bundle sono esclusi da Git.

Nella run locale, `evidence/supervisor-implementation-arbitration-r007/` contiene
arbitrato, ricevuta di migrazione, indice e `resolve_reference.py` per tradurre
le referenze storiche. I vecchi JSON/comandi restano identici: per rileggerli o
ripeterli occorre risolvere i path, senza attribuire nuove prove ai comandi storici.
L'indice comune di 8631 path si risolve senza assenze dopo lo spostamento;
8630 hash coincidono e la sola differenza preesistente è editoriale nel report
r007. 14 sorgenti/test e 218 file dei reviewer sono invariati.

Il comune r007 è ora storico STALE dopo governance/migrazione; nuovo ingresso
`pandoc-trace-r001-review-r002` dopo la consegna del trace; comune di review e ingressi
method-reconsideration sono ora storici. Nessuno snapshot equivale a GO. Le prove sono locali,
non trasferite con Git; hash e path non ne garantiscono disponibilità dopo futura
pulizia. Lo spostamento non libera spazio complessivo: pulizia selettiva solo
dopo verifica delle dipendenze. [Regola](../../development/run-lifecycle.md#archivio-e-pulizia).
