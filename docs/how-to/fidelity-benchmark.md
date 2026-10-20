# Eseguire il benchmark locale di conversione

**Stato2026-10-12: strumento sperimentale.** A002 è chiusa per accettazione
utente del perimetro ridotto, non per GOtecnico r014. Il reader può assegnare
COMPLETE/UNSAFE errati e omettere riferimenti nel modello osservato: risultati
provvisori nelle classi non convalidate, nessuna attestazione universale di
qualità/sicurezza. Codice e comandi non sono corretti da questa etichetta.
Usare gold/raw e valutazione umana; [debito e trigger obbligatorio](../../documentation/benchmarks/a002-residual-debt.md)
e [decisione parser](../../documentation/decisions/0008-maintained-parser-and-provisional-benchmark.md).
Le sezioni di implementazione precedenti descrivono prove storiche/domìni,
non la chiusura generale del debito. Nessuna nuova campagna avviata qui.


Guida per sviluppatori che dispongono della toolchain locale verificata e di
un target R con rete/socket isolati. Usa soltanto il corpus redistribuibile e
Pandoc già locale; non installa motori o scarica modelli. La
[reference](../reference/fidelity-benchmark.md) e il
[protocollo](../../scripts/benchmarks/PROTOCOL.md) definiscono formati e limiti.

## Preparare la prova

Seguire il preflight della [guida uv](uv-environment.md): origine managed CPython
3.12.13, startup verificato, sdist/wheel ricostruite, wheel canonica reinstallata
per ultima, dieci hash e input S/B/I confrontati. Disporre di Pandoc3.1.3 e
cache tokenizer locale. L'inventario esterno copre helper/corpus/tests/tool,
oltre ai moduli della wheel. Non trasferire un PASS a001 al nuovo codice.

Questi nomi sono variabili di lavoro da impostare ai **propri percorsi assoluti**:

```bash
RUN_BENCH_REPO=/percorso/markdown-for-llms
RUN_BENCH_STAGE=/percorso/stage-verificato
RUN_BENCH_PY="$RUN_BENCH_STAGE/dev/bin/python"
RUN_BENCH_OUT="$RUN_BENCH_REPO/temp/prova-benchmark-001"
RUN_BENCH_LAUNCH="$RUN_BENCH_REPO/scripts/benchmarks/launch.py"
```

Non creare RUN_BENCH_OUT prima di freeze. Su questo repository prepare può
creare uno stage offline usando i keeper a001 già esistenti; vedere i suoi
prerequisiti nella reference. Su altri cloni usare la procedura S/B/I locale,
senza copiare receipt di un interprete diverso.

Eseguire prima i controlli autonomi del reader/writer sotto R:

```bash
"$RUN_BENCH_PY" -I -B -m pytest \
  tests/unit/test_conversion_benchmark.py \
  tests/unit/test_conversion_comparison.py \
  tests/integration/test_conversion_writer.py -q
```

Il comando sopra è il payload pytest; per la prova ufficiale racchiuderlo nel
wrapper R come nel passo successivo. Un FAIL del test va corretto e ripetuto
con nuovi input/snapshot, conservando l'esito. Non cambiare attesi o writer per
far combaciare un output di conversione con il gold.

## Congelare e avviare i gruppi

Freeze è amministrativo: nessun motore avviato.

```bash
"$RUN_BENCH_PY" -I -B "$RUN_BENCH_LAUNCH" campaign freeze \
  --corpus "$RUN_BENCH_REPO/tests/fixtures/conversion-corpus" \
  --stage "$RUN_BENCH_STAGE" --output "$RUN_BENCH_OUT" \
  --experiment-id esempio-native-001
```

Prima del primo tentativo includere i nuovi JSON operativi negli input hashati
del target R e nello snapshot tecnico, senza sovrascrivere quelli precedenti.
La preparazione della run conserva gli S/B/I già immutati: il nuovo snapshot
aggiunge i config/manifest e il target include quei descrittori. Se cambia un
helper o un input prodotto, ricostruire/reinstallare e ripetere le prove invalidate.

Usare un gruppo per invocazione R, expected-exit0 fisso, output R e output
workload entrambi nuovi. Esempio di payload con direct-docx:

```bash
"$RUN_BENCH_PY" -I -B \
  "$RUN_BENCH_REPO/scripts/diagnostics/run-a001-fase0-uv/run_offline.py" \
  --runner firejail --python "$RUN_BENCH_PY" --repo "$RUN_BENCH_REPO" \
  --target /percorso/target-con-config.json \
  --snapshot /percorso/snapshot-tecnico.json \
  --preflight-config "$RUN_BENCH_STAGE/preflight.json" \
  --output /percorso/R-direct-docx-001 --cwd "$RUN_BENCH_REPO" \
  --timeout 600 --expected-exit 0 -- \
  "$RUN_BENCH_PY" -I -B "$RUN_BENCH_LAUNCH" campaign run-group \
  --manifest "$RUN_BENCH_OUT/manifest.json" --group direct-docx \
  --output "$RUN_BENCH_OUT/direct-docx"
```

Ripetere sequenzialmente per gli otto group_id elencati nella reference.
Per un fix degli input usare `campaign freeze --groups` con i soli percorsi
dipendenti: ciascuno mantiene quattro tentativi e nuove identità; il riepilogo
richiede tutti i gruppi dichiarati. Conservare la campagna precedente completa,
senza presentare il sottoinsieme come otto percorsi rieseguiti.
Verificare ogni receipt R, group.json e i quattro final.json. Un errore di
permessi del sandbox va diagnosticato usando l'escalation autorizzata degli
strumenti, mantenendo isolamento e quote; non equivale a R PASS. Non cambiare
expected-exit dopo aver visto l'esito. Timeout del motore può essere dato valido
solo con figli terminati/raccolti; timeout R o del harness invalida la campagna.

## Riepilogare e leggere le perdite

Passare gli otto group.json nello stesso ordine del manifest:

```bash
"$RUN_BENCH_PY" -I -B "$RUN_BENCH_LAUNCH" campaign summarize \
  --manifest "$RUN_BENCH_OUT/manifest.json" \
  --groups "$RUN_BENCH_OUT/direct-txt/group.json" \
  "$RUN_BENCH_OUT/direct-docx/group.json" "$RUN_BENCH_OUT/direct-epub/group.json" \
  "$RUN_BENCH_OUT/direct-rtf/group.json" "$RUN_BENCH_OUT/direct-tex/group.json" \
  "$RUN_BENCH_OUT/legacy-docx/group.json" "$RUN_BENCH_OUT/legacy-epub/group.json" \
  "$RUN_BENCH_OUT/legacy-rtf/group.json" --output "$RUN_BENCH_OUT/summary.json"
```

Per una prova ufficiale eseguire anche summarize dentro un target R dedicato.
Leggere execution_complete, conversion_batch e quality separatamente; aprire
metrics.json per diff e unavailable. Controllare A10: quattro vere ripetizioni
C02/C03 con Markdown e figura attesa referenziata/contenuto verificato. Un nome
PNG o il solo percorso TXT non soddisfano quel controllo. Confrontare mediane e
range delle tre ripetizioni con il primo avvio separato, senza chiamarlo cold OS.

## Raccogliere V e conservare evidenze

M e packet gold sono separati dal workload. Il confronto documentale v2 mostra
l'originale a sinistra e il Markdown di riferimento completo a destra. Aprire
prima C02, poi gli altri documenti dall'indice. Sono tredici coppie, con formule
MathML visibili, celle vere, asset e file Markdown scaricabili; nessun output
Pandoc viene promosso a gold. Per TeX si confrontano sorgenti non compilati.

L'utente risponde in chat, per documento: «C02: corrisponde», «C02: manca …»
o «C02: non riesco a valutarlo». Conservare esattamente il testo realmente
ricevuto in un nuovo file UTF-8. Una dichiarazione esplicita «Tutti i documenti
corrispondono» si riferisce alle coppie del packet: l'importatore conserva una
sola dichiarazione collettiva e ne attribuisce l'ambito ai documenti, senza
inventare righe originali o firme per proprietà. Un'approvazione vaga o limitata
ai soli documenti già verificati non accetta tutti i documenti.
L'operatore attribuisce autore e ora di ricezione
dalla risposta reale, senza chiedere timestamp per proprietà o una nuova
designazione del verificatore. Se manca l'ora della piattaforma, l'importatore
registra l'ora di importazione e dichiara tale limite. Importare il testo:

```bash
"$RUN_BENCH_PY" -I -B "$RUN_BENCH_LAUNCH" packet import-response \
  --packet /percorso/confronto-v2-immutabile --response /percorso/risposta.txt \
  --packet-sha256 DIGEST_DEL_PACKET --author "$RUN_V_AUTORE_REALE" \
  --received-at "$RUN_V_ORA_RICEZIONE_REALE" \
  --receipt-method "messaggio reale ricevuto in chat" \
  --output /percorso/risposta-importata-001
```

Il comando `verification` unisce M e il response.json importato per produrre
receipt separate per documento, senza modificare le annotazioni. `--response`
è ripetibile per conservare risposte parziali precedenti; un documento assente
da una nuova risposta non cancella il giudizio già ricevuto. Una risposta
esplicita successiva sullo stesso documento sostituisce la precedente.
Il testo originale e il suo hash restano conservati. Per il packet v2 non si
compila un TSV; il vecchio importatore schema1 resta per evidenze storiche.
`output_packet` prepara l'inventario passivo L04 da manifest/summary della
campagna effettiva; non assegna o accetta giudizi umani.

Verificare status e identità/hash, non solo exit. Vgold incompleto resta
PENDING; agent_visual e un accordo di durata non lo completano. Output formula
semantics/typography richiedono giudizi separati, non questa firma gold.
Conservare manifest, raw/errori, receipt R/S/B/I, osservazioni/metriche e ledger;
redigere il report effettivo in documentation/benchmarks dopo il workload.
Le due review del codice sono chat indipendenti del ciclo supervisionato;
commit/merge/push restano manuali dell'utente.

## Metadati aggiornati e rettifica di misure storiche

Includere `tests/unit/test_conversion_fix_r002.py` nei test mirati. Conservare
byte degli helper/config/test consumati e manifest prima/dopo, oltre agli hash.
Le receipt dei figli devono usare lo stesso interprete della wheel verificata.
La guardia attende anche writer successivi all'uscita del leader; un writer
entro cap e deadline deve completare normalmente.

Per aggiungere metadati header senza cambiare i contenuti approvati, preservare
prima il corpus e il packet precedenti. Controllare i nuovi header sul sorgente
con M; produrre il nuovo confronto con delta e checker indipendente. Usare
`derive_verification` con vecchio corpus/packet/risposta/verification e nuovo
corpus/packet: la receipt schema3 eredita la V reale soltanto sui contenuti
invariati, senza una nuova conferma umana. Non copiare VERIFIED su nuovi hash.
Non rigenerare PDF/canonici/render invariati. Se un contenuto visibile cambia o
l'invarianza non è dimostrabile, richiedere verifica mirata.

Passare `campaign freeze --verification /percorso/receipt-derivate` quando
la qualificazione di riferimento deve essere congelata prima dei motori.
Gli attempt_id della nuova campagna includono experiment_id: i precedenti
rimangono immutabili. Congelare attesi e input/helper byte prima del workload,
controllarli dopo; R expected-exit0 resta fisso, gli exit operativi restano dati.

Per rettificare osservazioni/audit/metriche senza motori, creare copie byte
identiche dei raw storici, nuove identità e nuovi observe/audit/score/final.
Pubblicare prima/dopo con collegamenti alle prove originali; engine_runs0 non
è un rerun della conversione. La provenienza helper storica non recuperabile
rimane un limite esplicito. Separare questa rettifica dalla nuova baseline.

La revisione r003 del misuratore conserva formule, unknown e risorse attraverso
HTML inline annidato e fallback passivi. Markup non bilanciato o risorse fuori
profilo rende il bundle INCOMPLETE; nessun target evasivo viene aperto. CER/WER
completi sono UNAVAILABLE anche per unknown testuale osservato unmatched; i
segmenti noti rimangono conteggi condizionati al matching strutturale, con
denominatori attesi invariati; non sono bound garantiti (rettifica r004).
Celle/caption/formule hanno domini locali distinti dal testo dei blocchi interi.
`WER_decomposition` espone edit aligned, omissioni e inserzioni strutturali con
ID/tipi/testo e candidate caption/titolo esatte, senza cambiare alignment.
Non sommare errori WER e gate caption/blocchi come perdite indipendenti. Per la
regola versionata e i limiti vedere il [protocollo](../../scripts/benchmarks/PROTOCOL.md).


## Misure e profili r004

Reader `observations-v4-html-coverage`: strutture di blocco inline unsupported
con foglie/raw conservati; residui table/figure espliciti, caption duplicate
senza scelta implicita, tutte le risorse auditate. `<` letterale in prosa resta
testo. Caption assente/vuota è equivalente nel solo dominio testuale; aggiunta
non vuota fallisce. Il [protocollo](../../scripts/benchmarks/PROTOCOL.md)
elenca i profili finiti e la politica di copertura.

Scorer `metrics-v4-conditional-and-bounds`: CER/WER completi restano None dove
il testo non è misurabile. Leggere `conditional_text_counts` e
`WER_decomposition` v2 come conteggi sotto il matching strutturale, anche per
`known_word_edits` compatibile; non sono bound garantiti. `text_lower_bound`
è separato: distanza globale esatta su flusso misurabile, zero garantito su
qualsiasi completamento del flusso non misurabile. Mantiene i denominatori
attesi, espone regola/scope/corrispondenze/ID e non modifica alignment o FAIL
per missing/pagina/celle/numeri/asset. La garanzia di zero deriva dalla
nonnegatività della distanza, non dall'interpretazione dell'unknown come
wildcard nel matching. Confrontare campi dello stesso tipo/versione.


## Correzioni r005: diagnosi HTML e domini testuali

Reader `observations-v5-html-diagnostics`: le diagnosi strutturali del parser
sono conservate indipendentemente dalla proiezione dei wrapper. Un contenitore
non chiuso o una chiusura incoerente conserva raw, offset e tipo della diagnosi:
audit INCOMPLETE, run/evaluate exit2. I wrapper bilanciati mantengono i contenuti
proiettati una sola volta. Gli heading HTML h1–h6 hanno coordinate HTML; soltanto
gli heading Markdown marcati `atx_heading` ricevono il mapping del prefisso ATX.
Heading HTML e code HTML sono host di `html-inline-v1`: strutture di blocco
annidate sono unsupported locali, conservando foglie note, formule e risorse.
Il codice Markdown letterale rimane inerte.

`html-tokens-v1` conserva commenti, dichiarazioni e processing instruction come
token fuori profilo, con raw e intervallo sorgente, all'inizio, in mezzo e in
coda. Non producono prosa o referenze dai caratteri racchiusi; un'immagine citata
in un commento non è una risorsa attiva. La diagnosi raggiunge audit/scorer e
impedisce COMPLETE/PASS. Le celle, caption e i residui conservano la posizione
locale del token. Letterali `<`, entità e HTML pulito mantengono il trattamento
precedente. Il profilo è finito, senza fetch o esecuzione dai dati.

**Dominio di `text_lower_bound`:** il campo limita esclusivamente la distanza
di edit del flusso globale dichiarato. NON è un limite inferiore di CER/WER
completi o di `conditional_text_counts`, che dipendono dal matching strutturale;
può superarli per ordine e separatori. Per esempio, due paragrafi invertiti
possono avere CER strutturale0 e distanza globale positiva; un'omissione può
aggiungere al flusso un separatore LF, escluso dal denominatore atteso.
`EXACT_GLOBAL_DISTANCE` è una misura esatta nel dominio globale misurabile;
`TRIVIAL_ZERO` è soltanto la garanzia conservativa nonnegativa sui completamenti
ammessi, non una misura completa0 né evidenza di fedeltà. Nessun cambiamento ad
alignment, normalizzazione, denominatori o calcolo della distanza r004.

## Rettifica r006: confini lessicali e contenuti inerti

Il reader corrente è `observations-v6-html-lexical-boundaries`. La descrizione
r005 di F25 era incompleta: CDATA con `>` interno e commenti attraverso blank/
marcatori Markdown potevano lasciare parte del corpo ai produttori successivi.
Gli stati INCOMPLETE già rilevati non rendevano corretti conteggi e risorse.
Il caso `</>` poteva invece risultare COMPLETE/PASS automatico senza diagnosi.
Le prove storiche restano conservate nel report r005; consultare i nuovi dati r006.

Ogni token del ramo finito conserva il costrutto chiuso completo oppure fino a
EOF. CDATA termina a `]]>`; commenti seguono `-->`, `--!>` e i rami abrupt del
tokenizer managed locale, compresa la sua precedenza effettiva fra chiusure.
Comment/declaration/unknown_decl/PI restano unsupported in `html-tokens-v1`:
audit INCOMPLETE, run/evaluate2. Il corpo di commenti e CDATA è inerte: zero
prosa/formule/reference/asset e nessuna apertura del target dal corpo. Un testo
presente soltanto nel commento non soddisfa il gold visibile né nasconde missing.

I confini dei produttori Markdown rispettano i token aperti, anche in celle,
caption, heading, liste, quote, note, wrapper e raw HTML fence. Raw, offset
Unicode/byte, righe fisiche e diagnostiche raggiungono audit/scorer una volta.
Le foglie con codice, formule, link e raw LaTeX conservano i propri delimitatori;
trim EOF, attributi, enfasi e label non possono troncare token inerti. Code inline
e fence Markdown letterali restano inerti. Il riepilogo senza foglie usa una reason
neutra e conserva le reason specifiche delle occorrenze. Nessun fetch o comando
dai dati; il subset non promette HTML universale.

F26 resta invariato: bound del flusso globale distinto da CER/WER strutturali e
conteggi condizionati, stessi algoritmi/normalizzazioni/denominatori. Il rescore
r006 crea nuove identità derivate dei 64 raw protetti, zero motori e nuova V0.

Report e dati r006 (archived: `documentation/benchmarks/2026-10-09-a002-r006-confini-lessicali-html.md`, tag `pre-docs-2026-10-18`).

## Rettifica r007: contesti Markdown e confini dei producer

Il reader corrente è `observations-v7-shared-markdown-contexts`. La chiusura
lessicale locale r006 non garantiva tutti i confini globali: lookahead di code,
math e tag poteva attraversare producer distinti o riprendere dentro fence;
autolink installava falso contesto HTML; metadata e comandi raw TeX consumati
inline potevano invece aprire token nello scanner. Sono F31–F33, con prima/dopo
reali nel nuovo report r007; report/dati r006 e FAIL/E9 restano immutati.

Il dominio è quello del producer effettivo prima del merge: paragrafo fino a
blank/marker di blocco, heading/cella/caption/label nel proprio valore, liste/
quote/note con le continuazioni realmente ammesse. Scanner e consumer condividono
atomi finiti code/math/autolink/link/span/metadata/command e pipe fuori dagli atomi.
Autolink non installa uno stack HTML; metadata sono opachi soltanto nell'host
sintattico riconosciuto; raw TeX unsupported resta locale, con ref/anchor separati.

Gli intervalli dei token inerti realmente riconosciuti sul sorgente arrivano ai
consumer anche attraverso merge, prefix stripping, celle/caption e label, inclusa
la label EOF senza destinazione. Un token chiuso intero o fino a EOF estende
l'unità Markdown; non estende il lookahead dei precedenti atomi nei suoi byte.
I consumer rimappano offset senza normalizzazione globale. Commenti/CDATA hanno
zero prosa/formule/reference/asset e aperture dal corpo, non soddisfano il gold
visibile. Comment/declaration/unknown_decl/PI restano unsupported html-tokens-v1,
audit INCOMPLETE e run/evaluate2. Codice letterale e metadata sicuri positivi
restano discriminanti; HTML vero/risorse attive/URI unsafe conservano le guardie.

È un subset finito, non HTML universale. F26/algoritmi/normalizzazioni/denominatori
immutati: bound globale distinto dai CER/WER strutturali e conteggi condizionati;
TRIVIAL_ZERO non è misura completa0. Nuovi derivati64 con conversioni r001/r002
originali preservate, nuova V0 e zero motori di campagna. Sonde CLI con writer
sintetico e gold PROVISIONAL: automatic PASS/INDETERMINATE/evaluate5, senza nuova V.

Report e dati r007 (archived: `documentation/benchmarks/2026-10-10-a002-r007-contesti-markdown.md`, tag `pre-docs-2026-10-18`).

## Rettifica r008: avanzamento e domini inline

Il reader r008 è `observations-v8-progress-and-inline-domains`; la sezione
r007 sopra documenta il giro storico e va letta con questa rettifica. R007
non chiudeva il dominio del raw LaTeX né tutte le interazioni nello stesso
blocco o le label code su soft-line. Le review r007 erano statiche: prima/dopo
r008 sono nuove prove autore, non runtime attribuito retroattivamente ai reviewer.

Raw fence LaTeX: l'intero corpo fino alla vera chiusura fisica/EOF è il dominio,
anche dopo blank/marker; token inerti attraversano false chiusure nel proprio
corpo. Ogni transizione dello scanner avanza strettamente. EOF conserva la
fence unknown e i token interni, senza rendere supportata tutta la TeX.

Delimitatori incompleti code/math/tag non saltano veri commenti/CDATA nello
stesso dominio. I consumer ricevono il token intero, corpo senza prosa/math/ref/
asset/aperture e missing/page2 indipendenti. Opening tag incompleto prima del
token resta diagnosticato `incomplete-start`; i tag riconosciuti sono validati
con lo stesso PassiveHTML prima di qualsiasi proiezione. Attributi quotati,
code/math chiusi, valuta/literal< e risorse attive restano discriminanti.

Gli intervalli code accettati dal walk label sono condivisi con lo scanner,
anche sui soft-line dei producer quote/list/note. Destinazione/titolo restano
separati; non si prende una chiusura da un blocco estraneo. Trim e prefix
stripping usano il ledger sorgente: niente riparsing di una riga parziale che
comincia dentro un literal. Raw/char/byte/testo/reference e blocchi esterni
sono verificati con Unicode, positivi e veri token EOF fuori dal codice.

Subset finito html-tokens-v1 unsupported, audit INCOMPLETE/run/evaluate2;
R PASS è isolamento/custodia, non qualità. CLI con writer sintetico e gold
PROVISIONAL: automatic PASS/qualità INDETERMINATE/evaluate5, senza nuovaV.
Riscoring64 con nuovi ID r008 e conversioni r001/r002 originali preservate:
zero motori/V nuova. Scorer/F26/normalizzazioni/denominatori invariati;
conteggi condizionati separati dal bound globale, TRIVIAL_ZERO non completo0.
L04outputUNAVAILABLE distinta da V13; L01/L02/L05/ChromeNONPASS/helper12gap restano.

Report e dati r008 (archived: `documentation/benchmarks/2026-10-10-a002-r008-avanzamento-scanner-e-domini-inline.md`, tag `pre-docs-2026-10-18`).

## Controllare un risultato con il reader r009

Questa sezione registra i controlli r009. Il reader corrente e le rettifiche
F43/F44 sono descritti nella sezione r010 successiva.
La rettifica r009 corregge la guardia HTML interrotta dagli errori dell'albero,
il falso tag incompleto con apice bare e la perdita delle label dopo stripping.
Le prove r008 rimangono riferite ai propri byte.

1. Leggere `audit.json`: `UNSAFE_REFERENCE/detail` precede unsupported quando
   esiste un'apertura unsafe reale, anche dopo duplicati o profondità eccessiva.
2. Controllare testo, reference e asset reali in `observations.json` e audit.
   Un commento successivo può lasciare il bundle INCOMPLETE anche se l'immagine
   precedente è conservata e verificata; non interpretare COMPLETE come fedeltà.
3. Nei trasporti quote/list/note confrontare `raw`, offset del blocco e
   `source_position.char/byte` con il sorgente UTF8. I marker dentro il code
   della label sono contenuto; un delimitatore in un altro producer resta esterno.
4. Leggere metriche ed exit separatamente. Un positivo sintetico PROVISIONAL
   ha automatic PASS, qualità INDETERMINATE e score/evaluate5; una perdita reale
   resta FAIL. Non assegnare V o fullPASS da un bound TRIVIAL_ZERO.

Per dettaglio dei confini, limiti e precedenze consultare la
[reference r009](../reference/fidelity-benchmark.md#reader-r009-contratti-storici-dei-confini)
e il report r009 (archived: `documentation/benchmarks/2026-10-10-a002-r009-confini-reader.md`, tag `pre-docs-2026-10-18`).
Le sonde CLI usano un writer sintetico dichiarato; le 64 misure derivate
riusano raw protetti e non eseguono nuovi motori o giudizi umani.

## Controllare i confini HTML e Markdown con il reader r010

Questa sezione conserva i controlli r010, confutati nelle interazioni F46–F48
e rettificati nella sezione r011. Usava i gate di installazione,
S/B/I e R correnti. Le review r009 hanno identificato tag nascosti dai
fallback e falsi unsafe nel code successivo a HTML bilanciato; i risultati
storici v9 non certificano questi confini.

1. Confrontare l'apertura HTML effettiva col suo dominio nel raw. Dopo un
   delimitatore incompleto o link malformato, traversal/script/event reali
   devono dare `UNSAFE_REFERENCE` con detail prima dell'unsupported. Se il
   fallback perde un'immagine sicura, deve restare almeno
   `UNSUPPORTED_RESOURCE_SYNTAX`, senza COMPLETE con risorsa scomparsa.
2. Per `<span>words.</span>` seguito da code chiuso, verificare il testo del
   literal, i blocchi esterni e l'assenza di reference/asset inventati. Nei
   positivi sintetici PROVISIONAL il risultato è run0, automatic PASS e
   qualità INDETERMINATE/score/evaluate5. Un vero tag unsafe fuori code deve
   ancora dare run/evaluate2 e score3.
3. Nei trasporti quote/list/note/cella verificare raw, coordinate char/UTF8 e
   ordine. Le chiusure di un altro producer non chiudono la foglia corrente;
   comment/CDATA conservano il corpo opaco, anche a EOF.
4. Leggere F36 come limite finito: B1–B4 possono essere UNSUPPORTED mentre
   `HTMLParser` puro riconosce aperture. Non estendere quel risultato a
   fallback Markdown né dichiarare parità universale col parser puro.

La [reference r010](../reference/fidelity-benchmark.md#reader-r010-confini-htmlmarkdown-inline)
e il report con dati (archived: `documentation/benchmarks/2026-10-10-a002-r010-confini-html-markdown.md`, tag `pre-docs-2026-10-18`)
identificano esempi, oracoli, CLI e limiti. Il writer delle sonde CLI è
sintetico dichiarato; il ricalcolo dei 64 raw è una nuova misura derivata,
senza motori né nuova V. TRIVIAL_ZERO non significa misura completa a zero;
L04 sull'output resta distinta da V13/A03 sul riferimento.


## Verificare ownership e metadati con il reader r011

Questa sezione conserva i controlli r011; le review hanno individuato
F50–F52. Usare la rettifica r012 successiva per il reader corrente.

1. Separare gli attributi del blocco heading dal testo `# … {…}` dentro la
   foglia del producer. Verificare H1/H6run2/evaluate2/detail unsafe, HC2
   metadata e H7 attributi `.c`. Per H3 il fallback sicuro deve almeno dare
   INCOMPLETE/UNSUPPORTED esplicito, senza COMPLETE con risorsa persa.
2. Nella label realmente accettata, verificare code chiuso su LF e formula
   raw integra, link, tre blocchi, coordinate char/UTF8 e nessuna lettura di
   img letterali. LF libera non diventa label. Il gold math include la
   formula indipendente; score5/INDETERMINATE non certifica il subset/L04.
   Comment/CDATA opachi non devono prestare chiusure al literal precedente.
3. Dentro l'owner HTML contare tutte le aperture effettive, compresa la
   prima unsafe interna, e verificare la diagnosi prima dell'unsafe esterno.
   Invertire l'ordine dei payload. Dopo chiusura dello span il code esterno
   rimane literal; una quote complessa già guardata non rilegge il raw HTML.
4. Controllare output/final ed exit OS reali; il writer CLI di questi casi
   è sintetico dichiarato, gold PROVISIONAL, senza motori o nuova V.
   Conservare H12FAIL previsti e prove precedenti invalidate. Ricalcolare
   i64raw protetti e verificare la catena originale/previousderived/nuova
   misura senza imporre conteggi. TRIVIAL_ZERO non equivale a fullPASS.

[Reference r011](../reference/fidelity-benchmark.md#reader-r011-ownership-delle-foglie-inline)
e report/dati (archived: `documentation/benchmarks/2026-10-10-a002-r011-ownership-inline.md`, tag `pre-docs-2026-10-18`)
identificano oracoli e limiti. F36 resta deliberatamente UNSUPPORTED; nessuna
parità universale HTML/CommonMark. Fallback observations-v1 è distinto dal
reader effettivo: controllare byte/R/snapshot. L04output distinta da V13/A03,
L01/L02/L05/NONPASSstorici e osservazioni non bloccanti restano dichiarati.


## Verificare label e raw nei container con il reader r012

Procedura storica: per guardie e fence applicare anche i controlli
[r013](#verificare-le-fonti-di-guardia-e-i-fence-con-r013).

Dopo managed3.12.13/S/B/I/R correnti, verificare il reader fisico
`observations-v12-shared-atoms` tramite byte e ricevute.

1. Provare una label accettata con backslash+LF, code su LF e una combinazione
   dei due. Per math chiusa confrontare formula raw, link, testo, ordine dei
   tre blocchi e coordinate char/UTF8. Il risultato sintetico è run0/evaluate5
   e qualità INDETERMINATE. Togliere il delimitatore finale: i tag reali devono
   ricevere guardia e dare UNSAFE con detail. LF libera e LF nel token opaco
   restano controlli distinti, senza prestito di chiusure dal token.
2. Inserire raw inline esatto `{=html}` nella label e due violazioni in ordini
   inversi. Controllare il detail della prima apertura, oltre all'exit2/2.
   Code ordinario resta letterale; owner HTML aperto possiede il contenuto.
   Le visite degli altri domini proseguono dopo prima unsafe o errore d'albero.
3. Nelle quote/list/note provare un fence `{=html}` con script, img evasiva e
   asset sicuro. I primi due danno UNSAFE con detail; il terzo resta
   INCOMPLETE/UNSUPPORTED poiché il raw block del container non è proiettato.
   Controllare la diagnosi e la posizione del raw, senza considerare `segments`
   una prova di copertura. Il fence ordinario chiuso mantiene il corpo letterale.
4. Leggere output/final ed exit OS, ricevute R e input congelati. Le CLI delle
   fixture sono writer sintetici dichiarati, senza conversioni nuove o V.
   Tenere separati H12FAIL previsti e fallimenti corretti. Il ricalcolo64 è
   una misura derivata: preservare l'originale conversion_final r001, il
   previous_derived_final e il nuovo risultato; non imporre conteggi di qualità.

[Reference r012](../reference/fidelity-benchmark.md#reader-r012-label-e-raw-nei-container)
e report con dati (archived: `documentation/benchmarks/2026-10-11-a002-r012-atomi-e-container.md`, tag `pre-docs-2026-10-18`)
identificano casi, decisioni e risultati. Le prove finite non certificano
Markdown/HTML universale, Pandoc sui formati raw esclusi o fedeltà semantica.
F36 e quattro osservazioni r010, L04 distinta da V13/A03, L01/L02/L05 e
NONPASS storici restano dichiarati.

## Verificare le fonti di guardia e i fence con r013

Procedura storica: integrare con i controlli r014 sotto.

Identificare `observations-v13-ordered-guards-fences` con byte, origini e
ricevute. Dopo ogni modifica ricostruire sdist/wheel, reinstallare la wheel
canonica, confrontare dieci moduli e helper effettivi e verificare S/B/I
prima di importare il workload sotto R. Un vecchio snapshot generale STALE
non trasferisce PASS alla nuova versione.

1. Congelare raw e attesi manuali prima del runtime. Provare HTML unsafe
   prima e dopo metadata attivi di heading/div/fence/figure e code/link/span
   inline, anche in cell/caption/quote/list/note. Registrare una singola
   observe con snippet, char/UTF8, sequenza e molteplicità esatte; eseguire
   audit separatamente. Verificare la coda anche dopo errore d’albero e
   il reset su una chiamata successiva sicura.
   Nei metadata su LF provare sia un separatore fra attributi sia un valore
   quotato multilinea: dopo stripping dei prefissi, confrontare la posizione
   di ciascun token attivo con il raw fisico, non con l’offset logico.
2. Variare contesto precedente, tilde/backtick, chiusura/EOF e container.
   Sotto R usare il Pandoc3.1.3 già installato (`-f markdown -t json`) per
   verificare l’owner del corpo discriminante. Un qualsiasi Code altrove
   nel documento non basta. Conservare input/AST/argv/versione/hash e le
   correzioni delle premesse; non derivare gli attesi dal reader provato.
3. Fuori da un vero owner code verificare script/URI con detail e l’immagine
   sicura come risorsa preservata o unknown esplicito con posizione. Separare
   il profilo raw HTML/LaTeX ricevuto dalla sintassi ordinaria; preservare
   LF label, F36, F44 e token opachi. Non aggiungere un secondo feed HTML
   indiscriminato ai residui Markdown.
4. Provare su copie private le due assunzioni sbagliate: attrs fail-fast e
   fence a ogni riga. Gli attesi devono rilevare errori di detail/visite e
   perdita asset/unsafe. Non alterare il corpus o i sorgenti per la prova.
5. Confrontare JSON osservazioni e19 campi metrici/gate separatamente. Il
   privato `_guard_coverage` deve sparire solo dalla serializzazione finale,
   mantenendo cell/location/reference. Controllare anche il limite A8:
   il reader precede il controllo audit dei path Markdown.

La consegna r013 (archived: `documentation/benchmarks/2026-10-11-a002-r013-guardia-e-fence.md`, tag `pre-docs-2026-10-18`)
contiene la tabella fonte → proprietà → prova e gli output eseguiti. La CLI
di prova usa writer sintetico dichiarato e gold PROVISIONAL; il rescore64
non avvia motori o nuove V. Tenere H12FAIL separato dai fallimenti ordinari
corretti. Suite veloce e matrice finita non sostituiscono nuove review,
collaudo completo o verifica semantica delle formule L04.


## Verificare accettazione e stato con r014

Dopo il preflight e R, confrontare lo stesso raw con reader e adapter V13
archiviati insieme: un vecchio reader importato con il nuovo adapter non
rappresenta la baseline originale. Conservare hash e copie prima dei cambi.

1. Per URI malformate, alternare script e risorse sicure/evasive prima e dopo
   l'errore. Verificare la prima diagnosi, ogni validazione effettiva e il reset;
   audit deve fornire INCOMPLETE/UNSAFE_REFERENCE, senza errore interno.
2. Per i fence, variare info valida/invalida e contesto esterno al producer:
   paragrafo attaccato/separato, quote/list/note lazy, figura. Verificare l'AST
   del corpo discriminante, conservando anche i veri Code inline backtick.
3. Per i metadata, provare gruppi validi/invalidi e HTML letterale nei valori
   in ogni producer. Confrontare intervalli, raw e coordinate UTF8/LF/CRLF;
   gli attributi attivi validi vanno rifiutati, le graffe invalide non nascondono tag.
4. Usare quattro copie private per smentire parsing URI sfuggito, qualsiasi
   info fence, stato azzerato nel chiamante e qualsiasi graffa metadata.
   Conservare diff, base e risultati; non mutare il prodotto o il corpus.

Gli esiti r014 (archived: `documentation/benchmarks/2026-10-11-a002-r014-contratti-sintassi.md`, tag `pre-docs-2026-10-18`)
distinguono attesi manuali, AST, writer sintetico e controlli sulle prove.
Per i confini consultare la [reference](../reference/fidelity-benchmark.md#reader-r014-accettazione-e-stato).
Non trasferire PASS da S/B/I o prove riferite a byte diversi.
<!-- r014: delta effettivi e attribuzione -->

Per confrontare i risultati r014 con r013 consulta anche i
delta attribuiti (archived: `documentation/benchmarks/2026-10-11-a002-r014-contratti-sintassi.md`, tag `pre-docs-2026-10-18`):
16 EPUB cambiano failures/details per la corretta lettura degli ID adiacenti,
senza modifiche ai valori numerici o ai gate. Otto TeX correggono soltanto le
classi dei div osservati. Non modificare il gold per annullare queste differenze.
