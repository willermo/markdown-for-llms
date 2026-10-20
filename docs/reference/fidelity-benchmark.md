# Reference del benchmark locale di conversione

**Stato2026-10-12: strumento sperimentale.** A002 è chiusa per accettazione
utente del perimetro ridotto, non per GOtecnico r014. Il reader può assegnare
COMPLETE/UNSAFE errati e omettere riferimenti nel modello osservato: risultati
provvisori nelle classi non convalidate, nessuna attestazione universale di
qualità/sicurezza. Codice e comandi non sono corretti da questa etichetta.
Usare gold/raw e valutazione umana; [debito e trigger obbligatorio](../../documentation/benchmarks/a002-residual-debt.md)
e [decisione parser](../../documentation/decisions/0008-maintained-parser-and-provisional-benchmark.md).
Le sezioni di implementazione precedenti descrivono prove storiche/domìni,
non la chiusura generale del debito. Nessuna nuova campagna avviata qui.


Per sviluppatori, protocollo `conversion-fidelity-v1`, schema1 per motori e
metriche; packet/risposte/verification documentali v2 hanno schema2. Gli helper in
`scripts/benchmarks/` misurano i percorsi leggeri locali della pipeline a script;
non sono servizi della nuova applicazione. [Protocollo](../../scripts/benchmarks/PROTOCOL.md)
per metriche, normalizzazioni, stati, limiti e responsabilità M/V.
[Guida pratica](../how-to/fidelity-benchmark.md) per eseguire una campagna.

## Launcher e comandi

Avviare con interprete verificato `-I -B` e percorso assoluto del launcher.
Aggiunge a sys.path solo la directory fidata degli helper, non la root prodotto.

| Script | Comando | Opzioni obbligatorie principali |
| --- | --- | --- |
| benchmark | inventory | --corpus |
| benchmark | profile | --format, --engine, --output |
| benchmark | run | --corpus, --document, --engine, --config, --output, --experiment-id, --attempt-id, --environment, --installation |
| benchmark | observe | --bundle, --output |
| benchmark | score | --gold, --observations, --audit, --assets, --format, --output; --verification facoltativo |
| benchmark | evaluate | opzioni run più --gold e --assets; --verification facoltativo |
| benchmark | summarize | --records (uno o più final.json), --output |
| campaign | freeze | --corpus, --stage, --output, --experiment-id; --groups facoltativo per rerun dipendente |
| campaign | run-group | --manifest, --group, --output |
| campaign | summarize | --manifest, --groups (tutti i group.json dichiarati, ordinati), --output |
| contracts | — | --case, --corpus, --output, --environment, --installation |
| inventory | capture / verify | capture: --repo, --output, --extra ripetibile; verify: manifest posizionale |
| prepare | — | --repo, --output, --label, --target |
| generate_corpus | — | --spec, --output, --environment |
| corpus | — | --first, --second, --destination, --external |
| reference_check | — | --corpus, --manifest, --output, --environment |
| packet | build | --corpus, --mechanical, --control-m, --output, --environment; --epub-render-packet facoltativo |
| comparison | — | --corpus, --mechanical, --control-m, --epub-packet, --output |
| comparison_check | — | --packet, --corpus, --output (nuovo file JSON) |
| verification | — | --packet, --response (response.json importato, ripetibile in v2), --corpus, --output |
| output_packet | — | --manifest, --summary, --output |
| packet | import-response | --packet, --response, --packet-sha256, --output; v2 richiede --author; --received-at e --receipt-method facoltativi |

Le directory di output devono essere nuove; overwrite/ripresa implicita non
sono ammessi. Il file environment è una mappa di stringhe o un target con
campo environment. L'installazione è la receipt I della wheel canonica,
controllata sullo stesso interprete dei figli. Prepare è il launcher operativo
**locale di questa run**: consuma keeper a001 già presenti, offline; non è
un installer portabile per un clone privo di quei keeper.

## Registry, profilo e corpus

| engine_id | Formati | Metodo |
| --- | --- | --- |
| txt-literal-v1 | txt UTF-8 con LF finale | fence più lunga di ogni run di backtick, payload esatto |
| pandoc-direct-v1 | docx, epub, rtf, tex | /usr/bin/pandoc assoluto/hashato, nessun filtro/template |
| pandoc-legacy-v1 | docx, epub, rtf | PandocConverter della wheel installata, child -I -B |

Gli otto group_id sono direct-txt/docx/epub/rtf/tex e legacy-docx/epub/rtf.
Freeze crea quattro attempt_id per percorso prima dei motori e protegge per
hash tutti i canonici, asset, include, gold, profili e installazione. Il manifest
non congela la page cache OS; ogni tentativo ha processo/workspace nuovo.
Il profilo bitset completo è in config, con22 enabled e complemento disabled.

Corpus permanente in `tests/fixtures/conversion-corpus`: dieci documenti
sintetici originali MIT, notice font separate, trascrizioni/AST/celle manuali,
controlli K01–K03. PDF/OCR non eseguiti nella baseline. TeX ha due include
letterali e non viene compilato. Numeri annotati nel testo e celle; numeri/operatori
math hanno il denominatore formula separato. Nessuna soglia OCR universale.

## Receipt e interpretazione

`run.json` registra identità, conversion/bundle, exit, actual argv, risorse,
audit e tempi. `observations.json` è gold-blind con raw SHA256 e offset.
`metrics.json` conserva denominatori, alignment, totals, failure/unavailable,
diff localizzati, CER/WER, asset content verification. `final.json` collega
per hash questi artefatti. `group.json` richiede quattro final validi e conserva
exit originali; il riepilogo richiede tutti i gruppi dichiarati. Il default è otto; un rerun
dipendente esplicito conserva quattro tentativi per ciascun percorso selezionato.

Exit benchmark evaluate2/3/5 può essere un esito legittimo rendicontato dal
harness con exit0. Un errore interno/preflight/schema/hash/record assente non
entra in questa categoria. R ha sempre expected-exit0. H01–H08 impiegano stub
dichiarati; H09–H16 richiedono NON PASS secondo la causa. Un riepilogo con exit0
non rende qualityPASS o conversion_batchCOMPLETE. A10 richiede quattro veri
C02/C03 Pandoc con figura attesa presente, referenziata e contenuto verificato.

Legacy: argv Pandoc interno ricostruito, engine_exit null se API non lo espone,
Markdown serializzato dal testo restituito dopo rimozione del temp originale,
API images distinta da inventory dei file reali. Non attribuire questa
configurazione extract-media al default legacy; latenza child/API distinta
dalla latenza Pandoc diretta.

## Gold e giudizi umani

Il packet documentale v2 contiene tredici coppie complete: originale effettivo
(o pagine/capitoli actual) e riferimento indipendente dal motore, più .md con
asset locali. Formule MathML native, celle vuote vere, figure/caption, note e
riferimenti sono visibili. Locator/AST/hash/M restano in `technical/`; niente
JavaScript/CDN. `comparison_check` legge HTML e Markdown effettivi con una
logica separata dal serializer e verifica contenuto, ordine, asset e sorgenti.
M canonico resta distinto: caption/celle/figure/target sui contenitori, AST
manuale contro fonte TeX, campi PDF contro locatori ed estrattori indipendenti.

Una risposta naturale per documento, ad esempio «C02: corrisponde», è sufficiente.
`response.json` schema2 dichiara granularità documentale: testo originale in
`received.txt` con SHA, autore/ora ricezione attribuiti dal chiamante e relativi
limiti, packet e coppia SHA, revisione, ambito completo, decisione/differenze.
Non contiene firme per blocco né `accepted_judgments`. Il digest della coppia
lega tutte le sorgenti/include/render/asset, il riferimento HTML/.md/mappa e il
CSS. Esclude la sola revisione amministrativa globale: prove di coppie con byte
identici possono restare attribuite alla revisione realmente vista.

«Tutti i documenti corrispondono» è una dichiarazione collettiva completa sul
packet. `statements` conserva un solo testo originale con `document_ids` e
`attribution=explicit_all_documents_statement`; le decisioni documentali
dichiarano `response_attribution=projection_of_single_explicit_all_documents_statement`.
Spazi e articolo attaccato («idocumenti») sono tollerati nell'interpretazione,
con byte originali immutati. Eccezioni, dubbi o approvazioni di sottoinsiemi non
diventano accettazioni collettive; la riclassificazione verifica anche l'ambito.

ACCEPTED richiede frase esplicita completa; «ok» resta PENDING, impossibilità o
dubbio AMBIGUOUS, differenze REJECTED, identità diverse STALE. Duplicate sullo
stesso documento sono AMBIGUOUS. ID ignoti restano non attribuiti. Un documento
respinto non annulla ACCEPTED sugli altri. `verification --response` ripetibile
conserva la storia: risposte successive esplicite sostituiscono quelle sullo
stesso documento, assenze non cancellano giudizi precedenti.

VERIFIED richiede M required PASS, vista completa PASS e V documentale esplicita
ACCEPTED sugli stessi hash/ambito; tutti tredici VERIFIED per A03. Importazione
riclassificata dai byte originali e hash della receipt impediscono di alterare
la decisione nel JSON. Non è autenticazione crittografica dell'autore. Le
annotazioni originali restano immutabili e la receipt separata può essere
passata a score/evaluate. Packet/import schema1 TSV restano compatibili per
artefatti storici, senza convertirli artificialmente in firme v2.

L04 per tipografia/semantica delle
**formule output** resta UNAVAILABLE se quei giudizi non sono assegnati,
anche quando il confronto strutturale passa. La durata concordata non è V.

## Limiti di misurazione

Conversion/audit/score/end-to-end sono tempi distinti. wait4 user/system e
maxrss individuale/reaped non coincidono con sampled tree RSS10ms: shared
pages, processi brevi, PID mancati e peak tra campioni sono riportati. Sampled
RSS stop6GiB, stream1MiB, file32MiB; nessun hard cap RAM aggregato certificato.
Solo Linux/toolchain locale osservata; niente certificazione GPU/altri OS.

## Profili r002 e receipt derivate

Reader `observations-v2-typed-inline`, scorer `metrics-v2-local-typed`:
contenuti tipizzati con raw/offset in celle e caption, link/immagini con titolo
separato, raw inline unsupported distinto dal code. L'audit vede ogni risorsa;
lo scorer confronta componenti inline e asset atteso senza doppio conteggio.
Gold/spec rev5 esplicitano header semantico da sorgente/policy; il placeholder
pipe vuoto per tabelle headerless non è corpo. Caption sono incluse anche nelle
trascrizioni. Le equivalenze AST finite valgono per alignment formula; WER
include inserzioni unmatched con denominatore atteso invariato. Unknown viene
attribuito soltanto localmente secondo `exact-visible-unique-v1`, senza
nascondere omissioni o pagine. Vedere le regole finite nel protocollo.

Opzioni aggiuntive:

| Script | Opzioni |
| --- | --- |
| campaign freeze | `--verification` directory receipt congelate prima dei motori |
| comparison | `--metadata-delta` delta verificabile rispetto al gold precedente |
| derive_verification | `--old-corpus`, `--corpus`, `--old-packet`, `--packet`, `--response`, `--old-verification`, `--output` |

`derive_verification` produce schema3 `derived_reference_content_qualification`.
Conserva risposta V originale, packet/pair/gold precedenti, nuovo M e checker
indipendente vecchie/nuove viste. `new_human_confirmation=false` rende esplicita
l'inheritance della V sul contenuto invariato; header tecnico è qualificato da
M. Una receipt VERIFIED derivata non è un nuovo giudizio dell'utente. L04 resta
UNAVAILABLE. Una modifica visibile sostanziale richiede perimetro umano mirato.

Il PATH effettivo Pandoc ammesso è `/usr/bin`: token legacy e percorso diretto
hanno realpath/hash/versione coerenti prima/dopo. I payload passivi fuori dal
subset HTML/CSS/SVG/ZIP sono INVALID_INPUT prima del tool. Il supervisore
risorse conserva propria config/receipt e tutta la discendenza fino a deadline
/reap/chiusura writer, senza segnali a figli concorrenti estranei.

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

## Reader r009: contratti storici dei confini

Il contratto r009 è storico: le review successive hanno individuato F43/F44,
rettificati nella sezione r010. `observations-v9-reader-boundaries` rettificava le affermazioni r008 sulla
guardia completa e sul dominio delle label dopo stripping. Il contratto
lessicale condiviso e la matrice finita sono nel
[protocollo r009](../../scripts/benchmarks/PROTOCOL.md#rettifica-r009-confini-del-reader-e-precedenza-della-guardia).

| Condizione | Osservazione e audit | Exit operativo/qualità |
| --- | --- | --- |
| Apertura unsafe dopo duplicati/depth/limite nodi | Prima diagnosi PassiveHTML in ordine sorgente, `UNSAFE_REFERENCE` con detail | run/evaluate2, score3/FAIL |
| Apice o virgolette interne a valore bare | Fine tag secondo tokenizer3.12.13; tag seguente validato e osservato | Dipende dalle risorse e dagli altri token reali |
| Token-looking bytes in valore attributo | Letterali, senza foglie/reference dal valore | Non introducono una nuova risorsa |
| Vero prefisso incompleto prima di token inerte | Diagnosi locale `incomplete-start` e token intero | INCOMPLETE/UNSUPPORTED, run/evaluate2 |
| Immagine reale prima di commento/CDATA/PI/declaration | URI/hash/dimensioni dell'asset e testo noti conservati; token opaco | Bundle ancora INCOMPLETE per il token |
| Label con code e marker nelle continuazioni quote/list/note | Dominio del producer trasportato, reference unica e coordinate originali | Nei positivi PROVISIONAL automatic PASS, score/evaluate5 |
| Chiusura in un producer successivo | Nessun prestito del delimitatore; blocchi esterni conservati | Unknown/perdite restano espliciti |

La guardia non costruisce l'albero e prosegue dopo la prima violazione.
La proiezione mantiene depth16 e massimo20000nodi; file32MiB. La mappa interna
dei prefissi serve `source_position.char/byte` senza modificare raw o UTF8.
Le prove finite non certificano un parser universale. Nessun cambiamento a
scorer, denominatori, F26 o qualificazione V dei riferimenti.

Report e dati r009 (archived: `documentation/benchmarks/2026-10-10-a002-r009-confini-reader.md`, tag `pre-docs-2026-10-18`).

## Reader r010: confini HTML/Markdown inline

Il contratto r010 è storico: le review hanno dimostrato F46–F48,
rettificati nella sezione r011. Il
[protocollo r010](../../scripts/benchmarks/PROTOCOL.md#rettifica-r010-domini-html-e-markdown-nelle-foglie-inline)
definisce i domini finiti. Il lexer fornisce aperture e intervalli dei subtree
HTML alla guardia e alla proiezione. Unknown Markdown non sopprime la guardia;
code/math chiusi restano letterali fuori dall'owner HTML. Il parser dell'albero
riceve la sola foglia HTML, compreso il subtree bilanciato, senza la coda
Markdown. Producer e mapping sorgente r009 mantengono i loro confini.

| Caso | Risultato richiesto |
| --- | --- |
| Unclosed code/math o broken link seguito da HTML unsafe reale | Audit INCOMPLETE/UNSAFE_REFERENCE e detail; score3, run/evaluate2 |
| Stesso fallback e risorsa sicura non rappresentabile | INCOMPLETE/UNSUPPORTED_RESOURCE_SYNTAX con diagnosi esplicita |
| `<span>` bilanciato e code chiuso con img/script/onclick literal, nei due ordini | Testo e blocchi esterni preservati, nessuna reference/asset; PROVISIONAL automatic PASS, qualità INDETERMINATE/score5 |
| Duplicati/depth HTML con literal chiuso esterno | Errore unsupported; il literal non introduce unsafe |
| HTML unsafe reale dopo un errore di proiezione | Prima violazione in ordine sorgente, senza fermare le visite successive |
| Token reale chiuso/EOF o chiusura in altro producer | Corpo opaco e coordinate originali; nessun prestito di chiusure |

F36 è una divergenza accettata: B1–B4 con opener in posizione sintattica del
prefisso possono produrre tag/attributi nel tokenizer puro, ma restano token
opachi e UNSUPPORTED nel profilo finito. Non c'è un secondo feed puro come
gate. Questo limite non ammette risorse perse con COMPLETE nei fallback
Markdown. Le prove finite non attestano ogni combinazione HTML/CommonMark.

Il report r010 (archived: `documentation/benchmarks/2026-10-10-a002-r010-confini-html-markdown.md`, tag `pre-docs-2026-10-18`)
registra matrice, oracoli originali, CLI con exit OS, regressioni e ricalcolo
storico. Nessun cambiamento a scorer/F26, denominatori o V; conteggi
condizionati e bound globale non sono intercambiabili. L04 resta UNAVAILABLE
sull'output, V13/A03 riguarda il riferimento; gli altri limiti storici restano.


## Reader r011: ownership delle foglie inline

`observations-v11-inline-ownership` (contratto storico, rettificato in r012), [contratto r011](../../scripts/benchmarks/PROTOCOL.md#rettifica-r011-ownership-e-metadati-inline).

| Dominio | Regola verificata |
| --- | --- |
| Blocco heading | Possiede gli attributi finali; HC2 mantiene metadata |
| Foglia già prodotta | `# … {…}` dopo stripping rimane testo; H7 distingue `.c` del blocco |
| Label finita | Code chiuso attraversa LF; la math chiusa mantiene corpo opaco, formula/link e char/UTF8. LF libera non è una label accettata |
| Token opaco | Non presta chiusure code/math; ricorsione conserva barriera sorgente |
| Owner HTML aperto | Backtick interni sono testo HTML, tutti gli start validati e prima unsafe stabile |
| Owner chiuso | Code esterno torna inerte; nessun feed HTML della coda o del raw già proiettato |
| Fallback incompleto | Unsafe reale prima di unsupported; H3 sicuro almeno INCOMPLETE/UNSUPPORTED esplicito |

F47 ha gold lessicale manuale con formula, score5/INDETERMINATE per subset
math unsupported e L04. F36 rimane divergenza deliberata/UNSUPPORTED.
Limiti backtick, raw fuori profilo e comment timing sono ricevuti non
bloccanti, senza renderer/ottimizzazione. Fallback observations-v1 distinto
dall'ID del reader eseguito; usare byte/R/gate per attribuirlo.
Report r011 (archived: `documentation/benchmarks/2026-10-10-a002-r011-ownership-inline.md`, tag `pre-docs-2026-10-18`)
confronta stessi153oracoli, CLI reali, regressioni e64misure senza motori/V.
Counts condizionati/globalbound/full distinti, TRIVIAL_ZERO non fullPASS;
L04 distinta da V13/A03, L01/L02/L05 eNONPASSstorici preservati.


## Reader r012: label e raw nei container

Contratto storico, rettificato per ordine completo della guardia e proprietà
dei fence dalla [reference r013](#reader-r013-guardie-fence-e-privati).

Il [protocollo r012](../../scripts/benchmarks/PROTOCOL.md#rettifica-r012-decisioni-di-atomo-e-raw-nei-container)
definisce `observations-v12-shared-atoms`. Descrittori interni condivisi
contengono tipo/intervallo/body/metadata/owner; le ricorsioni ricevono un
ritaglio delle stesse decisioni, con la source-map del producer.

| Classe | Contratto | Risultato operativo |
| --- | --- | --- |
| Label con LF in code chiuso o escape+LF, math chiusa | Formula raw opaca, link/testo/blocchi/char/UTF8 conservati | Nei gold manuali PROVISIONAL run0/evaluate5, qualità INDETERMINATE |
| LF di token comment/CDATA, delimitatore aperto | Token opaco non presta chiusure; tag reale fuori token visitato | Unsupported per token; unsafe reale con detail |
| Raw inline esatto `{=html}` nella label | Metadata appartiene all'atomo; guardia del corpo prima della proiezione | Prima unsafe globale, run/evaluate2 |
| Code ordinario o code dopo chiusura owner HTML | Corpo letterale, nessuna risorsa inventata | Audit coerente con le altre foglie |
| Fence `{=html}` in quote/list/note | Corpo HTML guardato; struttura raw del container non proiettata | Unsafe2/2 oppure sicuro INCOMPLETE/UNSUPPORTED2/2 |
| Fence normale chiuso nel container | Corpo letterale posseduto dal blocco | Nessuna apertura dai tag letterali |
| Proiezione incompleta o errore d'albero | Copertura del dominio distinta da segments; residuo unknown esplicito | Nessun COMPLETE con risorsa persa; guardie successive continuano |

Le LF accettate sono decise una volta dal walk della label; lookup per indice
sostituisce il loop LF/code ripetuto. Non è una prova di costo lineare globale.
Le misure finite registrano input, costo osservato, hardware/cache e limiti;
CPU lower/RSS sampled non certificano upper. Source-map e raw restano intatti.
Nessuno schema pubblico o parser universale nuovo. F36B1–B4, F40/F41/F46,
scorer/F26, V13/A03 e denominatori conservati. Formati html5/html-space,
lunghezza run backtick, costo storico commenti e fallback IDv1 restano limiti
r010. L04outputUNAVAILABLE distinta, L01/L02NOT_MEASURED, L05/NONPASS storici.

Report e dati r012 (archived: `documentation/benchmarks/2026-10-11-a002-r012-atomi-e-container.md`, tag `pre-docs-2026-10-18`)
riportano le prove effettive e i recuperi; nessun GO autore o PASS retroattivo
per le tracce statiche r011.

## Reader r013: guardie, fence e privati

Contratto storico, rettificato da r014 sotto.
`observations-v13-ordered-guards-fences`; [contratto r013](../../scripts/benchmarks/PROTOCOL.md#rettifica-r013-fonti-di-guardia-e-contesto-dei-fence).

| Elemento | Comportamento |
| --- | --- |
| HTML e metadata Markdown attivi | Eventi deduplicati con posizione assoluta; tutti visitati in ordine sorgente, prima Unsafe del reader come detail |
| Attributi attivi / errore d’albero | Attributi esclusi dai valori; errore gestito non tronca la guardia della coda |
| Metadata su LF | LF/CRLF separano token fuori da valori quotati; mappa per carattere conserva le posizioni fisiche dopo stripping quote/list/note |
| Reader/audit | Path Markdown verificati dall’audit dopo observe: limite A8, nessun ordine unsafe globale fra i due |
| Fence tilde | Richiede inizio o separazione rispetto al paragrafo precedente |
| Fence backtick ordinario chiuso | Può interrompere testo; Code inline iniziale nelle liste conserva lo stato pertinente al blocco successivo |
| Fence ordinario EOF | Nessuna ownership code; HTML reale guardato, residuo non rappresentato esplicito |
| Raw esatto HTML/LaTeX | Contratti ricevuti conservati, incluse differenze finite da Pandoc; raw container non proiettabile unsupported |
| Asset sicuro fuori code | Reference/asset preservato o unknown con raw/posizione/resource_syntax; mai COMPLETE con risorsa persa |
| `_guard_coverage` | Privato fino al consumo node/located/source-map, assente dal JSON finale anche nested |

La precedenza riguarda guardie del reader e non l’ordine di costruzione
dell’albero. Gli owner code/math veri e i token opachi mantengono inertness;
F36 e F44 non cambiano. La prima riga heading di una nota può essere Plain
secondo Pandoc3.1.3: il contesto non deriva dal solo marker dopo stripping.

Prove e limiti r013 (archived: `documentation/benchmarks/2026-10-11-a002-r013-guardia-e-fence.md`, tag `pre-docs-2026-10-18`):
285 casi principali e16 supplementari LF, 199 AST principali e16 AST attrs,
due controprove, regressioni, CLI sintetica
e rescore64. La sola rimozione di640 privati nested non cambia i19 campi
metrici/gate. Restano L04outputUNAVAILABLE, V13/A03 sul riferimento e i
limiti/NONPASS ricevuti. Nessuna equivalenza generale con Pandoc o qualità
semantica certificata; nuove review e arbitrato ancora necessari.


## Reader r014: accettazione e stato

`observations-v14-syntax-contracts`; il
[protocollo](../../scripts/benchmarks/PROTOCOL.md#rettifica-r014-accettazione-sintattica-e-rifiuti-uri)
definisce i confini dell'adapter e della grammatica finita.

| Decisione | Input consumato | Esito |
| --- | --- | --- |
| URI | Stringa e allow_external, parsing pinned | Path/None oppure Unsafe per rifiuti di policy e ValueError del parser |
| Metadata | Intervallo, token accettati, bound e host | Solo il gruppo interamente valido è opaco; HTML nei valori resta valore |
| Fence | Info, marker, chiusura, stato prima della riga | Code soltanto con owner dimostrato; fallback inline conserva la propria grammatica |
| Blocco radice | Record fisico dello scanner | Producer consuma lo stesso stato, anche dopo quote/note/figura |
| Container | Opener ammesso, host e mappa dei caratteri | Nessun inizio di blocco inventato dal solo stripping |

Non sono estesi l'ordine reader/audit A8, il subset raw/F36, le quattro
osservazioni r010 o la verifica semantica delle formule. L04outputUNAVAILABLE
resta distinta da V13/A03; L01/L02/L05 e NONPASS storici sono conservati.
Il report r014 (archived: `documentation/benchmarks/2026-10-11-a002-r014-contratti-sintassi.md`, tag `pre-docs-2026-10-18`)
contiene stato delle prove, correzione di HC2 e citazione precisa della
baseline132/133. Non certifica interoperabilità universale né un GO autore.
<!-- r014: semantica verificata dei token adiacenti -->

Nel contratto metadata r014 i token possono essere adiacenti:
`a="x"b="y"` è due coppie e `#chapter1.xhtml#start` è due dichiarazioni ID,
ultima `start`, secondo l'AST pinned. I target URI rimangono stringhe distinte.
I risultati r014 (archived: `documentation/benchmarks/2026-10-11-a002-r014-contratti-sintassi.md`, tag `pre-docs-2026-10-18`)
distinguono 16 delta EPUB di failures/details da zero delta numerici/gate e
dagli otto cambi delle sole classi div TeX. Non costituiscono nuova conversione.
