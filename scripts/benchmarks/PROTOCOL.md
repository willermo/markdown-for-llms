# Protocollo conversion-fidelity-v1

## Stato operativo2026-10-12 — accettazione sperimentale

A002 è chiusa per disposizione utente sul perimetro ridotto. Le garanzie
generali del reader/guardia restano provvisorie nelle classi non convalidate;
COMPLETE/UNSAFE non attestano universalmente qualità o sicurezza. Nessun
GO tecnico r014 retroattivo: ClaudeNO_GO statico eGPTreviewincompleta conservati.
Codice/prove non modificati; [debito permanente](../../documentation/benchmarks/a002-residual-debt.md)
e [ADR0008](../../documentation/decisions/0008-maintained-parser-and-provisional-benchmark.md)
vincolano il seguito. Le rettifiche storiche sotto valgono nei domini provati.


Protocollo di sviluppo a002, schema1 per conversioni/metriche; schema2 per
confronto e risposta V documentale. Misura la pipeline a script disponibile;
non definisce API, storage o modello documentale della futura applicazione.
La [reference operativa](../../docs/reference/benchmark-fedelta.md) descrive
CLI e artefatti. Il [mandato della run](../../documentation/development/next-run-a002-corpus.md)
identifica il perimetro; i report registrano quali prove sono effettivamente fatte.

## Identità e processi

Un experiment contiene document_id e attempt_id distinti, directory esclusiva,
source SHA256, gold SHA256, tool assoluto/hash/versione, config canonica SHA256,
protocollo, reader e wheel installata. `started.json` precede il motore.
`run.json`, `audit.json`, raw ed errori sono conservati; `final.json` è creato
atomicamente solo dopo observe/score e audit degli hash. Un nome di file non
identifica una prova. Rerun e fault injection usano directory nuove.

Run riceve un descrittore source senza gold/blocks. Observe legge solo bytes del
bundle: raw UTF-8, intervalli byte/char/line, blocchi, celle, segmenti/formule,
anchor e URI, unknown. Score è un altro processo e riceve il gold manuale.
Cambiare il gold non può cambiare observations: digest verificato. Gli asset
sono confrontati soltanto nello scorer, per SHA originale o pixel RGBA senza
perdite; nessun output viene corretto usando il gold.

## Writer e reader

`pandoc-md-bench-v1` usa l'inventario completo Markdown di Pandoc 3.1.3:
22 estensioni enabled in `profile.py`, tutte le altre disabled, ogni flag in
ordine lessicografico. Il file d'inventario, il bitset risolto e gli helper
sono hashati. Smart, auto identifiers, grid/simple/multiline tables sono off.
Opzioni ordinate: reader docx/epub/rtf/latex, writer completo, wrap none,
headings atx, data-dir vuota, resource-path source, extract-media relativo media.
Il reader Markdown di un controllo indipendente può disabilitare smart/ID
per isolare il comportamento writer; non cambia i reader della campagna.

Reader finito: paragraph/heading, enfasi, list item semplice, blockquote semplice,
link/image, span/attrs/div con contesto ordinato, fence/code prima di math,
pipe table con celle vuote/pipe escapata, caption adiacente, note e continuazioni,
HTML passivo, raw TeX label/ref/eqref, raw html/latex. Quote complesse, spanning
HTML tables, MathML, macro raw e altri formati sono unknown con contenuto,
non stringhe vuote. Anchor vuoto ha nodo raw proprio ed è legato alla successiva
foglia senza generarne il nome. Raw/offset non sono una serializzazione corretta
creata dal gold. Limiti: file 32MiB, depth16, formula4096char, matrice256celle.

`math-structure-v1` conserva Seq/Arg, Symbol/Number lessicale, Operator,
Scripts, Fraction, Sum, Text letterale, Aligned e Matrix con righe/celle e
delimitatori. Equivalenze: singolo argomento TeX/braced, ordine sub/sup,
spacing tokenizzato e meno Unicode/ASCII **solo math**, bmatrix o matrix
racchiusa in quadre. Gruppi non vengono cancellati; spacing non unisce numeri;
Text non normalizzato. Nessun CAS, float, semantica/algebra inferita o macro
espansa. Unsupported conserva raw/intervallo e struttura unavailable.

## Metriche e denominatori

Gold nasce da sorgenti/AST manuali, mai dall'output del motore. Allineamento
one-to-one: tipo, similarità minima .75, landmark visibili, margine .15;
monotone DP e matching libero separato per ordine a coppie. Doppioni,
missing/unexpected, ambiguità senza landmark e perdita pagina sono espliciti.
Denominatore ordine n(n−1)/2 comprende le coppie non risolte.

Testo: CER/WER Levenshtein; CRLF/LF e soft wrap→spazio solo prosa; raw edit
separato. Code, numeri e celle rimangono esatti. Nessun NFKC/casefold/float.
Numeri: occorrenza lessicale in coordinate di testo/cella; zeri/esponenti e
segni mantenuti. Formule: diff AST localizzato, raw diverso diagnostico fuori
CER prosa. Celle: coordinate e shape, vuoto presente distinto da assente.
Figure: immagine referenziata, contenuto atteso verificato e caption; riferimenti:
URI/fragment esatti, note con mapping bijettivo. Soglie zero-loss su questo
corpus sintetico; non sono soglie universali OCR. Missing/unknown mantengono
il denominatore. Confronti sopra 4 milioni di celle Levenshtein unavailable.

## Stati ed exit

| Dimensione | Valori |
| --- | --- |
| conversion | SUCCESS, ERROR, TIMEOUT, EMPTY, INVALID_INPUT, INTERRUPTED |
| bundle | COMPLETE, INCOMPLETE, UNAVAILABLE |
| quality | PASS, FAIL, INDETERMINATE |
| conversion_batch | COMPLETE, PARTIAL, FAILED, INTERRUPTED |
| execution_complete | tutti i record attesi validi, distinto da fedeltà |
| gold | PROVISIONAL/PENDING, VERIFIED, REJECTED, AMBIGUOUS, STALE |

Run exit0 success+bundle completo;2 errore/timeout/empty/unsafe/incompleto;
4 contratto/preflight/input invalido/internal error;130 interruzione.
Score exit0 tutti required passano;3 perdita critica;5 unavailable/indeterminate;
4 errore interno/contratto. Evaluate precedenza4→130(interruzione)→2→3→5→0.
FAIL critico precede indeterminate, con entrambi i dettagli conservati.
Summarize exit0 indica che ha letto/controllato record, non qualityPASS.

Harness congela otto percorsi × quattro tentativi e attesi **prima** del workload;
R expected-exit0 sempre. Un rerun dipendente esplicito seleziona un sottoinsieme
del registry mantenendo quattro tentativi per percorso e nuove identità;
non è una nuova campagna completa di otto percorsi. Evaluate2/3/5 legittimi sono dati del motore e possono
produrre harness0 con batchPARTIAL/qualityFAIL. Input/hash/schema/record/internal
inatteso→harness4, assert/pytestFAIL→1, deadline/monitor→2, interrupt→130.
H01–H08 sono controlli espliciti con stub; H09–H16 iniettano difetti e richiedono
R NON PASS. Il runner non deve trasformare un errore arbitrario in harness0.
Nessuno stub soddisfa il requisito A10 di conversione Pandoc con figura reale.

## Sicurezza e risorse

Registry solo TXT literal, Pandoc diretto e PandocConverter installato;
nessun PDF/OCR/provider remoto implicito. Argv assoluti, shell=False, ambiente
chiuso. Input ZIP/XML/HTML/CSS/SVG/passive validati prima del tool; niente DTD,
entity, active HTML, risorse remote, path encoded/assoluti/../, symlink/hardlink,
special files. ZIP256membri,16MiBcompressed/64MiBexpanded/ratio100. Auditor
lstat inerte prima di aprire asset, O_NOFOLLOW e controlli inode/mutazione.
TeX non compilato: due include letterali allowlisted, cicli/duplicati negati,
256KiB expanded, package/class/commands finiti. Non si eliminano wrapper nel
codice/LaTeX/URI tramite regex globali. La protezione non è una garanzia
universale filesystem del tool: R attesta rete/socket/namespace del target.

Deadline iniziali: motore60s, harness400s per gruppo, R600s (cap900). Stream
1MiB, file32MiB, sampled RSS stop6GiB; nessun hard cap aggregato RAM attestato.
Monotonic elapsed separa conversion/audit/scoring/end-to-end. Linux wait4 CPU
user/system del processo dedicato, RSS individuale/reaped max diverso da
sampled tree peak10ms. Shared pages possono duplicarsi; PID brevi/peak tra
campioni possono sfuggire. Kill/wait/reap copre gruppo, figli setsid e orfani
propri; SIGINT è rendicontato dopo raccolta. OS cache uncontrolled; primo
processo/workspace nuovo non viene chiamato cold OS. Nessun download/budget
remoto o privilegi persistenti per le prove.

## Verifica del riferimento e limiti umani

M è indipendente da generator/reader/scorer/Pandoc: locatori canonici,
OOXML/MathML/RTF/TeX, PDF Poppler+MuPDF e pixel raster. Confronta didascalie
reali separatamente dalle celle, target figure/note/link, AST annotato contro
fonte TeX e campi manuali PDF contro locatori poi verificati sul PDF effettivo.
Raster visibile non certifica code point NFC/NFD. Agent_visual non è V.

Il confronto v2 è un documento completo per coppia, originale a sinistra e
Markdown di riferimento a destra; deriva esclusivamente dal gold manuale.
Un checker indipendente legge i veri HTML/.md, matematica/celle/note/asset,
ordine e unità sorgenti/pagine. La presentazione MathML non prova la fedeltà del
canonico o del motore. TeX non compilato; limiti dei renderer originali restano
attribuiti, anche quando i PNG sono copiati in una composizione R PASS.

Vgold e Vcontrolli richiedono risposta reale sul packet passivo immutabile:
un giudizio naturale per documento, massimo tredici. Receipt schema2 con byte
ricevuti/hash, autore e ricezione reali attribuiti dal chiamante con limiti,
packet/coppia SHA, revisione, ambito visibile completo e differenze. Nessun
TSV o timestamp per proprietà; nessuna espansione in firme atomiche. Il digest
della coppia include sorgenti/include/render/asset/HTML/.md/mappa/CSS, esclude
la sola revisione amministrativa globale. Importatore legacy schema1 rimane
per record storici. Il nuovo import è riclassificato dai byte originali.
Una dichiarazione completa «Tutti i documenti corrispondono» resta un'unica
dichiarazione originale riferita agli ID del packet; le decisioni documentali
ne dichiarano la proiezione. Nessuna riga originale per documento è inventata;
vaghezza, eccezioni e sottoinsiemi non producono accettazione collettiva.

VERIFIED richiede M required PASS, vista completa PASS e V documentale esplicita
ACCEPTED sugli stessi hash. Assenza/vaghezza PENDING, dubbi AMBIGUOUS,
differenze REJECTED, hash/ambito diversi STALE. I giudizi sono indipendenti per
documento; risposte parziali successive conservano gli altri già ricevuti.
A03 richiede tutti tredici documenti VERIFIED. Accordo sulla durata non è
firma; i260min del vecchio workflow sono storici e non tempo umano svolto.
Formula output typography/semantics sono giudizi separati non assegnati:
L04 UNAVAILABLE anche con AST esatto. PDF/OCR L01 e compilazione TeX L02
NOT_MEASURED. Nessuna garanzia GPU/altri OS.

## Correzioni r002 e versioni del reader/scorer

`observations-v2-typed-inline` conserva contenuti inline tipizzati e offset
sorgente in celle pipe/HTML e caption, oltre a raw/coordinate. Immagini, link,
formule e unknown entrano nell'audit/scoring una sola volta; testo uguale non
basta a ignorare un'immagine o una formula aggiunta. Target e titolo sono
separati: titoli doppi/singoli/parentetici e destinazioni bilanciate/escape
rientrano nel profilo finito. Sintassi risorsa non supportata rende il bundle
INCOMPLETE; raw inline `{=latex}` resta unsupported localizzato, code vero resta
code. MathML con unica annotation TeX è osservabile; altri MathML restano unknown.

`metrics-v2-local-typed` usa le equivalenze AST finite anche nell'allineamento
formula, conservando raw e matching one-to-one. Le parole delle inserzioni
unmatched entrano nel numeratore WER sul denominatore atteso invariato; formule
raw equivalenti non diventano prosa. CER/WER completi sono UNAVAILABLE quando
l'attribuzione è ambigua/unresolved, con conteggi condizionati al matching strutturale, distinti dai limiti
inferiori garantiti introdotti in r004. `exact-visible-unique-v1` attribuisce unknown soltanto quando contiene
il testo visibile intero di uno specifico blocco (almeno8 caratteri), unico su
entrambi i lati; non copre tabelle/figure/formule, altre omissioni o pagine.
Unknown estranei non annullano missing/page loss FAIL.

Gold/spec rev5 dichiarano `header_rows` e policy indipendente dal motore.
DOCX/RTF/TeX del corpus sono headerless, EPUB ha th semantico; PDF/raster usano
policy manuale esplicita, senza attribuire proprietà encoded al formato.
`header-regions-v2` separa corpo e header. Soltanto per sorgente dichiarato
headerless, la prima riga pipe realmente vuota, senza contenuti inline,
è placeholder sintattico: tutte le righe del corpo mantengono coordinate e
numeri. Header reale anche vuoto, promozione indebita e cella assente restano
controlli distinti. Non si elimina genericamente la prima riga.

Sicurezza: HTML/XHTML ha attributi finiti, srcset e attributi risorsa ignoti
sono rifiutati; CSS ha proprietà/valori finiti, escape/commenti e funzioni non
ammesse sono rifiutati. Ogni SVG, anche XML con radice SVG, è validato con tag/
attributi passivi; membri ZIP ignoti sono rifiutati. I negativi vengono fermati
prima di estrazione/motore: nessun tool deve essere usato per dimostrare il danno.
Pandoc token legacy viene risolto nel PATH effettivo chiuso `/usr/bin`, con
which/realpath/hash/versione prima e dopo, coerenti con il percorso diretto.
Segmenti vuoti, relativi, cwd, user-bin o binari anteposti falliscono preflight.

Ogni monitor usa un supervisore stdlib proprio nello stesso interprete,
namespace e ambiente: raccoglie soltanto la sua discendenza, compresi setsid,
orfani e writer dopo exit del leader, fino alla deadline complessiva. Il writer
entro cap/deadline può finire normalmente. Identità PID tramite pidfd e
controlli finali dopo reap/chiusura stream precedono la receipt. Config/receipt
supervisore sono conservati; CPU wait4 del supervisore include la discendenza.
Restano i limiti sampled RSS/CPU lowerbound/cache OS già dichiarati.

La qualificazione derivata schema3 conserva V reale precedente, vecchi packet/
pair/gold e risposta originale, delta metadati, nuovo M sul sorgente e checker
indipendente dei contenuti visibili vecchi/nuovi. `derive_verification` produce
`derived_reference_content_qualification`, non una nuova risposta V:
`new_human_confirmation=false`. Header tecnico viene qualificato da M. Si
richiede conferma mirata solo se un contenuto visibile cambia sostanzialmente o
l'invarianza non è dimostrabile. L04 output resta UNAVAILABLE.

Campagne nuove usano attempt_id prefissati da experiment_id e possono congelare
`--verification` prima del workload. Conservare byte degli helper/config/test,
inventari prima/dopo e raw/errori. Nuova observe/audit/score dei raw storici ha
identità propria e engine_runs0; non riscrive metriche storiche. La mancanza di
byte helper storici resta un limite storico, non sanato dal nuovo scorer.

I puntatori src/href/full-path nei contenitori XML ZIP sono verificati prima
del tool indipendentemente dal suffisso casefold; xml:base è fuori subset.
HTML travestito XML attraversa lo stesso validator HTML passivo.

## Correzioni r003 del trasporto HTML e dominio testuale

`observations-v3-html-transport` trasporta tutte le foglie tipizzate attraverso
HTML inline nel profilo finito r004, anche span/a annidati: formule, unknown con raw,
segmenti prosa/math/code, risorse, anchor e offset. Il medesimo produttore
HTML copre celle/caption e paragrafi. Figure HTML finite hanno una immagine e
una figcaption; altre strutture passive conservano foglie/risorse e unknown.
Ogni src/href del subset viene validato e osservato anche in fallback; markup
non bilanciato o sintassi risorsa non rappresentabile rende INCOMPLETE. Nessun
browser, fetch o target evasivo viene aperto. Alt non sostituisce la caption.

`metrics-v3-text-domain` mantiene alignment e denominatori r002. La regola
`text-domain-unknown-v1` rende CER/WER completi UNAVAILABLE/None anche per
unknown testuali osservati unmatched, con ragione, observation ID, block ID
quando attribuito, offset e scope. Il dominio comprende i blocchi testuali
interi e il codice letterale; celle/caption di tabelle/figure e formule hanno
metriche locali distinte. Unknown esclusivamente math non invalida prosa
misurabile. Raw unknown non viene contato come parole: i segmenti noti di una
inserzione unmatched rimangono conteggi strutturali condizionati, con parole note non ridotte a
sola punteggiatura. Per confronto aligned incerto non si inventano edit; i
numeratori sono conteggi condizionati al matching strutturale, con denominatori
attesi completi; la precedente attribuzione di bound è rettificata in r004.
Unknown estraneo non cambia l'attribuzione `exact-visible-unique-v1`, né
copre omissioni, pagine mancanti o FAIL già dimostrati.

`WER_decomposition` (`wer-structural-components-v1`) registra per tentativo
edit nei blocchi aligned, omissioni attese e inserzioni unmatched, tipi,
ID, testo, offset, denominatori e conteggi noti. Le candidate caption/titolo
sono prove diagnostiche di identità testuale esatta al gold, senza mutare
alignment o numeratori. Caption divenuta paragrafo è inserzione nel dominio
dichiarato; titolo declassato produce omissione attesa più inserzione. Gli
stessi fenomeni possono fallire anche gate caption/blocchi/numeri: non sommare
i gate come perdite indipendenti. Nuove misure di raw r002 hanno identità
derivata r003 e engine_runs0, non sono nuove conversioni. V e gold rev5 restano
immutati, L04 resta UNAVAILABLE.

Attributi HTML duplicati sono fuori profilo: nessuna conversione in dict può
eliminare il primo src/href; fallback esplicito INCOMPLETE anche se l'ultimo
target esiste. Casefold, tag annidati e markup malformato hanno la stessa
regola lungo producer/transport/audit.

## Correzioni r004: copertura dei profili HTML e bound garantiti

`observations-v4-html-coverage` distingue `html-inline-v1`, `html-table-v1`
e `html-figure-v1`. Il profilo inline ammette span/a/img/math/code/br/em/strong/
sup/sub e i figli MathML finiti dentro math. Una struttura di blocco in un
contesto inline conserva l'intero sottoalbero testuale, le formule, gli unknown,
le risorse e il raw localizzato, ma aggiunge unsupported esplicito. Non usa la
proiezione text di table/figure. Il medesimo vincolo vale dentro celle, caption,
paragrafi e wrapper inline, senza reinterpretare il codice Markdown letterale.

La tabella ammette caption unica diretta, righe dirette oppure dentro thead/tbody,
celle td/th dirette nella riga, e whitespace strutturale. Ogni altro ramo è
consumato interamente una volta in `residual_contents`, con raw, posizione e
foglie tipizzate; tutte le risorse/formule/unknown delle celle, di tutte le
caption e dei residui raggiungono i consumatori. Caption duplicate sono fuori
profilo: non si sceglie l'ultima. Una figura ammette una img primaria diretta,
una figcaption diretta e whitespace; le foglie della caption sono inline e i
rami ulteriori sono residui espliciti. Unknown/profilo non rappresentabile non
ottiene PASS; aggiunte note nei residui danno FAIL, formule aggiunte supportate
falliscono il confronto e asset mancanti danno INCOMPLETE/exit2. Il validator
URI precede ogni apertura: nessun nuovo bypass URI storico è attribuito.
Il carattere `<` seguito da spazio/cifra resta testo; veri inizi di markup,
commenti/dichiarazioni e malformed mantengono i controlli di risorsa.

`metrics-v4-conditional-and-bounds` confronta la caption anche quando l'atteso
è assente/vuoto: `absent-empty-text-equivalent-v1` considera assenza e stringa
vuota equivalenti soltanto nel dominio testo della caption. Componenti tipizzate
(ref/formule/unknown) rimangono confrontate separatamente; una caption aggiunta
non vuota è FAIL3, senza spostare le celle/caption nel CER/WER della prosa.

Tre misure sono distinte:

- CER/WER completi: somma dei confronti sotto il matching strutturale registrato,
  con omissioni/inserzioni e denominatori attesi invariati; UNAVAILABLE/None se
  esiste testo non misurabile. Non sono distanze globali prive di struttura.
- `conditional_text_counts` e `WER_decomposition` v2: conteggi condizionati a
  free_pairs/attribuzione locale. `known_word_edits` è un alias compatibile del
  conteggio condizionato, con `guaranteed_bound=false`; le entry registrano ID,
  offset, testo e somme. Omissione e inserzione possono descrivere lo stesso
  contenuto quando il matching non accetta un nodo unknown. Non sommarle come
  edit certi del testo visibile o chiamarle lower-bound.
- `text_lower_bound`, `global-text-completion-bound-v1`: dominio dei blocchi
  testuali interi, codice incluso; esclude table/figure/formula/anchor e omette
  proiezioni testuali vuote. Char stream con LF fra blocchi, token stream con
  concatenazione dei token split per blocco; denominatori pieni del dominio
  originale (separatori LF non aggiunti al denominatore). Corrispondenze di edit
  globali libere, indipendenti dall'allineamento strutturale. Con foglie tutte
  misurabili si calcola la distanza globale minima esatta. Con qualunque testo
  non misurabile, o limite di calcolo globale, il bound è zero, TRIVIAL_ZERO.

Garanzia: ogni completamento ammesso degli unknown può emettere qualunque
stringa, anche zero testo, e modificare i confini dei token. Ogni distanza di
edit è nonnegativa, quindi zero non supera la distanza di alcun completamento,
anche con più unknown, cambi di posizione o corrispondenza. Nel caso misurabile
il flusso è fisso e Levenshtein è il minimo su tutte le corrispondenze di edit.
La garanzia riguarda questo dominio esplicito, non l'algebra delle formule o
l'HTML universale. Non si deduce la garanzia dal successo di sole due sonde.
Il bound conservativo può essere zero mentre il FAIL strutturale è certo:
missing/page loss/ordine/numeri/celle/asset e attribuzione locale F08 non cambiano.
Le inserzioni note restano discriminanti nei conteggi strutturali (C1 3/30,
Known words aggiunta 3/2); il loro valore non è attribuito come bound globale
quando una parte del flusso è non misurabile. Un bound zero non è CER/WER completo0.


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

[Report e dati r006](../../documentation/benchmarks/2026-10-09-a002-r006-confini-lessicali-html.md).

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

[Report e dati r007](../../documentation/benchmarks/2026-10-10-a002-r007-contesti-markdown.md).

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

[Report e dati r008](../../documentation/benchmarks/2026-10-10-a002-r008-avanzamento-scanner-e-domini-inline.md).

## Rettifica r009: confini del reader e precedenza della guardia

Contratto storico del reader `observations-v9-reader-boundaries`; le review r009
hanno individuato F43/F44, rettificati nella sezione r010 successiva. Le review r008 hanno confutato
la validazione di tutte le aperture dopo errore dell'albero e la conservazione
generale delle label dopo stripping: r008 rimane storia, senza PASS trasferiti.

`html_tag_boundary` usa i matcher tag/attributi del tokenizer managed locale
CPython3.12.13. Un apice interno a un valore bare è contenuto: il primo `>`
chiude il tag. `<` e token-looking bytes nei valori quoted/bare sono letterali.
La sola estensione finita F36 separa un prefisso da un token inerte in posizione
di nome tag/attributo, prima della fine del tag; non inventa attributi dal corpo.
Scanner, `FiniteHTML`, guardia e albero usano questa stessa decisione.

`HTMLGuard` visita le aperture reali sull'intero frammento HTML, senza albero,
prima della proiezione. Conserva la prima violazione PassiveHTML in ordine
sorgente e prosegue la visita. Duplicati, depth16 e limite20000nodi della
proiezione non interrompono la guardia. Un unsafe reale ha precedenza sulle
diagnosi unsupported strutturali o lessicali: audit `UNSAFE_REFERENCE` con
detail, FAIL score3 e run/evaluate2. Corpi comment/CDATA e valori letterali
non sono aperture; nessun fetch, esecuzione o apertura di path evasivi.
Un documento sicuro con token fuori profilo può conservare immagini/testo
reali e asset verificati, pur rimanendo INCOMPLETE/UNSUPPORTED/exit2.

Il producer passa `producer_limit` a `inline`: il consumer e le ricorsioni
label/span/enfasi non ricostruiscono il dominio dai marker dopo stripping.
`read_parts` costruisce durante il merge la mappa carattere logico→carattere
sorgente; `node` la consuma per `source_position.char/byte`, con UTF8 originale.
È una mappa interna, senza normalizzazione o ricerca di testo. Le barriere
dei token sorgente continuano a fermare i letterali incompleti. Anche la
classificazione della quote usa i code segment già riconosciuti: marker
letterali dentro code non generano quote complesse. Chiusure in un producer
successivo non vengono prese; blocchi esterni e raw rimangono distinti.

La matrice r009 copre interazioni finite con attesi sintetici indipendenti:
errore/ordine/safety/trasporto, valori quoted/bare, token chiuso/EOF,
link/image, uno/due backtick, quote/list/note e continuazioni tab, Unicode e
marker nei letterali. Il confronto con tokenizer fissato riguarda piccoli
input ammessi; non certifica HTML/Markdown universale. Gold PROVISIONAL:
automatic PASS mantiene qualità INDETERMINATE/score5, nessuna nuova V.
Scorer/F26, domini/normalizzazioni/denominatori sono invariati; conteggi
condizionati, bound globale e misure complete restano separati. TRIVIAL_ZERO
non è fullPASS. V13/A03 riguarda il riferimento; L04output UNAVAILABLE,
L01/L02 NOT_MEASURED e limiti L05/hardware/costi restano distinti.

[Report e dati r009](../../documentation/benchmarks/2026-10-10-a002-r009-confini-reader.md).

## Rettifica r010: domini HTML e Markdown nelle foglie inline

Contratto storico `observations-v10-inline-html-domains`: le review r010 hanno
individuato F46–F48, rettificati nella sezione r011. Le review r009
hanno confutato due parti del contratto precedente: un fallback Markdown
poteva perdere un'apertura HTML riconosciuta dallo scanner (F43), e il feed
HTML della coda poteva validare tag letterali in code successivo (F44).
Le prove v9 restano riferite ai loro byte; r010 usa nuovi S/B/I e ricevute R.

Nel dominio finito già deciso dal producer, `markdown_lexemes` fornisce anche
le aperture HTML e gli estremi dei subtree. La guardia inline visita tali
aperture prima di costruire un albero o restituire unknown, usando
`PassiveHTML` e la stessa decisione `html_tag_boundary`/F36. Conserva la
prima violazione in ordine sorgente e continua la visita. Code e math chiusi,
destinazioni/metadata, comandi e token opachi mantengono i propri intervalli;
un opener senza chiusura accettata non rende inerte il tag seguente. Non si
riclassificano i marker rimossi dai prefissi quote/list/note. La mappa r009
verso char/UTF8 originali resta la fonte delle coordinate.

Prima della proiezione, `HTMLTree` riceve soltanto l'apertura e il subtree
chiuso appartenenti alla foglia HTML. Un `<span>` bilanciato non possiede il
code Markdown che lo segue. Questa stessa proprietà delimita il lookahead
dell'enfasi. Un HTML owner ancora aperto conserva invece il dominio HTML
finito; il reader non promette l'intera semantica di un renderer CommonMark.
Nei producer HTML blocco/raw la guardia sull'intero frammento resta distinta.

| Condizione nel dominio inline | Audit e contenuto | Exit |
| --- | --- | --- |
| Code/math incompleto o link malformato seguito da apertura HTML unsafe riconosciuta | `INCOMPLETE/UNSAFE_REFERENCE`, detail della prima violazione; nessuna apertura del path evasivo | run/evaluate2, score3/FAIL |
| Stesso fallback con immagine HTML sicura non rappresentata | `INCOMPLETE/UNSUPPORTED_RESOURCE_SYNTAX`; resource_syntax esplicita, senza COMPLETE con risorsa persa | run/evaluate2, score3/FAIL |
| HTML bilanciato prima/dopo code chiuso contenente img/script/onclick letterali | Testo literal e blocchi esterni conservati, nessuna reference/asset inventata | Nei gold PROVISIONAL automatic PASS, qualità INDETERMINATE, run0/score/evaluate5 |
| Errore strutturale HTML e code chiuso esterno | Unsupported rimane esplicito; il literal non genera unsafe | run/evaluate2, score3/FAIL |
| Token comment/CDATA chiuso o EOF | Corpo opaco intero, raw e char/UTF8 originali; heading esterno soltanto dopo chiusura | INCOMPLETE/UNSUPPORTED, score3 |

F36 è una **divergenza deliberata** dal tokenizer CPython puro. Negli esempi
B1–B4 (`<div <!-- -- ><img ...>`, `<img <!-- src=...> -->`, variante raw con
script, `<alpha <!--` multilinea), il parser puro può riconoscere aperture e
attributi; il profilo finito separa il prefisso e conserva il successivo token
come opaco. Questi casi restano UNSUPPORTED, senza leggere risorse dal corpo.
Non attestano parità completa col tokenizer puro né validazione di ogni sua
apertura. Il limite accettato non si trasferisce ai fallback Markdown F43.

La matrice r010 ha oracoli manuali finiti: uno/due backtick, math chiuso/aperto,
link malformato anche con code interno, ordine HTML/code, safe/traversal/script/
event, HTML bilanciato/duplicati/depth17, token chiuso/EOF, sei producer e
Unicode. I 21 discriminanti originali sono legati agli stessi byte/oracoli;
le aspettative sono identiche prima/dopo. I controlli math con gold volutamente
senza formula sono negativi di scoring, non positivi di qualità. Le prove
non certificano grammatica universale, fedeltà di renderer o migrazione intera.
Scorer/F26, normalizzazioni e denominatori non cambiano. Le 64 misure derivate
conservano raw e V13/A03; conteggi condizionati, bound globale e TRIVIAL_ZERO
restano distinti. L04 output UNAVAILABLE, L01/L02 NOT_MEASURED, L05, hardware,
costi e gap storico dei 12 helper restano limiti separati.

[Report e dati r010](../../documentation/benchmarks/2026-10-10-a002-r010-confini-html-markdown.md).


## Rettifica r011: ownership e metadati inline

Contratto storico `observations-v11-inline-ownership`: le review r011 hanno
individuato F50–F52, rettificati nella sezione r012. Il producer possiede il
blocco, il dominio inline e la mappa esplicita al sorgente. Il riconoscimento
metadata heading è limitato al blocco sorgente: `# … {…}` in una foglia
list/quote/note/cella/caption o nel testo di un heading è contenuto. Il vero
heading mantiene i propri attributi, distinti da quelli del testo interno.

La label è accettata dal walk finito ricevuto. La LF dentro code chiuso
appartiene a quell'atomo e non taglia il lookahead math; LF libera resta
fuori dal profilo della label. LF dentro token opaco mantiene il limite:
comment/CDATA non prestano chiusure al literal precedente. Il token sorgente
è barriera nella ricorsione e conserva raw/char/UTF8. Una math lessicalmente
chiusa conserva tutto il corpo opaco anche quando la struttura è unsupported.

Il ledger code della label vale soltanto fuori dall'owner HTML aperto:
backtick interni sono testo HTML, le aperture interne sono reali. Guard prima
di proiezione, prima unsafe in ordine sorgente e tutte le visite successive,
anche con duplicati/depth error. Alla chiusura dell'owner il code esterno
rimane inerte. Il fallback di un blocco già proiettato/validato inline non
rilegge il raw come HTML. Un opener code/math non chiuso non nasconde
aperture; risorsa sicura non rappresentabile è almeno INCOMPLETE/UNSUPPORTED.

153 oracoli manuali finiti e controlli303r010/308r009; gold math include raw
indipendente, subset math unsupported/L04 mantengono INDETERMINATE/score5.
I gold reviewer negativi e le prove r010 sono storia, non PASS trasferiti.
F36 B1–B4 deliberatamente UNSUPPORTED, nessun secondo feed puro obbligatorio.
Limiti ereditati: lunghezze backtick diverse da CommonMark completo; formati
raw html5/html-space fuori profilo esatto, Pandoc attivo non confermato;
costo finito su commenti senza campagna prestazionale. Il fallback unsafe
observations-v1 in benchmark.py non è l'ID del reader eseguito; attribuzione
con byte, gate, R e snapshot. Scorer/normalizzazioni/denominatori invariati.

[Report e dati r011](../../documentation/benchmarks/2026-10-10-a002-r011-ownership-inline.md).


## Rettifica r012: decisioni di atomo e raw nei container

Sezione storica: la review r012 ha smentito la completezza della guardia
per attrs e la proprietà dei fence descritta qui. Per questi punti e per
la coverage privata vale la [rettifica r013](#rettifica-r013-fonti-di-guardia-e-contesto-dei-fence).

Reader `observations-v12-shared-atoms`, mandato F50–F53. `literal_atom`
registra tipo, corpo, intervallo, chiusura e metadata; `InlineDecision`
materializza una sola decisione nel dominio del producer, con source-map.
Guardia e consumer usano quel ledger. Le ricorsioni label/span/enfasi
ritagliano e traslano gli stessi descrittori; non riscoprono chiusure dopo
stripping. `inline_delimiter` consuma il ledger, `pipe_cuts` conserva il
proprio contratto di delimitatori usando la primitiva condivisa.

Il walk della label conserva la grammatica ricevuta: LF ammessa in code
chiuso, token opaco o backslash+LF. Metadata dopo code non ammette nuove
LF libere. LF possedute da code/escape non tagliano la math; le LF dei token
restano barriere e non prestano delimitatori. Le barriere sono registrate
una volta e consultate per indice, eliminando il precedente loop annidato
LF × code. Questo non certifica complessità lineare dell'intero reader:
lookup degli altri token, slicing e parsing HTML restano costi distinti.

Un code della label conserva i metadata: il formato raw inline esatto
`{=html}` genera un evento per il corpo HTML; code ordinario e raw LaTeX
non diventano HTML. Owner HTML aperto prevale sulle decisioni Markdown;
chiusure estranee non restituiscono quel dominio, chiusure riconosciute sì.

Quote/list/note costruiscono dominio e mappa prima della guardia. Il walk
riconosce i fence di quel dominio, inclusi prefissi quote annidati finiti.
Fence ordinario possiede il corpo letterale. Il fence esatto `{=html}`
possiede invece HTML: il corpo riceve guardia completa separata dall'albero.
Il raw block del container non viene proiettato come struttura completa:
conserva raw/posizione e una diagnosi `container raw block not projected`
con `resource_syntax`, dunque INCOMPLETE/UNSUPPORTED anche per l'asset sicuro.
Il target non viene seguito come reference; l’inventario ordinario può
comunque hashare i file già presenti nel bundle. Fence non chiuso conserva una diagnosi
strutturale; non certifica COMPLETE silenzioso.

La copertura privata registra dominio/range/hash ricevuti dalla guardia;
`node` consuma questa prova e la source-map. La sola presenza di `segments`
non esenta il fallback. Il residuo raw non proiettato rimane unknown esplicito;
unknown non certifica copertura. Il fallback senza dominio guardato conserva
la validazione prevista. Nessun secondo feed HTML indiscriminato sulla coda
Markdown, preservando F44 e l'inertness di code/math chiusi.

La sessione di osservazione conserva la prima Unsafe e continua la guardia
dei domini successivi in ordine sorgente. Il dominio già unsafe non viene
proiettato: gli eventi sono già stati visitati, senza ricontarli nell'albero.
Un errore di proiezione non tronca le guardie. Alla fine `observe` solleva
la prima diagnosi; audit resta UNSAFE con detail, run/evaluate2. Il fallback
`observations-v1` del benchmark resta distinto dal reader fisico.

Le prove r012 usano attesi manuali congelati e un generatore finito separato
(seed12012, selezione motivata, limiti espliciti), oltre alle regressioni.
Formule chiuse: raw/link/testo/blocchi/char/UTF8 preservati, run0/evaluate5,
qualità INDETERMINATE; nessun PASS semantico del subset math o L04.
Raw container unsafe: run/evaluate2/detail; sicuro non proiettato:2/2
UNSUPPORTED. I conteggi e gli input effettivi sono nel report, senza
trasferire prove statiche r011 o misure su precedenti versioni.

F36B1–B4 rimane deliberatamente UNSUPPORTED, nessun feed tokenizer puro.
Preservati F40/F41, metadata heading F46, scorer/F26, denominatori e V13/A03.
Restano le quattro osservazioni r010: run backtick ereditato; formati
html5/html-space fuori subset e Pandoc attivo non confermato; tempi storici
commenti v10/v9; IDfallbackv1. Microprove LF finite misurano soltanto gli input
registrati; O001 era stima statica, non timeout osservato. CPU lower/RSS sampled,
hardware/cache dichiarati, nessun upper certificato. Deadline della run rimossa;
quote CPU/disco/RAM/R/rete invariate. L01/L02 NOT_MEASURED, L04outputUNAVAILABLE,
L05 e ChromeNONPASS/helper12gap/H15/E9/errori storici conservati.

[Report e dati r012](../../documentation/benchmarks/2026-10-11-a002-r012-atomi-e-container.md).

## Rettifica r013: fonti di guardia e contesto dei fence

Contratto storico r013, rettificato dai contratti r014 sotto; le review r013
hanno confutato accettazione metadata, info fence e stato tra producer.
Reader `observations-v13-ordered-guards-fences`, F54–F57. Le fonti
Unsafe del reader confluiscono in eventi con posizione sorgente: aperture
HTML reali e attributi Markdown attivi di blocco/inline. Gli attributi
attivi sono esclusi dai valori proiettabili. Il collector deduplica per
tipo/posizione/snippet e drena gli eventi in ordine sorgente, proseguendo
dopo ogni violazione; il primo errore del reader determina il detail.
Gli errori gestiti dell’albero non troncano i domini successivi. L’albero
non rivalida aperture già registrate. La proiezione può essere costruita,
ma un risultato unsafe non viene pubblicato. `finally` azzera la sessione.
I validatori standalone mantengono il comportamento immediato.

Nei metadata su più righe `attrs` separa LF/CRLF fuori dai valori quotati;
`InlineDecision.attributes` trasporta la mappa di ogni carattere, compresi
i prefissi rimossi di quote/list/note. Le posizioni dei token attivi sono
fisiche, non offset logici sommati al solo inizio del gruppo. Sedici casi
LF aggiuntivi, congelati e confermati come Span/Link da Pandoc, verificano
prima causa, posizione char/UTF8 e visita della coda. Le prove complete
sono state ripetute dopo questo fix trovato nel controllo finale.

Questa precedenza riguarda il reader. Il controllo dei path Markdown
avviene nell’audit dopo observe: A8/CLA-O002 resta limite non bloccante,
senza una promessa di prima risorsa unsafe globale reader+audit.

La condizione `fence_decision` è riusata da radice, scanner, container,
boundary e consumer. Considera paragrafo precedente, separazione, marker,
metadata e chiusura: tilde non interrompe un paragrafo; backtick chiuso può
interromperlo; EOF ordinario non conferisce ownership code. Il vero owner
letterale resta inerte. Pandoc3.1.3 può leggere il backtick iniziale di una
lista come Code inline e la prima heading di una nota come Plain: questi
casi mantengono lo stato di paragrafo pertinente al fence successivo.

Raw esatto HTML/LaTeX mantiene i contratti preesistenti, anche a EOF; non
viene uniformato implicitamente a tutta Pandoc. In particolare il raw
list/backtick del profilo può differire dal Code inline dell’AST e LaTeX
radice conserva la guardia HTML ricevuta. Un marker tilde dopo testo non
nasconde il corpo HTML reale. Un asset sicuro è preservato, oppure resta
un residuo INCOMPLETE/UNSUPPORTED con raw, posizione e resource_syntax;
mai COMPLETE senza la risorsa. Raw container non proiettabile resta
unsupported. LF label, token opachi, F36 deliberato e F44 restano invariati.

`_guard_coverage` è consumata internamente da node/located/source-map e
rimossa ricorsivamente solo al termine di observe. La rimozione delle
640 copie nested nei 64 rescore è una differenza JSON separata dai 19
campi metrici/gate, risultati invariati. Byte ed esiti precedenti restano
storia. Nessun nuovo schema pubblico, modulo o parser universale.

La matrice285 usa raw e attesi congelati, massimo186B, seed13013, generatore
finito senza import prodotto e attesi semantici manuali. Pandoc locale
sotto R fornisce199 AST, incluse139 decisioni ordinarie e60 raw; le correzioni
delle premesse sono registrate. Due copie private reintroducono attrs
fail-fast e fence a ogni riga: entrambe falliscono3/3 discriminanti.
Il [report r013](../../documentation/benchmarks/2026-10-11-a002-r013-guardia-e-fence.md)
lega ogni fonte/callsite a proprietà e risultati, distinguendo autore,
helper, generatore e oracolo; non sostituisce nuove review indipendenti.

Le prove runtime identificano managed3.12.13, S/B/I reinstallata, dieci hash,
helper effettivi, snapshot tecnici e R prima degli import. CLI sintetica
dichiarata, gold PROVISIONAL, rescore64 senza motori/nuove V; H12FAIL rimane
deliberato. L04outputUNAVAILABLE è distinta da V13/A03; L01/L02/L05 e tutti
i NONPASS e limiti r010 ricevuti restano. CPU lower/RSS sampled, nessun
upper certificato, nessuna deadline o nuova quota. Il lavoro attende review
e arbitrato; non dichiara GO né integrazione.


## Rettifica r014: accettazione sintattica e rifiuti URI

`observations-v14-syntax-contracts` conserva l'adapter URI separato dal reader.
I ValueError prevedibili di `urlsplit` (bracket, IP literal e netloc NFKC)
diventano Unsafe nel solo adapter. Username/password sono semplici proprietà
stringa nel Python pinned; `port` non è utilizzata. Non si dichiara una
validazione universale RFC delle URI. Frammenti, relativi e URI esterne già
ammesse restano stringhe, mai fetched; la whitelist non viene ampliata.
Il drain continua sulle Unsafe contrattuali, conserva la prima causa e
azzera la sessione. Errori estranei al parsing non sono catturati globalmente.

I metadata diventano opachi solo dopo accettazione integrale della grammatica:
id, class, key=value (bare o quoted), `-` unnumbered e gruppo vuoto. I token
conservano intervallo e posizione; LF/CRLF soft sono separatori, non un blank
fra paragrafi. I valori quotati possono contenere graffe e HTML letterale;
anche un valore bare ammesso può contenere `<…>`. Non si effettua un secondo
feed HTML dei valori. Una coda o un gruppo invalido resta nel dominio inline,
con guardia dei tag reali e conservazione delle risorse o residuo esplicito.
Raw `=format` è distinto dagli attributi ed è accettato solo sui code/fence.

La info dell'opener ordinario deve essere vuota, una lingua singola o un
intero gruppo accettato. `~~~foo bar` non possiede il corpo; lo stesso testo
con backtick può ancora formare Code inline. Chiusura, marker e stato di
paragrafo sono parte della decisione. Il producer radice consuma il ledger
fisico dello scanner; heading/quote/list/note/div non interrompono un
paragrafo per il solo marker. I container ammessi trasportano host e source-map.
La continuazione lazy di quote/note e la figura seguita da tilde conservano
lo stato aperto; la lista lazy K2c resta un controllo code valido.

I limiti raw esatto HTML/LaTeX, EOF overlay, F36B1–B4, F44, A8 e quattro
osservazioni r010 restano quelli ricevuti. Lo schema pubblico non cambia.
La pulizia della coverage rimuove il record interno `domain=producer`,
conservando eventuali valori utente omonimi; `_guard_coverage=x` non è però
un attributo valido del dialetto pinned e non ottiene opacità metadata.

La baseline r013132 corrisponde a check-baseline-005/cases-005 sui raw/output
baseline-004; 133/check004 e freeze005 restano conservati nel
[report storico](../../documentation/benchmarks/2026-10-11-a002-r013-guardia-e-fence.md).
HC2 r011 viene rettificato con overlay separato: l'AST mostra RawInline HTML
dentro graffe invalide, non metadata. Raw, gold e atteso storico restano intatti.
[Prove, confini e stato r014](../../documentation/benchmarks/2026-10-11-a002-r014-contratti-sintassi.md).

Token metadata adiacenti seguono l'AST pinned: `#chapter1.xhtml#start` sono
due dichiarazioni ID, ultima `start`. Nei 64 ricalcoli r014 questo corregge
16 EPUB (failures/details, nessun delta numerico/gate); otto div TeX cambiano
da token unknown a classe valida. Non si normalizzano i target dei link per
compensare la nuova interpretazione. Delta, premesse rettificate e provenienza
sono enumerati nei dati del report r014; i conteggi4PASS/28FAIL sono osservati.
