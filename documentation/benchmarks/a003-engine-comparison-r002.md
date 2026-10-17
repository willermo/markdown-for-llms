# A003 r002 — Confronto reale dei motori e completamento tecnico

Nota di conservazione aggiornata il 2026-10-15: le prove citate sono locali e
non incluse in Git; risultati e attribuzioni di questo giro restano storici.
[Disponibilità e limiti](../runs/run-a003-fase1-engine-comparison/manifest.md#disponibilità-delle-prove).

2026-10-13 Europe/Rome. Implementatore Codex/OpenAI, stessa run e chat, senza
delega. **Confronto tecnico completato: cinque nativi, quattro PDF interi per
Marker e quattro per Docling, più due piloti per ciascuno.** Tutte le17 prove
correnti hanno SUCCESS operativo e isolamento PASS. Questo non è un GO del codice
né una chiusura della fase1: valutazioni dei contenuti AI, giudizi utente PENDING,
review ChatGPT/Claude ancora da svolgere dal supervisore. La consegna r001 rimane
progresso storico con matrice NOT_MEASURED, conservato integralmente.

Indice passivo completo (prova locale non versionata),
matrice dettagliata (prova locale non versionata),
copertura raw (prova locale non versionata).

## Risultati per formato

| Percorso/caso | Esito operativo | Contenuti osservati AI |
| --- | --- | --- |
| C01 TXT letterale | SUCCESS | Payload631byte esatto, CER/WER0, nove numeri e Unicode/code/URI intatti. |
| C02 DOCX diretto e legacy | SUCCESS entrambi | Prosa CER/WER0; due formule, nove celle incluso vuoto,13numeri, figura PNG byte-identica, tre riferimenti conservati. Raw e asset dei due wrapper identici. |
| C03 EPUB diretto e legacy | SUCCESS entrambi | **FAIL**: nota b11 e backlink mancanti; CER53/1223=4,334%, WER8/196=4,082%; numeri11/13. Due formule, nove celle, figura byte-identica e restante testo preservati. Stesso difetto nei due wrapper Pandoc. |
| C06/C07 Marker | SUCCESS | Prosa per entrambe pagine CER/WER0, ma frazione C06, limiti delle somme e matrice C06 perdono struttura. Aligned con contenuto di due equazioni presente, gruppo strutturato UNKNOWN. Tabelle conservano valori ma introducono header rispetto alla policy gold0. |
| C08 Marker | SUCCESS | **FAIL**: CER5/472=1,059% oltre1%, WER6/65=9,231% oltre2%; segni numerici alterati, frazione mancante e somma deteriorata. |
| C10 Marker | SUCCESS | NativeITp1/p3 CER/WER0; rasterENp2 CER2/600=0,333%, WER2/91=2,198% FAIL. RasterITp3 quattro unità prosa mancanti:175/175car e23/23token. Numeri25/29; matrice p2 mancante. Tre grafici presenti; quello p3 non contiene il testo perso. |
| C06/C07 Docling | SUCCESS | **FAIL**: nativep1− diventa -, caption cambia dash, CER3/586 e3/544; nativep2 prosa0. Formule non decodificate o limiti persi; frazione e matrice C07 restano leggibili in PNG, distinte da matematica strutturata UNAVAILABLE. Celle8/9 esatte, un segno alterato; numeri9/12. |
| C08 Docling | SUCCESS | **FAIL**: CER9/472=1,907%, WER9/65=13,846%; URI/n1 e ambiguità OCR alterati, frammenti riferimento/nota intercalati. Formule non disponibili, numeri7/9. |
| C10 Docling | SUCCESS | **FAIL**: nativeITp1 CER3/632 e p3 1/457, soglia0; rasterENp2 CER31/600=5,167%, WER7/91=7,692%; rasterITp3 CER2/175=1,143%, WER3/23=13,043%. Dodici celle mancanti p2/p3; numeri20/29. Didascalia EN riordinata, E=mc? errato; matrice p2 in PNG leggibile. |

C02/C03 confrontano lo stesso Pandoc3.1.3 attraverso due trasporti, non due
algoritmi indipendenti. Sul campione TXT e DOCX sostengono quei percorsi provvisori;
EPUB richiede recupero della nota nel seguito. **Nessun profilo PDF supera il gate
completo.** Marker preserva meglio la prosa nativa qui; Docling consuma meno nel
profilo misurato, con formula enrichment disabilitato. Non sono confronti di
capacità massima, altri formati/host o throughput statistico. Non scegliere un
motore definitivo o avviare la nuova applicazione da questi dati.

## Metodo e profili effettivi

Sette sorgenti/gold qualificati indipendenti, stessi denominatori del piano e
locatori pagina/lingua/native. I motori vedono solo input staged e runtime;
gold/trascrizioni/riferimenti restano fuori dal consumer. E01 copre ogni byte raw
UTF8 strict con span disgiunti e T00–T08 dichiarate. Attesi dal gold, emissioni dal
raw; allineamenti e tracce Levenshtein separati. Nessuna normalizzazione Unicode,
segni, numeri o LaTeX per produrrePASS. Missing conserva il denominatore; celle
vuote presenti distinte da celle mancanti. CER/WER sommano distanze e denominatori,
non medie per blocco; la partizione raster/native C10 rimane esplicita.

Annotazioni finite dell'autore, senza AST candidato come riferimento né nuovo
parser generale. Nei frammenti Docling le stringhe osservate si concatenano in
ordine di incontro, senza introdurre spazi: perdita di topologia/ordine registrata
separatamente. Il placeholder alt `Image` è custodito nel canale asset; i commenti
formula-not-decoded rimangono nel canale formula, senza nascondere omissioni.
Formule LaTeX/Unicode/PNG hanno osservazioni semantiche/visive AI; equivalenza AST
formale e giudizio utente restano non misurati. Nessun OCR aggiuntivo dei crop.
Tutti14assetunici sono stati visti e hashati, con confronti alle figure IT/EN.

Marker1.10.2, Surya0.17.1, torch2.7.1+cpu, cinque checkpoint default esatti e due
font GoNoto già acquisiti:50file/3.480.589.456byte riusati dopo verifica. CPU4,
interop1, batch1 **anche OCR-error nel builder**, pdftext_workers1, force_ocrFalse,
use_llmFalse, immagini/paginazione attive; strip_existing_ocrFalse e DPI96/192
identificati dal codice del pin, impostazioni runtime custodite. IR strutturale
nativo della stessa inferenza; buffer PIL lowres/highres di pagina/blocco esclusi
dal JSON e asset del renderer conservati. Warning ToC Pydantic visibile nei log,
non trasformato in una prova di equivalenza dell'IR completo.

Docling-slim2.138.0 con closure CPU separata73wheel:72originarie più extra OpenCV
headless4.11.0.86 ufficiale IBM Models, già disponibile; nessuna altra versione
cambiata. Torch2.7.1+cpu/torchvision0.22.1+cpu, Heron e TableFormer accurate, OCR
Tesseract5.3.4 ita+eng automatico, immagini referenziate, **formula enrichmentFalse**,
remote_services/external_pluginsFalse. Heron revision8f39ad3c0b4c58e9c2d2c84a38465abf757272d8,
Apache2.0; TableFormer revisionfc0f2d45e2218ea24bce5045f58a389aed16dc23,
CDLA-Permissive2.0. Sette file acquisiti gratuitamente, SHA pubblicati safetensors
verificati con header/offset/extent. Font OFL dal name table, Tesseract/tessdata
Apache2 da copyright locali. Marker codeGPL e pesi ModifiedOpenRAIL-M per uso
personale/ricerca secondo fonti del pin; nessuna decisione commerciale o promessa
di redistribuzione dei pesi. Provenienze/licenze e limiti nel bundle tecnico.

## Recuperi e prove invalidate

L'addendum autore r004 supera la nesting Firejail→bwrap, che falliva al mount
procfs. Il runner standard ora crea una sola barriera bwrap: net/PID/proc/mount
privati, cap-drop, input/runtime/cache/font ro, scratch rw, env chiuso e managed.
Preflight fidato S/B/I nella fase host; sonde nel consumer e figli Pandoc/Tesseract
reali prima del comando, monitor standard esterno con pidfd/subreaper/watchdog.
Nuovi namespace per tentativo, nessun confronto con ID obsoleti o proc host.
Socketpair e letture ammesse positive; egress, daemon/socket sintetico, traversal,
symlink, scritture ro, host PID/proc e FD negati. R registra prima/dopo e cleanup.
Nessun root/sudo, privilegio persistente o rifiuto di approvazione aggirato.

Errori importlib/FD urandom nel diagnostico, revalidazione closure Docling,
collisioni output setup e shadow stdlib/profile.py corretti nella stessa chat con
log conservati. Cv2 mancava all'inizializzazione TableFormer: extra ufficiale
risolto offline con tutte72versioni precedenti vincolate. I primi sei Marker
riusciti sono **invalidati per il profilo**: l'env Surya non controllava il batch
OCR-error del builder, default4. Corretta l'opzione esplicita e ripetuti due piloti
+quattro interi, con nuove sonde/risorse. Raw e immagini nuovi byte-identici:
riuso delle sole annotazioni dei contenuti verificato, nessun trasferimento del
PASS runtime. Non sono selezioni del risultato di qualità migliore.

Docling esportava URI asset assoluti perché il launcher passava un filename
assoluto senza artifacts_dir relativo. Corretto il serializer nativo; sei replay
sulla stessa IR, **zero nuova inferenza**, immagini byte-identiche e Markdown
identico salvo i target asset identificati. Raw iniziali e IR intatti, attribuzione
conversione/serializer e intervalli risorse distinti. Solo setup/gate invalidati
ripetuti; prefix, managed, wheel, pesi/font invarianti riusati.

## Risorse e verifica

| PDF intero | Marker consumer s / RSS GiB | Docling consumer s / RSS GiB |
| --- | ---: | ---: |
| C06 | 97,69 /7,817 | 22,17 /1,271 |
| C07 | 98,91 /7,819 | 19,86 /1,199 |
| C08 | 127,88 /8,145 | 20,88 /1,212 |
| C10 | 219,45 /8,242 | 26,56 /1,438 |

Intervallo consumer include sonde, startup, modelli, inferenza e serializer;
preflight host/driver/archivio separati. Cache OS non controllata, processi nuovi,
nessuna statistica su ripetizioni. Guardie450s/16GiB per ML, quattro CPU osservate
nel consumer, torch getter4/1; thread OS istantanei non campionati. RSS sampled,
non hard upper; wait4 maxrss individuale distinto dal picco dell'albero. Il nuovo
limite RSS ha positivi/negativo reali con raccolta figli; isolamento non allentato.

Ledger (prova locale non versionata):
8.496,364446CPU-s misurati in292intervalli selezionati, incluse prove invalidate,
senza sommare wrapper e consumer annidati. **Totale CPU run UNKNOWN**: hash/audit,
scoring/copie/editoria/tool esterni agli intervalli non misurati. Nessuna quota o
residuo gratuito inventato; r003 dispone tutte le risorse necessarie. Costo
monetario diretto€0, documenti inviati a servizi conversione/revisione0; wire/energia
non misurati. Runtime/pesi/cache non copiati nell'archivio.

Suite corrente (prova locale non versionata):
**3.184test +310subtest PASS,0skip**, input invarianti, R PASS e postcheck I PASS,
28 regressioni a003 incluse. Nuova sdist/wheel canonica e reinstallazione/I r007
per base/dev/Marker/Docling sui prefix riusati, dieci hash/origini verificati.
Il gate Docling verifica closure/versioni/wheel/METADATA e RECORD presenza/digest,
non tutti i binari terzi; ABI e inferenza reali sono prove separate. API/packaging/
Docker e root smoke esclusi, suite selezionata non collaudo completo. I risultati
3182+310 r001 non trasferiti. Le sole nuove spiegazioni editoriali rendono gli S
storici per futuri comandi: non autorizzano --no-sync senza preflight aggiornato.

## Consegna e portabilità

Bundle passivo r002 con raw completi, asset, IR, configurazioni, argv/exits,
errori/parziali, mappe/allineamenti/edit, sonde e consumi. Riferimenti sintetici
qualificati riusati dal sottoalbero `../evidence/references/` già archiviato:
portabilità della **lettura** verificata con queste dipendenze esplicite, non dei
runtime. Path assoluti nei metadata sono LOCAL_EXECUTION_REFERENCE, non link di
lettura o ambienti riproducibili già esportati. Originali r001/2116file e sette
record condivisi invariati alle verifiche specifiche; nessun MATCH generale
attribuito a comuni storici dopo nuovi registri.

Branch feature/run-a003-engine-comparison, HEAD=dev9f702a2, indice vuoto. Nessun
Git mutativo, deploy, nuovo motore/parser/fase o stato/arbitrato globale aggiornato.
Report autore e checkpoint r002 identificano consegna e snapshot tecnico autonomo.
**Prossimo ruolo: supervisore**, comune per due review reali indipendenti
ChatGPT/Claude, arbitrato ed eventuali fix; commit/merge rimangono manuali dopo GO.
L04 e fase1 restano aperti con i giudizi utente sui nuovi output pendenti.
