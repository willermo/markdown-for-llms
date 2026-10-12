# Mandato completo di implementazione — a002 corpus e misurazione — r001

Agisci esclusivamente come **implementatore** di `run-a002-fase1-corpus`.
Completa il risultato del brief, il piano approvato e tutti i fix ordinari nella
stessa chat, senza consegne parziali per test rossi, argv, permessi o freeze.
Usa le skill manage-implementation-run e verify-conversion-fidelity e il
protocollo del repository. La chat supervisore resta distinta; nessuna delega
automatica ai reviewer o simulazione di review. Git di scrittura manuale utente.

## Autorità, letture e identità d'ingresso

Leggi nell'ordine, senza scansioni indiscriminate di temp/storia a001:

1. AGENTS.md, temp/HANDOVER.md, sola sezione corrente iniziale
   temp/PROJECT-CONTEXT.md; STATE/HANDOVER/brief della run e questo prompt.
2. documentation/development/run-lifecycle.md, le due skill indicate,
   handovers/supervisor-implementation-r001.md e template implementation-report.
3. **Integrali** plans/plan-r001.md e plans/errata-plan-r001-r001.md,
   arbitrations/arbitration-plan-r002.md; arbitrato r001 solo per attribuzione
   e rilievi, handovers/planning-errata-r001.md e nota utente V.
4. documentation/README.md, decisions/README.md, ADR0001/0004, fase1 roadmap,
   development/next-run-a002-corpus.md, open-questions e workflow converter-benchmark.
5. docs/how-to/ambiente-uv.md e docs/reference/toolchain-legacy.md; pyproject,
   MANIFEST.in, converter e diagnostics a001 pertinenti a S/B/I/R/preflight,
   runner tests/launcher e inventory. Leggi codice prima di eseguirlo.
6. Manifest e note a001 solo per trovare risorse/keeper davvero consumati;
   inspection/tool-help planning e receipt locali estensioni per versioni/tool.
   A001 GO non è PASS del codice a002, cleanup già ricevuta non si ripete.

GO reale del supervisore in arbitration-plan-r002.md sul seguente oggetto:

- Piano r001 SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`.
- Addendum errata r001 SHA256 `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.
- Review originali reali ChatGPT GO/Claude NO_GO, conservate senza cambiare esito;
  NO_GO r001 superato da verifica locale E01–E08 del supervisore, non da nuove
  review inventate. Non è GO sul codice futuro.
- Feature `feature/run-a002-corpus`, HEAD/dev/merge-base e origin/dev **locale**
  `80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto, documenti locali
  ereditati attesi. Non serve commit preliminare su dev o chiusura aggiuntiva a001.
- Nuovo contesto tecnico `snapshots/implementation-context-r001.json`:
  receipt/hash/delta in evidence/supervisor-plan-errata-r001/context-transition.json.
  Verifica prima di modificare gli input:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-context-r001
```

Questo verify stdlib non attesta il prodotto Python. Dopo modifiche autorizzate
il contesto può essere STALE; registrare il delta, non ripristinarlo artificialmente
né chiedere una nuova approvazione. Gli snapshot di prove ufficiali richiedono
input stabili e sono creati autonomamente con label nuove. L'addendum prevale
nelle sezioni mappate, l'arbitrato r002 dispone quota/dipendenze, questo prompt
concretizza l'implementazione. Non modificare piano/addendum/report/arbitrati/
snapshot storici o registri comuni del supervisore.

## Obiettivo e risultato completo

Produrre corpus sintetico redistribuibile, gold controllabili indipendentemente,
protocollo e runner di misura, positivi/negativi che rilevano perdite, primo
esperimento reale dei percorsi leggeri locali. Prepara il futuro confronto
Docling/Marker/remoto sugli stessi input. Nessuna scelta anticipata di motore,
schema applicativo/API/worker/DB/Web o completamento dell'intera fase1.

Corpus **10 ID**, copertura normativa del piano §2, **9 pagine PDF** (2+2+1+1+3),
**12 render fissi** includendo DOCX2/RTF1, EPUB2capitoli, TXT e TeX2include:

| ID del piano | Input e fenomeni obbligatori |
| --- | --- |
| C01-txt-it | UTF-8 IT, NFC/NFD intenzionali, NBSP, segni/zeri/decimali, codice/URL literal |
| C02-docx-it | OOXML IT2p, OMML, tabella/cella vuota, nota/bib/bookmark/link, figura con caption |
| C03-epub-en | EPUB3 EN2spine, XHTML/MathML, nota/backlink, tabella, figura/caption e link capitoli |
| C04-rtf-it | RTF IT1render, Unicode/fallback gruppi, tabella/cella/nota/link, asset N/A dichiarato |
| C05-tex-en | main+2include EN, formula inline/frac/sum/aligned/matrice, tabular, figura/label/ref/bib |
| C06-pdf-text-it | PDF testo IT2p, una colonna, formule/celle/figure/note e contenuti esclusivi pagina2 |
| C07-pdf-columns-en | PDF testo EN2p, header full-width e due colonne, ordine sinistra→destra, formule/note |
| C08-scan-it | PDF image-only IT1p, segni/diacritici/ambiguità/numero/formula/figura/nota visibili |
| C09-scan-en | PDF image-only EN1p, contenuto diverso da IT, decimali/pedici/matrice/figura/link visibili |
| C10-pdf-mixed-it-en | PDF3p: native IT, raster EN, **nella stessa p3** native+raster IT; contenuto unico per pagina |

Sorgenti leggibili/generatori e manifest con ID/hash/derivazioni/licenze locali;
testi/figure originali sintetici MIT, notice font separati identificati, nessun
privato/font terzo senza provenienza. Due generazioni separate: byte-identical
quando dovuto; PDF/LO content-reproducible con pagine/contenuto/render/font/tool
controllati e hash entrambe, non rimozione metadata regex. Distingui capacità
locale osservata da runtime ancora da collaudare. Non installare nuovi renderer.

Gold per documento reale: trascrizioni, blocchi/ordine, numeri/segni, formule
con struttura attesa manuale, coordinate/celle vuote, asset/caption, note/link/
bib. Niente output Pandoc/reader/scorer trasformato in riferimento. Conteggi
effettivi dal sorgente verificato, mai solo quantità stimate nell'addendum.

## M/V e dipendenza umana concreta

Realizza E02: verificatore M separato dal generatore/reader/scorer/expander e
dal motore, con locatori sul canonico e proprietà encoded/visible/provenance.
TXT/OOXML/EPUB/RTF/TeX letti indipendentemente nel subset; PDF native controlli
incrociati Poppler/MuPDF e mapping ove osservabile. Divergenze non sanate dal
progenitore SVG. Raster non contiene code point: convenzione trascrizione e
provenienza distinte, niente certificazione NFC/NFD da una forma visiva.

**Utente già designato** per V: «Lo svolgerò io; concordiamo la durata sul
pacchetto pronto». Prepara prima bundle offline passivo leggibile: tutte le
pagine/render, capitoli actual, trascrizioni complete, ritagli/locatori, token
critici e code point a fianco, risultati M, checklist/judgment_id enumerati,
campi risposta vuoti. Includi controlli del misuratore solo nelle proprietà
visive pertinenti e separali dai giudizi output. Conserva i sorgenti canonici
e hash del packet. Niente JS/CDN/risorse remote o output unsafe reso attivo.

Quando il packet è pronto presenta percorso, conteggi reali, ambito e stima
aggiornata per concordare durata. Stima proposta Vgold185min/Vcontrolli25min
(intervalli120–260/15–45), nessuna finestra già accettata. Non richiedere di
nuovo chi svolgerà V; niente conferme per comando/documento. Continua sviluppo
indipendente mentre attendi risposta. Importa fedelmente file/testo realmente
ricevuto con hash, identità/data e limiti; risposta incompleta PENDING/AMBIGUOUS,
vecchio hash STALE. Agent_visual non human, mai firma simulata.

Gold VERIFIED solo con M required PASS e V required ACCEPTED per documento/
proprietà e identità coincidenti. A03 resta requisito di consegna: provisional
non è completo. Se dipendenza umana impedisce chiusura entro quota, registra
limite concreto e residuo al supervisore, continuando indipendenti compatibili.
Giudizi tipografici/semantici **sugli output** separati, non disposti ora: prepara
inventario/pacchetto/stima, L04 UNAVAILABLE/PENDING senza PASS semantico implicito.

## Runner, fedeltà, profilo e stati

Realizza specifica minima versionata del benchmark, separata dal dominio fase2.
Processi observe gold-blind e score distinti, raw/offset sempre conservati;
observations digest invariato con gold diverso. Reader limitato E05 tipizzato:
code/fence protetti, math/raw/div/span/attrs/note/anchor riconosciuti, unknown
visibile. Profilo pandoc-md-bench-v1:22enabled dell'addendum, ogni altro nome
dell'inventario markdown3.1.3 disabled esplicitamente; smart/multiline/simple/
grid off. Verifica su controlli autonomi **prima** della campagna; hash bitset,
tool/reader/config e argv risolto, niente tuning sugli output gold.

Math-structure-v1 E01: Arg/Seq, scripts/frac/sum, righe/celle/delimiter e numeri
lessicali; equivalenze limitate braced/singolo, spacing tokenizzato, meno solo
math, bmatrix/matrix con quadre. Raw differente è diagnostica, non duplicato
FAIL via CER prosa. Niente CAS, macro espanse, gruppi cancellati globalmente,
float o algebra inferita. FP01–FP13 passano gli automatici, FN01–FN14 (tutte le
varianti) rilevano perdita localizzata, FU01 unavailable; attesi hand-authored
e M sul sorgente indipendente. Text/prosa/codice/celle restano code point esatti.

Allineamento one-to-one con missing/unexpected/dup e ambiguità; ordine a coppie
su denominatore atteso completo, perdita pagina distinta da blocco. Metriche
per testo/CER-WER, numeri/segni, ordine, formule strutturali, coordinate/celle,
figure/caption e riferimenti: soglie del piano, nessuna media sintetica che
occulti perdite. Zero-error sintetico non soglia universale OCR. Formule
tipografiche/semantiche giudizi distinti. Missing/unsupported/unavailable
mantengono denominatori e non diventano N/A per migliorare punteggio.

Identità experiment/document/attempt/source/tool/config/protocollo/reader/
wheel/gold per ogni esecuzione, directory esclusiva, omonimia/rerun distinti;
raw inclusi parziali ed errori conservati, atomica final receipt solo dopo audit.
Stati conversione/bundle/quality/batch/coverage separati; processo0 non basta.
Exit run0/2/4/130, score0/3/5/4; evaluate precedenza4→2→3→5→0, dettagli score
anche se exit operativo prevale. FAIL critico precede indeterminate, entrambi
visibili. Riepilogo riuscito non significa batchCOMPLETE o qualityPASS.

## Campagna reale e harness R

Cinque percorsi diretti (TXT txt-literal-v1, Pandoc DOCX/EPUB/RTF/TeX) e tre
legacy installati DOCX/EPUB/RTF: **8 percorsi×4=32 tentativi previsti**,
1fresh-process/workspace+3ripetizioni. OScache uncontrolled, niente cold OS
inventato/drop_caches. Tentativi extra fino64 stimati con margine, non cap per
sonda né obbligo di raddoppiare se i32 sono conclusi. Ogni FAIL resta nel report.

Pandoc diretto binario assoluto/hashato, shell=False, input reader esplicito,
writer E05, wrapnone/headingsatx, datadir vuota confinata, resource source-copy,
extract-media relativo media/cwd raw proprio. TeX expander solo2include letterali
allowlisted, path/cicli/byte limitati; no TeX compilation/shellescape/macro
dinamica. Niente riparazione output con gold/media attesi.
Legacy solo PandocConverter della wheel installata, sys.executable -I -B,
PATH chiuso e Pandoc stesso hash; scratch source-copy nomi compatibili .split,
log/.temp.md confinati. E08 distingue argv observed/reconstructed, returned-text
serialized da temp originale non disponibile, engine_exit null se non esposto,
tempi end-to-end legacy da latenza Pandoc diretto. Nessuna discovery ML/cloud.

**A10:** almeno un percorso Pandoc C02/C03 con4 ripetizioni valide, Markdown
completo, asset figura attesa presente, referenziata e contenuto verificato
(hash/pixel/regioni con metodo). TXT, media qualunque o esistenza sola non basta.
Qualità motore FAIL pubblicabile anche quando percorso completo; non tollerare
FAIL del proprio misuratore/gold/harness come analoga lacuna. Diagnosi/rerun
con nuove identità nella stessa chat, vecchi tentativi preservati.

Harness ufficiale E03 con manifest congelato prima dei tentativi, schemi/attesi
assert a priori; **R expected-exit0 fisso**. Esiti benchmark2/3/5 rendicontati
integralmente possono dare harness0 e R PASS con batchPARTIAL/qualityFAIL:
execution_complete non è fedeltà o batchCOMPLETE. Preflight/tampering/schema/
record mancante/exception interna/assert/testFAIL danno harness nonzero e R
non accettato. Niente catch-all verde o expected scelto dopo l'esito.
H01–H16 verificano questo contratto; negativi controllano anche i loro attesi
nonzero come casi interni dell'harness positivo, mai rendendolo universalmente0.

Segmenta per gruppi/percorsi con manifest figlio e aggregato8percorsi, targetR
≤900s. Proposte iniziali motore60s/harness400s/R600s, render120s; correggi
autonomamente deadline/log/launcher/output entro cap e quota, registrando argv
reali. Stream≤1MiB e file guard32MiB; dettagli su file confinati compatibili.
Timeout del motore legittimo solo se terminate/wait/reap/assenza figli reali
entro deadline harness. Timeout/violazione R non è PASS di campagna ufficiale.

Misura tempi monotonic per conversione, audit, score/end-to-end separati;
CPU wait4/user/system figli con limiti, RSS individuale vs sampled tree peak
a10ms distinti, shared pages/processi brevi e missedPID espliciti. Non sottrarre
maxrss cumulativi né mescolare serie dirette/legacy. Hardware osservato locale
i7-1165G7/8CPU/~62,5GiB/nonNVIDIA esposto da riverificare, nessuna garanzia GPU/
Windows/WSL/macOS. Preparazione e workload costi separati.

## Confinamento, toolchain e prove necessarie

Contenuti non fidati: registry tool esplicito, shell=False, niente comandi
derivati da dati o fetch. Validatori ZIP/XML/HTML/SVG/include/font/path prima
di scrittura/esecuzione; root dedicati, symlink/hardlink evasivi rifiutati,
limiti finiti del piano. E06 input invalido4 prima tool; output unsafe2/score3,
INCOMPLETE/UNSAFE_REFERENCE, raw preservato inerte senza seguire target, lstat/
no-follow solo discendenti sicuri. Violazione scrittura/integrità è errore
harness4, non normalizzazione di un motor-loss. R rete non è sandbox universale
filesystem: non attribuirgli protezioni non collaudate. Sentinelle solo sintetiche.

Per progetto managed locale CPython3.12.13 origine verificata, venv locked
offline a002 o risorsa locale pertinente verificata, stesso interprete figli.
Nuovi input S/B/I+diecihash e preflight prima import/no-sync: build sdist/wheel
offline backend84, audit B, reinstall wheel canonica per ultima, I e RECORD
reali. Anche script/fixture fuori wheel hanno manifest/hash/versioni/config,
audit prima-dopo e tampering test, non attestati dalla sola wheel. Segui piano
§8/guida senza installer/lock/pin aggiornati o campagne a001 indiscriminate.

Snapshot di esecuzione tuoi nuovi con input reali/helper/config e output vivi
esclusi, poi R nuova nello stesso interprete della prova e figli con preflight
S/B/I + audit manifest esterno. Launcher -I degli script esterni carica solo
root scripts fidata/hashata; child converter installato senza root prodotto
in sys.path, verifica __file__/hash. Snapshot tecnici non nuove approvazioni.
Se codice cambia, nuova identità e prove dipendenti ripetute, non vecchie receipt
riattribuite. Promozione report/guide dopo workload può cambiare worktree: registra
delta documentale ed equivalenza degli input tecnici consumati; non proclama
preflight corrente della nuova impronta senza nuove prove se devi rieseguire.

Test significativi dei nuovi contratti e17famiglie originali: fedeltà/softwrap/
note, pagina/blocco/dup/numero/segno/ordine/formula/riga/cella/assets/caption/
link/empty/batcherror/crash/timeout/omonimia/rerun/ambiguity/goldblind/unknown/
safety; E01–E08 aggiungono i testimoni sopra. Assert contenuto, denominator,
offset/localizzazione, stato ed exit. Positivo discriminante un reject-all.
Stub dichiarati non conversioni reali. M/V/hash stale e harness false-green
testati; sentinelle non private, prove rete solo sintetiche.

Suite veloce selezionata una volta sugli input finali dopo installazione,
tokenizer cache/Pandoc pronti e R PASS; nel target isolato:

```bash
"$RUN_DEV_PY" -I -B -m pytest tests/ --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q
```

Conftest inserisce root repo: suite non dimostra il percorso installato, S/B/I
e child wheel lo verificano separatamente. API/campagne Docker/packaging complete,
smoke radice/discovery/ML esclusi. CollectionFAIL si diagnostica senza download/
installazione riparativa; non nascondere nodi o applicare xfail per chiudere.
Ripeti solo mirati/parti invalide dopo nuovi cambiamenti, non tutta a001.
Un FAIL della suite non è automaticamente una regressione della run: conserva
receipt/nodi e diagnostica l'attribuzione sugli input attuali. Nuovi test e
contratti a002 devono passare; difetti legacy fuori scope dimostrati e non
introdotti restano FAIL documentati, senza trasformare pytest/harness/R in PASS
né aprire una correzione generale del prodotto. I risultati storici non provano
da soli l'esito attuale. Le prove ufficiali della baseline hanno target distinti.

EPERM/EACCES: conserva errore, diagnostica e richiedi se necessario escalation
prevista **dal tool per azione già autorizzata**, mantenendo R/netns/isolamento.
Non è di per sé rifiuto approvazione. Nessun sudo/sysctl/AppArmor/setuid o
privilegio/daemon persistente cambiato. Rifiuto effettivo non si aggira: riferisci
azione/motivo, continua indipendenti; senza R official IMPEDITA, niente benchmark
fuoriR presentato equivalente. Errori ordinari non sono consegne parziali.

## Quota unica disposta e autonomia

Quota dell'intera implementazione/fix **52h wall cumulativo187200s**, prima
azione implementativa apre ledger UTC+monotonic. Base27h/margine22h/riserva3h;
include ragionamento/preparazione/attese M/V/test/workload/fix/registri anche
fra chat. Pianificazione/review/supervisione già svolte e future review codice
distinte; a001 resta storia senza sanatoria. Non azzerare contatori alle pause.
CPU aggregata processi propri21600s; crescita disco6GiB; RAM obiettivo8GiB,
stop prudenziale sampled RSS albero6GiB, non certificazione hard cap. Acquisizioni/
invio documenti0byte, nuovi provider/servizi€0; letture locali/cache consentite,
fonti primarie testuali per documentazione se necessarie con URL/data, nessuna
acquisizione software/pesi/dataset. Non archiviare segreti/env privato.

Ledger append-only con costi preparazione/workload/test/V/fix distinti, CPU
metodo/limiti, disco logical/allocated/peak/gross/retained e hardlink deduplicati;
R/file/stream caps distinti dal budget complessivo. Prima di stage verifica
residuo≥costo residuo essenziale+riserva. Stop nuovo workload a49h/19800CPU-s/
5GiB o prima; riserva3h/1800CPU-s/1GiB per report e consegna. Non eliminare prove
per rientrare o aumentare quota da solo. Superamento/assenza isolamento/risorsa
essenziale/V indisponibile entro quota sono limiti reali da segnalare al
supervisore con evidenze e lavoro residuo, senza falso completamento.

Autorizzati mezzi ordinari reversibili non enumerati, helper propri, directory
esclusive, timeout/log/deadline compatibili, fix reader/runner/generatori, prove
invalidate e loro snapshot autonomi; non chiedere mandato per ogni comando.
Layout/font/gold difettosi si correggono con revisioni/hash e M/V dipendenti
nuovi; canonici precedenti e fallimenti conservati, mai gold tarato sul motore.
Contenuti gold canonizzati sono protetti per ciascuna prova/mutazione. Scostamenti
sostanziali di metodo/perimetro, dati privati/nuove macchine/acquisizioni/ML/
servizi/quote o indebolimento criteri tornano al supervisore, continuando
indipendenti. Nessuna pulizia a001/cache/Docker/Git automatica.

## Artefatti, chiusura e consegna

Destinazioni permanenti autorizzate: tests/fixtures/conversion-corpus/{sources,
inputs,assets,gold,controls} e manifest/NOTICE/README; scripts/benchmarks con
generate/benchmark/adapters/observations/metrics/reference_check/campaign/
packet/helper e PROTOCOL.md; test unit/integration pertinenti. Nomi di helper
sono mezzi adattabili, non obbligo di introdurre apparato o schema applicativo.
Report reale datato documentation/benchmarks/<data>-a002-native.md creato solo
dopo workload; docs/how-to/benchmark-fedelta.md, docs/reference/benchmark-fedelta.md
e relativo indice basati su help/comandi verificati (skill write-diataxis-docs).
Non creare placeholder di risultati, non aggiornare changelog/stati del supervisore.

Output propri nella run:

- `reports/implementation-r001.md` secondo template, completo A01–A12/E01–E08/
  L01–L05, file cambiati/prove effettive/input hash, costi, decisioni/limiti e
  processi propri pendenti; nessun GO simulato.
- `evidence/implementation-r001/`: entry ledger, manifest/stage/input inventories,
  S/B/I/preflight/R, test veri, prove due generazioni, M/V/packet/risposta reale,
  attempt/raw/observations/alignment/metrics/audit/resources/report e fallimenti.
  Work dedicato `work/implementation-r001/`, nuovi ID/dir esclusivi; benchmark
  output vivi esclusi dagli snapshot degli input. File dettagliati sotto capR.
- `handovers/implementation-r001.md`: checkpoint aggiornato prima di cambiare
  chat, risultato/residuo, prova attribuita e prossimo ruolo supervisore.

Consegna completa richiede A01–A12 derivati nell'addendum e tutti i requisiti
non sostituiti del piano: corpus/copertura/provenienza/determinismo reale; M/V
gold e controlli; metriche/positivi/negativi/stati/test/preflight/hash/R reali;
32tentativi e A10 con figura; report/guide/ledger utilizzabili. FAIL di fedeltà
motore sono risultati validi, fallimenti del misuratore o gold non verificati
non sono successo. L01 conversioni PDF/OCR, L02 compilazioneTeX, L04 giudizi
output non disposti, L05 garanzie hardware/RSS mancanti restano precise e visibili.
Nessuna media o indisponibilità può nasconderne denominatori o dare qualityPASS.

Concludi in chat con risultato reale, percorsi report/checkpoint, lacune e costi,
prossimo ruolo **supervisore per ricezione completa, snapshot comune e due nuove
review indipendenti dell'implementazione**. Non avviare reviewer o fase2 e non
effettuare commit/merge/push; non consegnare solo richiesta di freeze/preflight.
