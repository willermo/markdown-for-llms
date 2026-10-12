# Consegna implementatore a002 r005

Mandato prompt18 e arbitrato reale r004 NO_GO, fix locali F22–F26; ruolo
esclusivo implementatore nella stessa chat autrice. Nessun nuovo piano,
review/arbitrato/GO autore o aggiornamento di registri condivisi. Feature
`feature/run-a002-corpus`, HEAD/dev/origin-dev locale/merge-base
`80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto, Git manuale utente.
Data delle nuove prove e misure: 2026-10-09 UTC/Europe/Rome.

## Ingresso e delta effettivo

[Verify ingresso](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/entry-verify.json) registra
argv/cwd/start/end/stdout/stderr/exit0 e MATCH del fix-context-r004, prima delle
modifiche. [Git ingresso](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/entry-git.json) include
status-uall, indice e file nuovi, non soltanto diff fra commit. Letti mandato,
arbitrato integrale, entrambe le review e checkpoint/prove r004, protocollo,
skill manage-implementation-run/verify-conversion-fidelity e input pertinenti.
CLA-R004-I002 rimane una deduzione statica r004: le prove A4/A5 dopo F23 sono
nuove, non attribuite retroattivamente al reviewer.

Common r004 e contesto fix sono storici preservati: SHA comune
7105ec0e91e0783a26e13623ecf9eff37211291323253e39e7dc815e92d768c7, SHA fix-context
cf9516fab267a241371e5556cbd8e39522102c30d60ac0cb3889d2a5cf3c0c88. Il MATCH tecnico
non approva il codice. Snapshot propri r005, target e manifest precedono i workload;
nessun nuovo freeze richiesto al supervisore.

Delta comune **7 file**, nei [controlli finali](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/final-checks-001.json):
observations.py, PROTOCOL, test r005, due guide e nuovo report/dati permanenti r005.
Scorer metrics.py byte-identico a r004; audit/safety/adapter/R/resources/writer,
registry/PATH/config, lock/pin e sorgenti corpus invariati. Nessuna nuova architettura.
[Byte ricevuti](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/received-bytes.json) preservano
reader/scorer e gli helper/test/guide prima dei fix. Registri condivisi, reviewer,
vecchie sonde/fallimenti/report/receipt/snapshot non modificati.

## F22–F26: scelte e prove prima/dopo

| Fix | Risultato nel profilo finito | Prova reale |
| --- | --- | --- |
| F22 | HTMLTree.close registra le diagnosi di contenitori non chiusi indipendentemente da parsed/visit. Il registro non contiene foglie semantiche, quindi non duplica prosa, formule o ref. Raw/offset/end/tag sono conservati; chiusure incoerenti restano distinte. Il tail guard tiene conto delle chiusure già diagnosticate, evitando il secondo errore sullo stesso token | GPT wrapper-unclosed/wrappers-unclosed/table prima COMPLETE/PASS, dopo INCOMPLETE/run2/evaluate2. Wrapper bilanciati div/section/body/article, table/figure, vuoti, annidati e fence raw: positivi misurabili; malformed nonPASS. Test per tipo/conteggio/ordine/coordinate delle diagnosi, contenuto e asset |
| F23 | block_syntax=atx_heading è emesso soltanto dal produttore Markdown ATX. Il mapping di prefisso usa questa identità, mai il semplice type=heading. Gli heading HTML mantengono gli offset del loro raw, inclusi wrapper e prefisso fence | Claude A3/A3a/b/c/A4/A5 prima TypeError in observe/audit; dopo nessun crash. h1–h6 positivi, differenze/aggiunte discriminanti, Unicode/byte/math/link/img e regressioni ATX/note/list/blockquote. Nove casi CLI reali con fixture writer dichiarato verificano receipt run/final ed exit effettivi |
| F24 | h1–h6 e code HTML sono host di html-inline-v1. Blocchi annidati diventano unsupported locali, con raw, foglie note, risorse, formule e unknown conservati una volta; il testo coincidente non basta al PASS | Nuovo runtime A4/A5 dopo F23: COMPLETE ma nonPASS automatico. Table/figure sotto heading/code, wrapper inline, MathML supportato/unknown e missing asset; phrasing span/enfasi positivo. Codice Markdown literal/fence resta inerte |
| F25 | Callback comment/decl/unknown_decl/PI producono token #token, senza testo o referenze. parsed conserva unknown/raw/intervallo e la proiezione cella/caption/residuo lo trasporta. Profilo html-tokens-v1 conservativo: outside-profile implica audit INCOMPLETE in ogni posizione. Per commenti il body del callback dà l'intervallo esatto, anche --!> o chiusura abrupt del tokenizer managed | GPT comment-middle/declaration-middle prima COMPLETE/PASS; dopo INCOMPLETE/nonPASS come il tail. Token inizio/interno/fine in wrapper/prosa/cella/caption/residuo/inline, multipli, Unicode, varianti di chiusura. Spy: nessuna apertura/fetch di immagini/URI racchiusi nei commenti, nessuna prosa secret estratta. Positivi `<`, entità, HTML pulito e literal code preservati |
| F26 | Definizioni documentali e dati nuovi chiariscono che text_lower_bound limita esclusivamente la distanza del flusso globale. Non limita CER/WER completi o condizionati strutturali e può superarli. Nessun cambio matematico/di alignment/normalizzazione/denominatore | D1 inversione: strutturale0 contro globale38char/10parole; D2 omissione26contro27char. Checker indipendente64: DOCX globale62contro strutturale60, EPUB54contro53. Definizioni/versione e numeri pubblicati confrontati con i file effettivi |

[Prima GPT](../../../../../temp/run-a002-fase1-corpus/work/implementation-r005/before-gpt/probe-summary.json) e
[prima Claude](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/before-claude-summary.json), sotto
R/stage-s001 corrente: esiti difettosi conservati, R PASS indica esecuzione della
sonda, non conformità del payload. Le copie proprie preservano gli originali;
Claude cambia soltanto la risoluzione del path repository prima della prova.
[Dopo GPT finale](../../../../../temp/run-a002-fase1-corpus/work/implementation-r005/after-gpt-002/probe-summary.json) e
[dopo Claude finale](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/after-claude-summary-002.json),
sotto R/stage-s004:16 e26 casi, tutti gli attesi correnti rispettati. L'atteso
A3c è ora positivo misurabile, A4/A5 nonPASS e B7/E2 INCOMPLETE, secondo mandato;
i vecchi file/attesi diagnostici del reviewer rimangono invariati.

Gold sintetici delle sonde PROVISIONAL, fixture di verifica/asset dichiarate
nei test, nessuna nuova receipt V umana. PASS automatico, qualità e R sono
campi distinti. Gli exit delle sonde reader/audit/score sono derivati dalla
regola; il [probe CLI](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/cli-probe-002/summary.json)
verifica invece18exit operativi effettivi run/evaluate in9casi, receipt/final e
traceback assenti, con writer **stub sintetico esplicito**, non motore nativo.
Positivi CLI run0/evaluate5 per gold PROVISIONAL, aggiunta evaluate3;
missing/unsafe/unclosed run2/evaluate2. Nessun engine baseline rieseguito.

La copertura è per produttore e tipo, non per stringhe della sonda: tutte le
diagnosi strutturali sono raccolte dopo close, qualunque sia la proiezione;
ogni callback lessicale crea una foglia token che parsed tratta senza prosa o
risorse; wrapper, contenuti inline e residui mantengono questa foglia o diagnosi.
Il walk table/figure r004 rimane invariato sulle foglie note. Una chiusura
incoerente non viene riusata per nascondere contenuto, e un commento non attiva
la sintassi racchiusa. Non è richiesto HTML universale; nessun nuovo bypass URI
storico è attribuito. Il flag resource_syntax trasporta la diagnosi conservativa
di markup all'audit, non dichiara una risorsa attiva dentro un commento.

## Misure, 64 raw e F16

[Manifest002](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/rescore-manifest-002.json) congelato
prima del workload, [rescore002](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/rescore-002/summary.json):
nuovi ID r005b,64 osservazioni/audit/score/final, engine_runs0. Conservate identità
di conversione r001/baseline003 r002, raw byte-identici, gold rev5/receipt V13,
vecchie osservazioni/metriche/final r004 e precedenti. Rescore001 r005 è storico
prima dell'ultima precisazione degli intervalli commento; non sovrascritto.
Due set32 hanno effettivamente4PASS/28FAIL ciascuno, non un assert imposto.

[Checker indipendente002](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/derived-check-002.json)
PASS64, zero differenze nei campi numerici/gate storici confrontati. Matrice
propria char/token, nessun import reader/scorer/generator, somme/denominatori,
role evidence, legami e raw verificati. Metadati ATX e hash delle osservazioni
sono nuovi. TXT0/89; DOCX11/182 caption2+9; EPUB8/186 omissioni; RTF19/162
condizionato=11aligned+3omissione titolo+5inserzioni (titolo3/caption2), contro
globale13/162; TeX completoNone/condizionato1/152/globaleTRIVIAL_ZERO0/152.
Celle corpo DOCX9/9, EPUB9/9, RTF6/6, TeX9/9 preservate; perdite reali restano
pubblicate. Caption divenuta prosa non significa perdita materiale; non sommare
gate come perdite indipendenti. F16 resta verificabile per tutti64 record.

CER/WER completi sono somme sotto matching strutturale; None dove il dominio
non è misurabile. conditional_text_counts/WER_decomposition v2 registrano
conteggi condizionati, guaranteed_bound=false, anche per alias known_word_edits.
text_lower_bound/global-text-completion-bound-v1 riguarda il flusso globale
libero: codice incluso, table/figure/formula/anchor esclusi, proiezioni vuote
omesse, char con LF fra blocchi/token concatenati, denominatori pieni originali
senza aggiungere i LF. Può superare la misura strutturale per ordine/separatori.
EXACT_GLOBAL_DISTANCE è esatta nel proprio dominio; TRIVIAL_ZERO deriva dalla
nonnegatività per ogni completamento unknown, anche zero. Non è completo0 o
prova di fedeltà. Missing/page loss, ordine/numeri/celle/asset e F08 rimangono
indipendenti. Il calcolo metrics.py e la garanzia r004 sono byte-identici;
la prova r00410casi42completamenti è riusata come storia con questa equivalenza,
non rivendicata come nuova esecuzione o giudizio umano.

[Report permanente r005](../../../../benchmarks/2026-10-09-a002-r005-diagnosi-html-e-domini.md)
e [dati](../../../../benchmarks/2026-10-09-a002-r005-diagnosi-html-e-domini-data.json)
rettificano la lettura del dominio nel report r004, conservato, con definizioni
esplicite ed esempi reali. PROTOCOL e guide coerenti. Il
[controllo dati permanenti](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/permanent-data-check-001.json)
verifica64record/12campi/384descrittori contro i file effettivi.

F01/F09/F13/F14 chiusi nel profilo finito per trasporto HTML/diagnosi/risorse e
heading; F17/F18 preservati, host completati F24; F21 letterale< preservato e
token/markup completati. F05/F15/F19/F20 chiusi nei domini dichiarati, F16
preservato. Altri fix r002 non riaperti senza dipendenza. Corpus/gold/source/
render/asset/packet/risposta V originale e13receipt derivate invariati: nessuna
nuova M/V, nessun VERIFIED copiato su hash nuovi. A03 chiuso nel perimetro delle
review precedenti; L04 formule output UNAVAILABLE distinta dalla Vgold.
L01PDF/OCR, L02TeXcompile e altro hardware L05 non misurati. ChromeNONPASS,
gap12helper delle conversioni storiche e fallimenti non sanati da questi score.

## Catena corrente, prove e conservazione

[Package attuale all'ingresso](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/entry-package-inputs.json):
3813 file, [delta rispetto a s008](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/entry-package-delta.json)
zero. I tre pyc encodings ricevuti r004 restano gli stessi, byte conservati,
causa/autore non attribuiti; nessun ripristino. Guardia startup host e snapshot
implementation-r005-managed-entry-001 prima del primo managed. Nuovi stage,
nessun trasferimento di vecchio PASS a input nuovi.

Stage finale [s005](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/stage-s005/preparation.json):
S/B/I nuovi, build offline sdist/wheel e canonical reinstall ultima, dieci hash,
origine managedCPython3.12.13, input package/startup/helper/config/tool/cache
verificati; preflight e [R gate finale](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/R-stage-s005-gate-001/receipt.json)
PASS prima degli import. R e figli nello stesso interprete, Firejail rete isolata,
registry/shell=False, filesystem e PATH chiusi preservati, --no-sync dopo gate.

652 mirati PASS/16F04 deselected in16.20s, stage-s004,
[receipt](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/R-r005-tests-003/receipt.json).
177 nuovi casi r005,475 precedenti pertinenti: F01/F05/F09/F13–F21 e confronto/
writer. [Equivalenza s004→s005](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/s004-s005-consumed-equivalence.json):
108 helper/test/input byte-identici; budget.py cambia solo la contabilità host,
non consumata dai workload; due nuovi report/dati permanenti aggiunti. Nuovi
S/B/I/gate finali, vecchie prove attribuite ai propri input validi.
Suite veloce prescritta sotto [R finale s005](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/R-r005-tests-fast-001/receipt.json):
**809 PASS +310 subtest**,27.33s. tests/ con ignore API/packaging/Docker;
i due smoke radice esclusi per la selezione tests/. È collaudo parziale.
Nessun install/download per collection, F03/H15/F04 standalone/API/Docker/ML,
a001 globale, browser/Pandoc su payload pericolosi o nuova campagna motori.

[Contratti002](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/contracts-002/summary.json):
H01–H08 R PASS/harness0, H12± R FAIL deliberati/expected0/harness1/wrapper2,
con attesi congelati prima. Tutti gli exit operativi, binding/reap/socket reali;
nessun expected-exit scelto dopo l'esito. Contratti001 preservati come prova
prima dell'ultima modifica locale del reader.

Byte helper fisici: [109 input prima delle prove s004](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/input-bytes-s004-before.json),
config/helper propri preservati nei file immutabili work e nei target/snapshot
prima/dopo; [168 copie alla consegna prima del controllo finale](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/input-bytes-delivery-before.json)
e [verifica dopo](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/input-bytes-delivery-after.json).
Le copie supplementari di consegna non sono dichiarate raccolte prima di ogni
workload. Recovery dei due input s002 solo dopo MATCH SHA con target/inventario
congelati: [receipt](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/recovered-consumed-s002.json).
Nessun diff inventato o recupero attribuito ai12raw storici senza byte.

[Final-checks001](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/final-checks-001.json) PASS:
42288artefatti protetti invariati,168byte verificati,19receipt correnti
(17PASS+2FAIL H12), binding prima/dopo, reap/socket, delta7, feature/HEAD/indice
vuoto/diffcheck, own_pending0 con /proc host read-only e soli prefissi propri.
Nessun segnale a processi estranei. Consegna tecnica completa, nessun GO autore.

## Fallimenti, quota e prossimo ruolo

Gate-s001001 IMPEDITA prima del workload per percorso snapshot inesistente
nel launcher proprio, corretto in chiamata nuova002 PASS; nessun rifiuto
approvazione. Admin plan NameError e recovery lookup StopIteration conservati/
diagnosticati, senza prodotto importato; correzioni nella stessa chat.
Test001630PASS10FAIL:8errori delle nuove fixture e2duplicazioni diagnosi tail;
corretti test e guardia con stage nuovi. Test002640PASS, poi precisione raw
commento+12casi, test003652PASS. Vecchie receipt/byte/fallimenti preservati.
Escalazioni tool autorizzate per R mantengono l'isolamento; nessun aumento quote,
privilegio persistente o allentamento di sicurezza.

[Ledger](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/ledger.jsonl) append-only, quota unica
52h/187200s dall'entry2026-10-08T07:06:54.369068Z, attese incluse; CPU21600s,
nuovi6GiB/RAMoperativa8GiB/stop sampled6GiB, stopnuovi49h/19800CPU-s/5GiB e
riserva3h/1800CPU-s/1GiB. Stopnuovi2026-10-10T08:06:54.369068Z,
deadline11:06:54.369068Z, nessun reset/incremento. CPU nota lower-bound,
RSS campionato, disco conservato/cacheOS: nessun hard cap universale inventato.
Costi reviewer includono misure reali senza sommare i figli già negli outer;
costi admin non misurati rimangono limiti. R≤900s/stream1MiB/file32MiB,
rete/acquisizioni/document-send/nuovi servizi€0. Budget e controllo documentale
finali nella sezione di chiusura, senza lavoro prodotto pendente.

Supervisore: ricezione completa, aggiornamenti condivisi/permanenti necessari
(changelog/indici/roadmap/stato nel proprio ruolo), nuovo snapshot comune e
**due NUOVE review indipendenti mirate**, poi arbitrato reale. Nessun GO futuro
attribuito, Git automatico/cleanup/nuovo lavoro roadmap/fase2. Archiviare prove e
byte prima della futura pulizia. [Checkpoint](../handovers/implementation-r005.md),
storia nel r004 precedente, preservata.


## Chiusura amministrativa finale

[Controllo documentale/processi](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/document-check-001.json)
e [ledger finale](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r005/delivery-budget-closing-001.json):
wall132920.871316s (~36.922h),
CPU nota8211.089263s lower-bound,
retained conservativo1646699588B,
RSS massimo noto campionato740855808B.
Quota originale/riserva invariata, attese incluse, nessun reset o nuovo workload.
own_pending0; consegna completa al supervisore per nuovo snapshot e due nuove
review indipendenti mirate, poi arbitrato reale. Nessun GO autore.
