# Consegna implementatore a002 r003

Mandato prompt12, arbitrato reale r002 NO_GO/NO_GO: fix locali F13–F16 e tutte
le prove dipendenti. Ruolo esclusivo implementatore; nessun GO autore, review
indipendente, arbitrato o aggiornamento dei registri comuni. Feature
`feature/run-a002-corpus`, HEAD/base/dev80c31da46e5caab990d3596c79c44e524ec1dc95;
indice vuoto, nessuna scrittura Git o cleanup.

## Ingresso e delta

[Verify ingresso](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/entry-verify.json): argv/cwd/
exit0/MATCH del contesto implementation-fix-context-r002 prima dei cambiamenti.
Snapshot storico r002 e contesto ricevuto non sono rigenerati né approvano i
nuovi output. [Byte ricevuti](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/received-bytes.json)
preservano reader/scorer/helper/test del giro precedente. Entrambi i report
review r002, arbitrato integrale, checkpoint supervisore e mandato sono input
reali letti; le loro evidenze rimangono protette.

Il delta comune esatto è nei [controlli finali](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/final-checks-002.json):
nove file, cioè observations/audit/metrics/PROTOCOL, test r003, due guide e due
nuovi report/dati permanenti. I registri comuni preesistenti rimangono identici.
Adapter/validator degli input/PATH/resources/R/writer/registry/lock/pin/prodotto
legacy invariati; nessuna nuova scelta di motore o architettura applicativa.

## Fix e prove discriminanti

| Fix | Risultato e regola | Prove correnti |
| --- | --- | --- |
| F13 | Trasporto completo delle foglie HTML attraverso span/a anche annidati, senza stringa text come unico contratto: formule/raw/AST, unknown/raw/scope, segmenti text/math/code, risorse/anchor, offset e coordinate. Mappe dei prefissi Markdown e fence riportano gli offset al raw effettivo | K01 vuota PASS automatico; A1 formula supportata aggiunta FAIL3; A2 MathML senza annotation resta unknown/INDETERMINATE5. Positivi autonomi nelle cinque collocazioni e quattro wrapper PASS automatico/L04 unavailable; duplicati/ordine/mixed/code letterale, prefissi e fence. Nessun doppio conteggio |
| F14 | Tutte le risorse HTML supportate o fallback raggiungono audit; strutture non rappresentabili sono unsupported/INCOMPLETE. URI vietate UNSAFE_REFERENCE prima di aprire target. Attributi duplicati respinti esplicitamente, senza perdere il primo src/href | B1/B2/B3/B4 asset assente INCOMPLETE/run2/evaluate2; B5 ../outside.png UNSAFE_REFERENCE/run2/evaluate2. Spy legge soltanto document.md. Positivi asset presenti/SHA e figura+caption indipendenti, case/nesting/celle/caption/paragraph/unsupported containers. Malformed/depth/raw fallback INCOMPLETE, vero code inerte |
| F15 | text-domain-unknown-v1 considera anche unknown testuale osservato unmatched nella disponibilità CER/WER complete: None/UNAVAILABLE con ragione/ID/offset/scope. Known typed leaves conservati come lower-bound; denominatori invariati. Formule/celle/caption hanno domini locali e non invalidano prosa estranea senza ragione | Known words positivo0; aggiunta nota3/2=1.5 FAIL; comando unknown estraneo completo unavailable con noto0. C0 resta3/30; C1 completo None con noto3/30 e FAIL certo. Unknown attribuito/estraneo, liste/inline/mixed/code/math/caption, missing/page loss ancoraFAIL3, limite Levenshtein esplicito |
| F16 | WER_decomposition v1 per tentativo: aligned edits, expected omissions, unmatched insertions, tipi/testi/ID/offset/denominatori/lower-bound e candidate caption/titolo esatte diagnostiche. Alignment e normalizzazioni r002 invariati | Checker stdlib separato64 tentativi: DOCX11/182=caption2+9 unmatched, aligned0; RTF19/162=11aligned+3omissione titolo+3inserzione titolo+2caption. Stesso fenomeno può fallire più gate; non sommare gate come perdite indipendenti |

373 test mirati PASS sotto [R corrente](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/R-r003-tests-007/receipt.json).
Sono123 test nuovi più3 regressioni attributi duplicati, insieme alle247
regressioni precedenti; attesi AST/testi/asset/counts autonomi. Il reader non
riceve gold. Scorer e audit consumano le vere osservazioni prodotte; gli attesi
non sono ridotti alle stringhe emesse dal producer.

Riproduzioni proprie [prima GPT](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/before-gpt/probe-summary-003.json),
[prima Claude](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/before-claude-summary.json),
[dopo GPT finale](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/final-gpt/probe-summary-003.json),
[dopo Claude finale](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/final-claude-summary.json).
Payload originali reviewer immutati; gli script/copied raw propri riproducono
gli stessi dati. R PASS della sonda significa esecuzione isolata, non qualità
dei payload. Gold GPT rimane PROVISIONAL; gold Claude usa V derivata reale,
senza nuova firma umana. L04 resta unavailable in entrambi.

Sonda aggiuntiva [src duplicato prima](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/duplicate-before-001/audit.json):
COMPLETE spurio quando il primo target manca e l'ultimo esiste. Il fix rifiuta
duplicati nel parser, con3 regressioni inline/blocco/caption. Non cambia il
validator degli input F04 né usa Pandoc/browser per dimostrare il danno.

## Prove dipendenti e misure dei64 raw

[Dieci contratti finali](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/contracts-002/summary.json):
H01–H08 R PASS, H12-negative/positive R FAIL deliberati/harness1/wrapper2;
R expected-exit0 fissato prima per tutti. H06 controlla UNSAFE_REFERENCE/run2/
score3/evaluate2. Receipt con binding MATCH prima/dopo, stream/risorse ed exit
operativi conservati; non scelto expected-exit dopo il risultato.

[Manifest congelato](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/rescore-manifest-002.json),
[rescore finale](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/rescore-002/summary.json),
[checker indipendente](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/derived-check-002.json):
32 raw della rettifica storica r002 e32 della baseline003, nuove copie byte
identiche e nuove observe/audit/score/final con ID r003b e **engine_runs0**.
26141 file protetti invariati prima/dopo. Vecchie osservazioni/metriche/final,
packet/receipt/gold/corpus e snapshot rimangono immutabili. Il precedente
rescore r003-001 è conservato, storico dopo il fix duplicati.

Ciascun set ha effettivamente4PASS/28FAIL; nessun atteso congelato imponeva
questa distribuzione. Tutti64 raw reali sono senza HTML; gli8 TeX contengono
solo unknown o0013 già attribuito. Lo scan non sostituisce la rimisurazione.
Il checker indipendente senza import reader/scorer/generator legge i file
reali e ricomputa distanze di token e somme, ruoli testuali esatti, denominatori
e confronto completo delle quantità/alignment/esiti r002. Totali/CER/WER,
blocks/order/pages/quality/gate/exit sono uguali sui64; FAIL cambiano soltanto
per nuove source_position nei dettagli. Disponibilità conserva i valori e
aggiunge regola/scope/provenienza, più text_complete unavailable nei TeX.
Hash osservazioni/audit/metriche nuovi sono attribuiti esplicitamente, senza
trasferire receipt verdi r002 ai nuovi byte.

| Percorso, per tentativo in entrambi i set | WER | Aligned | Omissioni | Inserzioni |
| --- | --- | --- | --- | --- |
| TXT | 0/89 | 0 | 0 | 0 |
| DOCX diretto/legacy | 11/182 | 0 | 0 | 11 |
| EPUB diretto/legacy | 8/186 | 0 | 8 | 0 |
| RTF diretto/legacy | 19/162 | 11 | 3 | 5 |
| TeX | completoUNAVAILABLE, noto1/152 | 1 | 0 | 0 |

DOCX caption «Celle misurate»2 e «Figura 1 — Tre regioni e assi del laboratorio.»9
sono paragrafi unmatched: non11 parole inventate nel testo aligned. RTF titolo
«Memoria del laboratorio» è declassato, quindi3 attese omesse più3 inserite,
e caption2;11 edit allineati restano reali. Candidate di ruolo diagnostiche
non mutano matching/counts. Perdite Pandoc e gate caption/ref/numeri sono
pubblicati separatamente; nessuna somma di errori sovrapposti.

La baseline003 resta conversione r002 del8ottobreUTC con misure derivate r003
del9ottobreUTC, non nuova baseline di conversione. Corpus rev5,13 qualificazioni
V derivate reali, sorgenti/PDF/render/include/asset/referenceMD/packet invariati;
nessun nuovo M/V, derivato visibile o render. L04 formule output UNAVAILABLE;
L01 PDF/OCR,L02TeXcompile,L05hardware nonmisurati. Vecchi12 helperbytes mancanti
non sanati. Chrome renderer NONPASS e mutazione esterna dei3binari restano
storici; il nuovo inventario usa byte esterni correnti, nessun Chrome ripristinato
o eseguito.

## Toolchain, byte e suite conclusiva

[Stage finale s010](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/stage-s010/preparation.json):
managed CPython3.12.13/origine/startup, nuovo S, sdist/wheel B, reinstall
canonica ultima, I/diecihash, target/inventario/preflight correnti e R PASS
prima degli import/prodotto/--no-sync. Interprete dei figli uguale a R; nessun
sync implicito o install/fetch per riparare collection. S010 incorpora anche
entrambi i documenti permanenti finali, evitando staleness documentale r002.
Test/sonde/rescore/contratti s009 consumano identici reader/audit/scorer/test/
helper; sole aggiunte editoriali ai dati permanenti richiedono nuovo stage,
senza invalidare quantità o codice provati. Equivalenza dei byte consumati
è verificata nei controlli finali e copie before/after, più
[59 input consumati equivalenti](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/stage-s009-to-s010-consumed-equivalence.json).

[Byte prima](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/input-bytes-delivery-before.json),
[dopo](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/input-bytes-delivery-after.json): helper
bench/diagnostici fuori wheel, test/config/plans e script propri, input S/B/I/
target/preflight/inventario fisicamente conservati. Byte s007/s009 e stage
precedenti restano distinti. Gli strumenti esterni sono identificati da hash;
non vengono acquisiti né attribuiti ad esecuzioni non svolte.

Suite veloce prescritta sullo stage finale, [receipt](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/R-r003-tests-fast-002/receipt.json):
python-I-B-m pytest tests/ --ignore=tests/api --ignore=tests/packaging
--ignore=tests/docker -q, basetemp proprio; **514 test e310 subtest PASS** in41,24s, con stdout e
checkpoint finale. I due smoke radice sono esclusi selezionando tests/.
Collaudo parziale, non convalida completa di migrazione/nuova architettura.
Nessuna campagna a001 globale, API/Docker/ML/OCR/Docling/Marker o nuovo motore.
Le regressioni unità esistenti di safety presenti nella suite vengono conservate;
non si ripetono le campagne/sonde autonome F03/H15/F04 senza dipendenza cambiata.

## Errori conservati e risorse

Primo giro mirato309PASS29FAIL: MathML interno era erroneamente marcato unknown;
corretto produttore, poi338PASS. Espansioni360/370PASS e ultima373PASS conservate.
S004 S/B/I completo ma gateFAIL STALE per ultimo fix scorer intervenuto dopoS:
ricevuta non riclassificata, nessun workload, nuovo stageS005. S006 soltanto
preparato e poi storico. S008 fast511/310PASS è storico dopo attributi duplicati;
prove dipendenti ripetute con nuovi ID, non sovrascritte. Nessun rifiuto effettivo
auto-review; launcher esterno autorizzato per socket sintetico con R/Firejail/
namespace senza egress e confini invariati.

Ledger append-only in evidence/implementation-r003/ledger.jsonl, bilancio finale
nel checkpoint. Entry unica2026-10-08T07:06:54.369068Z, include review waits,
quota52h/21600CPU/6GiBnuovi, stopnuovi49h/19800CPU/5GiB, riserva3h/1800CPU/1GiB.
Stop workload2026-10-10T08:06:54.369068Z, deadline11:06:54.369068Z, senza reset.
Retained include review/ricezione e snapshot successivi alla consegna storica;
CPU nota lower-bound include receipt nuovi e sonde reviewer conosciute, con
admin/preflight non interamente misurati. RSS sampled stop6GiB/RAMoperativa8GiB,
non hardcap aggregato/cacheOS controllata/certificazione universale filesystem.
Network/acquisizioni/invio documenti0, servizi nuovi0€, nessun pin/peso/macchina
o privilegio persistente nuovo.

## Ricezione richiesta al supervisore

Nuovo [report permanente](../../../../benchmarks/2026-10-09-a002-r003-html-e-metriche.md)
e [dati64](../../../../benchmarks/2026-10-09-a002-r003-html-e-metriche-data.json)
rettificano attribuzione/disponibilità dei tre r001 e r002, lasciandoli immutati.
Guide e protocollo coerenti. Controlli finali: file protetti/input byte/Git/
receipt/reap/snapshotbinding/link e nessun processo proprio pendente.

Supervisore: ricevere incremento completo r003, aggiornare CHANGELOG/STATE/
HANDOVER/indici di propria competenza, nuovo snapshot comune e due NUOVE review
mirate indipendenti, poi arbitrato reale. Non sono creati dall'implementatore.
Archiviare byte/evidenze necessari prima della futura pulizia temp; nessuna
pulizia o scrittura Git eseguita. Checkpoint proprio r003 conserva storia r002;
commit/merge/push restano dell'utente soltanto dopo GO finale.

Bilancio finale [budget006](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/budget-006.json):
26.222h wall dall'entry unica, CPU nota lower-bound7105.000s, retained conservativo1200929852B (più supplemento repo storico1890137B), RSSsampled noto735604736B. Riserva e deadline invariati; workload essenziali residui0.

[Final-checks002](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/final-checks-002.json):26141 protetti e136 input byte invariati,16 R finali con binding/reap corretti (14PASS+2FAIL deliberati H12),delta9file,indice vuoto/HEADfeature corretti/own_pending0.

[Conservazione con supplementi repo](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/delivery-budget-final-001.json): totale noto conservativo1204121202B, incluso il supplemento r0031301213B (dimensioni intere dei9 file modificati/nuovi). [Link locali](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r003/document-links-001.json):41 validi.
