# A002 r009 — confini del reader

Realizzato il 2026-10-10 (Europe/Rome), creazione UTC 2026-10-10T14:58:26.170486+00:00, run
`run-a002-fase1-corpus`. Reader `observations-v9-reader-boundaries`.
Rettifica le tre cause documentate nelle review r008 senza riscrivere il
[report precedente](2026-10-10-a002-r008-avanzamento-scanner-e-domini-inline.md).
[Dati r009](2026-10-10-a002-r009-confini-reader-data.json),
[report autore](../../temp/run-a002-fase1-corpus/reports/implementation-r009.md).
Consegna dell'implementatore: review indipendenti e arbitrato ancora da svolgere.

## Decisioni condivise

La fine del tag e la distinzione fra valori quotati e non quotati usano i
matcher del tokenizer CPython 3.12.13 fissato. Scanner, guardia e albero
condividono `html_tag_boundary` e l'estensione finita `FiniteHTML`. Un apice
dentro `title=it's` non apre una quote; un opener in un valore attributo
rimane valore. L'estensione F36 separa soltanto un vero opener in posizione
sintattica da un prefisso incompleto. Non interpreta il corpo del token come
attributi attivi e non promette HTML universale.

`HTMLGuard` valida tutte le aperture reali prima della proiezione, conserva
la prima diagnosi unsafe e continua a scandire senza creare un albero.
Duplicati, depth>16 e limite di 20000 nodi restano errori della proiezione;
una successiva apertura unsafe prevale su questi errori nell'audit. La
guardia usa lo stesso tokenizer della proiezione e `PassiveHTML`. Nessun
fetch o esecuzione da dati; corpi comment/CDATA e prefissi F36 restano opachi.

Il producer passa al consumer il limite del proprio testo inline e la mappa
dei caratteri verso il raw durante stripping dei prefissi e merge. Il consumer
non applica nuovamente i marker di blocco al testo trasformato. Anche la
classificazione della quote usa gli intervalli code già riconosciuti: `#`,
`>` e altri marker nel code rimangono contenuto. Le reference e i segmenti
ricevono char/UTF8 originali; gli intervalli opachi continuano a limitare i
letterali incompleti. Il mapping interno non viene serializzato come una
seconda osservazione.

## Matrice finita delle interazioni

Attesi piccoli scritti a mano prima delle esecuzioni, trasformazioni
deterministiche senza seed o fuzzing casuale. Nessun expected è calcolato dal
reader. 308 fixture audit→observe→score, più 31 test specifici; 339 test r009.
La selezione copre le interazioni indicate, senza una
pretesa di prodotto cartesiano completo o verifica universale.

| Invariante | Dimensioni ammesse e segno | Atteso | Prova |
| --- | --- | --- | --- |
| F39, aperture dopo errore | 89 unsafe: D1–D7, duplicati/depth17, traversal/script/onerror, due ordini; block/inline/raw/quote/cella/caption | UNSAFE_REFERENCE con detail; asset evasivo mai aperto; score3/evaluate2 | matrice `unsafe`, spy apertura risorse |
| F39, corpi inerti distinti | 12 safe con duplicati/depth, comment/CDATA, block/inline/raw; visita di tutte le aperture anche dopo primo unsafe; cap nodi safe/unsafe | errore strutturale resta unsupported; nessun tag dal corpo; unsafe reale prevale anche oltre cap | `inert-error`, tre test guardia/cap |
| F40, attributi reali | 56 safe: 10 valori quoted/bare × 4 host, più 16 EOF; quote interne, `<`/`>`, Unicode/entity, quattro famiglie di token; 20 confronti selfclosing/normali e 3 tag incompleti | immagine/testo/asset preservati, vero token unsupported; boundary conforme al tokenizer fissato | `safe-attribute`, oracle HTMLParser, negativi URI in `unsafe` |
| F41, dominio trasportato | 122 positivi: paragraph/quote/list/note spazi/note tab × link/image × 1/2 backtick × 6 marker; Unicode e quattro opener distribuiti; exact e controllo | testo/ref corretti, heading/paragrafo esterni, automatic PASS e qualità INDETERMINATE/evaluate5 | `domain`, raw e mapping char/UTF8, hash/dimensioni immagine |
| F41, mapping dopo stripping | 5 positivi, un per host; altro link e code sulla continuazione dopo Unicode | coordinate originali anche sulla seconda riga, zero errori testo/parole | `test_positive_mapping_after_stripped_continuation` |
| F41, chiusura estranea | 5 negativi, uno per host | nessuna reference costruita con chiusura del producer successivo; blocchi esterni conservati | `foreign` |
| F36, token reale intero | 24 negativi: comment/CDATA chiuso/EOF × prefissi producer Markdown e 7 trasporti HTML | token opaco intero; nessuna prosa/formula/asset del corpo; missing e score3, non wildcard | `token`, offset char/UTF8 e denominatori |

I 308 oracoli identici sul v8 conservato danno 178 rispettati; sul v9 finale
308/308. Le differenze reali sono 19 classificazioni unsafe, 14 attributi
safe e 97 domini. D1/D2/D3/D6/D7 erano unsupported sul v8 e sono unsafe sul
v9; D4/D5 rimangono controlli unsafe. Il minimo ChatGPT aveva COMPLETE ma
link perso e falso FAIL: ora conserva contenuto, ref e mapping. Il prima
non dimostra fetch o evasione riuscita.

La prima matrice292 aveva positivi paragraph con marker all'inizio della
riga fisica: quel producer conclude davvero il paragrafo. Fixture e mismatch
originali restano conservati; il confronto autorevole usa soft continuation
ammessa nel paragraph e gli stessi 308 byte/oracoli prima e dopo. Non è una
riclassificazione della prima prova come PASS.

## Prove e misure

1712 test mirati passano, 16 F04 deselected; r002–r008, confronto e writer
inclusi. Suite veloce: 1869 passed e 310 subtests passed, preflight e R
correnti. API/packaging/Docker e i due smoke radice sono esclusi: questa
selezione non certifica tutta la migrazione. Sono preservate le regressioni
F35/F36/F32/F27/F30/F28locale/F26/F16 nei domini selezionati.

35 fixture CLI, 70 comandi `run`/`evaluate` con exit OS reali e attesi
congelati prima dei figli. Writer sintetico dichiarato, nessun motore. Gold
PROVISIONAL: automatic PASS resta qualità INDETERMINATE/evaluate5; token e
unsafe reali danno run2/score3/evaluate2. I negativi con chiusura estranea
terminano senza inventare reference. Immagini inline controllate tramite
foglie, URI, hash, dimensioni e audit: il gold references del paragrafo
riguarda le reference non-image. H01–H08 R PASS; H12± restano R FAIL voluti,
expected R0/harness1/wrapper2. Il report autore conserva gli errori operativi
di fixture/label/directory e la ripetizione corretta.

I 64 raw vengono ricopiati e ricalcolati senza motori e senza nuova V.
Ogni set32 conserva 4PASS/28FAIL osservati, senza assert di quel conteggio.
Il checker indipendente verifica distanze con matrice completa char/token,
denominatori/decomposizioni, 19 campi/gate e identità dell'intero albero raw:
0 differenze numeriche/gate. Verifica aritmetica e catena, senza dichiarare
allineamento strutturale completamente indipendente. 56 misure testo sono
AVAILABLE con EXACT_GLOBAL_DISTANCE; 8 UNAVAILABLE con TRIVIAL_ZERO. Counts
condizionati, bound globale e metriche complete restano distinti.

`final` nei dati è il derivato r009; `conversion_final` e `run` identificano
gli artefatti precedenti, `previous_derived_final` e `old_*` il derivato r008.
Scorer, alignment, corpus/gold/raw/V13/A03 e receipt storici sono protetti.
Nuovi S/B/I, build sdist/wheel canonica, reinstallazione e dieci hash nel
medesimo managed3.12.13 dei figli precedono le prove. Le copie prima/dopo
degli helper e gli snapshot propri attribuiscono le esecuzioni; il gate
conclusivo e i costi aggiornati sono nel report autore. Nessun PASS trasferito
dal S007 storico, nessuno snapshot comune o GO creato dall'autore.

L01PDF/OCR e L02TeX compile NOT_MEASURED; L04 formule output UNAVAILABLE
distinta dalla V13/A03 reale. CPU lower, RSS sampled, cacheOS/hardware non
certificati, Chrome NONPASS e gap storici dei helper restano limiti. R non
è una sandbox universale del filesystem. Rete/acquisizioni/invii corpus e
servizi benchmark0/€0. Addendum24h senza reset: stop nuovi workload
2026-10-11T06:37:31Z, deadline09:37:31Z; riserva3h.

Contratto operativo: [PROTOCOL](../../scripts/benchmarks/PROTOCOL.md),
[reference](../../docs/reference/benchmark-fedelta.md) e
[guida](../../docs/how-to/benchmark-fedelta.md). Prossimo ruolo: supervisore
per nuovo comune e due nuove chat review indipendenti, quindi arbitrato.
