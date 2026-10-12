# A002 r008 — avanzamento scanner e domini inline

Consegna datata 2026-10-10 (Europe/Rome), creazione UTC 2026-10-10T10:23:10.347814+00:00, run
`run-a002-fase1-corpus`. Rettifica il [report r007](2026-10-10-a002-r007-contesti-markdown.md)
e le affermazioni sul dominio LaTeX, sui delimitatori incompleti nello stesso
blocco e sul codice soft-line nelle label. Le review r007 erano statiche; le
riproduzioni seguenti sono nuove prove dell'implementatore. Storia, fallimenti,
review e dati r007 restano immutati. [Dati r008](2026-10-10-a002-r008-avanzamento-scanner-e-domini-inline-data.json).

| Fix | Prima realmente riprodotto su r007 | Dopo sul reader v8 |
| --- | --- | --- |
| F35 | Observe/audit/measure del T1 non terminano: tre R FAIL/TIMEOUT a3s, figli reaped | Dominio dell'intero corpo raw fino alla vera fence fisica o EOF; tre R PASS nello stesso timeout. Blank/marker/backslash/backtick/math e false chiusure in token provati; raw TeX resta unsupported |
| F36 | 64/64 attesi disattesi; corpo commentato riattivato, reference/formule o q/page2 mascherati | Incompleto avanza sul solo opener; token intero trasmesso al consumer. 64/64 rispettati, zero aperture/reference/formule dal corpo, q/page2 missing. Produttori HTML bare/wrapper/cella/caption/fence conservano il prefisso incompleto e il token separati |
| F37 | 64/64 attesi disattesi; codice label troncato o newline spurio nei trasporti | Gli intervalli code accettati dal walk sono gli stessi del boundary scanner; trim di righe parziali usa il ledger sorgente. 64/64 rispettati, link/image/quote/list/note, code a1/2backtick e comment/CDATA/PI/declaration |
| F38 | Affermazioni globali r007 oltre la copertura effettiva | Nuove prove/ID/byte/receipt e rettifiche; nessun PASS o esecuzione reviewer inventata |

## Contratto finito e garanzia di avanzamento

Nel raw LaTeX il consumer riceve tutto il corpo: blank e marker interni non
finiscono il suo dominio. raw_fence_end cerca solo chiusure all'inizio di una
riga fisica; una chiusura apparente nel token inerte è attraversata insieme
al token e il bound successivo viene ricalcolato dal suo end. La vera chiusura
fisica è consumata prima degli atomi inline. EOF conserva anche la diagnosi
strutturale della fence non chiusa, con coordinate non negative e barriere
sui token. Non rende semanticamente supportati i comandi TeX.

Ogni transizione consuma almeno un carattere/opener oppure un intervallo
non vuoto già riconosciuto. Destinazioni/metadata/code accettati finiscono
strettamente dopo la propria chiave; token chiusi o EOF avanzano al loro end;
fence/righe avanzano all'end fisico. Il bound di un atomo nel raw fence è
futuro oppure è la vera chiusura, già consumata nel ramo precedente. Negli
altri producer è il confine corrente, ricalcolato dopo il token o prima della
riga seguente; i limiti label/cella sono futuri. Il cursore non torna su uno
stato visitato e resta nel raw finito. Quattro trace dei cursori reali verificano
l'avanzamento stretto; il timeout è una prova negativa del prima, non il fix.

Il delimitatore non chiuso lascia proseguire lo scanner; il consumer conserva
l'unknown locale sino alla barriera reale. Un opening tag incompleto prima di
un token reale non lo ingloba come attributo. Nei produttori HTML viene emessa
una diagnosi `incomplete-start`; ogni opening tag effettivamente riconosciuto
passa il medesimo PassiveHTML, prima delle proiezioni. I byte del commento/CDATA
non passano alla validazione risorse. Tag/attributi attivi e URI effettive
conservano i controlli; nessun indebolimento di safety/validator/R.

Il walk della label restituisce gli intervalli code nel dominio del producer,
che lo scanner consuma prima del limite a newline degli altri atomi. Non
allarga le chiusure a un blocco estraneo. Gli intervalli sorgente attraversano
prefix stripping e trim: una riga parziale non è riparsata come documento
separato. Cinque positivi Unicode controllano char/byte delle reference e del
code, tipi, testo, ordine e assenza di token, incluso code soft-line fuori label.
È un subset finito, non HTML/Markdown/TeX universale né whitelist dei payload.

## Prove, misure e limiti

1373 test mirati PASS,16 F04 deselected, regressioni r002–r007 incluse.
La matrice128 e le varianti HTML nei test sono audit→observe→score con foglie,
stati, raw/offset, risorse/aperture e missing. I positivi Unicode verificano
reader/audit e coordinate; le trace del cursore sono distinte dalle misure. Le 57 fixture CLI hanno
114 exit OS reali, run.json/final/errori e attesi congelati.
Writer sintetico dichiarato: non una campagna dei motori. Le immagini inline
sono controllate in foglie, URI/hash/dimensioni e audit asset; il campo gold
references del paragrafo appartiene alle reference non-image. La fixture
image002 errata in quel dominio resta R FAIL/evaluate3; image003 preserva il KeyError del harness sul campo non salvato; image004 mantiene
testo e immagine identici, attesi predefiniti e assert indipendenti sullo
stesso asset. Non modifica gold canonici o scorer per ottenere verde. Gold PROVISIONAL:
positivi automatic PASS ma qualità INDETERMINATE/evaluate5; token reali
INCOMPLETE/run2/evaluate2. Il T1 raw TeX chiuso termina con run0 e score3 sul
gold volutamente diverso, non con una qualità PASS inventata; EOF run2. H01–H08
R PASS; H12± R FAIL deliberati, R expected0/harness1/wrapper2 conservati.
[Report autore](../../temp/run-a002-fase1-corpus/reports/implementation-r008.md)
identifica gate finali, suite veloce parziale, costi cumulativi e fallimenti.

Nuove osservazioni/audit/score dei64raw, zero motori di campagna e nuovaV0.
Il checker aritmetico indipendente verifica denominatori, distanze globali
char/token, decomposizione condizionata, 19campi/gate e l'intero albero raw.
Differenze numeriche/gate: 0. Ogni set32 conserva
4PASS/28FAIL nei file effettivi, senza assert sul vecchio conteggio. final r008
è distinto da conversion_final/run originali e old_*r007. Corpus/gold/asset/
render/source/packet/V13/A03/receipt sono invariati. F16 rimane verificabile.

Scorer/alignment/normalizzazioni/denominatori/F26 immutati: il bound del flusso
globale non è il CER/WER condizionato al matching strutturale; TRIVIAL_ZERO
non è una misura completa0. Unknown non wildcard: missing/page/ordine/celle/
numeri/asset restano FAIL. F32/F27/F30/F28locale/F19/F20/F05/F15 preservati;
F31/F33 e residui globali F28/F29 sono verificati nei domini F35–F37. F13/F14/F09
conservano unsupported localizzati, senza promesse universali.

V13/A03 reale distinta da L04 formule outputUNAVAILABLE; L01PDF/OCR e L02TeX
compile NOT_MEASURED; L05 CPUlower/RSSsampled/cacheOS/hardware non certificati,
ChromeNONPASS e gap12helper storici non sanati. R non è sandbox universaleFS.
L'addendum utente24h dispone deadline 2026-10-11T09:37:31Z e stop nuovi workload
06:37:31Z, senza reset CPU/disco/entry originale; la proposta12h/PENDING è storia.
Nessun GO autore: nuovo comune, due nuove review indipendenti e arbitrato al supervisore.
