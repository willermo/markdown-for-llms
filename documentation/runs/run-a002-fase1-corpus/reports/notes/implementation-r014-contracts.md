# R014 — contratti prima delle modifiche

2026-10-11, implementatore OpenAI/Codex, GPT-6 esposto; deployment ed effort
non esposti. Mandato46, arbitrato r013 invariato. Nessuna delega o review.
Ingresso MATCH, reader e adapter originali archiviati insieme. Baseline e
AST locali: evidence/implementation-r014/baseline-001.json e grammar-001.json,
R-baseline-R-002 e R-grammar-R-001 PASS. Primo EPERM conservato; escalation
approvata senza cambiare R. 157 microcasi congelati in due tranche, nessun seed.
La tabella seguente è una specifica di lavoro, non una chiusura verificata.

## URI

`local_uri(str, allow_external)` restituisce un path relativo validato oppure
None per frammento/URI esterna ammessa, o solleva Unsafe. `urlsplit` può
sollevare ValueError per bracket IPv6/IPFuture, indirizzo invalido e netloc
non valida dopo NFKC. Il recupero è confinato al parsing URI; nessun catch
generale in observe/drain. Username/password conservano il divieto attuale.
Nessuna apertura URI. Errori di programmazione (tipo non stringa al di fuori
del controllo specifico PassiveHTML) non vengono occultati.

| Callsite | Input/esito ricevuto | Proprietà | Prova |
| --- | --- | --- | --- |
| PassiveHTML src/href → local_uri | Stringa oppure assenza rifiutata prima del parser | Unsafe per parsing invalido, prima causa e tutte le aperture drenate | GPT-I001, uri-113–120, adapter |
| SVG href / ZIP XML href/src/full-path / OOXML hyperlink | Stessa funzione, allow_external False salvo hyperlink | Nessuna modifica whitelist o accesso di rete | Adapter e regressioni safety |
| audit references → local_uri | Target Markdown dopo observe | INCOMPLETE/UNSAFE_REFERENCE; A8 resta separato | CLI URI / regressioni |
| adapters EPUB / comparison_check XML | Stesso adapter, parser error normalizzato | Rifiuto coerente prima di usare la risorsa | Adapter / suite veloce |

## Metadata

Un record accettato contiene intervallo completo e del corpo, token con
posizione, valori e grammatica. Soltanto il record accettato è opaco.
Gruppi non accettati restano testo e HTML reale riceve guardia/proiezione o
residuo con raw e posizione. Delimitare con graffe non prova l'accettazione.

Partizioni AST: id `#1`, class `.c` ma non `.1`, key con iniziale lettera
(`data-x`, `xml:lang`), `key=value` anche vuoto, doppi/apici con escape,
`-` unnumbered, gruppo vuoto; LF/CRLF soft ma non paragrafo vuoto. `x`, `_x=x`,
`a = b`, tag HTML come token sono negativi. Valore bare `a<b>` e quoted
`<img …>`/graffe sono valori, non aperture. Raw `=format` è un'accettazione
separata dei soli code/fence, non span/link/heading/div/figure.

| Callsite | Stato/decisione ricevuta | Proprietà | Caso |
| --- | --- | --- | --- |
| literal_atom → inline code/raw | metadata_at con permesso raw, bound del producer | Stesso intervallo scanner/consumer, code body inerte | metadata-89–92, info-backtick, grammar |
| markdown_lexemes span → inline span | Label e metadata accettati; niente regex autonoma permissiva | Invalidi conservano graffe/tag, validi quoted non guardati | K4a/e/f, metadata-81–84 |
| link scanner → inline link / pipe_cuts | Record dopo destination, limit effettivo | Non salta HTML del gruppo respinto | K4c, metadata-85–88 |
| peel_attrs → heading | Heading ammessa dallo stato fisico; metadata finale accettati | Testo letterale e attrs attivi distinti | K4b, metadata-93–96, heading-attached/separated |
| fence scanner → root/container consumers | Info interamente accettata prima dell'owner | Metadata non diventano code attraverso un fallback | info-57–80, grammar-25–30 |
| div opener/close → containers | Coda interamente valida o classe singola; stato fisico | Coda invalida resta testo, nesting/EOF espliciti | K3a/d, metadata-101–104, grammar-21–23 |
| figure → caption/final metadata | Record completo dopo link | Non perde tag nella coda né valore utente | metadata-97–100, K2e |
| attrs / InlineDecision.attributes | Soli token accettati, source_map originale | Eventi attivi in ordine char/UTF8, LF/CRLF | A1–A12, mapping16, metadata-lf/crlf |

## Blocchi e stato

Info fence: vuota, singola lingua (alfanumerici e segni del profilo), gruppo
attributi valido o raw esatto; info multipla/coda non valida non possiede un
CodeBlock. Il successivo riconoscimento Code inline resta indipendente e
preservato. Ordinario richiede chiusura con marker/lunghezza/indentazione
corretti; EOF overlay e raw esatto ricevuti restano invariati.

Il ledger fisico dello scanner registra lo stato prima della riga e le
decisioni di blocco. Il producer radice consuma il record, senza ricomputare
fence col default False. Heading/quote/list/note/div non interrompono il
paragrafo soltanto per il marker. Figure e continuazioni lazy quote/note
lasciano aperto il paragrafo; il controllo lista lazy K2c è un vero CodeBlock.
Il producer di container nasce solo da un opener ammesso e trasporta
host/posizioni; al suo interno conserva separazioni e stato del paragrafo.

| Callsite | Stato ricevuto | Proprietà | Caso |
| --- | --- | --- | --- |
| markdown_lexemes radice → _observe | Ledger per offset fisico, paragrafo prima e decisione | Unico owner radice anche dopo altro producer | K2a/b/d/e/f, boundary-111/112 |
| paragraph_boundary / inline_scope_end / source_scope_end | Paragrafo già aperto e sintassi accettata | Marker vietato non tronca il dominio | K1a–d/h, heading-attached |
| _observe quote/list/note → read_parts → InlineDecision | Opener ammesso dal ledger, host esplicito, mappa fisica | Nessun nuovo inizio inventato da container non separato | K1a–f, boundary-105–110 |
| markdown_lexemes container → inline atom consumer | Info/stato/owner del medesimo record | Tilde dopo testo attivo, code reale inerte | r013301, r012350/overlay, K2c |
| inline recursion / table cells / captions | Decision.slice del dominio oppure leaf senza block ownership | Non eredita falsi blocchi dalla forma della stringa | A10/P1, F50–F57, mapping16 |
| HTMLGuard → drain / node fallback / observe finally | Eventi source-owned, coverage fino alla fine | Validazioni effettive, prima causa, molteplicità/reset | URI ordine, A11/12, quattro controprove private |

Prima della consegna la tabella deve essere legata a risultati finali e alle
quattro controprove private. Gli attesi aggiuntivi owner=AST sono condizionali:
l'AST sceglie solo sintassi, mai sicurezza o fedeltà. I raw e gli attesi storici
restano immutati; eventuali correzioni vanno in overlay espliciti.
Baseline r013132 cita check-baseline-005/cases-005 sui raw/output baseline-004;
133/check004 e freeze005 restano storia. Non modificare il permanente r013.

## Riscontro finale sui byte r014

Questa sezione è aggiunta dopo le prove; non retrodata la specifica precedente.
Reader SHA256 `613aaa996fcf02200bc6a8ad499ba0ebc044f44ed94deae3f7e638905215cefd`,
adapter SHA256 `974656f69e022fcaa810c0828079e8672f59db8e8f8131e0b923f38b4e2e32b2`.
I percorsi seguenti sono relativi a `evidence/implementation-r014/`.

| Contratto / chiamanti delle tabelle | Riscontro e limite |
| --- | --- |
| URI adapter, PassiveHTML e audit | `check-after-final.json` 157/157; `test_conversion_fix_r014.py` esercita tutte le sorgenti ValueError del parser pinned e la whitelist invariata. I callsite XML/EPUB condividono l'adapter: non si dichiara una nuova integrazione end-to-end per ciascuno. |
| Info fence e stato ai confini | Originali GPT/CLA, controlli Code inline e K2c, grammar e partizioni host nella stessa matrice; regressioni storiche ripetute. La copia privata `info` fallisce script/asset, `caller` azzera il chiamante radice e fallisce figura→tilde e lazy quote. |
| Metadata nei sei producer | `metadata-preservation.json`: 22 controlli dei valori effettivi; sei quoted HTML non diventano tag. Invalidi, attributi attivi, char/UTF8 e asset sono controllati nella matrice. Copia privata `metadata`: due discriminanti rifiutano perdita di script/asset. |
| Drain / prima causa / reset | Spy della validazione effettiva in `after-final.json`; tre aperture nell'originale F58, comprese le due dopo la prima causa. Mutazione URI produce ValueError e tronca la coda: controprova negativa. |
| CLI e uscita contrattuale | `cli-final/summary.json`: sei fixture sintetiche, 12 veri subprocess; quattro negativi 2/2, due safe 0/3 con gold PROVISIONAL. H01–08 passano; H12± conserva FAIL deliberato 1/2. |
| Corpus ricevuto / metriche | 64 ricalcoli senza motori. Distanze/catene/19 campi verificati; 16 delta in failures/details EPUB attribuiti all'AST, otto cambi dei metadata div TeX senza delta metrici. `epub-delta-attribution.json` enumera tutti i cambi. |

La grammatica permette token adiacenti: `a="x"b="y"` è due coppie; anche
`#chapter1.xhtml#start` è due dichiarazioni ID, ultima `start`. La prima
premessa manuale del microcaso aggiuntivo 159 prevedeva l'ID composto:
`cases-003.json` e `compound-id-before.json` la conservano e l'AST la smentisce.
`compound-id-after.json` conferma il risultato senza modifica del prodotto.
La preservazione riguarda quindi i valori della sintassi accettata, non la
concatenazione inventata dal vecchio tokenizer. `::: thebibliography` ha una
classe div, come `grammar-21`, non un token attributo sconosciuto.

Gli attesi dettagliati `expectations-001.json` sono una materializzazione delle
premesse condizionali congelate e degli AST della baseline; lo script non legge
output after. La materializzazione segue la prima osservazione after-001:
non è dichiarata un elenco interamente manuale scritto prima di quel runtime.
La selezione è finita: 157 casi principali, HC2 ricevuto e un caso ID, totale
159 input, massimo 242B, nessun seed. Non prova un parser nested universale.

O003: il valore utente `x` resta presente; `_guard_coverage=x` è sintassi
metadata invalida e gli underscore seguono il parser inline preesistente.
Non si promette il mantenimento letterale di quel nome invalido. La rimozione
pubblica del record privato ora richiede il tipo dict e `domain=producer`.

Il [report autore](../reports/implementation-r014.md) lega comandi, gate,
incidenti conservati, budget e passaggio al supervisore. Questa nota non è GO.
