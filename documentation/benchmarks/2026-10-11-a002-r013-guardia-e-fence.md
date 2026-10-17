# Run a002 r013 — guardie ordinate e proprietà dei fence

Nota17ott2026: riferimenti operativi eliminati o già indisponibili sono
resi testo; esiti storici invariati. [Disponibilità delle prove](../development/cleanup-2026-10-17.md).

2026-10-11. Implementazione locale F54–F57, dopo NO_GO r012. Reader
`observations-v13-ordered-guards-fences`, SHA256 `27d1b5809bc2dca51b75e50222aaab1eef86b2c907732c33537fbb5df20c43ad`.
Questa è una consegna dell’implementatore: nuove review ChatGPT/Claude e
arbitrato del supervisore restano da svolgere. Nessun GO, commit o integrazione.

## Risultato verificato

| Prova | Risultato |
| --- | --- |
| Medesimi 285 raw, attesi finali indipendenti dall’output reader | V12 132/285; V13 285/285 |
| Metadata LF supplementari, attesi e coordinate fisiche | V12 0/16; copia precedente 0/16; finale16/16, AST Span/Link16/16 |
| Pandoc locale 3.1.3, formato `markdown`, sotto R | 199 AST conservati: 139 decisioni ordinarie verificate, 60 raw confrontati separatamente |
| Copia privata attrs fail-fast | 0/3: detail o visite errati |
| Copia privata fence a ogni riga/EOF | 0/3: unsafe/visite errati e immagine sicura persa |
| Regressioni r012/r011/r010/r009 | 350/350, 153/153, 303/303, 308/308 |
| Suite veloce finale prescritta | 2997 passed, 310 subtests passed, R PASS |
| CLI reale con writer sintetico dichiarato | 43 casi principali +8 LF, 102 comandi run/evaluate; exit e detail attesi |
| H01–H08 / H12± | Otto R PASS; due R FAIL deliberati, harness exit1 contro R expected0 |
| 64 misure derivate, zero motori/nuove V | 19 campi prima/dopo invariati; distanza e aritmetica indipendenti PASS |
| JSON osservazioni | 640 privati rimossi in 56 risultati, ID reader aggiornato; altri delta0 |

I conteggi delle due campagne restano ciascuno 4PASS/28FAIL, come risultato
osservato. Non indicano miglioramento semantico della conversione.

## Meccanismo e perimetro

HTMLGuard raccoglie aperture reali con snippet e posizione assoluta; attrs
raccoglie attributi Markdown attivi e li esclude dai valori proiettabili.
Il collector deduplica per tipo/posizione/snippet e drena in ordine sorgente
dopo la costruzione dell’osservazione. Tutte le guardie successive sono
eseguite, anche dopo violazioni o errori d’albero gestiti. La prima Unsafe
del reader è rilanciata, il risultato unsafe non viene pubblicato e la
sessione viene sempre resettata. Il percorso standalone dei validatori
resta immediato. Non è un ordinamento globale reader+audit: A8 rimane limite
esplicito e testato; un path Markdown precedente può essere diagnosticato
dall’audit solo dopo un HTML unsafe successivo già rilevato dal reader.

`fence_decision` è comune a scanner, radice, container, boundary e consumer.
Tilde richiede inizio/separazione rispetto al paragrafo; un backtick chiuso
può interromperlo. EOF ordinario non assegna code. Il vero corpo letterale
resta inerte. L’AST distingue anche Code inline iniziale nelle liste e
Plain della prima riga di una nota: il marker visibile nella riga privata
del prefisso non dimostra da solo l’inizio di un blocco.

Raw esatto HTML/LaTeX, label LF, mappe, token opachi, F36 e F44 mantengono il
profilo ricevuto. In quattro raw list/backtick Pandoc produce Code inline,
mentre il profilo raw già esistente tratta i metadata esatti: la differenza
è conservata, non è rivendicata equivalenza Pandoc. Nei dieci casi tilde
dopo testo l’HTML del corpo è reale; un suo RawBlock AST non dimostra che
il fence possieda quel corpo. Raw LaTeX radice conserva la guardia HTML
preesistente, distinta dal contratto dei container. Formati html5/html-space
restano fuori profilo. Code senza owner provato non certifica coverage;
risorsa non proiettabile rimane unknown con raw/posizione/resource_syntax.

La pulizia ricorsiva di `_guard_coverage` avviene soltanto all’uscita di
observe, dopo consumo node/located/source-map. Il confronto JSON è distinto
dal confronto metrico; le 640 aggiunte storiche r012 rimangono storia.

## Inventario → proprietà → prove

Il censimento iniziale precede le modifiche. Le righe raggruppano callsite
con il medesimo contratto; i dati allegati enumerano tutti gli ID associati
e ogni proprietà effettivamente controllata. Una visita indica una singola
observe, con audit separato: nessuna somma di visite di due osservazioni.

| Fonte/callsite | Proprietà | Prova e risultato | Limite o complemento |
| --- | --- | --- | --- |
| attrs → peel_attrs → heading | Ordine raw del testo, metadata finale e coda; UTF8 | 9 casi PASS: `attrs-block-heading-0-before` … | Attesi e identità completi nei dati. |
| attrs → div, incluso oltre depth | Metadata visitato anche senza proiezione del div; coda completa | 8 casi PASS: `attrs-block-div-0-before` … | Attesi e identità completi nei dati. |
| attrs → fence | Metadata attivo distinto dal corpo code chiuso inerte | 7 casi PASS: `attrs-block-fence-0-before` … | Attesi e identità completi nei dati. |
| attrs → figure | Caption, metadata e coda in ordine sorgente | 8 casi PASS: `attrs-block-figure-0-before` … | Attesi e identità completi nei dati. |
| attrs → inline code/link/span | Metadata nel ledger e nelle ricorsioni; ordine diverso dal parsing in due passate | 35 casi PASS: `attrs-inline-root-span-before` … | Attesi e identità completi nei dati. |
| inline_html_events, cell/caption/heading/quote/list/note | Una visita per apertura con snippet, char e byte esatti | 36 casi PASS: `attrs-inline-root-span-before` … | Attesi e identità completi nei dati. |
| validate_html → HTMLGuard → PassiveHTML | Visite successive dopo più cause, safe prima/dopo e attributi attivi | 35 casi PASS: `attrs-block-heading-0-before` … | Attesi e identità completi nei dati. |
| HTMLTree e catch di proiezione | Errore duplicate attrs non interrompe eventi metadata/HTML successivi; guardia non duplicata | 1 casi PASS: `attrs-tree-error` | HTMLTree è guardato da un ledger precedente nelle chiamate del reader; la modalità standalone conserva validazione immediata e non promette ordine globale. |
| inline raw HTML / html_nodes / inline_delimiter / fallback | HTML realmente attivo, raw esatto e niente secondo feed dei literal | 63 casi PASS: `attrs-inert` … | Formati raw esclusi, F36 e owner HTML mantengono il profilo ricevuto; regressioni r009–r012 completano le prove pertinenti. |
| _observe raw fence / HTML blocco e catch | Coordinate assolute, guardia indipendente dall’albero e residui espliciti | 71 casi PASS: `raw-root-`-begin-html` … | Attesi e identità completi nei dati. |
| node, located, source_map, pulizia finale | Coverage consumata prima della pulizia JSON, risorsa conservata o residuo esplicito | 7 casi PASS: `attrs-domain-cell` … | Test dedicato cell/location/reference e regressioni; il privato non compare nel JSON finale, restano privati interni fino al consumo. |
| observe/finally e reset | Prima causa reader ordinata; nuova chiamata senza sessione residua | 2 casi PASS: `attrs-inert` … | Tre test reset (UTF8, attrs e HTML), tre raw EOF vuoti; limiti d’ingresso non avviano una visita del documento invalido. |
| PassiveHTML adapter | Tag, attributo fuori subset/attivo, meta, URI assente/vietata/encoded/evasiva | 13 casi PASS: `validator-0` … | Validatore e audit non modificati; nessuna apertura URI o rete. |
| fence_decision: radice/scanner/container/boundary/consumer | Tilde non interrompe paragrafo; ordinario EOF non possiede code; vero owner inerte | 139 casi PASS: `fence-root-`-begin-False-False` … | 139 asserzioni AST sul corpo discriminante; nessuna grammatica Pandoc universale. List backtick iniziale può essere Code inline; heading iniziale note può essere Plain. |
| audit/observe_bundle/CLI | UNSAFE/detail/exit2; asset safe o unsupported esplicito | 41 casi PASS: `attrs-block-heading-0-before` … | A8 resta limite non bloccante: path Markdown è controllato dopo observe, non esiste ordine globale reader+audit. Test dedicato conserva ../second.png come detail. |
| attrs LF / InlineDecision.attributes / source_map | Separazione LF/CRLF e coordinate fisiche per ogni token attivo, anche dopo valori quotati su LF | 16 casi PASS: `metadata-LF-quote-span-0` … | 16/16 attesi manuali e AST Span/Link con onclick; V12 e copia precedente 0/16. Matrice supplementare separata dai285. |

## Attesi, correzioni e controprove

Generatore finito separato, nessun import prodotto, seed identitario13013,
285 casi principali, 16 supplementari LF e massimo186B contro limiti400/2048B. Sono297 raw unici contando anche le premesse intermedie e i due probe esplorativi (con sovrapposizioni), non297 PASS finali. Gli attesi di sicurezza,
asset e coordinate sono manuali; Pandoc è l’oracolo sintattico, non l’oracolo
semantico. Probe/test helper trasportano byte e registrano osservazioni.
L’autore resta lo stesso implementatore, non un revisore indipendente.

Prima di ogni nuovo runtime sono stati congelati raw e attesi. Le versioni
intermedie e i loro fallimenti sono conservati. Le correzioni delle premesse:

- List/backtick chiuso può essere Code inline invece di CodeBlock: corpo
  comunque letterale, nessun gold semantico copiato dal reader.
- LaTeX raw radice mantiene la guardia ricevuta; l’ipotesi iniziale di HTML
  sempre inerte era troppo ampia.
- Il caso tree-error iniziava un blocco HTML, dove i metadata Markdown
  erano letterali: aggiunto prefisso `È ` per selezionare il producer inline.
- Gli AST di list-code e delle due note-heading smentivano tre premesse:
  verificatore corretto per cercare il corpo discriminante nell’owner,
  non un qualsiasi Code precedente. Raw004/005 identici; attesi005 congelati
  prima del runtime finale, V12 riletto sugli stessi byte.
- Venti vecchi attesi r012 per fence ordinario EOF rettificati tramite
  overlay separato, preservando la fixture350 originale. La prima suite
  completa ha poi rilevato l’atteso F29 storico: AST aggiuntivo conferma
  assenza di Code/CodeBlock. Il commento aperto conserva il contratto opaco
  del tokenizer Python, che non è il parsing generale dei commenti Pandoc.

Il controllo finale ha trovato ancora offset logici negli attrs multilinea e
LF non riconosciuta come separatore di token:16 nuovi raw congelati e AST
Span/Link hanno confermato entrambe le cause. `InlineDecision.attributes`
trasporta la mappa di ogni carattere; attrs separa LF/CRLF fuori dalle
stringhe quotate. V12 e la copia precedente soddisfano0/16, il reader finale
16/16. Suite, matrice, CLI, rescore, controprove e contratti sono stati
ripetuti sulla nuova versione; dati e documenti provvisori conservati in
evidenza non vengono attribuiti al nuovo SHA.

Sono stati corretti nella stessa chat il trasporto delle righe vuote delle
liste, lo stato dopo list-code/note-heading e l’indice del raw HTML vuoto
a EOF. Nessun problema ordinario è stato rinviato al supervisore.

Le due mutazioni reintroducono solo le assunzioni richieste in copie private.
Diff, hash e output completi sono nei dati. La prima tronca detail/visite;
la seconda nasconde HTML reale e perde l’asset B4. I sorgenti correnti e il
corpus protetto non sono stati mutati per la controprova.

## Provenienza delle prove e limiti

Ingresso `implementation-fix-context-r012` verificato MATCH con argv/cwd/
exit/costo conservati. V12 fisico SHA256
`5cabe6778ed53149ad02e1f50aaea6f334fc1c7ff3f9ffd38dd7f87c010abc22`.
Pandoc SHA256 `3dd273647f0265cb439f22976d5366a54b071a3783f6fec50838b47fb53d701b`;
argv AST `/usr/bin/pandoc -f markdown -t json`, input stdin, exit0.
Gli AST sono del Pandoc già installato, sotto R senza rete.

Workload finali, suite e controprove in stage-s008 dopo entrambi i fix;
CLI supplementare LF in stage-s009 con gli stessi byte Python:
managed CPython3.12.13, sdist/wheel ricostruite e wheel canonica reinstallata,
S/B/I, origini/startup, dieci moduli e helper esterni identificati prima
degli import. Nessun nuovo modulo. I dati conservano ricevute R, snapshot
tecnici e installazioni dei workload. Il binding documentale finale e la
contabilità di consegna sono nel report autore della run; non si trasferisce
un PASS usando un general snapshot diventato STALE.

La suite esclude API/packaging/Docker e i due smoke radice; non è collaudo
completo della pipeline legacy. I gold CLI sono PROVISIONAL, il writer copia
fixture dichiarate: sono verificati contratti OS/artefatti, non qualità
nativa dei motori. I 64 rescore conservano conversion_final e precedente
derived_final; il checker indipendente usa distanza a matrice completa,
aritmetica e hash/catene, senza import reader/scorer/generatore.

L04 formule output resta UNAVAILABLE, distinta dalla chiusura V13/A03 sul
riferimento. L01/L02 NOT_MEASURED, L05 e gli altri NONPASS storici, helper12gap,
Chrome/H15/E9 non sono sanati. Nessun motore, modello, rete, acquisizione,
invio, costo servizio o nuova V. Nessuna deadline; quote ricevute invariate.
CPU lower e RSS sampled, amministrazione in parte non misurata, nessun
upper certificato. EPERM del socket diagnostico recuperato con escalation
autorizzata mantenendo Firejail netnone. Tre checker stdlib hanno avuto una
collisione fra nome risultato e receipt host: risultati/log preservati,
costo amministrativo non inventato. Non è stato un difetto del reader.

Il prossimo passo è del supervisore: congelare il comune, ottenere due
nuove review reali ChatGPT/Claude e arbitrare. Registri condivisi e CHANGELOG
sono riservati al supervisore. Nessuna promessa di ultimo ciclo.

Dati, AST e oracoli completi (riferimento storico non disponibile) ·
[Protocollo](../../scripts/benchmarks/PROTOCOL.md#rettifica-r013-fonti-di-guardia-e-contesto-dei-fence) ·
[Report precedente preservato](2026-10-11-a002-r012-atomi-e-container.md).
