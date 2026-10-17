# a002 r014 — contratti di sintassi

Nota17ott2026: riferimenti operativi eliminati o già indisponibili sono
resi testo; esiti storici invariati. [Disponibilità delle prove](../development/cleanup-2026-10-17.md).

2026-10-11. F58–F62 implementati e verificati nei domini finiti descritti,
consegna al supervisore; nessun GO, integrazione o nuova V. La nuova applicazione
resta in progettazione: questi risultati riguardano il reader del benchmark.

Fonti: [mandato46](../../temp/run-a002-fase1-corpus/prompts/46-implementation-r014-align-block-metadata-and-uri-contracts.md),
[registro dei contratti e callsite](../../temp/run-a002-fase1-corpus/notes/implementation-r014-contracts.md),
[report autore](../../temp/run-a002-fase1-corpus/reports/implementation-r014.md),
dati e provenienza (riferimento storico non disponibile).

## Contratti realizzati

`observations-v14-syntax-contracts` usa una decisione sintattica condivisa per
metadata e proprietà dei blocchi. Una info fence invalida o un gruppo di
graffe non accettato non rende inerte il suo HTML. Il ledger fisico conserva
stato del paragrafo e host tra scanner e producer; marker di quote/list/note/
heading non interrompono un paragrafo se il profilo pinned richiede separazione.
I controlli Code inline backtick e lista lazy K2c rimangono inerti quando
l'AST del corpo dimostra quell'owner.

Metadata accettati conservano valori, id, classi e posizioni; quoted HTML resta
un valore. L'adapter URI converte in Unsafe i prevedibili ValueError del solo
`urlsplit`, consentendo al drain di conservare prima causa e coda. Non cambia
whitelist, non apre URI, non cattura globalmente errori estranei al parsing.
Si verifica il contratto del parser URI utilizzato, non una validazione RFC
universale.

## Esiti osservati

| Prova | Esito |
| --- | --- |
| Originali V13+adapter → byte finali, stessi157 casi | 112/157 → 157/157; 45 violazioni riprodotte e corrette |
| AST pinned Pandoc3.1.3 | 157 casi principali, HC2 ricevuto, un discriminante ID; 159 input, massimo242B |
| Suite veloce selezionata | 3156 passed e310 subtests passed; r013301 inclusi |
| Matrici dedicate | r012350, r011153 con overlay, r010303, r009308 passano |
| Quattro copie private | Tutte respinte, sette discriminanti su URI/info/stato chiamante/metadata |
| Metadata effettivi | 22 controlli passano, inclusi sei valori HTML quoted nei diversi producer |
| CLI sintetica dichiarata | 12 processi reali: quattro unsafe2/2 e due safe0/3 run/evaluate, gold PROVISIONAL |
| H01–08 / H12± | Otto harness0 sotto R PASS; due R FAIL deliberati, harness1/wrapper2 |
| 64 ricalcoli senza motori | Catene/hash/distanze/19 campi verificati; due campagne da4PASS/28FAIL osservati |

La suite esclude API, packaging, Docker e due smoke radice; non è un collaudo
completo. Nelle prove prima/dopo sono identificati sia reader sia adapter:
la baseline non combina V13 con safety nuovo. Gli attesi dettagliati sono
materializzati dalle premesse congelate e dagli AST della baseline, senza
leggere gli output after; questa materializzazione avviene dopo la prima
esecuzione after-001 ed è distinta dal freeze iniziale.

## Delta e correzioni delle premesse

Nei 16 EPUB cambiano solo failures/details, con zero delta numerici o di gate.
`#chapter1.xhtml#start` è interpretato da Pandoc come due dichiarazioni ID,
ultima `start`; il vecchio reader conservava erroneamente un ID composto.
AST e output effettivi giustificano i nuovi attrs/anchors. Target dei link,
testo, asset e formule rimangono quelli ricevuti. Una mia premessa iniziale
contraria è conservata insieme all'AST che la smentisce. Otto TeX cambiano
solo metadata dei div `::: thebibliography`, ora classi anziché token unknown,
senza delta metrici. I dati enumerano questi cambi; nessun conteggio è forzato.

HC2 r011 è rettificato mediante overlay separato: l'AST mostra HTML reale in
graffe invalide. CASES, raw, gold e atteso storico sono conservati. O003 rimane
circoscritto: il valore utente sopravvive e si elimina solo coverage interna
tipizzata; per `_guard_coverage=x`, sintassi metadata invalida, non si promette
la grafia letterale del nome dopo la formattazione inline degli underscore.

La baseline r013 corretta **132** cita check-baseline-005/cases-005 sugli
output baseline-004. **133/check004/attesi004 e freeze005** sono storia
conservata nel [report r013](2026-10-11-a002-r013-guardia-e-fence.md), non riscritta.
La sua garanzia generale era parziale come stabilito dall'arbitrato r013.

## Riproducibilità, risorse e limiti

Managed CPython3.12.13, sdist/wheel ricostruite e wheel canonica reinstallata
offline in stage-s002; dieci hash e37 helper/Python esterni controllati.
Snapshot tecnici, copie puntuali e R Firejail/net=none identificano gli input
reali. I dati contengono gli hash finali; argv, cwd, exit e output completi
restano nelle evidenze della run. R final-001 passa; H12 mantiene i FAIL voluti.
Nessun nuovo motore, modello, dipendenza, download, servizio o nuova V.

Il [ledger finale](../../temp/run-a002-fase1-corpus/evidence/implementation-r014/budget-final.json)
mantiene contatori e limiti ricevuti, senza deadline o nuova quota. Il costo
proprio supera la stima circa180CPU-s; gli errori e le ripetizioni sono
contabilizzati, il margine per i ruoli successivi è ristretto. CPU lower,
RSS sampled e scope5 deduplicato non sono upper certificati. Nessun reset,
cleanup o sanatoria dei NONPASS. La ricevuta persa per collisione di nome
rimane un limite di misura esplicito, non un costo stimato presentato come reale.

Restano F36B1–B4/A8, quattro osservazioni r010, raw/EOF ricevuti, L01/L02/L05,
L04 output UNAVAILABLE distinta da V13/A03 chiusi, Chrome NONPASS, helper12gap,
H15/E9 e FAIL/STALE storici. Non è un parser nested universale.
Seguono ricezione del supervisore, nuovo comune, due nuove review indipendenti
ChatGPT/Claude e arbitrato; nessun commit o merge automatico.
