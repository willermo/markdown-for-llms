# A002 r010 — confini HTML e Markdown inline

Nota17ott2026: riferimenti operativi eliminati o già indisponibili sono
resi testo; esiti storici invariati. [Disponibilità delle prove](../development/cleanup-2026-10-17.md).

Realizzato il 2026-10-10, run `run-a002-fase1-corpus`, reader
`observations-v10-inline-html-domains`. Consegna dell'implementatore, ancora
da sottoporre a supervisione, due review indipendenti e arbitrato; nessun GO
autore o integrazione Git. Dati r010 (riferimento storico non disponibile),
[report autore](../../temp/run-a002-fase1-corpus/reports/implementation-r010.md).
Il [report r009](2026-10-10-a002-r009-confini-reader.md) conserva la storia.

## Cause e correzione

F43: `inline` restituiva unknown fino al limite della foglia per code/math
incompleto o link malformato, nascondendo tag che il lexer aveva riconosciuto.
F44: l'albero HTML riceveva la coda Markdown prima di selezionare il primo
subtree, validando img/script/onclick letterali nel code chiuso successivo.
La nota tecnica dell'autore precede le modifiche al prodotto.

Il medesimo walk `markdown_lexemes` espone aperture HTML e intervalli dei
subtree nel dominio già trasportato dal producer. La guardia inline valida
le aperture prima della proiezione o del fallback, conserva la prima
violazione PassiveHTML in ordine sorgente e prosegue le visite. Code/math
chiusi e token sorgente opachi non generano aperture. Un delimitatore senza
chiusura accettata non è un owner letterale. Se il fallback perde una
risorsa sicura riconosciuta, conserva almeno resource_syntax e un audit
INCOMPLETE/UNSUPPORTED; non diventa COMPLETE con asset scomparso.

Il feed dell'albero e il lookahead dell'enfasi usano la singola foglia HTML,
compreso il subtree bilanciato. Un `<span>` chiuso non possiede il code
Markdown successivo. Un owner HTML aperto conserva invece il dominio HTML
finito; non è una promessa di tutta la semantica CommonMark. I producer
HTML blocco/raw mantengono la guardia del proprio frammento. Il mapping
char/UTF8 r009 e la decisione tag/attributi F40 restano invariati.

## Matrice F43–F45

303 casi manuali deterministici, più due controlli delle visite dopo la prima
violazione. Sei producer: paragraph, quote, list, note con spazi, note con tab,
cella. Ordine HTML/code, uno/due backtick, chiusura accettata/estranea, math
chiuso/aperto, broken link anche con code interno, risorse safe/traversal,
script/event, HTML bilanciato/duplicati/depth17, comment/CDATA chiuso/EOF e
Unicode. Nessun atteso derivato dal reader, generazione casuale o nuova V.

| Gruppo | Casi | v9 rispettati | v10 rispettati | Contratto |
| --- | ---: | ---: | ---: | --- |
| Aperture unsafe e priorità | 142 | 15 | 142 | UNSAFE_REFERENCE/detail, score3/evaluate2, path evasivo mai aperto |
| Code letterale chiuso e HTML bilanciato | 78 | 38 | 78 | Testo/blocchi/map preservati, nessuna reference/asset, PROVISIONAL automatic PASS/qualità INDETERMINATE/score5 |
| Risorsa sicura persa dal fallback | 19 | 6 | 19 | INCOMPLETE/UNSUPPORTED esplicito, senza lettura o proiezione inventata |
| Errore HTML con literal esterno | 24 | 12 | 24 | Unsupported, senza falso unsafe dal literal |
| Math chiuso con tag letterali | 12 | 6 | 12 | Formula raw preservata, nessuna immagine; gold volutamente privo di formula è negativo di scoring |
| Token opaco chiuso/EOF | 24 | 24 | 24 | Corpo intero, coordinate originali, nessuna prosa/formula/risorsa del corpo |
| Divergenza F36 B1–B4 | 4 | 4 | 4 | Unsupported deliberato, senza apertura di risorse |
| Totale | 303 | 105 | 303 | Confronto sugli stessi byte e oracoli |

I 14 discriminanti Claude A/B sono identici ai byte originali, compresi i
terminatori di riga; i sette ChatGPT hanno gli stessi raw e blocchi gold.
A1/A2/A3/A4/A5/A6/A7/A8/A10 sono unsafe; A9 usa l'alternativa unsupported
esplicita consentita dal mandato. I positivi ChatGPT mantengono tre blocchi,
nessuna reference/asset e score5; il controllo con img reale resta unsafe.
La matrice r009 conserva 308/308 sui suoi oracoli immutati.

F36 è una divergenza deliberata dal parser puro CPython 3.12.13: B1
`<div <!-- -- ><img ...>`, B2 `<img <!-- src=...> -->`, B3 variante raw con
script e B4 `<alpha <!--` multilinea possono produrre aperture/attributi nel
parser puro. Il profilo finito separa il prefisso e mantiene il token opaco:
UNSUPPORTED, senza leggere risorse. I dati registrano le aperture pure reali.
Non attestiamo parità completa col tokenizer né trasferiamo questa eccezione
ai fallback Markdown F43.

## Verifiche e limiti

Prima selezione mirata: 2015 passed, 16 F04 deselected. Include r002–r010,
comparison e writer, con preflight e R correnti. Suite veloce finale: 2174 passed e 310 subtests passed, R PASS su stage-s007.
Il binding indipendente conferma dieci moduli source/wheel/installati uguali,
65963 input protetti, 129 file corpus e 92641 artefatti ricevuti invariati.
Il report autore identifica anche il binding amministrativo dopo queste
rettifiche documentali; i byte Python delle prove restano invariati. API/packaging/Docker e i due smoke radice restano esclusi;
questa selezione non certifica tutta la migrazione. Nessun download o modello.

41 fixture CLI, 82 comandi run/evaluate, exit OS reali e manifest di attesi
prima dei figli. Writer sintetico dichiarato, zero motori: non una nuova
conversione nativa. Unsafe/unsupported danno run/evaluate2 e score3; positivi
PROVISIONAL run0, score/evaluate5, qualità INDETERMINATE. H01–H08 R PASS;
H12± R FAIL voluti con expected R0/harness1/wrapper2, senza nascondere gli exit.

64 nuove misure derivate sui due set32, nessun nuovo motore o V. Ogni set
mostra 4PASS/28FAIL senza assert che imponga tali conteggi. Checker indipendente
con matrice completa char/token, decomposizioni e bound globali, catena e
identità dell'intero albero raw: zero differenze numeriche/gate rispetto a r009.
56 misure AVAILABLE con EXACT_GLOBAL_DISTANCE; otto UNAVAILABLE con
TRIVIAL_ZERO. I conteggi condizionati non sono il bound garantito; zero
triviale non è una misura completa. Scorer/F26/normalizzazioni/denominatori
restano immutati. V13/A03 riguarda il riferimento; L04 sull'output resta
UNAVAILABLE, L01/L02 NOT_MEASURED, L05/hardware/costi e gap storico dei 12
helper restano separati. Nessuna nuova qualifica del riferimento.

Preparazione managed locale CPython 3.12.13, build sdist/wheel, reinstallazione
della wheel canonica e confronto dei dieci moduli precedono i workload. Le
copie prima/dopo identificano i byte consumati. Gli errori iniziali del
collector, il percorso relativo rifiutato dal launcher e lo stage stale sono
conservati e corretti nella stessa chat; non sono PASS retroattivi. S/B/I e
snapshot propri attribuiscono le prove, senza approvare il risultato.

Risorse senza reset, anchor24h 2026-10-10T09:37:31Z, stop nuovi workload
2026-10-11T06:37:31Z e deadline09:37:31Z, riserva tre ore. CPU lower e RSS
sampled non certificano upper bound; disco misurato sullo scope attuale con
deduplica, senza sommare nuovamente il retained storico. Zero rete/acquisizioni/
invii/servizi e €0. Il ledger autore contiene misure e limiti effettivi.

Prossimo ruolo: supervisore, common nuovo, due review ChatGPT/Claude in chat
nuove sul medesimo risultato, arbitrato. Commit/merge/push restano dell'utente.
