# A002 r011 — ownership e metadati inline

Realizzato il 2026-10-10, run `run-a002-fase1-corpus`, reader
`observations-v11-inline-ownership`, SHA256 `af5299bb0c52bd4a0634a1aebb8a28743e2793c315eec5bbaf9b6424a98d8482`.
Consegna implementatore da ricevere e sottoporre a due NUOVE review
ChatGPT/Claude e arbitrato; nessun GO autore o integrazione Git.
[Dati](2026-10-10-a002-r011-ownership-inline-data.json),
[report autore](../../temp/run-a002-fase1-corpus/reports/implementation-r011.md).
Il [report r010](2026-10-10-a002-r010-confini-html-markdown.md) resta storico:
le due review reali NO_GO hanno dimostrato F46–F48.

## Correzioni e dominio effettivo

F46: i metadata heading si riconoscono soltanto nel blocco sorgente.
Una riga logica `# … {…}` dopo stripping list/quote/note o in cella,
caption e testo di heading rimane contenuto della foglia. HC2 mantiene i
metadata del vero heading; H7 distingue testo interno e attributi `.c`.
H1/H2/H4–H11 e HC1 sono UNSAFE con detail. H3 sicuro sceglie esplicitamente
INCOMPLETE/UNSUPPORTED per risorsa non rappresentata, senza apertura p.png.

F47: la label finita già accettata conserva la LF posseduta dal code span
chiuso anche nella guardia math. Una LF libera non diventa label accettata;
un token opaco non presta chiusure dal proprio corpo al literal precedente.
Gli otto originali e varianti conservano formula raw, link, testo, tre blocchi
esterni, coordinate char/UTF8 e nessuna apertura asset letterale. Gold manuale
include la formula lessicale: struttura matematica fuori subset ed L04 non
sono un PASS semantico, score5/INDETERMINATE, senza falso UNSAFE.

F48: il ledger code della label è subordinato allo stack HTML. Backtick
interni a uno span aperto sono testo HTML, quindi tutte le aperture reali
vengono validate in ordine, prima unsafe conservata e visite successive
continuate anche dopo un errore d'albero. Dopo chiusura dell'owner il code
Markdown torna inerte. Quote complesse già proiettate/guardate inline non
passano di nuovo il raw completo a HTML nel fallback node unknown.

## Prove effettive e oracoli

153 casi, 34 originali accolti (13 Claude H/HC e21 GPT cases003), poi39
metadata/chiusure/continuazioni,16 math-code label,48 ownership e16 ordine/
errore/controlli. Trasformazioni deterministiche dichiarate nel test, nessun
seed casuale o prodotto cartesiano illimitato. Attesi manuali precedenti,
stessi raw/gold/expected nel confronto finale: v10 27/153,
v9 37/153, v11 153/153. Il numero v9 include nuove asserzioni
sulle visite, non sostituisce il20/21 del reviewer o i suoi vecchi gold negativi.
Matrice r010303/303 e r009308/308 preservate; F36 B1–B4 ancora UNSUPPORTED.

Mirati dopo recupero:2102pass. Fast reale s005/RPASS:
2327pass/310subtest. Esclusi API/packaging/Docker e due smoke
radice; non collaudo completo della migrazione. CLI sintetica dichiarata:
37fixture/74exit OS, all_expected=True.
H1/H6run2/evaluate2; H3run2/evaluate2/score3 per alternativa unsupported;
F47run0/evaluate5; F48run2/evaluate2 e detail della prima apertura.
GoldPROVISIONAL, writer copia byte, motori0/nuovaV0. H01–H08R PASS;
H12±R FAIL deliberati con harness1/wrapper2 ed expectedR0, prove conservate.

64 raw ricalcolati senza motori: {'baseline-r002': {'attempts': 32, 'quality_counts': {'FAIL': 28, 'INDETERMINATE': 0, 'PASS': 4}}, 'historical-r001': {'attempts': 32, 'quality_counts': {'FAIL': 28, 'INDETERMINATE': 0, 'PASS': 4}}}, differenze effettive di19
campi numerici/gate rispetto v10: 0.
Checker indipendente per catena/albero raw, domini/denominatori/distanze
char/token e aritmetica, senza import reader/scorer; non alignment strutturale
interamente indipendente. Conversion_final originale r001 resta distinto
sia dal previous_derived_final r010 sia dal rescore r002 citato nei dati r009.
56EXACT_GLOBAL_DISTANCE/8TRIVIAL_ZERO osservati: conteggi condizionati, bound
globale e misura completa distinti, TRIVIAL_ZERO non fullPASS.

## Errori e limiti preservati

Raccolte iniziali con due errori miei di checker (visita immagine safe e
primo `[` della note) conservate; correzione senza cambiare raw/gold/expected.
Targeted0012098pass/4fail F31r007: eliminazione LF prestava chiusure nel token;
nuovo vincolo sugli atomi accettati, S/B/I e prove dipendenti ripetute.
S001 precede un cambio del checker e non viene presentato come gate corrente.
S009r010 generale storico STALE per4registri; gate reali origine/startup/
package3.12.13 prima del primo import, build/reinstallazione canonica e
10hash. Stage finale documentale lega gli stessi Python delle prove s005.
R Firejail/netnone,shellFalse,limiti/registry invariati; escalation tool del
socket già autorizzata, nessun rifiuto o privilegio persistente.

F36 separa il prefisso sintattico prima di un token opaco, anche dove CPython
puro riconosce aperture: limite deliberato senza secondo feed puro. Rimangono
le quattro osservazioni Claude non bloccanti: backtick-run ereditato diverso
da CommonMark; raw `{=html5}`/`{=html }` fuori profilo esatto con Pandoc attivo
non confermato; timing v10~7,3s/v9~1,4s su4000commenti, debito finito senza
soglia violata e senza ottimizzazione qui; fallback unsafe observations-v1
in benchmark.py fuori delta, distinto dal reader reale attribuito con byte/R.

Corpus/gold/raw/V13/A03 preservati, L04outputUNAVAILABLE distinta,
L01/L02NOT_MEASURED,L05CPUlower/RSSsampled eNONPASSstorici/gap12helper/H15/E9.
Budget senza reset/nuove acquisizioni, servizi0/€0. Risorse finali e processo
nel report/checkpoint/evidence autore; stop11ottobre06:37:31Z/deadline09:37:31Z.
Le prove finite non attestano ogni grammatica, renderer o formato.
