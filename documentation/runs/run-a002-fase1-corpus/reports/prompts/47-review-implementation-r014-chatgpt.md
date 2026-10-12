# Review implementazione a002 r014 — ChatGPT/OpenAI

Agisci esclusivamente come reviewer ChatGPT/OpenAI in una **NUOVA chat indipendente**,
mai chat autrice o supervisore. Non implementare, delegare o impersonare altri ruoli.
Non leggere report/evidenze/checkpoint del reviewer concorrente r014.

## Oggetto comune e identità

Run `run-a002-fase1-corpus`, r014, mandato46 F58–F62 dopo NO_GO r013 reale.
Consegna completa ricevuta; chiusure da giudicare. Due review, arbitrato e GO
r014 assenti alla preparazione. Stessa run e storia, nessuna ripianificazione.

- Comune `temp/run-a002-fase1-corpus/snapshots/implementation-r014-review-r002.json`.
- SHA256 `e0436b5ee2c1516beabfcaf5fd7c895f57c69ad5cecf71ea54a0d0c6b832cc56`; impronta `5e72dd4ea0204bd3b613338a2e6d603161e17c6a5616e4793806c937ffe79a5b`.
- 349 file generali / 2504 artefatti; snapshot 770964B.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`, indice vuoto alla ricezione.
- Reader `observations-v14-syntax-contracts`,83734B,SHA256
  `613aaa996fcf02200bc6a8ad499ba0ebc044f44ed94deae3f7e638905215cefd`.
- Safety13442B,SHA256 `974656f69e022fcaa810c0828079e8672f59db8e8f8131e0b923f38b4e2e32b2`.
- V13+adapter originali insieme in evidence/implementation-r014/received-bytes/,
  manifest.json: reader `27d1b5809bc2dca51b75e50222aaab1eef86b2c907732c33537fbb5df20c43ad`,
  safety `31aa1d14aeb8bf31894e3ff0b53765dbf1ea1ec52acf57028571f513231332d0`.
- Piano `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`,
  errata `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.

Receipt/inventario/transizione r002 in evidence/supervisor-implementation-review-r014/.
Comune completo originale implementation-r014 conservato byte per byte e già
MATCH alla ricezione; il comune r002 mantiene lo stesso prodotto/criteri e lega
quell’inventario per hash, con prove consolidate/decisive dirette. Non cleanup oGO.
Se usi un artefatto storico/per-caso non incluso direttamente nel comune r002,
confrontane i byte con il descrittore dell’inventario completo originale prima
di attribuirgli una prova. Non serve ricontrollare gli altri artefatti non usati.
Errore amministrativo supervisore sui link al ledger pendente conservato,
recupero stdlib senza prodotto; contabilità di quel costo parzialmente misurata.
Inventario non elenco letture obbligatorie. Storia originale legata ai byte
dei vecchi snapshot e ricevute; 1042 record ingresso immutati e sei delta
autorizzati verificati. R014 evidenze proprie correnti incluse; payload terzi
installati esclusi salvo dieci origini. Nessuna cancellazione/reset/checkout/query
remota. Registri vivi, prompt47/48, CP supervisore corrente e output reviewer
r014 esclusi. Il delivery autore MATCH alla ricezione non è il comune o un GO.

## Risorse: nessuna deadline, review mirata entro residuo CPU

Disposizione addendum-resources-r012-r001 e notes/user-no-deadline-r012-2026-10-11:
nessuna deadline/durata massima/stop temporale. Entry2026-10-08T07:06:54.369068Z
e contatori senza reset. Leggi clock reale per accounting e ledger corrente
**evidence/supervisor-implementation-review-r014/budget-final.json**.

CPU21600/**stop19800s**, disco6/**stop5GiB**, RAM8/stop sampled6GiB,
riserva1800CPU-s/1GiB, R≤900s/comando, stream1MiB/file32MiB,
rete/acquisizioni/invii0/€0benchmark e isolamento/SBI invariati.
Base autore19639,444469CPU-s lower; prima della ricezione160,555531s allo stop,
40,555531s dopo120s prudenziali per interruzione GPT non misurata. Il ledger
corrente sottrae amministrazione reale, non azzera contatori. Riserva120s non
è consumo misurato né margine da spendere implicitamente; 45sClaude r007approx
separati, admin parzialmente misurata, CPUlower/RSSsampled non upper certificato.

**Pianifica≤10CPU-s propri misurati**, inclusi entrambi verify, letture JSON/hash,
preflight/R/poche sonde/errori/output. È un tetto di pianificazione condiviso,
non nuova quota: verifica anche il residuo effettivo e i costi concorrenti noti,
senza leggere la loro review. Proteggi almeno il margine per l'altro reviewer
e arbitrato. Analisi statica approfondita; poche microsonde decisive consentite
solo se previsione e gate rientrano nel residuo. Non lanciare una campagna
che prevedibilmente non vi rientra. Non è obbligo di consumare10s o di sondare
ogni caso; documenta i limiti delle conclusioni statiche.

Nessuna suite/build/reinstallazione/rescore/H01–H12 completo/conversione/nuovaV/
motore/ML/download/cleanup. Per eventuali microsonde: R Firejail/net=none,
managed locale CPython3.12.13, gate prima import e input/copie effettivi.
Canonica stage-s002 autore già installata: non ricostruire byte che non hai
modificato. General S002 diventa storico STALE per quattro registri amministrativi
supervisore: conserva/attribuisci, non trasferire PASS o importare con gate fallito.
Nuovo comune/snapshot tecnico e preflight attuale legano input immutabili S/B/I,
dieci origini e Python/helper correnti prima import, come guida/protocollo.
Microparsing AST Pandoc3.1.3 locale sottoR solo per premise sintattiche decisive,
non nuovo motore/conversione. Attesi indipendenti prima delle esecuzioni;
baseline V13 eventuale usa **anche adapter V13**, non un confronto ibrido.

EPERM/EACCES: diagnosi e escalation tool autorizzata se necessaria, isolamento
invariato e rispetto di rifiuto effettivo. Recuperi ordinari nella stessa chat.
Al limite sostanziale reale CP/disposizione necessaria, nessuna estensione
implicita; lavoro indipendente coperto può continuare. Nessun NO_GO inventato
dalla sola assenza di sonde vietate dal budget, nessun GO se requisito bloccante
rimane o input non attribuiti. Giudica il prodotto sulla prova disponibile.

## Letture selettive e verify

Usa manage-implementation-run e verify-conversion-fidelity. Percorsi sotto
temp/run-a002-fase1-corpus salvo indicazione diversa. Nell'ordine:

1. AGENTS,temp/HANDOVER,prima parte corrente PROJECT-CONTEXT,STATE/HANDOVER/brief,
   questo prompt/CP supervisor-implementation-review-r014; skill, run-lifecycle,
   ADR0001, addendum r012 corrente/no-deadline, ledger.
2. Arbitrato r013, mandato46 e notes/recovery-method-r014 **integrali**;
   entrambi report/checkpoint reviewer r013 integrali e prove decisive ricevute:
   GPT cases001/002/probe-results001/002/owncheck001/AST/oraclecorrections;
   Claude cases.py/check002/out-probe001/AST K1–K4/A/P e correzione K2c.
   Sono storia comune ricevuta, non review concorrente r014.
3. Report/checkpoint autore r014 integrali, notes/implementation-r014-contracts
   anteriore al codice e sezione finale distinta; permanente2026-10-11-a002-r014-
   contratti-sintassi.md/-data.json; delivery-checks002, budget-final, snapshot
   delivery-r001/verify e runtime-byte-binding. Ricezione supervisore/inventario.
4. Reader/safety pertinenti integrali e delta dal V13+adapter fisici; nuovo test
   r014/fixture e overlay r011, PROTOCOL/due guide. Traccia scanner/ledger fisico/
   fence/metadata/token/producer/consumer/ricorsioni/guard/drain/fallback/node/
   located/source-map, PassiveHTML/audit/observe_bundle/CLI e callsite URI XML/EPUB.
5. freeze001/002,cases001/002/003,baseline001/grammar001/checks,after-final/
   check-after-final/expectations001 e generatore/checker in work/implementation-r014.
   mutations-inputs/results e quattro diff/copie, metadata-preservation/22 valori,
   hc2-ast/overlay, compound-id-before/after e premessa smentita. Attesi dettagliati
   materializzati DOPO primo after001 dai raw e AST baseline, senza leggere after:
   non dichiararli tutti manuali congelati prima, valuta indipendenza effettiva.
6. final-workload fast/summary19, matrici r012350/r011153/r010303/r009308,12CLI,
   H01–08/H12summary/receipt, R-final001/copie132/binding69; stage-s002 S/B/I/
   archivi/preparazione/interprete/origini10/helper37. Fallimenti EPERM, label,
   collisione receipt after001, regressione coverage/HC2 e checker delta conservati.
7. rescore001/summary,derived-check001/check-derived.py,epub-delta-attribution/
   finish-checks.py e manifest/catene/raw/gold. Ricevuta supervisore 19 campi:
   zero delta numerici/gate;16 EPUB failures/details eotto TeX metadata div.
   numeric_gate_differences=16 nel checker conta TUTTI i dizionari differenza,
   incluse diagnosi; non è16 differenze numeriche. Verifica dati e attribuzione,
   non correggere quel record storico. Distanze autore indipendenti ricevute,
   non una tua nuova prova. Piano/errata/mandati05–06 e F31/F36/F40/F41/limiti
   dove decisivi; non tutti i log storici o payload installati.

All'ingresso e dopo i tuoi output conserva argv/cwd/output/exit/costo:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r014-review-r002
```

STALE: conserva output, attribuisci delta e segnala; non rigenerare comune
o restaurare V13/V14 per forzareMATCH. Snapshot propri attribuiscono le prove,
non approvazioni; niente freeze per ogni argv/microcaso. Nessun test importato
durante letture statiche. Inventario è fonte di identità, non richiesta di leggere
ogni artefatto o di ricontrollare tutti gli archivi storici.

## Criteri comuni

Giudica chiusure e regressioni delle classi modificate nel profilo ammesso.
Finding: input, percorso raggiungibile, requisito, impatto, attribuzione,
prova e severità coerente. Sintetici rari e difetti ereditati possono violare
un requisito essenziale; non parser universale o ampliamento illimitato.

| ID | Criterio |
| --- | --- |
| C01 F58 URI | Errori prevedibili del solo urlsplit diventano Unsafe all'adapter. Nessun catch generale, whitelist invariata. Originale script→URI invalida→path conserva prima causa, tre visite char2/21/42 e byte3/22/43, coda e reset; ordine inverso e input safe. Altri callsite dello stesso adapter non sono nuovi test end-to-end impliciti. |
| C02 F59 info | Vuota/lingua singola/metadata accettati/raw esatto assegnano owner nel profilo; info multipla o invalida non nasconde HTML/asset. Owner CodeInline backtick distinto da CodeBlock, corpo AST discriminante. Marker/lunghezza/chiusura/EOF/indentazione e positivi ricevuti preservati. |
| C03 F60 stato | Scanner e producer condividono stato fisico, paragrafo precedente, host e decisione. Container attaccato, lazyquote/note, figura→tilde e root non azzerano lo stato. Quote/list/note richiedono separazione pinned; K2c lista lazy resta vero CodeBlock. Heading attaccata non è attributo attivo falso. Anche caller/ricorsioni/caption/celle, non solo interno al container. |
| C04 F61 metadata | Solo grammatica/intervallo accettati sono opachi in span/link/code/fence/heading/div/figure. Gruppi invalidi conservano HTML reale/asset/testo o residuo dichiarato. Quoted HTML/graffe/bare valori validi inerti. Token adiacenti/id ultima dichiarazione, LF/CRLF/posizioni e22 valori effettivi. Nessun feed raw indiscriminato dei valori. |
| C05 F62 prove | Registro callsite prima del codice, tre contratti finiti condivisi, AST sintassi e oracoli sicurezza/fedeltà indipendenti. Quattro copie private devono fallire proprio eccezione/coda, info, chiamante radice, metadata; sette discriminanti. Premesse condizionali congelate distinte da attesi dettagliati materializzati dopo after. Valuta copertura delle proprietà e decisioni, non solo157 payload o assenza di eccezioni. |
| C06 regressioni | F54 ordine/visite, F57 privati, F50 label/formule/UTF8, F43/F44/F46/F48 e F40/F41 domini chiusi, F31 tokenlabel/F35/F36/F32/F27/F30/F28locale/F26/F16. Scanner/consumer/guardia/fallback/progressione/ricorsioni preservano testo/formule/numeri/blocchi/celle/asset/ordine. Unknown non wildcard. |
| C07 byte/uscite | Managed3.12.13 e S/B/I source/archive/install10hash/helper37/copier132/binding69; stage-s002 finale. R finale PASS, H12 dueFAIL voluti harness1/wrapper2, H01–08PASS;12CLI reali con writer sintetico/goldPROVISIONAL, nessun motore/V. Prove autore ricevute distinte da statiche proprie e eventuali sonde proprie. |
| C08 derivati | 64 catene originali/derivate/raw/asset/gold,19 campi e distanze/bound/condizionato distinti.16 EPUB diagnosi eotto TeX metadata attribuiti da AST, nessun delta numerico/gate.4PASS/28FAIL osservati per set,56EXACT/8TRIVIAL_ZERO, mai numeri forzati oTRIVIAL_ZERO→fullPASS. Aritmetica checker indipendente da reader/scorer, alignment strutturale non totalmente indipendente. |
| C09 storia/docs | PROTOCOL/due guide/permanente precisi sui contratti e limiti. OverlayHC2/IDpremessa smentita ebaseline132/check005 vs133/check004 storici conservati. O003 coverage interna tipizzata distinta da utente invalido; quattro osservazioni r010/F36/A8 preservati. Risorse/clock/accounting/FAIL/STALE/NONPASS/incidenti senza sanatoria. |

L01PDF/OCR,L02compileNOT_MEASURED,L04outputUNAVAILABLE distinta daV13/A03
chiusi,L05CPUlower/RSSsampled/hardware/cache,ChromeNONPASS/helper12gap/H15/E9
e raw/gold/corpus ricevuti restano limiti. F36B1–B4 deliberate incomplete/
unsupported/nessuna apertura non impone secondo feed. Nessuna garanzia di
parser universale, nessun GO scontato dall'autore o NO_GO dalla sola scelta
di review statica. Se nuove premesse sintattiche sono incerte, distingui
inferenza statica, AST ricevuto e microprova effettivamente eseguita.

## Output e passaggio

Scrivi solo `reviews/review-implementation-r014-chatgpt.md`,
`handovers/review-implementation-r014-chatgpt.md`,
`evidence/review-implementation-r014-chatgpt/`. Snapshot tecnici propri
in snapshots/review-r014-chatgpt-*.json se necessari aR, non approvazioni.
Non modificare prodotto/comune/stato/arbitrati/output altri ruoli.
Registra provider/modello/effort se esposti, indipendenza, comune/SHA/verify,
letture/prove statiche e runtime separati, criteri/chiusure/finding ID/GPT,
impatto/attribuzione e GO/NO_GO con limiti espliciti. Costi propri misurati/
non misurati, processi, rete0/€0 e prossimo ruolo. Non leggere report corrente
concorrente, nessuna delega. Al limite reale conserva CP e disposizione
necessaria, non estensione implicita. Prossimo ruolo supervisore dopo entrambi
i report reali; Git manuale dopo GO finale, nessun ultimo giro promesso.
