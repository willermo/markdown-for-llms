# a002 r012 — Atomi condivisi e raw nei container

2026-10-11 Europe/Rome. Implementazione nel mandato40 F50–F53, stessa run e
incremento; consegna tecnica da ricevere e sottoporre alle due review reali.
Nessun GO autore, integrazione Git o scelta di motore. La nuova applicazione
resta progettata; il reader è uno strumento locale di misura del Markdown.

Il reader finale è `observations-v12-shared-atoms`. `literal_atom` decide
code/math, chiusura, corpo e metadata; `InlineDecision` trasporta il ledger
nel dominio del producer e nella sua source-map. Guard e consumer condividono
la decisione; ricorsioni label/span/enfasi ne consumano ritagli. La decisione
HTML/F36 preesistente resta comune al lexer, guard e albero.

La label mantiene le forme ammesse: LF via code chiuso, token opaco e
backslash+LF, incluse combinazioni. Metadata non ammette nuove LF libere.
Le LF code/escape non tagliano la math chiusa; token opachi non prestano
chiusure. Barriere LF indicizzate sostituiscono il loop ripetuto LF/code.
Non si certifica costo lineare dell'intero reader: scansioni di token,
slicing e parsing restano costi distinti.

Il code della label conserva `{=html}`: il corpo raw riceve guardia anche
prima di una seconda violazione esterna. HTML owner aperto prevale sui
backtick; chiusure estranee e ritorno al dominio Markdown restano distinti.
La prima unsafe viene differita fino al termine dell'osservazione: ogni
dominio successivo riceve guardia, mentre il dominio già unsafe non viene
proiettato e non duplica le visite nell'albero. Errori di proiezione non
troncano le guardie; il detail globale rimane quello della prima violazione.

Quote/list/note decidono il tipo di fence dopo lo stripping nel dominio
posseduto, prima del walk inline; quote annidate finite sono incluse. Fence
ordinario chiuso è letterale. Il raw block HTML esatto è guardato, conservato
con raw/coordinate e dichiarato non proiettato: asset sicuro INCOMPLETE/
UNSUPPORTED, script/img evasiva UNSAFE con detail. La presenza di segments
non certifica copertura; la prova privata registra dominio/range/hash/mappa.
Il residuo non rappresentato resta unknown esplicito. Il fallback senza
copertura mantiene la guardia, senza feed HTML indiscriminato della coda
Markdown che romperebbe F44. L'inventario audit può hashare file già presenti;
un asset non proiettato non viene seguito come reference URI.

## Prove effettive e attribuzione

| Prova | Esito sui byte runtime finali |
| --- | --- |
| Attesi manuali congelati, generator separato seed12012 | 350 casi; before v11 233/350, after v12 350/350, identici raw/gold/attesi finali |
| Regresso r011 / r010 / r009 | 153/153,303/303,308/308 sotto R |
| Suite veloce prescritta | 2688pass+310subtest,44.84s; API/packaging/Docker e smoke radice esclusi |
| CLI reale, writer sintetico dichiarato | 67fixture/134exitOS conformi, first detail e output/final verificati |
| Contratti H01–H08 | R PASS/harness0; stub sintetici, nessuna conversione generale nuova |
| H12± | FAIL deliberati R: wrapper2/harness1/expectedR0, contract-exit match=true |
| Microprove LF | N4/16/64/256, formula/link preservati; ~0.0006–0.005s wall osservati, cache calde |
| Ricalcolo64, nessun motore | zero differenze numeriche/gate nei19campi; 4PASS/28FAIL per set32 osservati |
| Checker distanza/aritmetica/catena | PASS64;56EXACT_GLOBAL_DISTANCE/8TRIVIAL_ZERO, alignment non interamente indipendente |

F50 preserva formula raw, link, testo, ordine e char/UTF8; run0/evaluate5 e
qualità INDETERMINATE nei gold manuali, senza PASS semantico del subset math.
F51 sceglie il residuo unsupported per raw container sicuro; unsafe2/2.
F52 verifica il detail della prima violazione e le visite successive, oltre
agli exit. F53 associa ogni classe alla proprietà e ai casi nei dati:
LF/math60, rawblock96, owner135, interazioni30, token10, freeLF5, ordineglobale1,
più13GPToriginali; altri controlli mirati nella suite. La conta non sostituisce
la mappa delle decisioni/proprietà. Non sono tutte le combinazioni Pandoc.

Gate origine managed3.12.13/startup/package prima del primo import e nuovi
S/B/I/sdist/wheel/reinstallazione canonica: dieci moduli source=wheel=install.
Helper fuori wheel, Python dei figli e input effettivi identificati nelle
copie/hash prima-dopo e snapshot R. Stage-s004 attribuisce i workload;
la chiusura documentale successiva richiede byte Python uguali, nuove receipt
e binding finale dell'autore. S006 r011 generale STALE resta storico, nessun
PASS trasferito. I finding r011 erano statici: questi sono workload r012 nuovi.

Errori e recuperi preservati: EPERM socket del sandbox prima del workload,
escalation approvata con R/Firejail/netnone invariati; dieci nuovi oracoli
avevano attributo `a` già unsafe prima dello script, sostituito con duplicati
`class` passivi mantenendo le vecchie versioni; visite ridondanti della
proiezione dopo unsafe eliminate; parent output contratti mancante creato
su nuovo percorso e prove ripetute. Nessun budget o isolamento indebolito.

## Limiti conservati

F36B1–B4 deliberatamente UNSUPPORTED, divergenza dal tokenizer puro, senza
secondo feed. F40/F41/F46, scorer/F26, denominatori, corpus/gold/raw e V13/A03
preservati. Quattro osservazioni r010: run backtick ereditato diverso da
CommonMark, html5/html-space fuori subset e Pandoc attivo non confermato,
timing storico commenti v10~7.3s/v9~1.4s su4000, fallbackIDobservations-v1
che non identifica il reader fisico. Nessun parsing Pandoc r012 eseguito.

CPU lower e RSS sampled; hardware locale e cache dichiarati nei dati,
nessun upper certificato o campagna prestazionale generale. O001 era stima
statica, non timeout osservato; le quattro microprove riguardano solo i loro
input. L01/L02NOT_MEASURED, L04outputUNAVAILABLE distinta dalla V sul gold,
L05/ChromeNONPASS/helper12gap/H15/E9/errori intermedi restano. Deadline della
run rimossa, entry/contatori e quote CPU/disco/RAM/R/rete invariati.

[Dati permanenti](2026-10-11-a002-r012-atomi-e-container-data.json) includono
inventario iniziale, sorgente del generatore, attesi, risultati effettivi,
CLI,64catene, limiti e hardware. [Fixture](../../tests/fixtures/conversion-reader-r012.json)
è sintetica; [test](../../tests/unit/test_conversion_fix_r012.py) verifica i
contratti. [Protocollo](../../scripts/benchmarks/PROTOCOL.md#rettifica-r012-decisioni-di-atomo-e-raw-nei-container),
[guida](../../docs/how-to/benchmark-fedelta.md#verificare-label-e-raw-nei-container-con-il-reader-r012)
e [reference](../../docs/reference/benchmark-fedelta.md#reader-r012-label-e-raw-nei-container).

Prossimo ruolo supervisore: ricezione completa, nuovo comune, due NUOVE chat
indipendenti ChatGPT/Claude e arbitrato reale. Nessun GO futuro o Git automatico.
