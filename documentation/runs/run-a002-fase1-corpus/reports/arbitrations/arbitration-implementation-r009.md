# Arbitrato implementazione a002 r009 — NO_GO, confini HTML/Markdown r010

2026-10-10 Europe/Rome. Supervisore Codex/OpenAI, deployment/effort/chatID
non esposti. Applicate manage-implementation-run e verify-conversion-fidelity.
Entrambi i report/checkpoint reali r009 letti integralmente; risultati, attesi,
receipt e sorgenti decisivi controllati. Nessuna delega o review impersonata.

**NO_GO** sul comune r009. Accolti due rilievi bloccanti ereditati:
F43 evita aperture reali nascoste nelle foglie Markdown unsupported;
F44 evita false aperture HTML dal code letterale successivo a HTML inline.
F45 prova insieme le due invarianti e rettifica documenti, incluso il limite
F36 osservato da Claude. [Prompt34](../prompts/34-implementation-r010-fix-html-markdown-boundaries.md)
è il mandato completo alla chat autrice r009, stessa run e incremento,
senza nuovo piano o reset. Non è un GO r010 anticipato.

## Identità, indipendenza e ricezione

[Comune r009](../../../../../temp/run-a002-fase1-corpus/snapshots/implementation-r009.json), SHA256
`bfc26ebc0db5946f30a0a47b6aa3a558674a167eb1c022dce1caa39fb6a03b9f`,
impronta `1c5d94c7db9f4a8508143a3bd1e7b72f559dfcaa6bb9909602ac12ebee20eb66`,
328 file/91038 artefatti, 20 PDF generali. Reader v9 SHA256
`a8ffa9eb44cf1695ca2e7e611c91ef8c7594605fe38a3bb194811f056cdbc737`.
Feature `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
`80c31da46e5caab990d3596c79c44e524ec1dc95`, indice vuoto. Nessuna query remota.

- [ChatGPT/Sol](../reviews/review-implementation-r009-chatgpt.md): **NO_GO**,
  OpenAI Codex basato su GPT-6, deployment/effort non esposti. Chat nuova,
  nessun report corrente Claude letto. MATCH prima e dopo, ultimo verify
  `verify-after-002.json`; 26 sonde, 8 oracoli attributi, 7 casi su v9/v8.
- [Claude](../reviews/review-implementation-r009-claude.md): **NO_GO**,
  Anthropic Claude Opus 5.5, effort/chatID non esposti. Chat nuova distinta,
  nessun report corrente ChatGPT letto. MATCH prima/dopo; 42 casi su
  v9/v8/r007, attesi fissati prima e copie degli input prima/dopo.
- [Ricezione supervisore](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-implementation-arbitration-r009/reception.json)
  e [premesse](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-implementation-arbitration-r009/static-premises.json):
  comune MATCH/exit0 prima dei registri, output reviewer legati ai byte,
  snapshot tecnici pertinenti controllati, sei receipt finali PASS con
  binding prima/dopo MATCH, input invariati, rete assente e socket ripulito.
  Receipt PASS attesta la raccolta, non conformità dei controesempi.

Supervisore: host stdlib/hash/JSON, manager invariato e Git read-only;
nessun import managed, reader/audit/scorer/R/suite/build/CLI/rescore/M/V.
Tracce statiche confermano la raggiungibilità; runtime è dei due reviewer.
Gli exit run/evaluate specifici dei rilievi sono dedotti dal contratto,
non exit OS riprodotti dal supervisore o dalle sonde reviewer.

## Decisioni su tutti i rilievi

| Rilievo | Decisione, requisito e attribuzione | Impatto e prove | Chiusura richiesta |
| --- | --- | --- | --- |
| CLA-R009-I001 | Accolto, **media, bloccante F43**. Ereditato: v9/v8/r007 identici. F39/C01/F42 e contratto r004/r009 applicabili, anche se non regressione del delta | `inline` :520/:540/:557 ingloba il resto della foglia dopo code/math non chiusi o link fuori profilo. Il fallback `node` :888 non valida unknown già muniti di lista. A1/A2/A4/A5/A6/A10 e A9: audit COMPLETE, nessuna diagnosi; A3 unsupported. Lo scanner riconosce invece aperture reali fuori dagli span chiusi. A9 perde una ref/asset sicura. Score FAIL, nessun fetch/file evasivo aperto: COMPLETE dell'audit non implica automatic PASS. | Ogni apertura reale nel dominio Markdown unsupported arriva alla guardia, con UNSAFE_REFERENCE/detail per i casi unsafe. A9 conserva e audita la risorsa oppure almeno INCOMPLETE con diagnosi esplicita. Non validare code/math realmente chiusi o corpi opachi; prove prima/dopo e regressioni F44 obbligatorie. |
| GPT-R009-I001 | Accolto, **alta, bloccante F44**, severità dichiarata conservata. Ereditato da v8, confermato v9; r007 non provato da questo reviewer. C01/C05 e F39/F42 richiedono che solo aperture reali vengano validate e che i letterali restino tali | `inline` :580–593 passa tutta la coda `raw[i:]` a HTMLTree, prima di scegliere il primo subtree. La validazione dell'albero solleva su `<img ...>` dentro backtick successivi; il fallback guardia riceve ancora tutta la coda. Quattro varianti su sette per reader producono falso UNSAFE, audit INCOMPLETE, FAIL/score3 e un solo unknown contenente tutto il documento. Raw conservato, ma tipi/text di heading/paragrafo persi. Code solo/prima dello span sono corretti; vero unsafe fuori code correttamente respinto. Nessuna apertura evasiva/esecuzione osservata. | Delimitare HTML inline con il dominio e gli span reali del producer prima della validazione/proiezione, evitando code esterno alla foglia HTML. Per i quattro falsi unsafe: tre blocchi e testo atteso, zero refs/assets, COMPLETE, automatic PASS, qualità INDETERMINATE/score5. Il controllo con unsafe reale resta UNSAFE/detail. |
| CLA-R009-I002 | Accolto come **bassa, non bloccante**, limite della scelta F36 da documentare in F45; nessuna guardia aggiuntiva obbligatoria. Regressione rispetto a r007, introdotta r008 e preservata esplicitamente dal mandato30/arbitrato r008 | B1–B4: nei producer HTML blocco/raw, prefisso prima di commento viene separato dall'estensione F36; il corpo resta opaco anche quando il tokenizer puro chiuderebbe un tag e riconoscerebbe img/script. v9/v8 INCOMPLETE/UNSUPPORTED, r007 UNSAFE. Nessun COMPLETE/PASS né fetch. Le dichiarazioni generali r009 non delimitano questa divergenza. | Conservare INCOMPLETE/UNSUPPORTED e zero aperture nei B1–B4, dichiarare la differenza rispetto al tokenizer puro nei soli prefissi F36, con confronto e attesi espliciti. Nessuna certificazione HTML5/renderer: non sono stati eseguiti. Questa scelta non si estende alle aperture reali secondo il lexer finito dopo errori Markdown o HTML; F43 resta obbligatorio. |

Esempio F44: `Known <span>words.</span> \`<img src="../outside.png"/>\``.
Il backtick chiuso rende il tag letterale; prima/dopo uno span valido non
deve cambiarne la classificazione. Esempio F43: `See \`x <script>alert(1)</script>`.
Il backtick non chiuso non autorizza a rendere opaco tutto il producer:
lo scanner stesso riconosce il tag, ma il consumer lo perde. I due difetti
sono cause operative distinte di una stessa incoerenza di domini. Un feed
HTML indiscriminato per chiudere F43 peggiorerebbe F44; fermare ogni feed al
primo tag senza validare i successivi rischierebbe di riaprire F39.

L'alta severità GPT riguarda la perdita della proiezione dell'intero documento
e il falso rifiuto di input ammesso, non un exploit. Il rilievo Claude media
resta bloccante per classificazione/risorsa perduta nel contratto essenziale.
Non cambio severità per ottenere GO; la preesistenza non esclude il requisito.

## Chiusure confermate, limiti e metodo

**F40 chiuso** nei domini revisionati: decisione tag/attributi condivisa,
14 nuove forme Claude conformi al tokenizer fissato e oracoli ChatGPT,
variante D7 sicura conserva immagine/testo. Estensione F36 distinta e limitata.
**F41 chiuso** nei domini revisionati: domini/source-map e ricorsioni reggono
E1–E7 Claude e casi ChatGPT; posizioni char/byte, testo, link e blocchi esterni
esatti. Non riaprire questi fix in assenza di nuova prova pertinente.

F39 è chiuso sui frammenti HTML dopo errori di struttura, **parziale** fra
HTML e Markdown per F43/F44. F42/C06 **parziali**: matrice 308 conforme ai
propri attesi, ma mancano le due interazioni accolte e il limite F36.
F35/F36/F32/F27/F30/F28 locale/F26/F16 preservati nei domini verificati.
Nota tecnica prima del codice r010: esplicitare quale producer possiede ogni
intervallo, aperture reali e span letterali, anche in fallback unsupported;
poi prove finite con attesi indipendenti, comprese trasformazioni di ordine
e chiusura. Refactoring locale autorizzato, non nuovo parser universale.

64 osservazioni/metriche invariate salvo readerID, catene e distanze
controllate dai reviewer: 4PASS/28FAIL per set, 56 bound esatti/8TRIVIAL_ZERO.
Resta risultato r009, non attesa da forzare in r010. Corpus/gold/raw/V13/A03
protetti; L04 formule output UNAVAILABLE distinta da V dei riferimenti,
L01/L02 NOT_MEASURED, L05 lower/sampled/hardware. Chrome NONPASS/helper12gap/
H15/E9 e fallimenti intermedi non sanati. S006 STALE per soli quattro
registri: preservato, nuove prove con gate reali dipendenti dai byte finali.

Errori propri dei reviewer conservati: EPERM recuperato mantenendo isolamento,
gold/checker/preflight errati distinti dai difetti prodotto. Errore proprio
del supervisore nel checker dei descrittori: KeyError su campo `bytes` assente,
prima verifica comune MATCH; script/fallimento conservati, controllo corretto
su sha256 dei descrittori standard e ricezione ripetuta nella stessa chat.
Il checker chiedeva anche MATCH attuale allo snapshot del preflight Claude
v9-001 fallito: la sola differenza è il launcher rrun.py, i cui byte storici
sono in rrun-001-failed-preflight.py. Attribuzione corretta e preservata,
nessun PASS di workload assegnato a quel preflight.
Nessun input prodotto restaurato o mutato, nessuna richiesta di nuova quota.

## Risorse e prossimo passaggio

Contabilità alla ricezione in reception.json; finale in budget-final.json.
Base comune 12864,569662 CPU-s visibili, più ChatGPT r009 185,908129 e
Claude r009 106,65 misurati lower; CPU supervisor misurata separatamente.
45s Claude r007 storici approssimati restano distinti, non aggiunti di nuovo.
Admin non integralmente misurata, nessun upper certificato; retained corrente
misurato senza sommarlo alla stima storica dello stesso scope.

Disposizione24h effettiva, entry2026-10-08T07:06:54.369068Z invariata.
Stop nuovi workload **2026-10-11T06:37:31Z =08:37:31 Europe/Rome**,
deadline09:37:31Z =11:37:31, riserva3h. CPU21600/stop19800s,
disco6/stop5GiB, RAM8/stop sampled6GiB, riserva1800CPU-s/1GiB,
R≤900s/stream1MiB/file32MiB, rete/acquisizioni/invii0, €0benchmark.
Preparazione/attese/review/registri incluse, stop al primo limite reale.

Comune r009 storico dopo soli quattro registri generali, non rigenerato.
Nuovo contesto tecnico `implementation-fix-context-r009`, identità nella
receipt e [checkpoint supervisore](../handovers/supervisor-implementation-arbitration-r009.md),
non GO. Utente invia prompt34 alla chat autrice r009. Alla consegna completa
r010: nuovo comune, due NUOVE chat indipendenti ChatGPT/Claude con criteri
comuni e poi arbitrato reale. Non promettere ultimo giro o futuro GO.
Git manuale dopo GO finale, nessun cleanup/a001/fase2/nuovaV/scelta motore.
