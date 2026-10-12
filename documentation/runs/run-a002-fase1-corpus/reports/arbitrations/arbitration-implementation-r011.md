# Arbitrato implementazione a002 r011 — NO_GO

2026-10-11 Europe/Rome. Supervisore Codex/OpenAI, ruolo esclusivo.
Due review reali indipendenti ricevute: [ChatGPT](../reviews/review-implementation-r011-chatgpt.md)
e [Claude](../reviews/review-implementation-r011-claude.md), con checkpoint
e tracce. Entrambe **NO_GO**, solo statiche dopo lo stop. Nessuna sonda
prodotto del supervisore, delega o review simulata. Skill manage-implementation-run
e verify-conversion-fidelity; mandato37 F46–F49, stesso incremento.

## Oggetto e attribuzione

Comune [implementation-r011](../../../../../temp/run-a002-fase1-corpus/snapshots/implementation-r011.json), SHA256
`29efea9167ad4809fc08e1ec9a1923998c6434d2f518a92c9c663811f53c16f5`,
impronta `525b853f32493abe8c3637bcb455219f557442cc032ac69d6b4358dcb370b680`:
334 file generali/103576 artefatti, 20 PDF. MATCH ingresso/finale dei reviewer
e supervisore prima dei registri; receipt effettive, argv/cwd/output/exit in
[evidenze](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-implementation-arbitration-r011/reception.json).
Reader observations-v11-inline-ownership, SHA256
`af5299bb0c52bd4a0634a1aebb8a28743e2793c315eec5bbaf9b6424a98d8482`.
Feature feature/run-a002-corpus; HEAD/dev/origin-dev locale/merge-base
80c31da46e5caab990d3596c79c44e524ec1dc95, indice vuoto, delta autorizzato
preservato. 232 file reviewer identificati per hash. Nessun remoto/Git mutativo.

## Decisioni sui finding

| Finding | Decisione | Causa, requisito e attribuzione |
| --- | --- | --- |
| CLA-R011-I001 | Accolto, alta bloccante → F50 | Label accettata con backslash+LF, ma bound math della guardia attraversa solo LF appartenenti a code. Consumer chiude la formula; guardia vede il tag nel corpo come attivo. F47/C02/C05/C06, regressione r010 non chiusa da r011. |
| CLA-R011-I002 | Accolto, media bloccante → F51 | Fence raw HTML esatto nella quote complessa trattata come code inline. segments esenta il fallback, senza guardia del corpo come raw block. Audit COMPLETE silenzioso deducibile. C04/C05/C06 e contratto risorse, regressione r011. |
| GPT-R011-I001 | Accolto, media bloccante → F52 | Ledger code nella label non possiede metadata raw inline {=html}; salto precede raw_html e la seconda unsafe alza prima della ricorsione sulla label. Perde prima diagnosi globale/visita. F48/C03/C05, ereditato r010. |

**F50.** T1 usa una label con formula chiusa su due righe, la prima termina
con backslash seguito da LF; contiene un img escaping. markdown_label accetta
l'escape (riga434), guardia tronca il limite alla LF (240–242), consumer
math_end può chiudere oltre quella LF. È raggiungibile e produce staticamente
il falso unsafe e la perdita di formula/link/blocchi tipizzati già vietata
da F47. Gli attesi devono includere formula e posizioni, non il vecchio gold
negativo. Il fix del solo LF dentro code non copre questo vettore ammesso.
Nessun nuovo risultato runtime attribuito: confronto v9/v10/v11 da sorgenti.

**F51.** T2 quote con fence esatto ```{=html}, script o img nel corpo.
Scanner block non vede il marker preceduto da >; producer strip quote, guardia
inline interpreta la fence come code chiuso; nodo unknown con segments,
unknown=[] salta il fallback. Per script/escaping atteso UNSAFE con detail;
per asset sicuro non proiettato almeno INCOMPLETE/UNSUPPORTED, mai COMPLETE
silenzioso. La prova autore metadata-closed-quote-1 conferma il ramo
unknown/segments/COMPLETE per code letterale, **non** l'esecuzione dei nuovi
T2. Raw esatto appartiene al profilo locale; Pandoc non eseguito, nessun
output Pandoc/browser certificato. Non ripristinare un feed raw indiscriminato
che riaprirebbe F44 o renderebbe attivo il normale code.

**F52.** Esempio GPT: label con backtick-script-backtick{=html}, poi img
escaping esterno. Raw inline esatto è già riconosciuto dal consumer.
Il ledger salta il code prima dell'emissione raw_html; guardia esterna segnala
l'img e interrompe prima della ricorsione che avrebbe validato lo script.
In ordine inverso deve diagnosticare prima l'img interno. Documento comunque
bloccato: non inventare fetch indebito o falso PASS. Otto discriminanti
(2 ordini × 4 producer) e cinque controlli, statici non eseguiti.

Le tracce verificano percorsi reali e contratti essenziali; l'assenza di runtime
vietato dal budget non è di per sé finding. Non riabbassare severità per GO.

## Osservazioni e chiusure

CLA-R011-O001: non bloccante. Ciclo LF/lookup code ripete scansioni e può
crescere circa cubicamente nella classe descritta; nessun timing né timeout
900s misurato. Consentito refactoring locale per indice/visita monotona nel
fix F50/F53; eventuali sonde piccole solo dopo autorizzazione temporale.
Non nuova campagna prestazionale o previsione presentata come misura.

CLA-R011-O002: non bloccante, ereditato. Fence raw esatto in list/note con
body riproiettato inline: stessa radice di F51. Includere come trasformazioni
finite della classe del fix, senza promuoverlo a finding bloccante autonomo,
senza parser universale o ampliare raw html5/htmlspace. Nessuna esecuzione
Pandoc ricevuta o richiesta come prova storica.

F46 metadata chiuso nei domini esaminati. F47 chiuso per LF in code, **parziale**
nel dominio label ammesso. F48 chiuso sui precedenti owner HTML, **parziale**
per raw inline della label. F49 e proprietà fallback quote **parziali**.
F40/F41 e originali F43/F44 restano preservati nei domini revisionati; F36
deliberatamente UNSUPPORTED, divergenza tokenizer dichiarata, nessun secondo
feed imposto. Quattro osservazioni r010 conservate senza ampliamento:
backtick-run, html5/htmlspace Pandoc non confermato, timing commenti,
ID fallback observations-v1 distinto dall'ID fisico del reader.

## Prove e limiti della ricezione

153/153 autore, 303r010/308r009, 2327fast+310subtest, 74 exit CLI e 64 derived
ricevuti, non rieseguiti. S005 runtime distinto da S006 documentale; MATCH
generale S006 storico diventa STALE per quattro registri supervisore,
non PASS trasferito. 64 misure 4PASS/28FAIL per set32, 56 bound esatti e
8TRIVIAL_ZERO; 19 campi/gate invariati, originali/derived distinti.
Non forzare conteggi futuri né trasformare math INDETERMINATE in PASS semantico.

GPT ha letture statiche e controlli propri di identità/JSON/catene; fallimenti
di schema propri conservati. Claude dichiara letture incomplete dei dettagli
r010, delle due guide e S/B/I/receipt individuali: non attribuirgli copertura
integrale di quei criteri. Limite registrato; i percorsi decisivi dei finding
sono comunque documentati e verificabili, non serve ripetere review completa
per ricevere questo NO_GO. Nuove review sul fix completo dovranno coprire i
criteri comuni pertinenti con prove realmente consentite dalla disposizione.
Nessuna suite/R/import/CLI/Pandoc/render/conversione di questa review statica.

Errore proprio del ricevitore: chiave exit invece di exit_code nella receipt
GPT; fallimento conservato, schema corretto, verify MATCH reale riusato sugli
stessi byte senza mutare prove/reviewer. Errore di sintassi apply_patch prima
della scrittura arbitrato: nessun file/prodotto modificato dal tentativo.
CPU di amministrazione non tutta misurata. Corpus/gold/raw/V13/A03 e limiti
L01/L02/L04outputUNAVAILABLE distinta/L05, ChromeNONPASS/helper12gap/H15/E9
e FAIL/STALE/errori intermedi intatti.

## Passaggio e risorse

[F50–F53 nel prompt40](../prompts/40-implementation-r012-align-label-atoms-and-container-raw.md):
fix locale r012 nella chat autrice r009/r010/r011, nota tecnica prima dei
cambiamenti, invarianti comuni e oracoli indipendenti. Non nuovo piano.
Mandato completo **preparato, esecuzione in attesa di disposizione temporale**.
Alla consegna completa: nuovo comune, due nuove chat indipendenti, arbitrato
reale; nessun GO futuro o ultimo giro promesso. Git manuale dopo GO finale.

Clock ingresso 2026-10-11T09:17:02Z. Stop06:37:31Z già passato, deadline
09:37:31Z (11:37:31 Roma), stessa disposizione24h da10ott09:37:31Z ed entry
08ott07:06:54.369068Z. Nessuna risposta temporale assunta dal messaggio di
consegna. CPU21600/stop19800s, disco6/stop5GiB, RAM8/stop sampled6GiB,
riserva1800CPU-s/1GiB, R≤900s/stream1MiB/file32MiB, rete/acquisizioni/invii0/€0.
Receipt ricezione:15459,709673 CPU-s lower, base15314,903874 + GPT36,526462
+ Claude76,19 + propria verify31,956206/receipt0,133131. GPT primo verify
non misurato; 45sClaude r007approx distinti non risommati. Finali nella receipt
di budget, retained attuale scoped/dedup, niente R/base duplicate o upper
certificato. Anche amministrazione deve fermarsi alla deadline reale.
Nessuna estensione CPU/disco/RAM/rete implicita dalla richiesta di tempo.

