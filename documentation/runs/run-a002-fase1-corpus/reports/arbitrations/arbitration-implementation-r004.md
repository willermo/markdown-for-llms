# Arbitrato — implementazione — run-a002-fase1-corpus — r004

- Supervisore Codex/OpenAI, deployment/ID chat precisi non esposti, nessuna delega.
- Oggetto: fix implementazione r004, prompt15/F17–F21, non nuovo piano.
- Snapshot `snapshots/implementation-r004.json`, SHA256 `7105ec0e91e0783a26e13623ecf9eff37211291323253e39e7dc815e92d768c7`, impronta `3976b26937e2f80ee03fb089e7e18a4fd8782edd96c0f821536c367a65064edf`;313file/41997artefatti.
- Report reali [ChatGPT](../reviews/review-implementation-r004-chatgpt.md) e [Claude](../reviews/review-implementation-r004-claude.md), entrambi **NO_GO**, chat nuove/indipendenza dichiarate, concorrente r004 non letto. Nessuna sostituzione/delega: Codex/OpenAI famiglia GPT-6 e Claude/Anthropic Opus5.5 come dichiarato; deployment non verificati esternamente.
- Feature `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`, indice vuoto. Nessuna query remota/Git di scrittura.
- Esito del supervisore: **NO_GO locale**, due blocchi accolti e tre completamenti non bloccanti; F22–F26 alla stessa chat implementatrice per consegna r005. Nessun nuovo piano.

## Validità e ricezione

Entrambi i reviewer verificano MATCH/exit0 prima/dopo il medesimo comune.
Ricezione supervisore in [reception.json](../../../../../temp/run-a002-fase1-corpus/evidence/supervisor-implementation-arbitration-r004/reception.json),
verify prima e dopo l'arbitrato **prima** della transizione amministrativa.
Report integrali, checkpoint, evidenze e snapshot tecnici propri ricevuti.
Supervisore: letture statiche del codice e delle prove, hash/Git/manager stdlib;
nessuna suite/build/R/conversione/rescore/V nuova.

GPT R-probes-002 PASS,16casi, gate propri S/B/I validi e due snapshot tecnici:
il primo tentativo EPERM è IMPEDITA conservata; escalation approvata senza
rifiuto automatico. Claude R-probe-002 PASS,26casi; R-probe-001 FAIL al crash
heading conservata. Suoi snapshot propri legano la sonda, preflight s008
STALE per quattro documenti supervisore conservato; non trasferisce PASS.
Gold sintetici PROVISIONAL, gate automatici distinti dalla qualità globale;
exit evaluate delle sonde derivati dalle regole, non nuove conversioni.
CLA-I002 è **solo statico**, runtime bloccato dall'heading: non si attribuisce
una prova eseguita dopo un fix mai fatto.

## Decisione sui cinque rilievi

| ID | Decisione | Raggiungibilità, attribuzione, requisito ed evidenza | Azione e prova di chiusura |
| --- | --- | --- | --- |
| GPT-R004-I001 | Accolto, media, **bloccante** | `observations.py:307,316,323`: parsed genera unknown/resource_syntax sul wrapper non chiuso, visit ne scarta base e visita figli; parser.errors non recupera il fatto. Prosa e table in div/section non chiusi risultano COMPLETE/PASS. R valido con positivi bilanciati/negativo inline. Ereditato r003, requisito corrente F13/F14/F21/C06. Non perdita di prosa visibile o bypass URI. | **F22**: diagnosi strutturale indipendente dalla proiezione, propagata ad audit/scorer con raw/offset e senza duplicare foglie. Wrapper div/section/body/article annidati, prosa/table/figure/raw HTML fence; malformed INCOMPLETE/2, bilanciati invariati. |
| CLA-R004-I001 | Accolto, media, **bloccante** | `observations.py:389–390,411,416,460,507`: ogni heading HTML viene trattato come heading ATX per mappare prefissi; re.match ritorna None e causa TypeError. Runtime A3/A3a/b/c/A4/A5 in blocco/div/fence {=html}: crash observe e audit, CLI exit4/run.json assente. Introdotto r003, ereditato r004, h1–h6 nel subset dichiarato. | **F23**: distinguere sintassi di origine nel mapping, nessun prefisso Markdown su heading HTML. Test h1–h6, wrapper/fence, contenuto noto/aggiunto e risorse, offset Unicode/fence corretti; heading/note/list/blockquote Markdown invariati. Non basta intercettare TypeError e rendere tutto unsupported. |
| CLA-R004-I002 | Accolto, bassa, **non bloccante** da solo; completamento del fix heading | `observations.py:278,309`: h1–h6/code esclusi dai contenitori inline. Dopo F23, block table/figure nell'heading può essere appiattito senza unsupported. Statica, foglie/risorse comunque conservate; non aggiungere un nuovo requisito generale sul confronto immagini nei paragrafi. Contratto finito F17. | **F24**: includere heading HTML/code HTML nella politica inline o motivare una precisa eccezione coerente; in questo mandato applicare la regola inline alle strutture di blocco in questi host. A4/A5 non PASS spurio dopo F23, positivi validi ed HTML nel codice Markdown letterale inerte. |
| GPT-R004-I002 | Accolto, bassa-media, **non bloccante** | `HTMLTree:227, covered_end:257`: callback commenti/dichiarazioni interni non rappresentati; massimo end rileva la coda ma non intervalli interni. R comment-middle/declaration-middle COMPLETE/PASS contro tail INCOMPLETE. Ereditato omissione callback, nuova guardia r004 parziale. Raw generale conservato, nessuna perdita di prosa/asset attivo dimostrata. F21/C06 e coerenza profilo. | **F25**: conservare token/raw/offset con diagnostica consumata e classificazione coerente inizio/interno/fine. Per il profilo conservativo ricevuto, comment/declaration outside-profile restano unknown espliciti, senza attivare img/fetch dentro commenti. Positivi literal< e code invariati; nessun parser universale. |
| CLA-R004-I003 | Accolto, bassa, **non bloccante documentale** | `metrics.py:112–143`, guide/PROTOCOL/report/dati: text_lower_bound limita la distanza del flusso globale dichiarato, non la somma strutturale. Può superare CER/WER completi: DOCX62>60char, EPUB54>53, D1 boundCER0,667 contro completo0. Ricalcolo64 e sonda D1/D2 validi. Garanzia nel dominio globale corretta, F20 non matematicamente invalido. | **F26**: frase esplicita su dominio/relazione in protocollo/guide/nuovo report e definizioni dati; non bound del CER/WER completo o condizionato, può superarli per ordine/separatori. Non ridisegnare metrica/allineamento o riscrivere report r004. Numeri invarianti verificati con il nuovo reader. |

## Motivazione e divergenze

NO_GO non deciso a maggioranza: i due report osservano difetti distinti e
raggiungibili nel produttore HTML condiviso. F22 ha falso stato completo;
F23 impedisce ogni misura su un tipo supportato e interrompe la produzione
della receipt del tentativo. Quest'ultimo è bloccante anche senza falso PASS:
il precedente F21 era una classificazione conservativa errata di un carattere,
qui un'intera classe supportata va in crash e manca il risultato tracciabile.
Il fix è ordinario/localizzato; non serve una nuova pianificazione.

Sui commenti Claude vede correttamente assenza di perdita visibile (come nel
render HTML), mentre GPT rileva l'incoerenza col profilo/diagnostica ricevuti.
Non inventiamo risorse attive in un commento o rischio di fetch: F25 è un
completamento della rappresentazione e del contratto, non una nuova barriera
di sicurezza né un blocco aggiuntivo. F24 completa gli host inline dopo F23.
F26 chiarisce due metriche con corrispondenze diverse: non impone che il
bound globale sia ≤ alla misura strutturale e non ne cambia la garanzia.

La ricorrenza della copertura richiede verificare **diagnosi/token del parser
→ proiezione → consumatori**, oltre alle foglie semantiche. Un errore già
prodotto non può scomparire quando il visitor attraversa il wrapper; un estremo
finale non prova la copertura degli intervalli. Correzione finita, positivi e
varianti dei produttori, senza introdurre un parser universale o un apparato
di approvazioni per singoli comandi.

F17–F21 originali sostanzialmente chiusi nei casi controllati, con questi
residui: F13/F14/F09 HTML aperti su F22/F23/F25; host F17 da completare F24.
F19 e F20/F05/F15 chiusi nel dominio dichiarato; F21 letterale< chiuso,
diagnostica markup da completare. F16 **preservato sulle64misure**, entrambe
le verifiche indipendenti coincidono:4PASS/28FAIL per ciascun set32,
engine_runs0, RTFcondizionato19/162 vsbound13/162, TeXNone/1/152condizionato/
bound0. Gli altri fix r002 non riaperti senza dipendenza reale.
18PASS+2FAIL H12 deliberati, suite475/632+310subtest ricevute, non nuove
esecuzioni del supervisore. Nessuno dei64raw contiene HTML interessato.

Corpus/gold/raw/V13 protetti, A03 chiuso, nessuna nuova V umana generale;
L04 formule output UNAVAILABLE distinta, L01/L02/L05/ChromeNONPASS/helper12gap
e fallimenti preservati. Nessun confronto nuovo motori o scelta vincitore.

## Risorse e passaggio successivo

Quota unica52h/187200s da `2026-10-08T07:06:54.369068Z`, attese incluse,
CPU21600s/6GiB/RAM8GiB/stop sampled6GiB, stop nuovo49h/19800CPU-s/5GiB,
riserva3h/1800CPU-s/1GiB. Stop nuovi `2026-10-10T08:06:54.369068Z`,
deadline `2026-10-10T11:06:54.369068Z`. Nessun reset o incremento.
Ledger closing autore120881s/CPUnota7524,45s/retained1479827278B è storico.
Costi reviewer reali conservati (GPT sottoinsiemi misurati, Claude circa20min
wall/55CPU-s); non sommare CPU già inclusa nei wrapper né inventare upper.
Receipt supervisore aggiorna wall; ledger cumulativo da aggiornare alla ripresa.

Stato **IMPLEMENTATION / NO_GO_LOCAL_FIX**, [prompt18 completo](../prompts/18-implementation-r005-fix-html-diagnostics-and-headings.md)
alla stessa chat implementatrice. Consegna r005 F22–F26, test/regressioni,
catena attuale e prove dipendenti, rescore o invarianza64raw senza motori
per il delta previsto. Contesto tecnico nuovo implementation-fix-context-r004
dopo la sola transizione amministrativa; comune r004 e prove preservati.
Poi supervisore per snapshot comune e due NUOVE review indipendenti mirate,
quindi arbitrato reale. Nessun GO r005 anticipato, Git manuale dopo GO finale,
nessuna fase2, acquisizione o nuova V implicite.
