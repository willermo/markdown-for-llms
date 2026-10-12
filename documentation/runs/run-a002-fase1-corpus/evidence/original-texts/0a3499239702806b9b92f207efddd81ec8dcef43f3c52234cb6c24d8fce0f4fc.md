# Review indipendente implementazione a002 r009 — ChatGPT

Agisci esclusivamente come **reviewer ChatGPT**, in una NUOVA chat indipendente
da autore,supervisore e altro reviewer. Registra provider/modello effettivi;
nessuna sostituzione fittizia,delega o subagente.

Output tuoi sotto temp/run-a002-fase1-corpus/:

- Report: `reviews/review-implementation-r009-chatgpt.md`.
- Evidenze: `evidence/review-implementation-r009-chatgpt/`.
- Checkpoint: `handovers/review-implementation-r009-chatgpt.md`.
- ID rilievi: `GPT-R009-I001`, progressivi.
- Prompt origine: `prompts/32-review-implementation-r009-chatgpt.md`.

## Oggetto comune e identità

Run `run-a002-fase1-corpus`, implementazione **r009**, fix F39–F42 dopo
arbitrato r008 NO_GO reale. Stato IMPLEMENTATION_REVIEW/FIX_REVIEW_PROMPTS_READY.
Consegna completa ricevuta; questa ricezione non chiude i rilievi. Nessuna
review/arbitrato/GO r009 presente alla preparazione del mandato.

- Comune: `temp/run-a002-fase1-corpus/snapshots/implementation-r009.json`.
- SHA256: `bfc26ebc0db5946f30a0a47b6aa3a558674a167eb1c022dce1caa39fb6a03b9f`.
- Impronta worktree: `1c5d94c7db9f4a8508143a3bd1e7b72f559dfcaa6bb9909602ac12ebee20eb66`.
- 328 file generali / 91038 artefatti espliciti,20PDF generali.
- Feature `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`, indice vuoto alla ricezione.
- Piano r001 SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`;
  errata SHA256 `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.
- Mandato30/arbitrato r008 F39–F42, addendum-resources-r008-r001 DISPOSED24h.

Inventario/receipt/transizione in evidence/supervisor-implementation-review-r009/.
Gli artefatti identificano byte e prove, non una lista di letture obbligatorie.
Registri vivi, prompt32/33, checkpoint corrente supervisore e nuovi output
reviewer r009 esclusi; payload dev/cache terzi esclusi salvo dieci origini
installate. Gli helper/test/documenti del prodotto restano inclusi. Non
nascondere codice nelle esclusioni, non ricaricare tutta la storia/temp/a001.

Ingresso autore fix-context-r008 SHA256
`f6cd0f93f5dc764725b79f83b393e5cbb29c03df89425672563bd2b6c6cc99aa` storico;
snapshot tecnico delivery-001 SHA256
`1e54271bdf8dfc0a2f1ed6a7212b57c4ee2427c2ace5ce6c486dbb67f7c11892`
MATCH alla ricezione. Questi snapshot attribuiscono prove, non sono GO.
Nuovo comune:7file autore +4registri generali supervisore, nessun altro delta
prodotto. Non pretendere MATCH dei vecchi snapshot dopo cambi autorizzati.

## Letture selettive e verifica iniziale

Usa manage-implementation-run e verify-conversion-fidelity. Percorsi sotto
temp/run-a002-fase1-corpus salvo indicazione diversa; leggi nell'ordine:

1. AGENTS,temp/HANDOVER,parte corrente PROJECT-CONTEXT,STATE/HANDOVER/brief,
   questo prompt e handovers/supervisor-implementation-review-r009.md.
2. Skill,protocollo run-lifecycle,template review; arbitrato r008 e prompt30
   integrali, disposizione24h e nota utente originale. Nota metodo r009 solo
   per il consolidamento; modello/effort sono raccomandazioni non selezioni.
3. Entrambi i report/checkpoint reviewer **r008**, finding GPT-I001 e casi
   Claude D1/D2/D3/D6/D7/safe, premesse statiche e ricezione r008. Sono
   storia comune. Non leggere report/evidenze concorrenti o arbitrato r009.
4. Report/checkpoint autore **r009 integrali**, inclusa nota tecnica iniziale;
   report/dati permanenti2026-10-10-a002-r009-confini-reader*,ricezione attuale,
   delivery-check-002,manifest47input,author-delivery-receipt001,budget/ledger,
   process-and-resources-final001 e transizione del comune.
5. Reader observations.py integrale e delta dal v8 fisico in
   work/implementation-r009/r008-observations.py; nuovo test r009 e regressioni
   r003–r008 pertinenti. PROTOCOL/due guide correnti. Segui audit/observe_bundle,
   safety/PassiveHTML,metrics/alignment/input-validator/CLI raggiungibili.
6. Prove autore mirate: matrix-before-corrected001/matrix-after003,work/matrix.py,
   expected.json/byte/oracoli,CLI002/manifest,stdout targeted003/fast004,
   contracts002/H12, rescore001/derived-check001,stage-s006 S/B/I/startup/
   package/archivi/target e before-after dei helper. Leggi i fallimenti quando
   incidono sulle conclusioni; non tutte le copie degli ambienti.
7. ADR0001/indice/registro ADR/fase1roadmap,piano/errata e mandati05/06 nei
   contratti pertinenti,workflow benchmark,guida uv/toolchain per i limiti.

Verifica Git/delta non committato e hash prima/dopo la review, registrando
argv/cwd/stdout/stderr/exit e costo nelle tue evidenze:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r009
```

Manager host stdlib, non prova prodotto. Se STALE, conserva il risultato,
attribuisci le differenze e segnala al supervisore; non rigenerare, ripristinare
o correggere input ricevuti, né dichiarare GO su un oggetto diverso.

## Delta e prove ricevute

Reader `observations-v9-reader-boundaries`, SHA256
`a8ffa9eb44cf1695ca2e7e611c91ef8c7594605fe38a3bb194811f056cdbc737`.
Sette file:reader,PROTOCOL,nuovo test r009,due guide,report/dati permanenti.
Scorer v4/alignment/normalizzazioni/denominatori,safety/audit/adapter/R/
resources/config/lock/PATH/registry/writer/corpus/gold/raw/V13/A03 invariati.
La nuova applicazione della guardia nel reader fa parte del delta da valutare.

Autore:308/308matrice contro178/308 sul v8,339test r009 (308+31),1712mirati/
16F04deselected,1869fast+310subtest,35fixtureCLI/70exitOS. Sei classi:
89unsafe,12inert-error,56safe-attribute,122domain,5foreign,24token.
Prove sintetiche audit→observe→score,goldPROVISIONAL,writer dichiarato,
automaticPASS può accompagnare INDETERMINATE/evaluate5. Nessuna V o motore.
H01–H08 R PASS,H12±FAIL deliberati con expectedR0/harness1/wrapper2.

Rescore64:19campi/gate invarianti,4PASS/28FAIL per set32 osservati;
56AVAILABLE/EXACT_GLOBAL_DISTANCE,8UNAVAILABLE/TRIVIAL_ZERO. Il checker
indipendente ricalcola distanze/aritmetica/catena/raw, non tutto l'allineamento.
Nei dati permanenti final=r009,conversion_final/run originali,
previous_derived_final/old_*=r008; summary originale conserva final storico.
Non confondere i contratti di questi campi né imporre qualità4/28 future.

Ricezione supervisore solo stdlib:152371confronti hash/83283path unici,
77011artefatti storici invariati,65963descrittori protetti,47input/copie,
dieci origini source=wheel=installato,18receipt selezionate16PASS+2FAILH12
con binding/reap/byteunchanged. Raw/catene e19campi verificati dal supervisore,
nessuna suite/reader/R/CLI/build/rescore/motore/M/V rieseguita. Le verifiche
di ricezione non sostituiscono il tuo giudizio tecnico né il secondo reviewer.

## Criteri tecnici comuni

Valuta chiusure,classi modificate e regressioni raggiungibili. Non limitarti
ai tre payload o al conteggio dei test; nessuna espansione a parser universale.
Per ogni rilievo verifica input ammesso,percorso/requisito/impatto/attribuzione.
I sintetici restano validi anche se rari o assenti nel corpus32.

| ID | Criterio sostanziale |
| --- | --- |
| C01 F39 guardia completa | HTMLGuard/FiniteHTML validano ogni apertura reale fuori dai corpi inerti, anche dopo prima violazione,duplicati,depth>16/cap20000nodi? Proiezione non interrompe la guardia. D1/D2/D3/D6 e ordini/trasporti unsafe→UNSAFE_REFERENCE/detail; D4/D5 invariati. Prima diagnosi unsafe in ordine sorgente precede unsupported. Nessun path evasivo/fetch aperto. Safe structural errors e token inerti non diventano unsafe inventato. |
| C02 F40 semantica lessicale | html_tag_boundary/matcher3.12.13,scanner/FiniteHTML/guardia/albero condividono fine tag e attributi? Apice/virgolette in bare value non aprono quote; quote solo valori quotati, `<`/`>`/token-looking bytes dentro valori non diventano aperture. D7 unsafe; variante p.png conserva testo/ref/asset pur con token reale unsupported. F36 prefisso veramente incompleto resta locale e token intero,compresi selfclosing/EOF/comment/CDATA/PI/declaration nei contesti ammessi. Oracle tokenizer indipendente sui piccoli input. |
| C03 F41 domini e mapping | Producer passa dominio effettivo al consumer/ricorsioni e mappa sorgente dopo stripping/merge; nessuna inferenza discordante dai marker trasformati. GPT-exact e quote/list/note/tab/paragraph ammessi,link/image/1-2backtick/marker/Unicode: testo/ref/ordine e raw/char/UTF8 esatti,automaticPASS/evaluate5PROVISIONAL. Mapping anche delle reference/code sulla seconda riga e in sottocontesti. Chiusura in producer estraneo non costruisce link né assorbe heading/paragrafo. Token vero fuori code intero/opaco. Non basta COMPLETE. |
| C04 F42 classi e oracoli | Nota tecnica precedente al codice,decisioni duplicate eliminate/vincolate,matrice finita e attesi fissati prima,trasformazioni conservative ammesse; controlli di contenuto/ref/asset/safety/mapping/metriche/exit lungo audit→observe→score. Prove incrociate delle invarianti,positivi/negativi discriminanti,assenza di expected derivati dal reader. Verifica i confini non coperti dai soli exact e la progressione terminante; nessun fuzzing illimitato/prodotto cartesiano imposto. |
| C05 regressioni | F35/F36,F32autolink,F27closevuoto,F30reason,F28CDATA locale,F26,F16 preservati nei domini modificati. Scanner/code/math/rawfence/metadata/attrs/HTML/TeX e trasporti pertinenti non riattivano corpi inerti; unknown non wildcard,missing/pagine/ordine/numeri/celle/asset restano FAIL. Nessun falso FAIL o perdita su input ammesso. F28/F29 globali e suggerimenti r008 su costo/rawTeX fuori scope, salvo regressione dipendente nuova. |
| C06 documenti e storia | PROTOCOL/dueguide/report/dati descrivono il comportamento e la copertura effettivi; chiusure r008 rettificate,storia e errori conservati. EPERM iniziale/matrice292/CLI001/H01label-parent/deliverychecker-link/docwrappercollision/H12 non sanati. Correzione degli expected/fixture motivata senza gold canonico o scorer cambiati; nuova308 usa identici raw/oracoli prima/dopo. GoldPROVISIONAL non V. |
| C07 64misure e V | Catena/source/gold/raw/asset/verification/run/old_*/final e identità derivate coerenti,19campi/gate/denominatori/arithmetic verificabili. Counts condizionati,bound flusso globale,CER/WER completi distinti;TRIVIAL_ZERO non fullPASS. Nessuna qualità forzata,motore o nuovaV. V13/A03 riferimento non L04formuleoutput,UNAVAILABLE distinta;L01PDF/OCR/L02compile NOT_MEASURED,L05lower/sampled/hardware precisi. |
| C08 toolchain e attribuzione | Stage-s006 S/B/I,snapshot/input/installazione canonica/diecihash/startup/stesso managed3.12.13 dei figli e preflight/R attuali; proof code.py finale uguale agli input realmente consumati. Helper fuori wheel/copie prima-dopo e argv reali,stage intermedi storici/equivalenze selettive dichiarate. Dopo4registri supervisore S006 integrale è STALE: non trasferire PASS generale,non ripristinare pyc/documenti; nuovi gate circoscritti sui byte delle sonde proprie. Binding/reap/exit reali,non solo assertion. |
| C09 risorse | Quota24h effettiva/entry originale/contatori inclusi;wall non somma parallela,CPU lower+45sapprox separati,RSSsampled/discoretained nonupper. Stop al primo limite,riserva3h,stream/file/caps e rete0/€0 preservati. Costi amministrativi non tutti misurati dichiarati,non stime trasformate in misure. Timeout breve per sospetti cicli con reap verificato. |
| C10 ciclo | Chat indipendente,new common valido,nessun GO autore/review concorrente/delega. Fix locale non nuovo piano/fase2. Nessun Git automatico/cleanup/campagnaa001/motore/V. GO o NO_GO fondato sui criteri essenziali,non assenza dell'altro report o numero test; ogni finding raggiungibile e attribuito,prossimo ruolo supervisore. |

## Sonde proprie consentite e limiti

Letture/hash/link/aritmetica/manager stdlib e propri output autorizzati.
Piccole sonde sintetiche reader→audit→score e CLI mirata con writer sintetico
se necessarie: input/attesi propri fissati prima,copie,gate/R/Firejail/rete0
e binding attuali,stesso interprete dei figli. Non rieseguire suite/campagne,
build/reinstallazioni/contratti H generali/baseline/motori/render/browser/M/V,
download/pagamento o Git di scrittura. Nessun fix al prodotto o input altrui.

Prima del primo managed verifica origine locale3.12.13,startup/package,
archivi e installazione/diecihash,S/B/I/preflight delle dipendenze reali.
Conserva STALE del S006 generale dovuto ai quattro registri del supervisore;
identifica manifest/target/gate propri circoscritti,non un PASS storico.
Snapshot tecnici tuoi con label `review-r009-chatgpt-*`,output vivi esclusi,
autorizzati senza freeze al supervisore. Nessuna mutazione dei comuni ricevuti.

Errori ordinari di argv/helper/log/timeout/path si correggono nella stessa
chat,versioni/failures preservati,prove invalidate ripetute. EPERM/EACCES:
diagnosi ed escalation tool della stessa azione autorizzata mantengono
R/no-network/shell=False/registry/PATH/confini; un rifiuto effettivo blocca
quell'azione,nessun aggiramento o privilegio persistente. Review statica
ammessa con limiti; prova decisiva mancante non diventa GO.

Leggi clock reale all'ingresso. Disposizione24h ancorata alla conferma
osservata2026-10-10T09:37:31Z,entry2026-10-08T07:06:54.369068Z senza reset.
Stop nuovi workload **2026-10-11T06:37:31Z =08:37:31 Europe/Rome**;
deadline **09:37:31Z =11:37:31**,riserva3h. Wall max268236,630932s/
stop257436,630932s,CPU21600/stop19800s,disco6/stop5GiB,RAM8/
stop sampled6GiB,R≤900s/1MiBstream/32MiBfile,riserva1800CPU-s/1GiB,
rete/acquisizioni/invii corpus/€0. Stop al primo limite; dopo stop letture/
registri/consegna nella riserva,nessun nuovo workload.

Budget autore a15:05:03Z:wall201488,809292s/55,969h,CPUaccounting12793,622141s,
propria963,321937s;45sClaude r007 approx distinti,retained2696144109B,
RSSsampled749760512B. Ricezione supervisore a15:24:40Z:wall202665,639366s/
56,296h,accounting12835,064462CPU-s aggiunge35,71815s supervisione ingresso
non inclusi dall'autore e5,724171s ricezione. Letture/assembly/admin non tutta
misurata;accounting non upper. Misura retained attuale circoscritta e costi
finali nella receipt budget-review-handoff.json. Non sommare retained storico
alla misura attuale dello stesso scope,wall parallele o CPU R annidate.
Aggiorna propri costi,saldi visibili e limiti prima delle sonde.

## Consegna ed esito

Usa template review; dichiara identità/provider/modello/effort/chatID solo se
esposti,indipendenza,prompt/snapshot/hash/delta,letture,criteri,prove proprie
e ricevute distinti,argv/exit/input/costi/limiti. Esito **GO** o **NO_GO**
sul comune r009;GO non certifica parser universale,L04,altri motori/fase1.
Identità invalida: impedimento motivato,nessun GO su stato diverso.

Ogni finding:ID/severità/blocco,posizione,input/percorso raggiungibile,
requisito/impatto,attribuzione r009/ereditato/mandato,evidenza statica o
runtime,correzione minima e criterio di chiusura. Valuta le chiusure F39–F42
e le invarianti condivise,non un voto medio o rilievi inventati.

Non modificare STATE/HANDOVER/eventi/artifacts/changelog/indici/arbitrati/
CP supervisore,snapshot comuni/storici o input concorrenti. Non leggere
review/evidenze concorrenti r009 o arbitrato corrente r009 anche se appaiono.
Checkpoint tuo con input/output/esito/costi/processi pendenti e prossimo
ruolo **supervisore per ricezione di entrambi i report e arbitrato**.
Concludi in chat con esito fattuale e percorsi. Nessuna delega/implementazione.
