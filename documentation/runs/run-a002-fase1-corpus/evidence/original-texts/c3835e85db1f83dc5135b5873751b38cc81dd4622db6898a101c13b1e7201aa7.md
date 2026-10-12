# Implementazione a002 r013 — chiudere le fonti di guardia e il contratto dei fence

Agisci esclusivamente come implementatore della run a002,nella chat autrice
che ha consegnato r012. Mandato completo di fix locale F54–F57 dopo
NO_GO r012 reale;stessa run/incremento,non pianificazione nuova.
Non delegare/impersonare reviewer o supervisore,non scrivere stato/arbitrati.

## Decisione e perimetro

Leggi [arbitrato r012](../arbitrations/arbitration-implementation-r012.md) e
[diagnosi/metodo](../notes/recovery-method-r013.md) integralmente.
Due cause bloccanti,F54 e F55;F56 gate delle prove/F57 documenti e privati.
Correggi le classi prima della consegna,non solo sei input dimostrativi.
R012 ha chiuso F50 e raw esatto/label originali,ma il consolidamento è
parziale:attrs bypassa la sessione e il generatore assume la grammatica
che intende verificare. Ripetere «atomi condivisi e più test» non basta.

Nessuna deadline della run:addendum-resources-r012-r001 e fonte utente no
deadline prevalgono sui gate temporali storici. Leggi clock per accounting;
entry08ott07:06:54.369068Z e contatori senza reset. CPU21600/stop19800s,
disco6/stop5GiB,RAM8/stop sampled6GiB,riserva1800CPU-s/1GiB,
R≤900s/comando/stream1MiB/file32MiB,rete/acquisizioni/invii0/€0benchmark,
isolamento e canonicalS/B/I invariati. Ledger corrente in
evidence/supervisor-implementation-arbitration-r012/budget-final.json.
Base ricezione17301,553741CPU-s lower comprende i due reviewer una volta;
aggiungere costi propri soltanto al ledger finale effettivo,non alle due
basi dei reviewer separatamente. 45sClaude r007approx distinti,admin in
parte non misurata,non upper certificato,retainedscoped dedup non risomma.
Al primo limite non temporale reale conserva CP/chiedi disposizione
necessaria;continua lavoro indipendente consentito. Nessuna nuova quota.

## Identità e letture

Repo /home/davide/workarea/markdown-for-llms,feature/run-a002-corpus.
HEAD/dev/origin-dev locale/merge-base80c31da46e5caab990d3596c79c44e524ec1dc95,
indice vuoto all'arbitrato;controlla delta esistente,nessun reset/checkout o
remoto/Git mutativo. Ingresso corrente
`snapshots/implementation-fix-context-r012.json`,identità nella receipt
evidence/supervisor-implementation-arbitration-r012/snapshot-receipt.json.
Comune revisionato r012 SHA256a8ad87d33d3d5ea6fecf40557cb0e470b1282bd043c312b005fd7ad17b048625
è storico dopo soli quattro registri supervisore,non rigenerare per MATCH.
Reader ricevuto observations-v12-shared-atoms,74476B,SHA256
5cabe6778ed53149ad02e1f50aaea6f334fc1c7ff3f9ffd38dd7f87c010abc22.
V11 fisico work/implementation-r012/r011-observations.py,
SHA256af5299bb0c52bd4a0634a1aebb8a28743e2793c315eec5bbaf9b6424a98d8482.
Piano3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8,
errataa09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9.

Usa manage-implementation-run e verify-conversion-fidelity. Nell'ordine:

1. AGENTS,temp/HANDOVER,prima parte corrente PROJECT-CONTEXT,STATE/HANDOVER/
   brief,questo prompt/CP supervisor-implementation-arbitration-r012,skill,
   run-lifecycle/ADR0001,addendum r012/fonte no deadline.
2. Arbitrato r012,nota recovery-method-r013,entrambi report/checkpoint
   reviewer r012 **integrali**. GPTcases001/probe-results001/own-results-check/
   current-gate-binding/Rprobe003;Claudecases.py/freeze-receipt/summary
   out-probe005/check005/corrections/Rprobe005 e AST B1–B9/D1 integrali.
   Sono storia ricevuta,non riaprire la review r012 o ignorare correzioni
   oracoli A12/A13/PandocRawBlock. GPT confronto V11 statico,Claude runtime.
3. Report/checkpoint autore r012,nota inventario,PROTOCOL/due guide/report
   permanente/dati e delta reader,test/fixture/generate002/350attesi,
   proofbinding/finaldeliveryinput/S004runtime/S005documentale. Criteri
   prompt41/42 e mandate40 integrali,arbitrati precedenti solo pertinenti.
4. Rami raggiungibili attrs/peel_attrs/located/guard/PassiveHTML/_observe/
   node/source_map/fence/paragraph-boundary/producer/consumer/ricorsioni/
   audit/benchmark/CLI. Contratti precedenti F31/F36/F40/F41/F43/F44/F46
   e piano/errata/mandati05–06/guida uv per il profilo ricevuto,non tutti i log.

Verifica all'ingresso,registra argv/cwd/output/exit/costo:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r012
```

## F54 — protocollo unico per ogni fonte Unsafe del reader

Prima di cambiare codice,nota breve con censimento completo di tutte le
fonti/callsite di Unsafe del reader e adapter dei validatori:HTML/raw HTML,
attributi Markdown di blocco e inline,eccezioni di proiezione/fallback,
ordine del producer e posizione raw disponibile. Individua ogni raise
diretto/catch/ritorno che può troncare guardie;includi attrs442 e chiamanti
div/fence/heading/figure/span. Per ogni punto indica proprietà e prove.

Tutte le violazioni **del reader** confluiscono in una sola politica con
posizione sorgente:prima causa in quell'ordine e guardie dei domini
successivi continuate,anche con più violazioni/proiezione impossibile.
Scelta dei mezzi libera:raccolta eventi/diagnosi separata dall'albero o
equivalente meccanismo circoscritto. Non basta sostituire raise con
report_unsafe senza ordine verificabile,né catturare l'ultima eccezione e
rilanciare la prima interrompendo comunque la visita della coda.
Rifiuta ancora gli attributi attivi,non proiettarli come valori sicuri.
Reset su errori/chiamate successive e nessuna apertura URI evasiva.

Originali GPT26/27 e ClaudeA1–A6:prima HTML unsafe resta detail,heading/div/
fence/figure onclick non interrompono le guardie successive;ordine inverso
segnala attributo per primo. ClaudeA7 attrs span prima dell'HTML nel medesimo
dominio è nella classe:ordine raw corretto per inline e blocco,non ordine
del parsing a due passate. Include trasporti/UTF8,attributi sicuri e più
violazioni,safe prima/dopo,errore d'albero,owner/code/math inerti.
Spy registra identità sorgente/snippet/posizioni,molteplicità/sequenza esatte
per singola observe;non membership dei tag né somma observe_bundle+audit.

Confine fermo:CLA-O002/A8 ordine path Markdown controllato dall'audit dopo
observe rimane limite non bloccante;non estendere a globale prima risorsa
di reader+audit o cambiare scorer/safety per uniformarlo implicitamente.

## F55 — proprietà dei fence decisa dal contesto di blocco reale

Una sola condizione riusabile nei rami radice/scanner/container/consumer
per le classi modificate. Non decidere code da solo marker all'inizio
della riga proiettata. Stato precedente paragrafo/inizio/separazione,
backtick/tilde,chiusura/EOF e raw metadata sono parte della decisione.
Non importare una regola generale CommonMark per analogia:il dialetto
effettivo è quello pinned/Pandoc3.1.3 già presente e del profilo ricevuto.

ClaudeB1–B4 quote/list/note testo→tilde:lo script reale riceve guardia
UNSAFE/detail;img sicura conservata come reference/asset nei rami già
supportati,oppure INCOMPLETE/UNSUPPORTED esplicito con risorsa/raw/posizione
se non proiettabile,mai COMPLETE vuoto. B7 radice stessa condizione ereditata
inclusa. B9 non chiuso non presta dominio code per nascondere HTML reale.
B5 backtick dopo testo/B6 tilde a inizio e fence realmente posseduti
mantengono corpo inerte. Code senza proprietario dimostrato non certifica
coverage:residuo non rappresentato resta unknown,non falsa riuscita.
Preserva raw esatto {=html}/{=latex},LF label e source-map,F36deliberato,
tokenopachi/barriere,F44code inerte dopo HTML. Nessun secondo feed HTML
indiscriminato,parser nested universale o html5/htmlspace fuori profilo.

## F56 — gate oggettivo prima della consegna,non solo più casi

È obbligatorio pubblicare tabella **ogni fonte/callsite censito → proprietà
→ caso/risultato**,incluse le due decisioni che r012 non aveva testato.
Non accettare un punto senza prova pertinente o limite preciso già ammesso.
Distinguere autore/testhelper/generatore/atteso indipendente:non usa il
prodotto come oracolo della stessa assunzione.

1. Congela nuovi attesi e raw prima del runtime,confronta V12 sui medesimi
   casi per attribuzione. Generatore finito separato,limiti/seed espliciti,
   massimo400nuovi microcasi/2048B cadauno salvo motivata riduzione dello
   scope di prova;nessun target numerico minimo. Fonti Unsafe×producer×
   ordine;fence×contesto precedente×chiusura×container×raw/literal×body;
   non ogni combinazione invalidamente grammaticale.
2. **Oracolo sintattico obbligatorio**:microparsing AST Pandoc già installato
   sotto preflight/R/no rete delle decisioni fence modificate. Salva argv,
   readerformat/versione/hash,input e AST. Classificazione CodeBlock contro
   RawBlock/RawInline/Plain/Para decide l'ownership del microcaso;gold testo/
   asset/coordinate e politica sicurezza restano manuali/indipendenti.
   Se una premessa Pandoc differisce dall'atteso,non correggere il gold
   copiando output reader:diagnostica e correggi la premessa con traccia.
3. **Due controprove obbligatorie su copie private**,mai sorgenti/corpus
   protetti:reintroduci in modo controllato (a) attrs fail-fast che scavalca
   il collector, (b) fence accettata ad ogni riga. Le proprietà indipendenti
   devono fallire rispettivamente su detail/visite e ownership/asset/unsafe.
   Conserva diff/hashes/output delle copie. Non intera campagna mutation:
   due assunzioni,pochi discriminanti,non nuova review simulata.
4. Ricerca/riduci/correggi i controesempi della classe nella stessa chat
   prima di report finale. Suite/regressioni sole non sono gate di copertura.
   Refactoring interno reversibile ammesso;nessuna modifica dialetto/schema
   pubblico/engine/modelli/goldcanonical per far passare le prove.

Non chiediamo certezza su tutta Pandoc né un LLM che si auto-approvi:le
condizioni d'uscita sono eseguibili e indipendenti dalle assunzioni nuove.

## F57 — documenti,coverage privata e prove dipendenti coerenti

PROTOCOL/due guide/nota tecnica e permanente descrivono esattamente ordine
del reader,limite reader/audit,condizioni fence,coverage dimostrata,residui.
CLA-O001 non bloccante:rimuovi _guard_coverage dalla serializzazione nested
finale tenendola per la guardia,oppure motiva/documenta una compatibilità
esplicita senza nuovo schema pubblico. located/source-map/cell/caption e
consumo node non si devono rompere. Confronta JSON e19campi metriche/gate
separatamente:vecchie640aggiunte e byte precedenti rimangono storia.
Nuovo report permanente r013 e dati con provenienza,non riscrivere r012.

## Prove finali e consegna completa

Canonical managed3.12.13:S→B→I/wheel reinstallata/stessi10hash,origini,
startup,copie helper/Python effettivi/gate prima import/R PASS,guida uv.
Se estrai un modulo interno aggiungilo a packaging/origini/copie/gates:
dieci hash da soli non coprono nuovi import. S004runtime/S005documentale
r012 e prove vecchie rimangono attribuiti;generalSTALE non trasferisce PASS.
Gestisci autonomamente snapshot di esecuzione e prove invalidate,non freeze
per ogni argv,approvazione supervisore non richiesta per errori ordinari.
EPERM/EACCES diagnostica/escalation strumenti disponibile mantenendoR;
rispetta un rifiuto effettivo,non aggirarlo o aumentare limiti.

Mirati classi/casi/regressioni350r012/153r011/303r010/308r009 e pertinenti
F31/F36/F40/F41/F43/F44/F46/F48;poi suite veloce prescritta sottoR con
API/packaging/Docker/due smoke esclusi. CLIreale run/evaluate/detail/exit
per nuovi discriminanti e controlli,writer sintetico dichiarato/gold
PROVISIONAL,nessuna qualità semantica inventata. H01–H08/H12± con attesi
contrattuali;H12FAILdeliberati non sanati.64rescore su raw esistenti,
zero engine_runs/nuovaV,19campi numerici/gate prima/dopo,catene/distanza/
aritmetica indipendenti;non forzare4PASS/28FAILse cambiano,attribuisci delta.
L04formuleUNAVAILABLE distinta,V13/A03chiuse,L01/L02/L05 e altriNONPASS
preservati. Ripeti solo prove invalidate dopo cambi di codice/input.

Output tuoi reports/implementation-r013.md,handovers/implementation-r013.md,
evidence/implementation-r013/,nota iniziale in notes/,work propria e
permanente documentation/benchmarks/2026-10-11-a002-r013-guardia-e-fence.md
e dati (data effettiva aggiornata se necessario). Riporta inventario,
matriceperramo/AST/controprove/errori/copie/identità/input/argv/cwd/exit/
limiti/costi/provenienza,cosa è eseguito e cosa no. Report non sostituisce
due nuove review reali. Nessun GO autore/Git/cleanup/fase2/ML/motore/nuovaV.
Prossimo ruolo supervisore sulla consegna completa,nuovo comune,due NUOVE
review ChatGPT/Claude e arbitrato;nessuna promessa ultimo ciclo.
