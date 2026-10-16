# A003 — Passaggio corrente e riesame del metodo

Aggiornamento 2026-10-15. La run è già avviata su
`feature/run-a003-engine-comparison`, base/dev `9f702a2`; questo documento
non è più un mandato di preparazione della run.

## Passaggio vigente: priorità prodotto e validazione differita — 2026-10-16

Decisione utente registrata in ADR0008: trace r005 ricevuto/non revisionato,
nessuna prosecuzione automatica del ciclo di validazione. A003 non diventa GO
né fase1 completa. Preparazione a004 e mandato in temp per piano del primo
percorso PDF/CLI/Web UI, seguito dagli incrementi della roadmap. Debito a002
aperto e differito; controlli operativi preservati. Le indicazioni sotto sono
storia, non mandati attivi in contrasto con la nuova priorità.

## Passaggio precedente: trace r004 NO_GO, fix r005 — 2026-10-16

Due review arbitrate: confini Pos ancora errati per CDATA/entity, origini dei
break troppo ampie; derivata parziale e dominio T07 inclusi nel fix. Mandato
`41-implementation-pandoc-trace-r005-native-boundaries-and-exclusive-witnesses.md`
nella stessa chat autore, ingresso `pandoc-trace-r005-fix-context-r001`.
Prima fonte nativa dei termini e provenienza esclusiva, poi campagne dipendenti.
Nessun nuovo piano generale o scanner. Enfasi/note e nucleo non invalidato
preservati, sufficienza a002 non confermata. Proposte esterne non adottate,
nessun GO prodotto/debito/fork permanente. Consegna completa, poi due review.

## Passaggio precedente: due review del trace r004 — 2026-10-16

Consegna mandato38 ricevuta: HTML/TagSoup, break, conteggi, note e T07 bilanciato.
Nuovo comune `pandoc-trace-r004-review-r001`, prompt locali
`39-review-pandoc-trace-r004-chatgpt.md` e `40-review-pandoc-trace-r004-claude.md`
in due NUOVE chat indipendenti, stessi criteri. A003 parziale: proposte
C02/T05 e T07 inline non adottate. A002 sufficiente per futuro prototipo è una
nuova affermazione autoriale da valutare, non chiusura del debito. Nessun GO,
integrazione o fork permanente; ricevute e codice solo in temp. Poi arbitrato.

## Passaggio precedente: fix trace r004 dopo NO_GO r003 — 2026-10-16

Arbitrati entrambi i report: blocco HTML GPT-R3-01/CLA-R3-01, più break/T00
e conteggi enfasi. Mandato `38-implementation-pandoc-trace-r004-html-and-selection-contract.md`
nella stessa chat autore, ingresso `pandoc-trace-r004-fix-context-r001`.
Prima relazioni token/consumo e risultato effettivo, poi gate finali attribuiti;
nessun nuovo piano o scanner. Questioni esterne distinte da fix tecnici,
nessuna riduzione adottata. A003 parziale non approvata e a002/D04 insufficiente
fino al fix HTML; originali e GO r002 preservati. Prossime review mirate.

## Passaggio precedente: review ruoli r003 e gap — 2026-10-16

Mandato35 consegnato: codice dei ruoli e prove, a003/a002 parziali con gap
T00, T07 e selezione delle note annidate. Prompt locali
`36-review-pandoc-trace-r003-chatgpt.md` e `37-review-pandoc-trace-r003-claude.md`
in due NUOVE chat indipendenti, comune `pandoc-trace-r003-review-r001`.
Verificare codice nuovo, copertura e limiti, poi arbitrato; nessuna riduzione
o integrazione approvata. Chiusure r002 non invalidate conservate.

## Passaggio precedente: GO nucleo r002 e prova dei ruoli — 2026-10-16

Due GO reali r002 e arbitrato positivo sul nucleo; blocchi r001 chiusi,
CLA-R2-01 documentato come limite non bloccante delle fasi dichiarate.
Mandato locale `35-implementation-pandoc-trace-r003-role-coverage.md` nella
stessa chat autore: ruoli essenziali per i due consumatori, C01 e risorse
HTML/blocchi per primi, poi famiglie pertinenti e copertura sul campione.
Ingresso `pandoc-trace-role-context-r001`. Nessuna integrazione o adozione
permanente; nuovi output in temp e successive review del codice nuovo.

## Passaggio precedente: review mirate dei fix trace r002 — 2026-10-16

Mandato32 completato: Gate1 PASS72casi/34mutazioni e Gate2 PASS_AST381raw,
ruoli parziali e limiti espliciti. Prompt locali
`33-review-pandoc-trace-r002-chatgpt.md` e `34-review-pandoc-trace-r002-claude.md`
in due NUOVE chat indipendenti; comune `pandoc-trace-r002-review-r001`.
Verificare chiusure r001, dipendenze cambiate e regressioni, poi arbitrato.
Ricezione con ricostruzione di risultati salvati, nessun nuovo parser/build;
originali preservati. Nessun GO prodotto, adozione permanente o chiusura a002.

## Passaggio precedente: fix locale del nucleo trace — 2026-10-16

Due review r001 ricevute e arbitrate NO_GO: trim non equivalente alla base,
scope frame orfani e owner lessicale accettati dal verificatore. Mandato locale
`32-implementation-pandoc-trace-r002-local-fixes.md` nella stessa chat autore;
include precisazioni su contrasti, ordine/categorie, eventi non selezionati e BOM.
Prima nucleo e negativi, poi gate finali attribuiti; nessun nuovo piano o
estensione ai ruoli futuri. Ingresso `pandoc-trace-r002-fix-context-r001`.
Originali in temp preservati; dopo consegna due review indipendenti mirate.
Nessuna integrazione, adozione permanente o chiusura a002 approvata.

## Passaggio precedente: esperimento di provenienza interna — 2026-10-16

Errata E01–E06 verificate e accolte. Il prossimo mandato è
`29-implementation-pandoc-trace-feasibility.md` nella stessa chat implementatrice
del prototipo: trace nei rami originali di Pandoc3.1.3, acquisizione/build in
temp ora autorizzati, grammatica e AST invariati, contabilità raw/residui.
Prima un piccolo nucleo decisivo; ampliamento soltanto dopo PASS. Integrazione
e adozione permanente restano fuori dal GO dell'esperimento; debito aperto.
Nessun nuovo wrapper/stringificatore o scanner proprio dei prefissi.

## Passaggio precedente: errata locale del piano r002 — 2026-10-16

Review ChatGPT/Claude completate e arbitrate. Diagnosi negativa circoscritta
accolta; piano NO_GO come mandato di prosecuzione non delimitato. Non è
impossibilità generale del metodo mantenuto. Per successiva richiesta utente,
nuova chat implementatrice del prototipo con Astra: prompt locale
`28-implementation-method-prototype-astra.md`, che incorpora i criteri E01–E06
del precedente mandato27: matrice
dei requisiti, correzioni di attribuzione/conteggi/URL, gate statico senza
toolchain Pandoc, testimone strutturale e casi D03/D04 con risorse eccedenti.
Prove minime e criteri di scarto prima di build/campagne; nessuna r008 o scelta
di fork autorizzata. L’errata si verifica localmente; solo un cambio sostanziale
richiede riesame del delta, non nuove review integrali automatiche.

I mandati24–26 sotto e nei registri sono storia del piano già consegnato.

## Decisione dopo r007

Due review indipendenti NO_GO arbitrate: **FIX_PLANNING /
METHOD_RECONSIDERATION_REQUIRED**. Si interrompe la sequenza automatica di fix
per esempi. Il comparatore ricostruisce sintassi tramite stime che diventano
prove e ruoli dipendenti dall'annotazione; la ricorrenza richiede cambiare il
metodo di dimostrazione. Le chiusure operative e i risultati del campione si
conservano. Nessun GO, accettazione ridotta o trasferimento del debito a002.

Il passo successivo è una NUOVA chat del pianificatore con il prompt locale
`temp/run-a003-fase1-engine-comparison/prompts/24-planning-r002-comparator-method.md`:
fattibilità di un parser mantenuto per la sintassi e delle coordinate sul raw,
gold indipendenti per la fedeltà, copertura necessaria e limiti espliciti.
Per disposizione utente si anticipa anche la ripresa del debito a002: nucleo
condiviso, due adattatori e piano di chiusura D-A002-01–05 con prove distinte.
Un prototipo circoscritto deve superare controprove e positivi prima di build,
suite complete o integrazione. Se mancano informazioni essenziali, proporre
un'alternativa motivata; non aggiungere un'altra grammatica propria.

Segue piano r002 sul metodo invalidato e sulla risoluzione del debito a002, due review indipendenti
ChatGPT/Claude in nuove chat e arbitrato del supervisore. Nessuna r008 integrata
prima di questa valutazione. La direzione [ADR0008](../decisions/0008-maintained-parser-and-provisional-benchmark.md)
era già accettata; candidato e adattatore ancora da convalidare. Fasi 2/3 non
avviate, human PENDING e fase1 aperta. Risorse gratuite necessarie già disposte,
nessuna spesa o trasmissione remota dei documenti autorizzata.

## Contesto da usare

Leggere temp/HANDOVER.md e STATE della run, l'arbitrato r007, prompt24 e il
comune corrente `pandoc-trace-r001-review-r002`. I comuni precedenti rimangono
attribuzioni storiche: non ripristinarli per ottenere MATCH. I vecchi prompt
di fix non sono il mandato corrente. Prove migrate in temp con mappa dei path,
senza alterare i byte; nessun nuovo bundle operativo in documentation.

- [Sintesi, risultati e disponibilità delle prove](../runs/run-a003-fase1-engine-comparison/manifest.md).
- [Brief iniziale](a003-engine-comparison-brief.md) e [ricognizione iniziale](a003-local-reconnaissance.md), storici.
- [Risorse già disposte](a003-resource-disposition.md).
- [Protocollo](run-lifecycle.md#review-triage-e-uscita), [conservazione](run-lifecycle.md#archivio-e-pulizia).
- [Roadmap](../roadmap.md), [changelog](../CHANGELOG.md).
