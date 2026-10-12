# Implementazione a002 r012 — consegna completa F50–F53

2026-10-11 Europe/Rome. Ruolo esclusivo implementatore Codex/OpenAI nella chat
autrice, mandato40, stessa run e incremento; nessuna delega, review o GO autore.
Modello esposto GPT-6; effort, ID chat e deployment non esposti.
Skill manage-implementation-run, verify-conversion-fidelity e write-diataxis-docs
applicate; istruzione esplicita assegna stato/arbitrati/comune al supervisore.
Nessun aggiornamento di quei registri o del CHANGELOG condiviso da questo ruolo.

## Identità e input

Feature `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
`80c31da46e5caab990d3596c79c44e524ec1dc95`, indice vuoto al ricevimento e alla
consegna. Delta già presente conservato. Nessun checkout/reset/commit/merge/push
o interrogazione remota. Ingresso fix-r011-unlimited-r001 verify MATCH reale:
[receipt](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r012/entry-verify.json).
Il vecchio comune r011 e il suo S006 generale STALE restano storici.
Il reader v11 è conservato fisicamente in
[originale](../../../../../temp/run-a002-fase1-corpus/work/implementation-r012/r011-observations.py), SHA256
`af5299bb0c52bd4a0634a1aebb8a28743e2793c315eec5bbaf9b6424a98d8482`.

Reader finale `observations-v12-shared-atoms`, SHA256
`5cabe6778ed53149ad02e1f50aaea6f334fc1c7ff3f9ffd38dd7f87c010abc22`. Inventario dei punti decisionali scritto prima
del codice: [nota iniziale](../notes/implementation-r012-atom-inventory.md).
Input/output, byte degli helper, copie prima/dopo, gate e argv effettivi:
[manifest finale](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r012/final-delivery-inputs-001.json).

Snapshot tecnico autore [delivery-001](../../../../../temp/run-a002-fase1-corpus/snapshots/implementation-r012-delivery-001.json):
SHA256 `fd15440f45ba8dcc106fa097edfcbe8691ea52d408d69bedefae028f1fe9dbc5`, impronta `84cd6590fc62a88526b7731a73ec3680d1b09902a3e3f42b8ab074c23dd52533`.
Verify reale MATCH in [receipt](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r012/delivery-verify-001.json);
controllo finale dopo report/CP in delivery-verify-002.json. Questi snapshot
identificano input, non costituiscono approvazioni o nuovo comune del supervisore.

## Risultato e scelte

`literal_atom` decide intervallo/code/math/raw metadata; `InlineDecision`
conserva atomi, owner, eventi, limiti e source-map nel dominio del producer.
Guardia, consumer e ricorsioni label/span/enfasi consumano quella decisione o
ritagli, eliminando la seconda scoperta dei confini dalla stringa strippata.
LF da code chiuso e backslash+LF rimangono ammesse; math chiusa resta opaca
anche oltre LF. Token opachi/barriere F31 non prestano chiusure. Metadata
non allarga l'accettazione a LF libere. Un indice locale delle barriere LF
sostituisce le scansioni ripetute; non è prova di complessità dell'intero parser.

Raw inline esatto nella label riceve guardia sul corpo e coordinate fisiche;
owner HTML e ritorno a Markdown restano distinti. La prima unsafe è differita
fino alle guardie dei domini successivi: il detail resta quello della prima
violazione; il dominio già unsafe non duplica le visite attraverso l'albero.
Errori di proiezione non interrompono le guardie.

Fence quote/list/note viene classificato nel dominio proiettato del container,
incluse quote annidate finite. Code ordinario chiuso rimane letterale. Raw
block HTML esatto è guardato e conservato; la risorsa sicura non proiettata
è INCOMPLETE/UNSUPPORTED esplicito (`container raw block not projected`).
Script o reference evasiva sono unsafe con detail. Questa scelta reversibile
chiude il contratto senza aggiungere un parser nested universale. Copertura
privata range/hash/source-map e residui unknown sono separati da `segments`;
il fallback senza copertura continua a guardare il dominio, senza feed della
coda Markdown che violerebbe F44. Audit può hashare file regolari nel bundle:
non si deduce assenza di ogni lettura da asset non proiettati o spy parziali.

Nessun nuovo schema pubblico, modulo, dialetto, motore, corpus, V o scorer.
Decisioni architetturali dell'applicazione non modificate. Il prodotto Python
toccato è solo observations.py; test e fixture nuovi sono sintetici.

## Prove effettive sui byte finali

| Prova | Esito | Evidenza propria |
| --- | --- | --- |
| Finito indipendente seed12012, limite400casi/2048byte | 350 raw/gold/attesi congelati; v11 233/350, v12 350/350 sugli stessi byte | before-v11-final, after-final-001 |
| Originali GPT statici | 13 raw esatti, prima diagnosi e visite controllate | cases-final-before-002.json, after-final-001 |
| Regresso r011/r010/r009 | 153/153,303/303,308/308 | regression-r011-001, regression-r010-001, regression-r009-001 |
| Suite veloce prescritta | 2688 passed,310 subtests passed,44.84s, R PASS | R-r012-fast-001 |
| CLI reale con writer sintetico dichiarato | 67fixture/134comandi OS conformi, prima diagnosi oltre exit | cli-001 |
| H01–H08 | R PASS, wrapper0/harness0/expectedR0 | contracts-002 e R corrispondenti |
| H12 negativo/positivo | R FAIL deliberato, wrapper2/harness1/expectedR0, contract-exit.match=true | contracts-002/H12-negative e H12-positive |
| Microcosto locale LF | N4/16/64/256,106/226/706/2626byte, formula/link preservati | cost-001.json |
| Rescore64 | zero differenze19campi/gate;4PASS/28FAIL per32 osservati | rescore-001 |
| Checker indipendente distanze/aritmetica/raw/catena | PASS64,0delta;56EXACT_GLOBAL_DISTANCE/8TRIVIAL_ZERO | derived-check-001.json |
| Documenti | 46link locali/ancore e whitespace, git diff --check PASS | docs-001.json |
| Binding finale | PASS_SCOPED_BYTES,10moduli/37Python esterni uguali,350oracoli,67CLI | delivery-check-001.json, R-r012-final-binding-001 |

Comando veloce reale: stage-s004/dev/bin/python `-I -B -m pytest tests/
--ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q`.
API/packaging/Docker e due smoke radice esclusi; suite selezionata, non collaudo
completo o convalida migrazione. F50: run0/evaluate5 con INDETERMINATE e gold
manuale, nessun PASS semantico; F51 unsafe2/2 e sicuro unsupported2/2.
F52 controlla detail e visite, non solo exit. Matrice oltre payload:
LF/math60,rawblock96,owner135,interazioni30,token10,freeLF5,ordineglobale1,
13GPT. La mappa decisione→proprietà→casi nei dati permanenti distingue
copertura delle classi dalla semplice conta. Generator stdlib separato,
nessun helper reader nell'oracolo; gold/attesi fissati prima dei workload.

Microtempi wall osservati ~0.0006–0.005s su hardware/cache locali, nessun
timeout O001 osservato o benchmark generale. Scansioni/slicing diversi
restano fuori questa conclusione. Checker usa matrice completa edit distance
e catene byte original conversion_finalr001 → precedente derivedr011 → r012;
alignment strutturale non interamente indipendente, non confondere i due limiti.

## Interprete, gate e conservazione

Origine managed CPython3.12.13 verificata prima del primo import; package
installato nello stesso interprete dei figli. Nuovi sdist/wheel e reinstallazione
canonica negli stage: s004 per i workload finali; s005 dopo solo documenti/dati.
S/B/I e dieci hash source=wheel=install, inventari Python/copie prima-dopo
e R Firejail/netnone reali, nessun --no-sync. Binding s005 confronta direttamente
i dieci moduli e37Python esterni di s004 e s005, rendendo valide le prove
runtime senza attribuirle a uno stage documentale non usato allora.
Preflight/gate in stage-s004 e stage-s005; argv/cwd/exit/costi/log nella receipt
host e R, descriptor del checker inserito nel target prima del workload.

Binding conserva 65963file protetti, 129file corpus,
334file del contesto comune e 103849artefatti iniziali; sole quattro
modifiche ai file preesistenti autorizzate dal mandato, nuovi test/dati/documento.
Confronto raw/gold/attesi before/after esatto;52coordinate safe raw/code/block
controllate nel binding, altre coordinate/visite nelle proprietà delle fixture.

## Fallimenti preservati e recuperi

EPERM socket sotto sandbox prima del workload: receipt IMPEDITA conservata,
escalation tool approvata e Firejail/netnone/limiti invariati; nessun rifiuto
di approvazione aggirato. Prime prove before231/350,after340/350 e targeted
1315pass/11fail conservate. Dieci nuovi attesi errati avevano attributo `a`
già unsafe prima dello script: ridotti a duplicati `class` passivi, versioni
originali intatte; caso finale prima di ogni nuova prova. Un vecchio test
ha rilevato visite ridondanti dopo unsafe: corretto il prodotto, ripetute tutte
le prove invalidate sul reader finale. s003 invalidato da successiva source-map,
non trasferito al prodotto finale. Contratti001 falliti per parent output
mancante prima del test: log preservato, percorso contracts002 creato, ripetuti
i dieci contratti con limiti identici. Nessuna consegna parziale o nuova fase.

## Risorse e limiti

Base effettiva supervisor-resources-r012-r001/budget-final.json:
15672.319041 CPU-s lower; nessun reset, non la vecchia base15535.
Budget prima della consegna: 16578.371753CPU-s lower cumulativi,
3947462656byte allocati correnti deduplicati;
receipt [budget](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r012/budget-before-delivery-001.json).
Accounting amministrativo finale in budget-delivery-final-001.json dopo i
report: include comandi host misurati una sola volta; alcune letture/edit e
amministrazione non misurate. ~45s Claude storico separato, non risommato.
CPU21600/stop19800s, disco6/stop5GiB, sampledRAM8/stop6GiB, riserva1800CPU-s/
1GiB, R≤900s/stream1MiB/file32MiB. Rete/acquisizioni/invii/servizi0,€0.
Deadline/stop wall della run rimossi; entry2026-10-08T07:06:54.369068Z
conservata per accounting, non nuova finestra24h.

Process check: 26receipt, socket rimossi, children reaped,
nessun processo workload osservato pendente; identità PID/starttime controllate.
Massimo RSS sampled r012 664543232byte;
storico749760512byte preservato. CPU lower/RSS sampled e hardware/cache
non sono upper certificati. [Processi](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r012/process-and-resources-final-001.json).

F36B1–B4 restano INCOMPLETE/UNSUPPORTED, nessun secondo feed; F31/F40/F41/
F43/F44/F46 e F48 preservati con regressioni pertinenti. Quattro osservazioni
r010 conservate: backtick run diverso da CommonMark, html5/html-space fuori
subset/Pandoc attivo non confermato,4000commenti v10~7.3s/v9~1.4s storico,
fallbackIDobservations-v1 distinto dal reader fisico. Nessun AST Pandoc r012.
L01/L02 NOT_MEASURED, L04outputUNAVAILABLE distinta da V13/A03 sul gold,
L05/ChromeNONPASS/helper12gap/H15/E9/errori intermedi restano. Nessun modello
pesante, download, installazione di recupero o nuova conversione generale.

## Artefatti e prossimo ruolo

[Report permanente](../../../../benchmarks/2026-10-11-a002-r012-atomi-e-container.md)
e [dati permanenti](../../../../benchmarks/2026-10-11-a002-r012-atomi-e-container-data.json)
includono attesi, copertura, generator, hardware, risultati e64catene;
protocollo e due guide aggiornati. Dati permanenti precedono il binding
amministrativo finale; questa consegna registra la receipt successiva reale.
[Checkpoint](../handovers/implementation-r012.md) autosufficiente per ricezione.
Stato condiviso/arbitrati/comune/CHANGELOG riservati al supervisore.

Implementazione completa consegnata per ricezione; nessun GO autore.
Prossimo ruolo supervisore: nuovo comune sui byte ricevuti, due NUOVE chat
indipendenti ChatGPT/Claude e arbitrato. Nessuna review futura inventata,
nessuna azione Git automatica; eventuali comandi solo dopo GO reale.
