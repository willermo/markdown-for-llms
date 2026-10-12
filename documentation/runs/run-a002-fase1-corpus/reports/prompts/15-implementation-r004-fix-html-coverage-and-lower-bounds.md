# Implementazione a002 r004 — copertura HTML e bound testuali

Agisci esclusivamente come **implementatore**, nella stessa chat autrice di
r003. Completa F17–F21, regressioni, prove e report r004 nella stessa chat.
È un mandato di fix locale per risultato, non un nuovo piano o una catena
di approvazioni per comandi. Non scrivere review, arbitrati o registri comuni.

## Mandato, identità e ingresso

Run `run-a002-fase1-corpus`, branch `feature/run-a002-corpus`;
HEAD/dev/origin-dev locale/merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`;
indice vuoto alla ricezione. Delta locale previsto, Git manuale dell'utente.
Piano r001 SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`,
errata SHA256 `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`,
GO piano r002, prompt05/06/09/12 e addendum V restano il mandato approvato.
Prevale ora l'arbitrato implementazione r003 NO_GO sui sette rilievi reali.
F17–F20 bloccanti, F21 completamento non bloccante accolto. Nessun nuovo piano.

Percorsi relativi della run sotto `temp/run-a002-fase1-corpus/`. Leggi:

1. AGENTS.md, temp/HANDOVER e parte iniziale PROJECT-CONTEXT; STATE, HANDOVER,
   brief e `handovers/supervisor-implementation-arbitration-r003.md`.
2. Le skill manage-implementation-run e verify-conversion-fidelity, protocollo
   run-lifecycle, questo prompt e `arbitrations/arbitration-implementation-r003.md` integrale.
3. Entrambi i report `reviews/review-implementation-r003-chatgpt.md` e
   `reviews/review-implementation-r003-claude.md`, checkpoint ed evidenze
   citate: sei autore dei fix, entrambi sono input ammessi.
4. Report autore `reports/implementation-r003.md` e sezione finale Consegna
   completa del tuo checkpoint r003; PROTOCOL, guide benchmark, report/dati
   permanenti r003. Piano/errata/mandati precedenti solo per requisiti pertinenti.
5. Reader, scorer, audit, safety, adapter/R e dipendenze realmente consumate;
   test unitari/fix/contratti pertinenti, gold/manifest/receipt V, rescore64.
   Lettura statica senza importare automaticamente prodotto o ML.

Il comune review r003 è `snapshots/implementation-r003.json`, SHA256
`6dc951642d3108b6b5ed31cc36677ed6c1187b849812474fe74e1d7644f6d6e8`, impronta
`9eb2033e07e62ba509be8a795308f1dc0c2d362caaf79745eff9d4249251ddd1`;
310 file/37588 artefatti. MATCH nelle review e nella ricezione del supervisore;
poi soli registri documentali aggiornati. Non rigenerare il comune storico.

Ingresso corrente: `snapshots/implementation-fix-context-r003.json`, hash e
impronta nella receipt `evidence/supervisor-implementation-arbitration-r003/snapshot-receipt.json`
e nel checkpoint supervisore. SHA256 `2d9c06062f1b16b8b6279240fbc14da8a70e85023d503ed1c229e953018efb31`, impronta
`fe75ffddf343897b9f4ae051fae2ebe4128ebe576646d29b370bf170ddf548f4`, 310 file/38018 artefatti espliciti.
Verifica identità Git/delta/indice e contesto:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r003
```

Registra argv/cwd/output/exit. Questo MATCH attribuisce il contesto, non è
un GO o un test prodotto. Dopo cambi autorizzati crea autonomamente nuovi
snapshot di esecuzione, target, inventari e manifest prima dei workload.
La directory `snapshots/` può contenere questi **nuovi** oggetti tecnici
dell'implementatore, con etichette uniche r004; non modificare snapshot comuni
o vecchie prove, non chiedere un nuovo freeze come consegna intermedia.

## Contratto da completare

I casi originali r002 sono corretti, ma F01/F05 ricorrono in nuove forme.
Controlla il contratto produttore → trasporto → audit/scorer: ogni foglia
semanticamente consumata deve arrivare una sola volta al dominio corretto,
oppure risultare unsupported localizzato con raw conservato e risorse
auditate. Raw/segments presenti ma ignorati dai consumatori non chiudono il
requisito. Definisci un subset finito verificabile: non è richiesto HTML
universale. Sono ammesse scelte reversibili non enumerate, motivate e provate;
non adottare whitelist delle sole stringhe delle sonde o rifiuto di tutti i positivi.

### F17 — strutture di blocco nel contesto inline

CLA-R003-I001, regressione r003. `transport` prende il `text` di table/figure
senza le foglie delle celle o del resto della figura. In una cella pipe vuota,
`<table><tr><td>99 inventato</td></tr></table>` risulta vuoto/esatto e PASS.
In un paragrafo, table con `Parole inventate` produce WER0/30 disponibile.

Stabilisci esplicitamente che cosa è accettato inline. Puoi trasportare tutte
le foglie correttamente, oppure rendere la struttura di blocco inline
unsupported: conserva raw, coordinate, contenuto noto, formule e risorse;
mai testo vuoto o PASS spurio. Copri table/figure in cella, caption, paragrafo
e wrapper annidati, senza doppio conteggio o contaminazione prosa/math/codice.
Con unknown testuale CER/WER completi indisponibili secondo il dominio.
Positivi span/a/enfasi/MathML supportato e i normali blocchi restano corretti.

### F18 — foglie residue nelle proiezioni table e figure

GPT-R003-I001 e CLA-R003-I002. Table conserva solo tr>td/th e l'ultima caption;
figure supportata conserva asset e caption, scartando ulteriore testo.
La struttura accettata deve rendere conto dell'intero sottoalbero, non solo
delle foglie proiettate. Puoi restringere esplicitamente il profilo e conservare
il residuo come unsupported, mantenendo risorse/localizzazioni. Non è necessario
accettare HTML strutturalmente invalido, ma non può passare silenziosamente.

Prove prima/dopo sul percorso reale osservazione/audit/scoring:

- Table equivalente positiva e figure singola immagine/caption positiva PASS
  automatico; whitespace strutturale innocuo non diventa contenuto spurio.
- Img locale mancante fuori celle, in tr, sotto wrapper o in caption duplicata:
  INCOMPLETE/exit2; risorse presenti auditate, evasive respinte prima dell'apertura.
  `../outside.png` è già respinto: non dichiarare un bypass URI storico inesistente.
- Testo/numeri, formule supportate e unknown fuori td/th/caption e testo extra
  in figure: aggiunta nota rilevata oppure unsupported esplicito, mai PASS
  automatico. Formula supportata aggiunta, dove supportata nel profilo, FAIL3.
- Caption duplicate: nessuna scelta implicita dell'ultima elimina testo,
  formula o asset della prima. Gestione esplicita di malformed/residui, con
  raw e ordine conservati; nessun asset mancante risulta COMPLETE.
- Le stesse foglie dentro celle, caption o paragrafo restano discriminanti;
  codice/fence con HTML letterale resta inerte. Duplicati attributi, offset
  con prefissi e fallback già corretti non regrediscono.

Usa varianti per tipo di foglia e posizione, e spiega perché i consumatori
coprono anche i rami non enumerati. Nessun nuovo apparato generale obbligatorio:
una politica finita con test pertinenti e motivazione di conservazione basta.

### F19 — caption aggiunta con atteso assente o vuoto

GPT-R003-I002, scorer `if g.get('caption')`. La table HTML ben strutturata
con singola caption `Added text` è letta correttamente e passa il gate quando
il gold non ha caption. Confronta presenza e contenuto anche nei casi
atteso assente/vuoto. Non usare la truthiness per saltare il dominio.

Positivi senza caption e con caption equivalente; aggiunta nota FAIL3,
caption attesa omessa/differente FAIL, unknown locale mai PASS spurio.
Mantieni riferimenti/formule inline, differenze tipizzate e denominatori;
non riversare tutte le celle/caption nel CER/WER di prosa per aggirare il difetto.
Documenta se assente e vuoto sono equivalenti nel contratto del benchmark.

### F20 — bound garantiti distinti da conteggi strutturali

GPT-R003-I003 (deduzione statica) e CLA-R003-I003 (runtime R valido).
L'allineamento rifiuta un nodo unknown, poi omissioni gold + inserzioni del
testo noto possono contare due volte lo stesso contenuto. Un unknown può
produrre zero testo visibile: il noto20/30 di L1 supera il suo completamento
1/30; `Known \foo{x}words.` può avere distanza0 e non ammette bound4/2.

Definisci e versiona separatamente misura completa, conteggio condizionato
al matching strutturale e **limite inferiore garantito**, specificandone
dominio/corrispondenze consentite. Un bound deve restare non superiore alla
vera distanza in tutti i completamenti ammessi, non solo essere chiamato
KNOWN/LOWER_BOUND. Componenti omissione/inserzione ambigue possono contribuire
0 al bound oppure rimanere diagnostici strutturali separati. Fornisci la
motivazione della garanzia della politica scelta; non basta superare due esempi.
Le cifre complete restano UNAVAILABLE dove il testo non è misurabile.

Non rendere unknown una wildcard globale: perdite note, pagine/blocchi
mancanti, ordine, aggiunte certe, numeri, celle e asset conservano FAIL e
denominatori. La decisione strutturale F08 può rimanere FAIL anche se il bound
testuale conservativo è zero. Non alterare alignment/gold per ottenere verde.

Prove discriminanti sotto catena nuova valida:

- Riproduci GPT gold `Known words.` / raw `Known \foo{x}words.`, variante
  list e HTML unknown intercalato; foglie note equivalenti non generano
  omissione+inserzione presentate come bound certo.
- Riproduci CLA L1/L1c: bound non supera1/30 nel completamento a zero testo;
  verifica anche CER, non solo WER. Conserva la diagnosi di missing distinta.
- Confronta altri completamenti sintetici dell'unknown (zero/parole), posizioni
  e più blocchi, con un calcolo atteso indipendente dalle funzioni sotto test.
- C0 positivo0, C1 inserzione nota3/30 e Known words inserzione certa3/2
  restano discriminanti nei loro domini/corrispondenze; unknown solo non
  diventa misura completa0, unknown estraneo non occulta perdita di pagina.
- Mantieni decomposizione per ID/offset/testo, somme e denominatori coerenti
  con il tipo di misura. Le categorie strutturali possono essere conservate
  come tali: non spacciare la somma per bound se la garanzia non vale.

### F21 — `<` letterale distinto da risorse

CLA-R003-I004, non bloccante accolto nello stesso fix del reader.
`a < b` e `p < 0.05` non devono generare INCOMPLETE per asset/sintassi di
risorsa. Trattali come testo fedele, oppure unknown non resource dove il
profilo lo richiede; audit COMPLETE e classificazione motivata. Preferisci
il positivo fedele quando previsto dal reader. Conserva i confini per veri
tag/commenti/dichiarazioni, tag malformati, URI/fonti non fidate; code literal
inerte. Non estendere il subset con fetch o comandi dai dati.

## Prove, input esterni e conservazione

F16 è chiuso sulle64 misure; F02/F03/F04/F06/F07/F08/F10/F11/F12 e V derivata
restano chiusi nel perimetro delle review, salvo regressioni del delta reale.
L04 formule output è UNAVAILABLE, distinta dalla V originale sui13 riferimenti.
Nessuna nuova V generale. Proteggi corpus/sorgenti/render/asset/gold/packet,
V ricevuta/receipt derivate,64raw, vecchi report/sonde/stage/fallimenti.
Le copie proprie per le nuove sonde sono consentite, senza mutare i canonici.

Prima di avviare managed/import/prodotto prepara gli input correnti reali.
Tre `.pyc` managed encodings (`__init__`, `aliases`, `utf_8`) sono cambiati
nel corso della review GPT. Vedi `external-package-delta.json`, diagnosi
statica e osservazione del supervisore; autore/causa non accertati. Non sono
i tre Chrome storici. Vecchi R/S/I FAIL restano conservati; la diagnosi
statica non è una nuova catena valida. Identifica origine e byte realmente
consumati, vincola il nuovo manifest package/startup agli snapshot tecnici
r004, quindi usa nuovi gate. Non ripristinare byte/descrittori per nascondere
differenze, saltare preflight o trasferire il PASS precedente agli input nuovi.
Se un problema rivela un rischio sostanziale reale, documentalo al supervisore;
il solo aggiornamento di inventari/snapshot e gate è lavoro ordinario autorizzato.

Segue il protocollo managed locale CPython3.12.13: S/B/I, sdist/wheel
ricostruite e wheel canonica reinstallata dopo modifiche; dieci hash/origini
e input di tool/helper/fixture esterni alla wheel verificati prima/dopo,
preflight prima degli import e --no-sync. R nello stesso interprete dei
figli con rete isolata, registry/shell=False e confini filesystem preservati.
Niente installazioni/download per rimediare collection, acquisizioni/pesi o
nuovi strumenti non necessari; usa input offline già disponibili.

Esegui riproduzioni prima/dopo, test pertinenti nuovi e precedenti F01/F05/
F09/F13–F16, audit su positivi/negativi, contratti dipendenti H01–H08/H12 e
suite veloce prescritta, dopo gate validi. H12 R FAIL deliberati restano
FAIL con esito contrattuale atteso: non cambiare expected-exit a posteriori.
Test di contenuto/localizzazione/stato/exit, non sola presenza file o conteggi.
Non ripetere F03/H15/F04/API/Docker/ML o campagne a001 senza dipendenza reale.

Per i due set da32raw conserva l'identità di conversione r002 e genera nuove
osservazioni/score r004 oppure una prova di invarianza verificabile per tutti
gli input/campi dipendenti. Quantità4PASS/28FAIL attuali sono storia, non
assert da mantenere a ogni costo. Motori non da rieseguire per il delta
reader/audit/scorer previsto; se cambi adapter/validator/R/writer/config,
analizza e ripeti solo le prove effettivamente invalidate nel mandato.
Legami raw→observation→audit→score, versioni, hash, denominatori, disponibilità,
decomposizione, batch/failure e costi nei dati permanenti su prove reali.
F16 deve restare verificabile, con eventuali differenze spiegate.

## Quota unica e recupero ordinario

Quota originale52h/187200s da `2026-10-08T07:06:54.369068Z`, attese/review
incluse. CPU21600s,6GiB nuovi, RAM operativa8GiB/stop sampled6GiB;
stop nuovo workload49h/19800CPU-s/5GiB, riserva3h/1800CPU-s/1GiB.
Stop nuovo workload `2026-10-10T08:06:54.369068Z`, deadline
`2026-10-10T11:06:54.369068Z`. Nessun reset o incremento.
Il ledger autore94489,40259s wall/CPUnota7104,999667s/retained1204121202B è
storico alla consegna r003. Aggiorna alla ripresa con attese e nuovi costi;
ricezione del supervisore contiene il wall effettivo, non upper-bound CPU.
Distinguere CPU nota, RSS campionato, disco conservato, tempi reali e non
misurati; rete/acquisizioni/costi remoti0, niente dati privati o altre macchine.

Timeout/argv/log/output/launcher, snapshot e manifest propri sono mezzi
ordinari reversibili entro i cap. R ≤900s per comando,1MiB per stream,32MiB
per file; gestisci batching e riserva senza ridurre l'obiettivo a sonde parziali.
Correggi errori/test rossi/strumenti nella stessa chat, conservando i fallimenti.
Per EPERM/EACCES diagnostica e richiedi l'escalation tool della stessa azione
autorizzata se necessaria, mantenendo isolamento; rispetta un rifiuto effettivo,
non aggirarlo. Non fermarti per un freeze tecnico o singolo comando fallito.
Limite sostanziale reale/riserva raggiunta va riportato senza GO inventato;
prosegui nel frattempo il lavoro indipendente consentito.

## Consegna completa

Scrivi soltanto output del tuo ruolo:

- `reports/implementation-r004.md`: mappa F17–F21 con prova prima/dopo,
  scelte/garanzie/scope, chiusure/residui F01/F05/F09/F13–F16, delta effettivo,
  identità attuali e controlli invalidati/ripetuti o riusati per equivalenza reale.
- `evidence/implementation-r004/`: byte ricevuti/protetti, argv/cwd/output/
  exit, snapshot tecnici/manifest/target, S/B/I/R, prove e fallimenti,
  metriche/invarianza64, ledger cumulativo e own_pending finale.
- `handovers/implementation-r004.md`: input/output, consegna finale distinta
  dai passaggi in corso, costi/limiti/processi, prossimo ruolo supervisore.
- Guide/PROTOCOL e nuovo report/dati permanenti benchmark r004 con date
  effettive, misure condizionate distinte dai bound; non riscrivere report
  storici r003 o lo stato condiviso/changelog/arbitrato del supervisore.

Controlla link locali, whitespace e git diff --check; indice vuoto, nessun
commit/merge/push automatico. Alla consegna completa: supervisore per nuovo
snapshot comune e due NUOVE review indipendenti mirate, poi arbitrato reale.
Nessuna review/GO autore, nuovo lavoro roadmap o fase2 anticipati.
