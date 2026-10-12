# Implementazione a002 r005 — diagnosi HTML, heading e domini delle metriche

Agisci esclusivamente come **implementatore**, nella stessa chat autrice di
r004. Completa F22–F26, regressioni, prove dipendenti e report r005 nella
stessa chat. Il mandato copre il risultato, non approvazioni per comandi.
Non scrivere review/arbitrati/registri comuni, nessun nuovo piano.

## Mandato e identità

Run `run-a002-fase1-corpus`; feature `feature/run-a002-corpus`,
HEAD/dev/origin-dev locale/merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`;
indice vuoto alla ricezione, delta locale previsto. Git manuale dell'utente.
Piano r001+errata/GO piano r002, prompt05/06/09/12/15 e addendum V restano
il mandato; prevale ora `arbitrations/arbitration-implementation-r004.md`.
F22 wrapper non chiusi e F23 crash heading sono bloccanti; F24host inline,
F25token comment/declaration e F26dominio bound sono completamenti non bloccanti
accolti da chiudere nello stesso giro. Nessun nuovo piano o campagne a001.

Percorsi relativi della run sotto `temp/run-a002-fase1-corpus/`. Leggi:

1. AGENTS.md, temp/HANDOVER e parte iniziale PROJECT-CONTEXT, STATE/HANDOVER/
   brief; `handovers/supervisor-implementation-arbitration-r004.md`.
2. Skill manage-implementation-run e verify-conversion-fidelity, protocollo
   run-lifecycle, questo prompt e arbitrato implementazione r004 **integrale**.
3. **Entrambi** i report reviews/review-implementation-r004-chatgpt.md e
   reviews/review-implementation-r004-claude.md, checkpoint e prove citate;
   autore dei fix, puoi leggere entrambi. CLA-I002 è statico, non runtime.
4. Report/CP finale autore r004, guide/PROTOCOL, report/dati permanenti
   `documentation/benchmarks/2026-10-09-a002-r004-profili-html-e-bound*`;
   reader/node/HTMLTree/visitor, audit/scorer/safety e test pertinenti.
   Lettura statica senza import automatico di prodotto o ML.
5. Sonde GPT wrapper-unclosed/positive/inline/table/comment e Claude A3a/b/c,
   A4/A5, D1/D2, precedenti regressioni, rescore64/manifest/receiptV/
   inputbytes/protected. Piano/errata/mandati antecedenti solo per requisiti.

Il comune r004 revisionato è `snapshots/implementation-r004.json`, SHA256
`7105ec0e91e0783a26e13623ecf9eff37211291323253e39e7dc815e92d768c7`, impronta
`3976b26937e2f80ee03fb089e7e18a4fd8782edd96c0f821536c367a65064edf`;
313file/41997artefatti. MATCH prima/dopo delle review e supervisore prima
dei nuovi registri; ora storico. Non rigenerarlo o alterare vecchie prove.

Ingresso corrente `snapshots/implementation-fix-context-r004.json`: identità
nella receipt `evidence/supervisor-implementation-arbitration-r004/snapshot-receipt.json`
e checkpoint supervisore. SHA256 `cf9516fab267a241371e5556cbd8e39522102c30d60ac0cb3889d2a5cf3c0c88`,
impronta `563a352193a209595d42ec03f9ce6b6fc9e285ac561d3fb1c713865fadd4b328`,
313file/42288artefatti espliciti. Verifica Git/base/indice/delta e contesto:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r004
```

Registra argv/cwd/output/exit; MATCH tecnico non è GO. Dopo cambi autorizzati
crea autonomamente target/manifest/inventari e snapshot tecnici r005 con
label uniche sotto snapshots/, output vivi esclusi; non mutare ricevuti o
chiedere freeze intermedio al supervisore.

## Risultato F22–F26

Completa la conservazione delle **diagnosi/token del parser** anche quando
il visitor proietta solo i figli. Il contratto delle foglie corretto in r004
deve includere errori strutturali, host dichiarati e token fuori profilo:
raw conservato o massimo end da soli non provano che i consumatori li vedano.
Resta subset finito; nessun parser universale o schema applicativo nuovo.
Scelte ordinarie reversibili non enumerate sono ammesse se motivate e provate;
non whitelist delle sole stringhe dei reviewer né rifiuto di tutti i positivi.

### F22 — wrapper non chiusi: diagnosi indipendente dalla proiezione

GPT-R004-I001. `<div><p>Known words.</p>` e wrapper annidati/table senza
chiusura risultano COMPLETE/PASS: parsed produce unknown/resource_syntax,
visit scarta base, parser.errors non conserva il fatto. Non perdita di
prosa visibile o bypass URI, ma stato falso rispetto al profilo ricevuto.

Rendi gli errori strutturali indipendenti dalle scelte del visitor, oppure
trasportali esplicitamente con raw, offset e tipo ai consumatori. Un errore
deve sopravvivere alla proiezione senza duplicare testo/ref/math/unknown;
gestisci anche wrapper senza foglie, annidamenti e chiusure incoerenti
raggiungibili nel profilo. Conserva risorse e l'ordine delle foglie note.

Prove sull'effettivo observe→audit→score/exit:

- Div/section/body/article bilanciati con prosa/table/figure positivi restano
  misurabili e PASS automatico dove conformi, senza unknown spurii.
- Stessi wrapper non chiusi, multipli/annidati, con testo/table/figure e
  raw fence {=html}: INCOMPLETE/exit2, diagnosi localizzata presente,
  nessun COMPLETE/PASS per mancata propagazione. Controllo inline esistente
  rimane negativo, malformed/risorse mancanti/evasive non perdono i confini.
- I casi originali F17/F18 di residui e caption duplicate restano corretti;
  la nuova raccolta errori non conta nuovamente l'intero testo del wrapper.

### F23 — heading HTML supportati senza prefisso Markdown

CLA-R004-I001. Node calcola il prefisso ATX per qualunque heading, anche
HTML; `re.match` ritorna None e genera TypeError in reader e audit.
Il CLI restituisce exit4 e manca run.json: va corretto, non rietichettato.

Distingui sintassi di origine (o equivalente criterio preciso) nel mapping:
prefisso Markdown solo per heading Markdown, coordinate HTML ricavate dal
raw reale. Non mascherare il crash con un catch generico e unsupported
per ogni h1–h6, che sono dichiarati supportati. Non cambiare il protocollo
dei receipt/batch o il convertitore per riparare un errore del reader.

Prove h1–h6 a inizio blocco, sotto wrapper bilanciati e in {=html}:
positivo noto misurato, aggiunta/differenza rilevata, audit/receipt/exit
coerenti e nessun TypeError. Verifica source_position char/byte con Unicode,
prefissi del fence e foglie/ref/math; risorse presenti/mancanti/evasive
seguono audit. Heading ATX, note/list/blockquote Markdown e loro coordinate
restano corretti. Un normale heading non diventa unsupported solo perché HTML.

### F24 — host heading/code HTML nel contratto inline

CLA-R004-I002 statico. Dopo F23 riproduci A4/A5 e varianti: table/figure
in heading HTML oggi sono appiattite senza unsupported perché child_inline
esclude h1–h6/code. Applica la politica inline anche a questi host: struttura
di blocco non ammessa esplicita, raw/foglie/resources conservati, non PASS
spurio quando il testo coincide. Non richiediamo nuovo confronto generale
delle immagini nei paragrafi, osservazione fuori scope del reviewer.

Test heading/code **HTML** con block annidato e wrapper, positivi phrasing
validi, risorse/formule/unknown e offset. Distingui il codice Markdown
letterale/fence non raw: HTML in quest'ultimo resta inerte, senza risorse
attive o esecuzione. Nessuna sanatoria di A4/A5 dal runtime r004, che era
bloccato dal crash; prove dopo il tuo fix nuove e realmente eseguite.

### F25 — token commenti/dichiarazioni e copertura interna

GPT-R004-I002 non bloccante. Il tokenizer ignora comment/declaration nel
mezzo, mentre la guardia covered_end vede solo code finali. Raw generale
presente, nessuna perdita visibile/fetch dimostrati; serve diagnosi coerente.

Conserva token/raw/intervalli/offset e politica esplicita del profilo,
consumata da audit/scorer. Per il profilo conservativo ricevuto, questi
token fuori profilo danno unknown localizzato e stato coerente in
inizio/interno/fine: non COMPLETE/PASS per essere seguiti da altro testo.
Non basta il massimo end: verifica i rami/callback o intervalli interni
realmente consumati. Una soluzione locale finita è sufficiente.

Prove comment-middle/declaration-middle/comment-tail, in wrapper/cella/
caption/residuo e più token. Commenti e dichiarazioni non diventano prosa,
non estrarre come asset attivi img/font/URI citati **dentro** un commento;
niente fetch/execute. Test positivi literal `<`, entità, codice/fence e
HTML supportato senza commenti invariati, malformed tail continua a essere
rilevato. Conserva evidenze anche di callback ignorati prima, non inventare
una perdita materiale o una vulnerabilità non dimostrata.

### F26 — dominio del bound globale, chiarimento documentale

CLA-R004-I003. La garanzia di text_lower_bound è corretta sul **flusso
globale** dichiarato; non è un limite inferiore del CER/WER completo o
condizionato, che usa un diverso matching. Può superarlo: DOCX62>60char,
EPUB54>53, paragrafi invertiti D1 CERglobale0,667 contro completostrutturale0.

Inserisci questa distinzione esplicita in PROTOCOL, guide, nuovo report e
definizioni dati. Non imporre global≤structural, modificare allineamento,
normalizzazioni/denominatori o ridisegnare metrica per rendere vera quella
disuguaglianza. Conserva il campo compatibile e precisa a quale distanza
si riferisce; con testo non misurabile TRIVIAL_ZERO non è un risultato
completo0 o evidenza di fedeltà. Bound stretto su unknown non richiesto.
Pubblica nuovo report/dati r005 o errata versionata collegata, lasciando
immutati report/dati r004 e vecchie prove. Verifica gli esempi/campi e
l'invarianza delle cifre, senza nuova misura motori per una frase documentale.

## Perimetro e prove dipendenti

F19/F20/F05/F15 chiusi nei domini dichiarati; F16 preservato sulle64misure.
F17/F18 originali corretti, F21literal< corretto: preservali mentre completi
F22–F25. Gli altri fix r002 non riaperti senza dipendenza reale. L04output
UNAVAILABLE resta distinta dalla V reale sui13 riferimenti; A03 chiuso.
Nessuna nuova V generale, M/render o motore per questo delta previsto.

Proteggi i41997input del comune ricevuto, corpus/sorgenti/render/asset/gold/
packet/Voriginale/13receiptderivate,64raw e vecchi report/sonde/fallimenti.
Copie proprie per sonde consentite, nessuna mutazione dei canonici. Preserva
i snapshot reviewer e le prove precedenti, non riclassificare i loro FAIL.
I tre pyc managed ricevuti e causa storica non attribuita restano tali;
per input attuali aggiorna manifest/snapshot/gate propri prima di workload,
senza ripristino per nascondere differenze o download per riparare collection.

Catena managed CPython3.12.13: preflight prima degli import/--no-sync,
S/B/I attuali, sdist/wheel offline ricostruite e canonical reinstall dopo
modifiche, dieci hash/origini/stesso interprete dei figli. Helper/tests/
fixture fuori wheel con inventari/hash/byte prima-dopo e provenienza.
R rete isolata/registry/shell=False/confini mantenuti, no input comandi dai dati.
Vecchi stage sono storia, non devono tutti MATCH; riusa prove solo per
equivalenza reale delle dipendenze consumate, non per il nome del stage.

Riproduzioni prima/dopo dei nuovi casi, test pertinenti del reader/diagnosi/
offset e regressioni F17–F21, audit/scorer, suite veloce prescritta dopo
gate validi e contratti dipendenti. H12FAILdeliberati/expected0 prefissato
rimangono RFAIL con esito contrattuale atteso. Non ripetere F03/H15/F04/API/
Docker/ML/campagne a001 senza dipendenza reale, né ampliare dopo verifiche
concluse senza nuovo difetto/modifica. Test di contenuto/diagnosi/stato/exit,
positivi discriminanti e attesi indipendenti, non solo creazione dei file.

Per tutti i64raw dei due set32 crea nuove osservazioni/score r005 oppure
prova di invarianza completa verificabile dei campi dipendenti, con versione/
hash input/metriche e identità originarie. Zero motori per reader/diagnosi/
documentazione; se modifichi adapter/validator/R/writer/config, analizza e
ripeti solo prove invalidate nel mandato. Le cifre4PASS28FAIL sono storia,
non un assert da forzare. Legami raw→observe→audit→score→final, conditional/
global domain, disponibilità/denominatori/decomp/exit/batch verificabili.
Costi reali, limiti storici L01/L02/L04/L05/ChromeNONPASS/helper12gap
conservati; nessun GO o collaudo di altri motori/macchine.

## Quota unica e recupero

Quota52h/187200s da `2026-10-08T07:06:54.369068Z`, attese/review incluse,
CPU21600s/6GiB/RAM8GiB/stop sampled6GiB, stop nuovo49h/19800CPU-s/5GiB,
riserva3h/1800CPU-s/1GiB. Stop nuovi `2026-10-10T08:06:54.369068Z`,
deadline `2026-10-10T11:06:54.369068Z`. Nessun reset o incremento.
Ledger closing r004120881s wall/7524,45CPU-s lower/1479827278B retained
è storico. Aggiorna alla ripresa con attese e costi reviewer reali, senza
doppio conteggio CPU già inclusa nei wrapper; receipt supervisore aggiorna
wall, non inventa un saldo CPU/disco upper-bound. RSS campionato/retained
conservativo/cache OS hanno i loro limiti, niente certificazioni hardware.
Nessuna rete/acquisizione/invio/servizio/costo esterno o altro hardware impliciti.

Snapshot/manifest/argv/timeout/log/output/launcher propri sono mezzi ordinari
autorizzati entro cap. R≤900s/comando,1MiB/stream,32MiB/file; batching/riserva
senza consegnare a ogni sonda. Correggi errori ordinari/test rossi/freeze
tecnici nella stessa chat preservando i FAIL. EPERM/EACCES: diagnosi ed
escalation tool per la stessa azione autorizzata, mantenendo R; rispetta
un rifiuto effettivo, non aggirarlo. Limite sostanziale/riserva raggiunta
si riporta senza GO inventato, continuando il lavoro indipendente consentito.

## Consegna completa r005

- `reports/implementation-r005.md`: mappa F22–F26/prove prima-dopo/scelte,
  chiusure e residui, delta reale, catena/identità, prove ripetute/riusate
  con equivalenze, misure/invarianza64 e limiti. No GO autore.
- `evidence/implementation-r005/`: byte/protetti/input/argv/cwd/output/exit,
  snapshot tecnici/target/manifest, S/B/I/R, test/contratti/sonde e FAIL,
  misure/invarianza64, ledger cumulativo e own_pending finale.
- `handovers/implementation-r005.md`: input/output, consegna finale
  distinta da storia in corso, costi/limiti/processi e prossimo ruolo.
- Guide/PROTOCOL e nuovi report/dati o errata permanenti r005 su prove
  reali, date reali e domini distinti. Non sovrascrivere storia r004,
  STATE/HANDOVER/eventi/changelog/arbitrati/checkpoint del supervisore.

Verifica link/whitespace/git diff --check, indice vuoto; Git manuale dell'utente.
Alla consegna completa: supervisore per nuovo snapshot e due NUOVE review
indipendenti mirate, poi arbitrato reale. Nessuna review/GO autore, nuovo
lavoro roadmap, fase2, scelta motore o pulizia automatica.
