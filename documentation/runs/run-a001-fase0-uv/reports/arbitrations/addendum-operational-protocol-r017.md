# R017 — Risorse ricevute per concludere core, suite e Docker CPU

2026-10-06, supervisore. Ricevuti report-r015 e proposta unica r002. Il piano
r003 resta GO; R016 continua a delegare fix e snapshot tecnici all'autore.
Nessuna nuova pianificazione, review preventiva o attesa di freeze.

## Ricezione e risultato

Ricezione mirata in `evidence/supervisor-implementation-r001/completion-resources-r001/`:
59 identità di report/proposta/receipt/input/archivi ricalcolate; 22 operazioni
PASS e quattro I correnti ricevuti. Ricevuta di fedeltà PASS: 12 casi e **67**
chiamate, non 64 come nel report; rettifica editoriale delegata nella consegna
successiva. Non riscrivere report-r015 o la delivery ricevuta.

Snapshot product-s021 e cache-s030 verificati MATCH prima degli aggiornamenti
del supervisore. L'identità dei wheel proposti coincide con il lock: core36
wheel/10659742byte; Docker94wheel/184806462byte noti, Torch/OCI/APT esclusi da
quel totale parziale. Nessuna prova applicativa rieseguita dal supervisore.

La quota precedente è realmente superata: H672907264 alla lettura, oltre
Hentry553541632+112MiB. V1 NON PASS, parziali conservati. Le altre prove non
diventano FAIL per questo limite, ma non dimostrano i criteri mancanti.
Nessun processo proprio attivo dichiarato nella raccolta finale dell'autore.

## Disposizione delle risorse

Autorizzo la proposta nel perimetro del piano, con ammissione runtime verificata
dall'implementatore. Fonte operativa:
`evidence/supervisor-implementation-r001/completion-resources-r001/authorized-scope.json`.
Superati i precedenti divieti di acquisizione dev/API/Docker e il cap112MiB
nei soli punti qui esplicitamente estesi. Non cambia il significato dei FAIL
precedenti né viene azzerato il pregresso.

- **Core +160MiB:** incrementale totale272MiB dal medesimo Hentry553541632;
  pool1GiB/stop896MiB/riserva16MiB invariati. Formula del nuovo scope, ledger
  cumulativo e controlli di spazio libero; alla ricezione165847040byte residui.
  Acquisizioni locked dev/API previste dalla proposta ora autorizzate, preparate
  separatamente dalle suite; nuovi prefix sotto run/work, quattro basi conservate.
  Tempo7200s cumulativo: ingresso1523.5855519899271s, residuo5676.414448010073s.
- **Docker CPU distinto:** massimo16GiB storage e7200s cumulativi per preparazione,
  acquisizioni, build e prove; massimo1GiB di traffico acquisito, inclusi metadata,
  OCI/APT/Torch, retry e non soltanto il totale noto dei wheel. Paid0. Prima dei
  payload verificare provenienza/hash/dimensioni e prima della build toolchain
  OCI, firme/versioni apt e backend/requisiti EbookLib. Metadati mancanti e
  strumenti propri si completano nella stessa chat entro questi limiti.

I due ledger devono essere distinti: il core esclude esclusivamente i due nuovi
subtree Docker nominati nello scope; Docker include tali subtree e l'incremento
attribuibile a immagini/layer/build-cache/container propri nell'engine esistente.
Non spostare vecchi dati nel nuovo pool per liberare quota. Registrare baseline
engine/data-root e spazio sul suo filesystem; usare contabilità conservativa
se i byte condivisi non sono separabili. Non inventare misura o garanzia atomica.

Il cap32MiB per singolo file resta del core; i payload Docker verificati possono
superarlo entro dimensione dichiarata e residui reali. La build Docker sceglie
deadline finite compatibili con tool e quota7200s distinta; non eredita per
errore il cap900s dei figli core. Log normali e contabilità dei retry invariati.
Non applicare la soglia libera del repository al diverso filesystem /tmp;
i nuovi payload pesanti restano nei subtree dedicati e nello storage dell'engine.

## Operazioni e confini

Autorizzate acquisizioni pubbliche necessarie dai pin/fonti configurati e relativi
redirect verificati; niente upload di documenti o segreti. Preparazione in rete
delimitata e distinta dai workload offline. Nessun backend ignoto o resolver
alternativo implicito: derivare export nativo, marker/ABI e input prima dell'uso.
Acquisire digest OCI e metadata apt firmati nella stessa chat; non è richiesto
un handoff per ricevere ciascun digest. Se tutti i requisiti sono soddisfatti
entro il costo, l'autore prosegue direttamente con build e prove.

Engine Docker locale già presente: consentite letture di preflight e operazioni
solo su immagini/build/container di prova identificati per la run, senza mount
di dati privati, socket host nei container, privilegi, rete host, modifiche
persistenti o startup del servizio normale. V7 include build CPU e config dei
due Compose; V8 diagnostico installato offline `network none`/`pull never`.
Il traffico di preparazione Docker non è una prova R/D4. Le prove host conservano
R/D4 e le prove container il confinamento richiesto dal piano.

Niente pesi, font remoti, inferenza Marker, GPU/V10/V11 o benchmark paid.
Usare solo librerie native/font di sistema pianificati e dati sintetici. Niente
prune/cleanup globale o Git del repository. Fermare/raccogliere propri workload
al limite; una CLI Docker terminata non prova che la build daemon sia terminata.
Usare il sistema ufficiale di approvazione quando richiesto, rispettarne il rifiuto.

## Mandato e seguito

Prompt51 copre l'intero lavoro restante: V1, cinque CLI, supplementi di fedeltà
e negativi, preparazione profili dev/API, suite/discovery/governance, V7/V8 e
documentazione. Fix e catture tecniche autonomi; consegna report-r016 al risultato.
Gli input precedenti diventano storici dopo i delta dichiarati: identificare
gli input finali e ripetere soltanto le prove invalidate, motivando equivalenza
quando si riusa B canonica. Conservare il superamento di quota e tutti i FAIL.

Si torna prima al supervisore solo per un limite sostanziale effettivo oltre
questo scope, continuando il lavoro indipendente. Nessuna richiesta di sola
preparazione o permesso per una scelta ordinaria. GO finale aperto: dopo risultato
completo, snapshot comune e due review reali ChatGPT/Claude; arbitrato e Git manuale.
