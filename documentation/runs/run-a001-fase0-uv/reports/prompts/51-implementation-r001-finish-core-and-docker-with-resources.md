# Implementazione r001 — Concludere core, suite e Docker CPU con risorse ammesse

Prosegui come implementatore nella stessa chat di report-r015. **Completa il piano
approvato e le verifiche mancanti**, correggendo ordinariamente nella stessa chat.
Nessuna attesa di freeze o nuova pianificazione. Non ripartire dall'installazione
legacy o dai vecchi prompt. Supervisore e reviewer restano ruoli separati.

Repository `/home/davide/workarea/markdown-for-llms`, branch `feature/run-a001-uv`,
HEAD/dev/base `66ba82200e5def5a4db76f9bafccb0731b506091`; worktree modificato
e nuovi file autorizzati, indice vuoto alla ricezione. Preserva il lavoro esistente.

## Input correnti

Leggi AGENTS, skill manage-implementation-run e protocollo. Sotto
`temp/run-a001-fase0-uv/`, recupera:

- `STATE.md`, il tuo checkpoint, `implementation/report-r015.md` e
  `arbitrations/addendum-operational-protocol-r017.md`;
- `evidence/supervisor-implementation-r001/completion-resources-r001/authorized-scope.json`
  e `reception.json`: **nuovi costi e acquisizioni autorizzati**;
- `evidence/implementation-r001/completion-r001/delivery-r015-r002.json` e
  `additional-resources-proposal-r002.json`, con i tuoi driver/V1 già preparati;
- `plans/plan-r003.md`, `arbitrations/arbitration-plan-r003.md` e R016 per criteri
  e autonomia. Se già letti, recupera le parti pertinenti; in chat fresca leggi
  il piano completo. Non caricare cinquanta prompt storici.

Piano SHA462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b;
arbitrato SHAf14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d.
R017 supera cap112MiB e divieti dev/API/Docker del precedente scope nei punti
esplicitamente estesi; R016 supera i gate di attesa del freeze del supervisore.

## Risorse: aggiornare il driver e proseguire

**Core:** +160MiB, totale incrementale272MiB dal medesimo Hentry553541632,
pool1GiB/stop896MiB/riserva16MiB invariati. Tempo7200s cumulativo, già
1523.5855519899271s addebitati alla consegna; nessun reset o doppio conteggio.
Il vecchio gate112MiB è storico e false: aggiorna driver/monitor a usare lo scope
R017 effettivo, registra il delta e verifica l'ammissione prima dei workload.
Acquisisci ora le dipendenze locked dev/API della proposta, tramite gli strumenti
standard e fonti configurate; prepara prefix nuovi sotto run/work. Confronta
selezione statica con export uv nativo, marker/ABI e hash reali prima dell'uso.

**Docker CPU:** pool separato16GiB, quota distinta7200s, acquisizioni<=1GiB.
Comprende wheel/Torch, OCI, apt, cache/layer/container propri, log/preparazione e
retry. I184806462byte noti non sono il totale completo. Completa nella stessa
chat metadata/dimensioni, digest OCI e apt firmati, ispezione EbookLib e backend
vincolato. Se tutto rientra nei limiti, acquisisci, costruisci e prova direttamente:
nessuna nuova richiesta al supervisore per quei prerequisiti ordinari.

Partiziona i ledger come nello scope: escludi dal core solo i due nuovi subtree
Docker nominati; contabilizzali insieme all'incremento dei propri oggetti engine
nel pool Docker. Non trasferire i parziali storici per liberare quota e non
scaricare payload pesanti su /tmp. Rileva data-root, baseline e spazio libero
del filesystem engine senza modificare l'host; contabilità conservativa quando
necessaria. Non applicare ai wheel/layer Docker il limite32MiB del core, né alla
build Docker il timeout900s dei figli core: usa dimensioni e deadline ammesse
nello scope, compatibili con tool e residui reali.

## Completa i criteri, conserva i risultati validi

22/22 S/B/I/E e12 casi installati sono ricevuti; la receipt conta67 chiamate,
rettifica il riepilogo64 nella nuova consegna senza riscrivere report-r015.
La ricezione del supervisore ha verificato s021/cache-s030 MATCH prima dei propri
delta documentali. Ora sono ingressi storici: crea tu snapshot tecnici nuovi,
senza pretendere MATCH del vecchio worktree o attendere il supervisore.

1. Riprendi V1 dalla diagnosi e dai driver corretti: canonica iniziale, prova
   config-only distinta dal flag, negativo stale prima del sync, rebuild e
   ripristino effettivamente verificati. Conserva i tre tentativi e lo stop
   storage; raccogli i processi e le ricevute di ogni nuova chiamata.
2. Chiudi le cinque CLI e i negativi, origini/transitive dei figli, workspace,
   precedenza dotenv/shell e supplementi V5 previsti. Contenuti, numeri, formule,
   chunk e riferimenti restano confrontati senza normalizzazioni. Preserva la
   baseline62PASS/5FAIL e le perdite legacy; non correggere algoritmi fuori piano.
3. Prepara davvero i profili dev/API, progetto installato dallo stesso interprete
   dei figli, cache tokenizer/Pandoc e preflight pronti. Poi esegui suite veloce
   con gli ignore di AGENTS, API, packaging, governance e discovery previste.
   Preparazione distinta dai test: niente installazioni/download come riparazione
   della collection. Correggi difetti ordinari del codice/harness e ripeti.
4. Esegui V7 build CPU con input OCI/apt/backend reali vincolati, S/B/I e
   dieci hash nel builder/runtime, config di entrambi i Compose. V8 verifica
   codice realmente installato, Marker/Surya/torch CPU, contratto wrapper con
   fake, librerie native/rendering e constructor senza pesi o font remoti.
   Usa il diagnostico offline del piano, senza avviare il servizio normale.
5. Aggiorna guide/configurazione pertinente e matrice finale A1–A7/V0–V12,
   distinguendo prove dirette, suite, mock e inferenza non eseguita.

R/D4 e Firejail restano obbligatori per i workload host offline; V8 conserva
`network none`/`pull never` e i controlli container del piano. Preparazione in rete
autorizzata distinta dalle prove offline. Tool approval ufficiale se necessaria,
senza handoff per richiederla; rispettare un rifiuto reale.

Engine esistente: solo oggetti di prova identificati della run, niente privilegi,
mount di dati privati/socket host, rete host, cambi persistenti o servizio normale.
Fermare e raccogliere propri container/build se un limite viene raggiunto; non
considerare la sola terminazione del client come raccolta del daemon. Niente
prune/cleanup globale, Git del repository, pesi/font remoti, inferenza/GPU/V10/V11
o paid. I Git sintetici delle fixture governance rientrano nei test autorizzati.

Usa gli strumenti esistenti e correggi i tuoi driver nel perimetro. Ogni FAIL
ordinario richiede diagnosi/fix/ripetizione, non consegna al supervisore. Nuovi
snapshot di esecuzione autonomi, output esclusivi, nessun PASS trasferito a input
modificati. Riusa B canonica solo con equivalenza dei reali input di build
verificata; altrimenti rebuild/reinstall e nuove prove dipendenti.

## Consegna al risultato

Concludi con `implementation/report-r016.md`, evidenze sotto
`evidence/implementation-r001/completion-r002/` e sotto i subtree Docker ammessi,
aggiornando il tuo checkpoint. Registra criteri/prove, errori/soluzioni, identità,
costi cumulativi distinti e processi propri raccolti. Non modificare registri
comuni o arbitrati. Nessuna sola request di preparazione/freeze.

Solo un limite sostanziale dimostrato oltre R017 richiede supervisione anticipata;
completa comunque il lavoro indipendente e riferisci il confine effettivo.
Quando i criteri sono verificati, il supervisore riceve il risultato, prepara
snapshot comune e due review reali indipendenti ChatGPT/Claude. Nessun GO finale,
commit/merge/push/deploy può essere dedotto dai PASS dell'implementatore.
