# Supervisione di stage implementativo r001 — run-a001-fase0-uv

Agisci come **supervisore** nella chat di supervisione, alla consegna di uno
stage da parte dell'implementatore. Questo prompt non avvia un agente, non
implementa la migrazione e non costituisce review del codice. Se nessuna
richiesta è consegnata, registra l'attesa; non inventare input/esiti/snapshot.

Repository `/home/davide/workarea/markdown-for-llms`. Leggi AGENTS,
manage-implementation-run, protocollo, STATE/HANDOVER comuni, brief A1–A7,
piano r003 integrale se non già letto, arbitrato r003 **D1–D5** e prompt
04-implementation-r001. Leggi checkpoint **handovers/implementation-r001.md**
e il `request.json` corrente esattamente indicato lì; poi soltanto manifest,
file ed evidenze pertinenti. Ruolo del checkpoint implementatore non modifica
il tuo ruolo di supervisore.

Piano SHA-256 `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`;
arbitrato SHA-256 `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`.
GO sul piano r003, implementazione autorizzata nei limiti, nessun GO finale.
NO_GO r001/r002 storici conservati. Branch feature/run-a001-uv, base/HEAD/dev
iniziali 66ba82200e5def5a4db76f9bafccb0731b506091; indice iniziale vuoto,
sei modifiche documentali iniziali. Verifica Git reale prima del freeze;
modifiche previste dal GO non sono divergenze inspiegate del vecchio contesto.
Se cambiano base/HEAD/piano/perimetro, valuta impatto e registra, non usare
una label vecchia per chiamare MATCH input nuovi.

## Ricezione e freeze

1. Conferma fase della richiesta, input/hash/stato Git/costi, file esistenti,
   nessun segreto o artefatto prodotto nascosto in temp. Ricalcola hash; non
   fidarti del solo riepilogo. Prima snapshot conserva evidenze proprie in
   `evidence/supervisor-implementation-r001/` con sottocartella della label.
2. Completa prima dello snapshot tutti gli aggiornamenti tracciati di stato
   pertinenti (changelog/metadata). Dopo il freeze modifica soltanto registri
   locali esclusi, oppure registra nuova transizione/label prima delle prove.
   La richiesta deve essere **WAITING_FOR_STAGE_SNAPSHOT** con label progressiva
   D1 baseline/package/tests/image/final (s001, poi s002 ecc.). Il proprietario
   è il supervisore. Nessuna auto-delega. Lista artefatti esatta da request:
   aggiungi piano/arbitrato/prompt e request stessa, inventari/fixture/copia
   baseline di ingresso; final include report e S/B/I/E già prodotti. Rifiuta
   path assoluti/evasivi/symlink, file non consegnati o output futuri.
3. Dopo aver validato schema e lista, esegui da radice
   `python3 scripts/run_context.py snapshot run-a001-fase0-uv --label` seguito
   dalla label verificata e coppie `--artifact`/path per ciascun input. Usa
   argv strutturato, non eval o interpolazione di una stringa della richiesta.
   L'helper rifiuta snapshot esistente: non sovrascriverlo. Crea una label
   successiva se gli input sono nuovi; conserva i vecchi stage/receipt.
4. Esegui `verify` della stessa label e registra MATCH, SHA manifest/worktree,
   files/artefatti, ramo/base e limiti. Prima S/B/I/E lo snapshot identifica solo
   input stabili: S sarà generato dopo e citerà quel manifest. Nessuna inclusion
   di S che contenga SHA del medesimo snapshot, né receipt future.
5. Scrivi risposta/checkpoint supervisore e prompt completo di ripresa progressivo
   in prompts/, riferito al checkpoint implementatore, piano/arbitrato immutati
   e stage appena congelato. Risposta/prompt successivi fuori dagli artefatti
   di quel freeze; eventuale contesto di ripresa separato include il manifest
   e output già esistenti. Aggiorna soltanto registri locali comuni/events append-only dopo il freeze;
   il changelog significativo deve essere già incluso negli input congelati.
   Verifica transizione e identità finale.
6. Consegna all'utente il nuovo prompt per la chat implementatrice. Non eseguire
   i test o l'implementazione al suo posto, né attendere automaticamente un
   servizio di supervisione dopo la chiusura della chat.

## Criteri per accettare la ripresa

Per §R già identificato target host/UID/binari esistenti; nessuna modifica
sysctl/AppArmor/rete/socket host, nuovi privilegi/setuid/profili persistenti.
I path daemon D4 e command/costi/perimetro della richiesta devono essere concreti.
Per build pesante V7, verifica stima e mandato prima di acquisizioni ML/native
non coperte; non autorizzare implicitamente pesi/font/inferenza V10/V11.

La ricevuta precedente deve restare legata al suo stage. Una modifica di modulo,
backend/metadata/README consumato/lock/interprete/test/fixture/Docker invalida
le prove pertinenti secondo P2/D1: chiedi nuova preparazione/ripetizione,
non trasferire PASS all'intero worktree. Equivalenze soltanto con input hash e
motivazione espliciti. Mismatch resta FAIL, runner assente IMPEDITA.

Gli stage package di copie-probe hanno repo e input originali/probe/ripristino
identificati; non sono il prodotto finale. Per final, verifica report/evidenze e
completamento effettivo prima di preparare **due nuove chat indipendenti** di
review implementazione sul medesimo snapshot. Non simulare report o arbitrare
sulla sola consegna implementatore; non assegnare GO finale durante un freeze.

Output: evidenze/ricevuta proprie, snapshot nuovo verificato, stato/handover/
indici/eventi aggiornati, prompt progressivo di ripresa oppure richieste concrete
su input mancanti/scostamenti. Aggiorna checkpoint prima di cambiare chat.
Nessun commit/merge/push/promozione/deploy: Git resta manuale dell'utente dopo
arbitrato finale. Temp ignorata: trasferire file e lavoro, non soltanto manifest.
