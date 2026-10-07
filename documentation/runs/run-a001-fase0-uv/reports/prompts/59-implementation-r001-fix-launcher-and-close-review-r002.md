# Implementatore r001 — Chiudere il residuo R e le guide dopo le review r002

Prosegui **nella stessa chat implementatrice** di report-r019. Completa questo
fix in un'unica consegna, risolvendo errori ordinari e ripetendo le prove
invalidate nella stessa chat, senza richieste di freeze intermedio.
Repository `/home/davide/workarea/markdown-for-llms`, branch `feature/run-a001-uv`,
HEAD/dev/base `66ba82200e5def5a4db76f9bafccb0731b506091`; indice vuoto alla
ricezione, worktree modificato autorizzato. Nessun reset/Git del repository.

## Input e confini

Leggi AGENTS, skill manage-implementation-run/protocollo e il tuo checkpoint.
Input correnti, sotto `temp/run-a001-fase0-uv/`:

- [Arbitrato r002](../arbitrations/arbitration-implementation-r002.md),
  [R021](../arbitrations/addendum-operational-protocol-r021.md) e
  [scope autorizzato](../../evidence/supervisor-arbitration-implementation-r002/authorized-scope.json).
- Entrambi i report `reviews/review-implementation-r002-{chatgpt,claude}.md`,
  report-r019 e prove `evidence/implementation-r001/fix-review-r002/`.
- Piano/arbitrato piano r003 e arbitrato codice r001 per i requisiti pertinenti
  di D2, R e documentazione; nessuna nuova pianificazione.

final-s002 SHA `e0cb33d16a5766f653923c2d25cf365f910d7e738d542893db2a1ac589272c1d`
era MATCH alla ricezione prima degli aggiornamenti propri del supervisore.
Ora è storico: non sovrascriverlo e non fermarti perché i registri sono cambiati.
Crea autonomamente snapshot tecnici/input S necessari, escludendo output futuri
e checkpoint vivi; sono attribuzioni delle prove, non nuove approvazioni.

**+600 s core ora autorizzati: cap cumulativo 8880 s.** Usa come ingresso
prudenziale `max(8380 s, maggiore costo storico ricostruito)` e conteggia tutto
il nuovo lavoro attivo: preparazione, tentativi, guardie, prove e consegna.
Al minimo ingresso hai **500 s** nuovi, non 600 s per comando; tieni almeno
60 s per raccolta/chiusura, singolo figlio<=180 s e deadline compatibili col
residuo. La vecchia quota resta NON PASS: storico almeno8321,418589 s /
sforamento almeno41,418589 s, upper ignoto; non ripetere il vecchio dato5,287
come sforamento finale. Registra questa ripresa e ogni script/riscrittura finale.

Storage352MiB/Hentry553541632/pool1GiB/stop896MiB/riserva16MiB invariati, radici
ed esclusioni dello scope. Rimisura con output review/supervisione; niente reset,
trasferimenti nel pool Docker o cancellazione di FAIL/baseline. Nuovi output
compatti, nuovi output entro4MiB e comunque entro il residuo misurato; nessuna
copia intera dell'ambiente quando basta una fixture sintetica.
Nessuna rete/download/ML/GPU/Docker/build o nuova dipendenza. Usa managed locale
3.12.13 verificato, B56 e prefix già pronti; mantieni i controlli R/D4.

## Risultato completo

1. **GPT-I002 / residuo CLA-I001.** Generalizza `snapshot_check()` in
   `scripts/diagnostics/run-a001-fase0-uv/run_offline.py`: resta un controllo
   vincolante dello snapshot dichiarato, con schema corretto, run_id/label
   validi, path confinato al clone/stage, symlink respinti e verify reale.
   Preserva binding prima/dopo e tutte le guardie R. Puoi riusare/condividere
   validazione permanente coerente col produttore S, oppure correggere il ramo
   locale con test; non aggiungere dipendenze circolari. Il nome storico della
   directory diagnostica non impone a001. Non eliminare la verifica per
   accettare qualunque JSON, non usare il bundle preliminare per workload.
2. **Prove mirate del launcher.** Test permanenti sulla validazione per a001
   e un secondo ID valido, e sui confini negativi: schema bool/invalido,
   ID/label/path evasivi, symlink e snapshot stantio. Dimostra il rifiuto prima
   dell'avvio del workload/verificatore quando appropriato, senza falsi PASS.
   Ripeti i tre negativi di cleanup già pertinenti dopo il cambio dello stesso
   file. I mock sono legittimi per questi rami e distinti dalle prove reali.
   Esegui **il launcher R reale corretto** almeno per a001 e una seconda run
   valida (es. b002) con `run_context.py verify` reale, target installato
   verificato e comando sintetico minimo offline; ricevute binding/preflight,
   R/D4, exit, postcheck e raccolta propri. Non basta richiamare soltanto S/I
   o simulare subprocess come nella sonda del reviewer. Riusa strutture pronte;
   genera i nuovi S/I/binding richiesti per i soli target effettivamente usati,
   verificando payload B56 e interprete. Non attribuire I/R vecchi a S nuovi.
   Un piccolo clone sintetico è ammesso se serve al confinamento del secondo
   snapshot; nessun commit/remoto, né modifica degli input di confronto.
3. **GPT-I003 / CLA-I007 e residuo CLA-I004.** Correggi in
   `docs/how-to/ambiente-uv.md` il flag inesistente di verify_distribution:
   I deriva lo scope da S. Aggiungi `RUN_IMAGE_CONTEXT_RECEIPT` alla reference
   `docs/reference/toolchain-legacy.md`, allineandola al test/guida Docker.
   Controlla opzioni/variabili delle sezioni toccate sul codice, link e whitespace.
4. **CLA-I008: alternativa documentale arbitrata.** La guida uv distingue
   esplicitamente C-docker: richiede `RUN_TEST_MODE=official|standalone`;
   il solo flag pytest non seleziona la modalità del driver Docker. Ripeti
   la precisazione nella reference e, se necessario, nella guida Docker,
   conservando il rifiuto di scelte discordanti. Non modificare test/driver
   Docker: niente nuova V7/V8 o build per una nota. Non dichiarare corretto
   il comportamento CLI escluso: il supporto è quello esplicitamente descritto.
5. **CLA-I006 e riusi.** Il supervisore ha già corretto il ledger corrente;
   riportane fonte, limite inferiore e incertezza nel nuovo report, senza
   riscrivere report-r019/receipt storiche. Conserva il comando reale di ogni
   nuovo script e la misura di chiusura comprensiva delle riscritture.
   Identifica delta e input consumati: riusa B56, Docker r007/C-docker,
   fedeltà/CLI/host/V1 ai loro stage quando invariati. Le vecchie prove sotto R
   restano storiche; le nuove prove mirate attestano il launcher corretto.
   Per soli diagnostici e guide non consumate, non ricostruire o reinstallare
   la wheel e non ripetere tutte le campagne. Se una scelta tecnica modifica
   davvero input consumati da altre prove, gestisci le sole invalidazioni
   necessarie entro il mandato; non toccare README/pyproject/lock/dieci moduli.

Errori ordinari, log/timeout/path e launcher propri sono scelte operative
delegate entro requisiti/costi. EPERM/EACCES: conserva errore, diagnostica ed
eventuale escalation dei tool per l'azione autorizzata, mantenendo isolamento;
solo rifiuto effettivo o limite sostanziale reale richiede supervisione.
Continua il lavoro indipendente ammesso se una parte incontra un limite reale.

## Consegna

Scrivi **`implementation/report-r020.md`**, evidenze nuove in
`evidence/implementation-r001/fix-review-r003/` e aggiorna il tuo checkpoint.
Matrice dei rilievi: fix, prove reali/mock distinte, riusi con identità,
invalidazioni, costi storici corretti e nuovi costi completi, limiti e processi
raccolti. Non modificare stato condiviso, arbitrati, report/review precedenti
o snapshot congelati. Nessun GO finale o Git.

Poi il supervisore prepara snapshot comune e **due review nuove mirate** alle
chiusure e regressioni di questo delta; la futura chiusura/Git resta manuale
dell'utente dopo GO. Non consegnare una richiesta di freeze invece del risultato.
