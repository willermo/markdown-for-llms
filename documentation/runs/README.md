# Archivio permanente delle run

Qui si conservano sintesi dello sviluppo, decisioni e report selezionati. Dal
2026-10-15 le prove operative restano in `temp/` e non vengono copiate qui a ogni
giro. Gli archivi storici già tracciati conservano la propria struttura; la nuova
regola non ne dispone la cancellazione. La prima run è integrata in `dev` a `b90c6d4`.

Ogni run archiviata ha una directory `<run-id>/`, un `manifest.md` derivato dal
[template](../development/templates/archive-manifest.md) e sottocartelle per
report e note effettivamente selezionati. Aggiungere qui il collegamento alla run
solo quando esiste un archivio reale, con stato e risultato di integrazione.

Probe e test riutilizzabili vanno nelle posizioni eseguibili indicate dal
[protocollo](../development/run-lifecycle.md), collegandoli dal manifest. Questa
cartella contiene documentazione permanente. Stream, matrici, receipt e copie
di esecuzione restano nella run locale; il manifest ne dichiara disponibilità e limiti.

I file di questa cartella sono esclusi dall'impronta del codice del helper di run
per consentire l'archiviazione dei report dopo le review. L'integrità dell'archivio
va verificata separatamente e registrata nel manifest.

## Run archiviate

- [run-a003-fase1-engine-comparison](run-a003-fase1-engine-comparison/manifest.md):
  sintesi della run con validazione differita per priorità utente al prodotto.
  NO_GO storici preservati; trace r005 ricevuto/non revisionato, prove migrate
  in temp e debito aperto. Nessuna chiusura o integrazione dichiarata.

- [run-a001-fase0-uv](run-a001-fase0-uv/manifest.md): toolchain uv, fase0.1;
  GO finale r003 dopo review GO/GO, quote temporali NON PASS conservate.
  Archivio selettivo verificato; feature `8479f97` integrata manualmente in dev
  a `b90c6d4`, albero identico e 405 file MATCH. Registri e push verificati
  a `80c31da`; [pulizia selettiva manuale ricevuta](run-a001-fase0-uv/notes/cleanup-completion-2026-10-07.md)
  (193 file e tre tag superati). Nessun deploy/main o pulizia generale della run.

- [run-a002-fase1-corpus](run-a002-fase1-corpus/manifest.md): primo incremento
  fase1, chiusa12ott2026 per accettazione esplicita del perimetro ridotto;
  reader sperimentale/debito obbligatorio ADR0008. Archivio selettivo verificato,
  64bundle eindice umano;ClaudeNO_GO/GPTINCOMPLETA conservati,nessunGOtecnico,
  integrazione o pulizia dichiarati. Fase1complessiva ancora aperta.

Integrazione a002 ricevuta12ott2026: dev/feature a `7caff336119b6d7d78978204142cfdc16d620fff`,
352filegenerali/2artefatti identici; [ricevuta](run-a002-fase1-corpus/evidence/integration-reception-2026-10-12.json).
Registri/24log predisposti per commit amministrativo,push/main/deploy assenti.
