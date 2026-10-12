# Pulizia selettiva eseguita dall'utente — 2026-10-07

La supervisione a002 ha ricevuto le receipt dell'esecuzione manuale delle
17:02 Europe/Rome: entrambe `PASS`, `apply: true`. Sono esecuzioni reali,
distinte dal dry-run descritto nella [valutazione precedente](cleanup-assessment-2026-10-07.md).
Il supervisore non ha effettuato rimozioni.

- [Receipt file](../evidence/cleanup-files-receipt-2026-10-07.json): 193 file
  identificati rimossi, nessun file già assente alla chiamata. Lista immutata;
  ricontrollate tutte le 193 assenze e hash/dimensioni delle 91 copie conservate
  uniche. Byte logici rimossi: 334.918.780 (~319,40 MiB).
- [Receipt Docker](../evidence/cleanup-docker-receipt-2026-10-07.json): tre tag
  r004/r005/r006 rimossi con output di untag/delete. Il ricontrollo corrente
  mostra soltanto `run-a001-marker-cpu:r007 b3db6dc22d92` per quel repository.
- [Ricezione e limiti](../evidence/cleanup-reception-2026-10-07.json): copie
  archiviate byte per byte con hash degli originali in maintenance. Prima query
  Docker fallita per accesso al socket; escalation read-only approvata e riuscita,
  senza cambiamenti ai permessi persistenti o rifiuti dell'approvazione.

Il recupero fisico di spazio non è stato misurato; non sommare le dimensioni
nominali delle immagini. Non è stata ripetuta una campagna a001 né attribuito un
nuovo PASS a prove storiche. Le vecchie ricevute/snapshot restano riferiti agli
input originali; i path eliminati sono provenienza, le copie conservate sono
indicate nella [mappa esatta](cleanup-duplicate-payloads.json).

Questo avanzamento amministrativo è successivo alla chiusura tecnica e non
modifica i 266 record originali di archive-files.json. Nessuna pulizia generale
è necessaria per a002, nessuna rimozione residua identificata da questi controlli.
