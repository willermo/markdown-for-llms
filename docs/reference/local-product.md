# Reference del prodotto locale I1

Configurazione amministrativa: [.env.example](../../.env.example). Precedenza:
default → file selezionato con `--env-file` → ambiente → `--data-root` e
`--model-root`. Il file non viene cercato implicitamente. Prefisso `MDF_`,
nomi sconosciuti e valori non validi rifiutati, nessuna configurazione JSON
aggiuntiva per l’utente. Le identità dello storage, dello schema e della
configurazione sono verificate da CLI, API e worker. Lo schema iniziale è1;
schemi futuri sono rifiutati. Non sono previste migrazioni distruttive.

| Variabili MDF | Default | Significato |
| --- | --- | --- |
| `DATA_ROOT`, `MODEL_ROOT` | `./mdf-data`, `./mdf-models` | Directory distinte, senza symlink |
| `HOST`, `PORT` | `127.0.0.1`,8765 | Solo loopback IPv4; Host localhost o127.0.0.1 sulla porta scelta |
| `BIND_PROFILE`, `REVIEW_PROVIDER` | `marker-cpu-v1`, `disabled` | Unico profilo ammesso, nessuna revisione AI |
| `TIMEOUT_SECONDS` |450| Termine del figlio, errore visibile; non una SLA |
| `MEMORY_MIB`, `THREADS`, `PROCESS_LIMIT` |16384,4,512| RLIMIT_AS per processo, RSS del gruppo campionata; affinità CPU e limiti processi |
| `UPLOAD_MIB`, `MAX_PAGES` |50,50| Rifiuto esplicito; nessun taglio `page_range` |
| `OUTPUT_MIB`, `LOG_MIB` |512,10| File size limit più guardia totale scratch; log limitato |
| `STORAGE_MIB`, `RESERVE_MIB` |20480,1024| Quota archivio e spazio libero; nessuna cancellazione automatica |
| `MAX_PIXELS`, `MAX_DIMENSION` |25000000,16000| Limiti prima del decode PNG/JPEG; altri formati esclusi dalla galleria |
| `POLL_SECONDS`, `LEASE_SECONDS`, `GRACE_SECONDS` |1,15,5| Poll worker, lease, grazia prima dell’arresto forzato |

Tutti gli interi sono positivi e finiti. Il lease deve essere almeno tre volte
il polling. Le opzioni di conversione sono passate al pin e registrate separando
richieste, traduzione nativa e valori effettivi. `force_ocr` e `strip_existing_ocr`
sono booleani; gli altri parametri del profilo sono fissi. Il mapping
`disable_image_extraction=false` usa il campo nativo `extract_images=true`.
Batch layout/recognition/detection/table/OCR-error1, pdftext worker1, CPU,
DPI96/192, interop1, `use_llm=false`, `paginate_output=true`.

| Comando `mdf` | Funzione |
| --- | --- |
| `serve` | API/UI loopback; nessuna inferenza nel processo |
| `worker [--once]` | Worker separato; una sola proprietà ML per storage |
| `convert FILE... [--wait] [--force-ocr]` | Submit separati; attesa opzionale del worker; batch nonzero per errori |
| `configure --expected-config HASH [--storage-mib N] [--timeout-seconds N]` | Cambio amministrativo con vecchia configurazione verificata e archivio fermo |
| `jobs`, `show ID` | Stato, identità, problemi, configurazione e manifest |
| `retry ID` | Nuova esecuzione collegata a quella terminata |
| `cancel ID` | Annulla coda o richiede arresto dei processi attivi |
| `download ID --output PATH [--overwrite]` | ZIP dopo verifica identità; sovrascrittura solo esplicita |

Ogni comando accetta prima del sottocomando `--env-file`, `--data-root`,
`--model-root`. Equivalente: `python -I -B -m markdown_for_llms.cli`.
Gli esiti CLI sono JSON; gli assi `state`, `bundle`, `quality` e il messaggio
restano distinti. `unchecked` scaricabile è un avviso; `incomplete`/`invalid`
non sono successi del batch.

API sotto `/api/v1`: `GET session`, `GET readiness`, `GET runs`, `GET runs/ID`,
`POST runs`, `POST runs/ID/retry`, `POST runs/ID/cancel`, `GET runs/ID/text?offset=N`,
`GET runs/ID/artifacts/ARTIFACT_ID`, `GET runs/ID/source`, `GET runs/ID/bundle`.
Upload: body PDF binario, header `X-Filename` codificato percent-encoding e
`Idempotency-Key` opzionale. Mutazioni: Origin esatto e `X-MDF-CSRF` dalla sessione.
CORS chiusa; OpenAPI JSON locale, nessuna Swagger/CDN. Gli asset sono indirizzati
per ID, mai per path della richiesta. Il testo è paginato a256KiB con offset byte
allineato UTF-8; `complete` segnala la lettura integrale. Offset negativi, oltre
fine o su byte di continuazione restituiscono HTTP400, senza invalidare il
bundle; l’offset di fine documento restituisce una porzione vuota completa.
`next_offset` è sempre riutilizzabile. UTF-8 realmente corrotto resta un
errore d’integrità e blocca lo ZIP.

Il manifest schema1 include documento/run/tentativo, source hash, inventario
con ID/hash/byte, profilo/tempi, controllo riferimenti e limiti. Esclude segreti
ambientali e path host. Il JSON del modello nativo e i log limitati restano
nelle directory diagnostiche dei tentativi; non sono inseriti automaticamente
nel download ordinario. Il raw non viene normalizzato o ripulito.

La directory dati deve appartenere all’utente corrente e non essere scrivibile
da gruppo o altri utenti. Le radici dati e modelli non possono coincidere né
contenersi. L’isolamento protegge dai documenti importati e dai figli di
conversione; non protegge da un altro processo ostile con lo stesso account che
modifichi direttamente database, runtime o directory amministrative.


La capacità condivisa comprende sorgenti, tentativi, log, metadata, SQLite,
ZIP e staging. Ogni writer prenota lo spazio prima della crescita: upload fino
al limite più64KiB, worker `2 × OUTPUT_MIB + LOG_MIB + 1`MiB, pubblicazione
con margine aggiuntivo per il DB.256KiB restano per il controllo amministrativo.
I byte già presenti nella prenotazione non sono sommati una seconda volta.
La riserva libera del volume si aggiunge agli impegni ancora da scrivere.
L’inferenza non detiene il lock di ammissione o una transazione DB.
La dimensione scratch e RSS restano controlli campionati; prima di pubblicare
si ricontrollano quota e impegni. Durante i tick si controllano solo il budget
proprio e la riserva fisica; la scansione globale avviene all’ammissione e
alla pubblicazione, fuori dall’inferenza. Metadata motore/raster/manifest
hanno limite1MiB e profondità massima128contenitori, inventario4096voci.
Il limite comprende i byte UTF-8 effettivi e il newline finale. Il manifest
completo è validato prima di `publishing`: l’envelope aggiunge profondità ai
metadata del motore. Nuovi manifest in file, DB e ZIP condividono la stessa
serializzazione UTF-8; quelli storici non vengono riscritti.
NaN, Infinity ed esponenti che producono numeri non finiti sono rifiutati. Un limite superato produce un errore visibile.

L’upload HTTP scrive direttamente nello staging attribuito, senza seconda copia.
Rifiuti, stream interrotti e conflitti vengono raccolti; le fonti accettate
restano conservate. Un replay con chiave già registrata confronta firma,
dimensione, hash e opzioni leggendo lo stream senza scriverne una copia,
anche quando non c’è spazio per un nuovo upload.

Lo ZIP è una cache immutabile unica per tentativo, con digest e controllo CRC
ad ogni consegna; tutti gli originali e il manifest vengono verificati anche
quando la cache esiste. Usa ZIP_STORED: lo spazio necessario è la somma dei membri,
del manifest e degli header, con margine64KiB. Le creazioni concorrenti dello
stesso ZIP sono serializzate; download ripetuti e disconnessioni riusano il file.
I `.part` appartengono a una prenotazione e vengono rimossi su errore o recupero
post crash. Nessuna raccolta basata soltanto sul suffisso; i writer vivi sono
protetti da flock. La cache conta nella quota, senza nuova copia ad ogni click.

Un file assente, non regolare, non sicuro o alterato invalida il bundle.
`EMFILE`, `EIO`, `ENOMEM` e permessi transitori rendono la lettura indisponibile:
HTTP503 con `retryable=true`, senza invalidazione permanente. Dopo il recupero
si ritenta lo stesso risultato. Gli errori SQLite sono espliciti; il worker
limita i tentativi di recupero e, con DB non scrivibile, termina dichiarando
la riconciliazione necessaria al riavvio, senza inventare uno stato persistito.


Un errore di capacità, lettura o sincronizzazione durante il recupero lascia
`publishing` recuperabile: dettaglio e lista espongono `error` e `recovery`.
La lettura mostra l’ultimo motivo registrato, quando disponibile, senza
scansione globale né lock di ammissione. Se la quota impedisce di registrare
la diagnosi, compare un motivo generico di pubblicazione differita quando
il worker non detiene il suo lock; nessuna scrittura diagnostica oltre quota.
Se non può registrare la diagnosi, il worker termina liberando la proprietà:
lo stato differito resta osservabile senza un controllo di capacità in lettura.
La pubblicazione attiva senza errori non propone automaticamente un riavvio.
Il riavvio del worker tenta una
riconciliazione finita dello stesso ID, senza nuova inferenza. Prima di
`completed` ripete fsync dei file/albero e dei parent `attempts`/`results`,
anche quando il rename risulta già eseguito. Hash leggibili non attestano
da soli la durabilità. Un guasto persistente non abilita il download.
Il risultato completo può trovarsi ancora in `attempts` dopo un errore prima
del rename, oppure già in `results`: entrambe le posizioni sono recuperabili.
Entrambe presenti/assenti, manifest incongruente o file alterati producono
un errore d’integrità esplicito, conservando i byte rimasti. Uno scratch
incompleto in `running` resta invece `interrupted` con retry distinto.
Prima del controllo di capacità il recupero raccoglie i figli identificati
del worker morto e rilascia i relativi impegni; i proprietari vivi restano protetti.

Voci symlink/hardlink/FIFO/socket nello scratch bloccano il tentativo. Dopo
il fallimento e la raccolta dei figli, il conteggio dei parziali attribuiti
nel DB include le voci non sicure senza seguirle o aprirle (minimo4KiB per
voce speciale); altri submit/ZIP/retry possono proseguire entro quota.
Percorsi non attribuiti, sorgenti, risultati pubblicati, DB e runtime restano
strettamente controllati. Nessuna rimozione automatica del parziale.

Una cache ZIP corrotta o priva della ricevuta rimane indisponibile; I1 non
include un comando di ricostruzione/rimozione. Le guardie sono campionate:
la crescita WAL e il consumo di altri programmi non sono una quota kernel.
Il controllo finale impedisce `completed` quando la capacità risulta superata.
