# Addendum operativo r006 — baseline completa e audit fonte EbookLib

2026-10-06. Supervisore; ricezione prompt37. Stesso obiettivo del mandato36,
piano/arbitrato r003 invariati. Nessun GO finale né nuova pianificazione.

## Ricezione

Baseline-s006 conserva FAIL_INTERRUPTED_BY_AUTHOR_GUARD. R diagnostico e S
esistono, ma wrapper/V0 non sono completi; collection67 non è esecuzione della
suite. Errore ordinario del monitor riconosciuto, senza reinterpretare la receipt.
Package-s008 conserva lock FAIL/exit1, lock-check NOT_EXECUTED; managed e backend
indipendenti ricevuti come acquisizioni, non convalida del prodotto.

Il supervisore ricalcola gli input delle receipt,4772 record managed,364 backend,
330 payload wheel/cache/installato e1892 file/lock s007. Entrambi i freeze MATCH
all'ingresso; Git e codice invariati. Il checksum del download managed resta
**inferenza dal contratto uv pinned**, non hash indipendente del flusso originale.
La wheel setuptools84 ha archivio pubblico conservato/hash e payload verificati.
Le osservazioni parziali F1–F6 non sostituiscono V0. Non occorre reinstallare gli
input acquisiti o ricostruire i precedenti PASS.

## Un solo ledger per la prossima tranche

Per baseline-s007 e audit EbookLib vale **run intera + .venv-python**, inclusi
storico/cache/tmp/parziali/evidenze e blocchi directory, senza seguire symlink.
Max1GiB/stop896MiB, libero minimo1GiB. Ammetto74MiB incrementali dalla misura di
ricezione:64MiB prove baseline,2MiB audit,8MiB registri/snapshot/launcher/report.
Riserva esterna16MiB. Le allocazioni sono limiti, non consumo sostenuto.

Misurare logical/allocated di entrambi i root; usare il massimo per i gate e
contare i link come metadati senza seguirli. All'ingresso includere tutta la
quota74MiB più16MiB. Durante le operazioni sottrarre il consumo incrementale
osservato dalla riserva74MiB residua, senza contarla due volte; fermare per
incremento74MiB, cap specifico dell'attività o cumulativo896MiB con riserve.
Monitor periodico entro1s con tempi/gap reali, non quota atomica. Log1MiB/stream,
JSON8MiB,file32MiB; nessun cleanup/spostamento per passare il gate.

Questa disposizione **sostituisce prospetticamente il budget baseline500/384**
di r005/prompt37. Il main storico check_preserved non decide l'ammissione nuova:
usare soltanto verify per integrità s007 e il ledger qui definito. Anche il ramo
baseline=True del launcher corretto di prompt37 usa ancora il vecchio gate:
va adattato come strumento d'autore prima della nuova chiamata, senza toccare
helper/input congelati. Non trasferire limiti nuovi alle receipt storiche.

## Baseline e correzioni locali

Freeze baseline-s007 sul codice stabile corrente, target s007 in sola lettura,
nuovi output r002: R→S→V0 nel wrapper Firejail/D4, argv/env della nuova request.
S e receipt future restano fuori dal freeze. Prima/dopo verificare snapshot e
integrità target; nessuna installazione per riparare una collection.

Correggere sintassi/guardie/strumenti propri nella stessa chat. Se un errore
dimostrato del solo launcher interrompe la chiamata, raccogliere i figli e
preservare tutto: è ammessa una sola ripetizione locale con output esclusivi
r003, sostituendo soltanto i tre path di output r002 e la preflight derivata.
La scope congela la ricetta di sostituzione; registrare argv e hash della nuova
config prima dell'avvio. Codice, target, fixture, timeout, env e gate non cambiano.
Nuovi FAIL reali R/V0/integrità/cap non si ritentano con questa eccezione.
Un input congelato modificato richiede nuova label, senza nuovo piano per fix locali.

## Audit ammesso, backend ancora escluso

La fonte primaria PyPI conferma EbookLib0.18 source-only: archivio115484byte,
SHA25638562643a7bc94d9bf56e9930b4927e4e93b5d1d0917f697a6454db5a1c1a533.
Ammetto acquisizione pubblica HTTPS **solo del JSON versione e dell'archivio
identificato**, body cumulativo fino1MiB, storage audit fino2MiB,120s.
URL e input esatti nella scope. Nessun documento/credenziale o proxy ereditato.

Ispezione statica con stdlib: controllare nomi/tipi/dimensioni dell'archivio,
leggere soltanto file di metadata/build pertinenti, senza extractall, import,
exec/setup.py, hook/backend/installazione o risoluzione. Distinguere backend
dichiarato da fallback standard inferito, dipendenze statiche da quelle dinamiche
ignote. Conservare bytes/hash/fonte e formulare un comando/costo concreto per
il gate successivo. Non rimuovere no-build, cambiare pin o restringere il grafo
universale. L'audit può proseguire dopo baseline fallita se integrità/costi validi.

L'audit non necessita di un secondo freeze package preliminare: input/fonte e
ammissione sono nello stesso freeze; il suo risultato reale entrerà nel gate
package successivo. Questo non è tests-s001 o autorizzazione product build/sync.
V7/V8 restano obbligatorie con mandato pesante distinto; V10/V11 escluse.
Due review indipendenti, arbitrato finale e Git manuale restano necessari.
