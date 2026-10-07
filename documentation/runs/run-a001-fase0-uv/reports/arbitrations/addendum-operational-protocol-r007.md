# Addendum operativo r007 — baseline ricevuta e backend EbookLib circoscritto

2026-10-06. Supervisore; ricezione mandato38. Piano/arbitrato r003 immutati,
stesso obiettivo36. Nessun GO finale o esecuzione backend del supervisore.

## Esiti ricevuti

R/S/wrapper e V0 **PASS caratterizzazione degli originali** su baseline-s007.
Suite realmente eseguita **FAIL/exit1**,67 nodeid/201 eventi,62callPASS/5callFAIL,
setup/teardown67PASS ciascuno, zero skip/error. Receipt, binding/S.id, namespace,
output/inventari e215 file referenziati verificati dal supervisore, senza riesecuzione.
Perdite originali conservate. Questa baseline serve al confronto futuro: nessun
PASS trasferito al prodotto migrato, nessuna nuova baseline per soli metadata.

Audit source-only EbookLib0.18 ricevuto:115484byte/hash pubblico,66membri,
271811byte dichiarati e copie di metadata verificate. Setup importa setuptools.setup,
legge README e dichiara lxml/six; non importa runtime, non dichiara estensioni o
setup_requires. Fallback setuptools.build_meta:__legacy__ confermato nel sorgente
uv0.10.10, ma **selezione/esecuzione effettiva ancora non osservata**.
Versione interna0.18.1 non cambia quella distribuzione/setup0.18.

## Ammissione nuova

Accolgo la proposta concreta di due build native **solo EbookLib0.18**, isolate,
offline/no-index, con setuptools84 dalla sola wheel pubblica già acquisita e
build-constraints.txt invariato. Cache/tmp/output a e b distinti sotto
work/ebooklib-native-r001. R→uv nello stesso namespace Firejail/D4 prima di
ciascuna; target nuovi predisposti dal supervisore nel freeze package-s009.
Il resolver interno uv per le sole dipendenze di build locali/pinned è ammesso;
nessun resolver runtime/prodotto. Nessun giro intermedio di sola preparazione.
Zero download, installazioni runtime,
import EbookLib, build/sync del prodotto o test/Docker in questa tranche.

La richiesta d'autore resta non autorizzata alla sua consegna; nuova request/scope
del supervisore identificano i comandi ammessi. `--verbose` rende visibile la
selezione backend. Un launcher stdlib inline -I -B usa execve per passare l'ambiente
nativo chiuso esatto, poiché il wrapper R ammette16 chiavi e il comando uv necessita
anche UV_PYTHON_INSTALL_DIR. I valori rilevanti HOME/PATH/TMPDIR/cache/config e
controlli privacy sono identici; tre variabili uv aggiuntive non cambiano namespace,
UID, startup del bootstrap o daemon. Non modificare il wrapper per accettarle.

L'isolamento di build resta quello nativo uv: creazione venv temporanea e
installazione **solo backend setuptools84** ammesse, mai dipendenze runtime.
I figli degli hook uv pinned usano Python -c, senza -I/-B: lo attesta il sorgente
del frontend. Ambiente chiuso PYTHONDONTWRITEBYTECODE=1/PYTHONNOUSERSITE=1 rimane
effettivo per questi figli; non dichiararli -I -B. I lanci Python controllati
dall'agente/runner conservano -I -B. Startup nativo della venv (_virtualenv e
setuptools pinned) è accettato entro questa scope; nessuno shim dell'interprete,
patch backend o hook sostitutivo per imporre flag artificiali.

Conservare native trace/receipt e inventariare cache/output effettivamente rimasti.
La normale rimozione dei temporanei **da parte di uv stesso** è lifecycle nativo
ammesso, distinto dal cleanup dell'agente vietato. Se il ritorno raw di un hook
non sopravvive, dichiararlo non osservato; non inventarlo né costruire un altro
backend per conservarlo. Wheel/METADATA/RECORD/fonti e log restano prove reali.
Un requisito dinamico non disponibile nella sola wheel locale resta FAIL,
senza acquisizione/riparazione o allentamento di constraint.

## Risorse, risultati e confini

Ledger run+.venv-python invariato: max1GiB/stop896MiB, directory/link metadata
senza seguire link; libero minimo1GiB. Nuova tranche32MiB incrementali dalla
ricezione (24 build/cache/temp,8 registri/strumenti) più16MiB esterni, non riuso
della precedente quota74MiB. Headroom e quota residua contati una sola volta;
log1MiB/stream,JSON8MiB,file32MiB, child120s/outer300s per chiamata, monitor≤1s.
Nessuna quota atomica o picco istantaneo inventato. Nessun cleanup per passare.

Confrontare due wheel e payload/RECORD contro fonte: outer hash e timestamp
possono differire, non normalizzare per dichiarare riproducibilità byte.
PASS vale **solo backend/wheel della dipendenza**, non B/I/E del progetto.
Fix dei launcher ordinari locali; input congelati difformi richiedono nuova label.
Nessun retry cieco su target/parziali di un comando iniziato.

Il lock resta FAIL storico, lock-check NOT_EXECUTED. Questa ammissione non rimuove
`--no-build` dal resolver e non cambia pin/indici/grafo universale. Due build da
file locale non dimostrano cache metadata del registry: il seguito deve identificare
una via nativa verificabile per il registry originale, senza forgiare cache/metadata,
promuovere una find-links locale a fonte prodotto o abilitare backend sconosciuti.
Consentita nella stessa chat l'analisi read-only di sorgenti/help/cache per formulare
quel gate concreto, senza avviare resolver, altri backend o nuove acquisizioni.

V7/V8 restano obbligatorie con mandato pesante distinto; V10/V11 escluse.
S/B/I/E, prove del codice migrato, due review indipendenti e arbitrato finale
restano aperti. Git manuale dell'utente, nessun commit/merge/push/deploy.
