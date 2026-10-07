# Implementazione r001 — completare lock/check con autonomia operativa

Agisci esclusivamente come implementatore nel repository
`/home/davide/workarea/markdown-for-llms`. Puoi continuare nella stessa chat.
**AUTORIZZATO A IMPLEMENTARE nel perimetro già approvato.** Completa lock e check
offline nella copia esistente. Non aprire un altro giro di sola preparazione;
correggi gli errori ordinari e prosegui fino al risultato o a un blocco sostanziale.

## Autorità e letture pertinenti

Leggi AGENTS, skill manage-implementation-run e protocollo aggiornato, STATE,
HANDOVER e checkpoint `handovers/implementation-r001.md`. Recupera brief A1–A7,
indici architettura/decisioni/roadmap, piano r003 integrale e arbitrato D1–D5 se
non già letti. Non ricaricare tutta la storia. Percorsi seguenti relativi a
`temp/run-a001-fase0-uv/`:

- `implementation/report-r007.md` e consegna `evidence/implementation-r001/resume-package-s011/delivery.md`;
- `arbitrations/addendum-operational-protocol-r010.md`, disposizione corrente;
- `implementation/stages/impl-r001-stage-package-s012/request.json`;
- `evidence/supervisor-implementation-r001/impl-r001-stage-package-s012/authorized-scope.json`,
  reception.json, transition.json, checks.json e freeze-verify.json;
- `handovers/supervisor-operational-autonomy-r001.md`.

R010 supera esplicitamente i vincoli operativi incompatibili di41/r009/s011.
Le policy sostanziali sono conservate nella scope s012; non usare vecchie formule
«argv immutabile» o «qualsiasi FAIL ⇒ consegna» per fermare un fix ordinario.
La richiesta dell’autore resta storica: la patch alternativa è stata ricevuta,
il budget nuovo proposto NON è stato ammesso. Nessun GO finale.

Piano SHA256 `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`;
arbitrato `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`.
S012 manifest126455byte/SHA
`1ef1a9a3d0e36107b1f4ef9d1a3452b4475f10fab7fd26495b4849eec146f900`,
worktree `1c829980984ba9420ce8dccb05cd7add975bf116af875c6b608f706f63401414`;
123file/375artefatti. RequestSHA
`3a87c1c27fcef0e235dc45c7ee3b3eba7539f7a7c4fae81590904402735b7cf9`;
scopeSHA `7f3be026277271b89f93f0c26f4efe7e3c20c1f33d41dafdd729e3b912d64b49`.

Verifica hash pertinenti e da radice prima delle attività e alla consegna:
`python3 -B scripts/run_context.py verify run-a001-fase0-uv --label impl-r001-stage-package-s012`.
Git feature/run-a001-uv,HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto. S011 è storico per i sei delta dichiarati in transition.json;
i suoi360artefatti intatti. Nessun falso MATCH del vecchio helper dopo il fix.

## Esecuzione e correzioni delegate

Il wrapper tracciato è già corretto e congelato, SHA
`860d3b6a2c9f54f1b3b39f0bfb58564882bf22593789a97a0845399f366e6c83`:
CLI ammette fino a900s, default120 invariato. Il precedente rifiuto CLI/exit2
non ha eseguito R/uv/lock e non è un timeout reale; log e receipt restano intatti.

Usa i due template strutturati **scope.operations**, cwd/env espliciti,
shell=False/close_fds=True; niente merge con ambiente ereditato. Ogni operazione
ha una chiamata distinta; raccogli una sessione viva senza rilanciarla. I target
s011 restano input validi; lo snapshot consumato ora è s012. Directory work/tmp/
config esistenti si verificano e si riusano, non si ricreano come assenti.
Nuove evidenze/output in `evidence/implementation-r001/resume-package-s012/`;
non creare le directory output del wrapper prima della chiamata.

1. Lock offline: deadline900s, esterna1080s, R/Firejail net=none/D4 → uv nel
   medesimo namespace; receipt reali, nessun hook/build/install, audit lock.
2. Check offline: solo dopo lock exit0 e audit coerente; deadline120s,
   esterna300s. Stessa policy, SHA lock invariato.

**Adatta senza tornare al supervisore**: deadline compatibili entro900s per lock
ed entro120s per check, tempi esterni deadline+180, suffissi esclusivi degli
output nelle evidenze s012 e launcher/reader propri non congelati. I template
identificano il punto di partenza; salva l’argv reale, delta e motivazione.
Conserva il FAIL precedente; correzione diagnosticata e nuovo tentativo sono
ammessi, nessun retry cieco. Tempi workload cumulativi900s lock/120s check;
un errore CLI prima del workload non li consuma. Per timeout reale verifica
fine dei processi propri, parziali, gate e ripetibilità prima di riprovare.

Non riscrivere scope/request/receipt congelate. Le variazioni delegate dei
parametri non richiedono un nuovo freeze. Se occorre correggere un input della
prova, fallo e verifica nel perimetro nella stessa chat; consegna insieme fix
completo e input pronti se serve nuovo binding ufficiale. Non trasferire PASS
alla versione cambiata né dichiarare MATCH/R ufficiale contro un helper diverso.
Non creare snapshot da implementatore. Adatta i vecchi reader allo schema e
alla label correnti: un reader proprio difettoso va corretto, non consegnato.

## Vincoli sostanziali ereditati

Native argv/env, pin/fonti/metadata/grafo, baseline e target protetti restano
quelli della scope. Gatecache114nomi e source/build/Git vuoti prima/dopo,
EbookLib0.18 metadata osservati, divieti degli altri package, zero rete operativa/
backend/build/install/acquisizione. Startup bootstrap/managed e a1_coverage.pth,
origine/binari/config chiusa e integrità baseline restano da verificare secondo
scope. Nessun nuovo probe applicativo, import native o modifica al root progetto.

Budget cumulativo s011 invariato: Hentry525762560byte,16MiB totali già aperti
(8attività/8registri),16MiB esterni,pool1GiB/stop896MiB/libero1GiB. Ledger run+
.venv-python e formula/caps in scope.budget; nessun azzeramento dei costi già
sostenuti, nuovo budget, cleanup o scrittura fuori ledger. Monitor periodico non
quota atomica. Nessun nuovo costo remoto. Non ripetere baseline o build EbookLib.

Blocca soltanto l’attività dipendente per input protetto alterato, policy nuova,
namespace/privacy/integrità non rispettabili, budget esaurito o rifiuto sandbox;
conserva ragione/azione e continua lavoro indipendente ammesso. Un semplice
errore di argomento, sintassi o reader non è uno di questi arresti.

## Consegna completa

Report nuovo `implementation/report-r008.md`, checkpoint implementatore con
copia d’ingresso, `implementation/stages/impl-r001-stage-package-s012/completion-r001.json`
legata a request/scope/manifest, `evidence/implementation-r001/resume-package-s012/delivery.md`.
Esiti separati R/wrapper/lock/check, actual argv/env/cwd/tempi/sessioni/exit,
receipt/log, startup/gatecache/namespace/integrità/budget, audit e hash del lock.
Non modificare report/receipt s011 o registri condivisi.

Se riesce, consegna insieme richiesta concreta per promozione degli input/lock
e seguito S/B/I/E con comandi/costi/prove; non eseguirlo implicitamente. Se un
blocco è sostanziale, consegna causa verificata e soluzione concreta. Esiti
storici baseline/suite62pass5fail, s009 byteFAIL e lacune rimangono tali. Review
ChatGPT/Claude e arbitrato finale futuri; niente GO finale/commit/merge/push/
deploy/cleanup. Git manuale dell’utente. Stato finale WAITING_FOR_SUPERVISOR_RECEPTION;
prossimo supervisore con prompt05+r010 e checkpoint corrente, nessun servizio
automatico da attendere. Trasferire file reali ignorati insieme al lavoro.
