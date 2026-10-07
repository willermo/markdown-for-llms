# R015 — restrizione di syscall, approvazione e ripresa prodotto

2026-10-06, supervisore. L'utente chiede direttive e prompt effettivi dopo
l'arresto48. Non è una review del codice o un GO finale. Piano/arbitrato r003,
r013/r014, requisiti e costi rimangono invariati.

## Ricezione e responsabilità

Report-r013/delivery reale: S tentativo1 exit1, wrapper IMPEDITA per bind
AF_UNIX/EPERM prima di Firejail. Diagnosi minima conferma; altri21 NOT_RUN.
Nessun S/B/I/E prodotto, runner nuovo o processo residuo. Nessuna richiesta
di approvazione effettuata e nessun rigetto auto-review osservato.

Ricezione propria in `evidence/supervisor-implementation-r001/sandbox-recovery-reception-r001/`:
1027record s016 ricalcolati prima dei nuovi delta documentali; unico mismatch
checkpoint autore, con copia frozen identica verificata. Inventory805file e
bindings delivery ricalcolati; moduli/metadata/lock/backend/diagnostici invariati.
L'autore ha seguito la clausola generica di arresto di48/protocollo: la
responsabilità della formulazione insufficiente è del mandato supervisore.
EPERM prova una restrizione del contesto corrente, non l'impossibilità di
ottenere tramite gli strumenti un contesto autorizzato per la medesima azione.

## Disposizione operativa

Supero esplicitamente le formule «rifiuto sandbox» come STOP automatico dopo
una syscall o un comando negati, in48 e nei mandati/addenda precedenti per
questa ripresa. Vale il protocollo aggiornato: diagnosi già disponibile,
richiesta dell'escalation prevista dallo strumento se disponibile, entro il
mandato, poi esecuzione soltanto se approvata. Non serve un altro intervento
supervisore prima di presentare quella richiesta. La sessione e il sistema di
approvazione conservano autorità sui permessi effettivi.

Per Codex che espone l'opzione, usare exec_command con
`sandbox_permissions="require_escalated"` e giustificazione specifica dopo
avere preparato comando/launcher reviewabile. L'azione è il launcher esterno
di una prova già ricevuta che crea il solo socket sintetico e avvia
Firejail/R, poi S/B/I/E soltanto dopo PASS R/D4. Sono ammesse le connect/close
diagnostiche ai canali già inventariati, senza richieste daemon. Nessun
workload di conversione/test può passare fuori da R; nessun socket operativo,
sysctl/AppArmor/rete host/setuid/profilo persistente modificato, niente sudo.
Usare solo l'escalation ufficiale, non strumenti alternativi per aggirare
la restrizione o una decisione negativa. Non ampliare rete/download/privilegi
del workload o budget. Un'approvazione non è un PASS R o prodotto.

Al rifiuto effettivo conservare azione/motivo, non aggirare la decisione;
proseguire il solo lavoro indipendente coperto. Se l'escalation non è
disponibile, registrare la limitazione esplicita. Nessun retry cieco o
richiesta equivalente ripetuta per ottenere una decisione diversa.

## Freeze senza nuova preparazione autore

S016 rimane immutato e storico per checkpoint/delta di governance dichiarati.
Non è legittimo rieseguire il launcher hardcoded s016 sul worktree nuovo.
Il supervisore riceve ora gli stessi input reali da request-r012, con il
report-r013 e la presente variazione; non chiede un'altra sola preparazione.
Dispongo un nuovo freeze **product-s017** prima del primo S riuscito.
La lista effettiva è in freeze-disposition.json della ricezione propria,
distinta dalla vecchia request autore e non presentata come una nuova request
scritta dall'implementatore. Tutti gli input sono già presenti; niente output
futuri o codice operativo ancora da scrivere.

Rimuovo dalla lista il checkpoint autore vivo `handovers/implementation-r001.md`:
si congelano invece la copia originale s016 dell'autore e una copia immutabile
del checkpoint-r013 ricevuto dal supervisore. I digest delle copie identificano
l'ingresso; l'autore può aggiornare il suo checkpoint alla consegna senza
rendere STALE il freeze. I registri comuni vivi restano esclusi. Gli altri
artefatti originali e la request-r012 rimangono nella lista come provenienza;
R015/disposition identificano label, budget e metodo di ripresa effettivi.

Adattare in una versione operativa separata il dispatcher hardcoded: snapshot
usato nei check e nel wrapper =s017, argomento --snapshot del produttore S =s017.
Consentiti adapter/in-memory override dei riferimenti, senza riscrivere il
launcher/template s016 né i diagnostici ufficiali. Valida prima i bytes del
template storico, poi registra la configurazione effettiva s017/argv/env/cwd
con identità dell'adapter. Il semplice cambiamento di label/output e contabilità
del driver è operativo: nessun ulteriore freeze per quel solo adattamento.
La semantica delle guardie e l'attribuzione degli input restano quelle originali.
Modifiche a un input della prova si correggono autonomamente e si raggruppano
per il freeze ufficiale necessario, senza nuovo piano per difetti ordinari.

## Costi, seguito e uscita

Tutti i22ID della request/sequence restano ricevuti: S→buildoffline attestata
pre-startup/import/get_requires/rawhook84→B→root/base-a/base-b runtime+canonica
I/E→esterno exporthashlocked/venv/runtime/canonica/pip-check/I/E.
S ricomincia con tentativo esclusivo>=2; gli altri ID non erano stati eseguiti.
Rootvenv/S/dist/base-a/base-b/esterno ancora assenti alla ricezione.

Pool112MiB/Hentry553541632 e ledger run/.venv-python/.venv/tmpesterno invariati,
residuo storage live nelle verifiche. Nessun reset/cleanup/residuo32 sommato.
Workload7200s, già **111.10300820460543s**, residuo **7088.896991795395s**;
delivery-r013 contiene il driver comprensivo del launcher annidato e i costi
aggiuntivi. Non aggiungere di nuovo3.1342911049723625s del launcher.
Il dispatcher s016 non legge quei costi esterni: il nuovo driver deve applicare
il cumulativo autorevole all'esterno, sottrarre nuovi intervalli non sovrapposti
e imporre deadline reali. Non fidarsi del remaining interno sovrastimato.
Singolo workload<=900s nel residuo, outer+180, monitor0,5s/file32MiB/JSON8MiB/
stream1MiB, repository>=1GiB e /tmp>=128MiB, paid0; nessuna quota atomica.

Consegna report-r014 e checkpoint autore dopo esecuzione completa oppure
blocco effettivo documentato del sistema di approvazione/confini. Non una
nuova proposta astratta per chiedere il permesso dell'escalation. Prove V1,
V3–V9/import/CLI/fasi/dev/API/comparativi e negativi restano aperte; V7/V8
pesanti distinte, V10/V11 escluse. Due review reali/arbitrato/GO finale futuri,
FAIL storici preservati, niente Git/deploy/cleanup automatici o agenti.
