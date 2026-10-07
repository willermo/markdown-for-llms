# Ciclo di implementazione supervisionato

Stato: procedura adottata con [ADR 0007](../decisions/0007-supervised-development-runs.md).
Vale per le run di sviluppo del progetto. Le normali domande o letture della repository
non aprono automaticamente una run. La supervisione è una chat distinta con memoria
persistente, non un servizio che continua a lavorare dopo la chiusura della chat.

## Organizzazione dei file

```text
temp/                              # esclusa da Git
  README.md                        # come recuperare il contesto
  PROJECT-CONTEXT.md                # contesto comune, decisioni e vincoli
  RUNS.md                          # indice delle run locali
  HANDOVER.md                      # prompt di ingresso alla run corrente
  run-a001-fase0-uv/
    STATE.md                       # fase, revisioni, branch, proprietari, prossimo passo
    HANDOVER.md                    # checkpoint compatto del supervisore
    brief.md                       # obiettivo, perimetro e criteri iniziali
    events.md                      # cronologia dei passaggi, append-only
    artifacts.md                   # indice degli artefatti e della loro versione
    prompts/                       # un prompt per ruolo, revisione e revisore
    plans/                         # plan-r001.md, plan-r002.md, ...
    reviews/                       # review-plan-r001-chatgpt.md, ...-claude.md
    arbitrations/                  # arbitration-plan-r001.md, ...-implementation-r001.md
    implementation/                # report-r001.md, report-r002.md, ...
    handovers/                     # checkpoint del singolo ruolo, aggiornabili
    snapshots/                     # impronte degli artefatti sottoposti a review
    evidence/                      # output dei controlli, comandi e risultati
    probes/                        # prove esplorative e script temporanei
    notes/                         # osservazioni accessorie
```

Le directory operative vengono inizializzate insieme alla run; i report vengono
creati soltanto quando il lavoro corrispondente è stato svolto. Non compilare
segnaposto con `GO`. I nomi dei revisori identificano chat distinte; nel report
registrare provider/modello effettivo e, se disponibile, riferimento alla chat.

I file globali contengono indicazioni valide per tutte le run; `STATE.md` contiene
lo stato corrente della singola run. Gli ADR sono la fonte delle decisioni stabili.
Un conflitto tra file locali e repository va segnalato e riconciliato dal supervisore,
senza perdere il contesto delle richieste successive dell'utente.

Per gli avanzamenti significativi aggiornare anche il [changelog permanente](../CHANGELOG.md):
fase/run, esito verificato, limiti e prossimo passo. Il supervisore vi registra gli
arbitrati e i cambi di stato; commit, integrazione e deploy vanno riportati solo dopo
verifica. Il changelog sintetizza la storia, mentre `STATE.md` conserva il dettaglio operativo.

## Ruoli e responsabilità

| Ruolo | Scrive | Responsabilità |
| --- | --- | --- |
| Analisi | `brief.md`, evidenze iniziali | Delimitare obiettivo, dipendenze, branch e criteri |
| Supervisore | Stato, handover comune, prompt, indice, eventi e arbitrati | Coordinare la run e decidere i passaggi |
| Pianificatore | Piano versionato e proprio checkpoint | Produrre il piano implementabile e verificabile |
| Revisore ChatGPT | Proprio report e proprie evidenze/checkpoint | Revisionare indipendentemente il medesimo oggetto |
| Revisore Claude | Proprio report e proprie evidenze/checkpoint | Seconda valutazione indipendente |
| Implementatore | Codice, report e proprio checkpoint | Eseguire il piano approvato e fornire prove |
| Utente | Operazioni Git di integrazione e rilascio | Eseguire commit, merge, push/promozione secondo i comandi concordati |

Solo il supervisore modifica lo stato condiviso. Gli altri ruoli consegnano i propri
artefatti e comunicano il percorso nel riepilogo in chat. Usare cartelle di evidenze
distinte per ruolo e revisione. Non avere due implementatori che scrivono nello stesso
working tree contemporaneamente. I revisori non correggono il codice oggetto di review.

ChatGPT e Claude ricevono lo stesso piano o snapshot del codice e gli stessi criteri;
prima di consegnare il loro report non leggono quello dell'altro né l'arbitrato corrente.
I fix approvati di un giro precedente sono invece parte della specifica condivisa.
L'autore del piano/codice non impersona i due revisori. Se una chat o un provider non
è accessibile, produrre il prompt e attendere il report reale: non simulare l'esito.
Sostituzioni di provider/modello si concordano e registrano; nessun modello futuro è
considerato automaticamente disponibile.

## Sequenza e criteri di passaggio

| Passaggio | Attività e output | Prossimo stato |
| --- | --- | --- |
| Preparazione | Recuperare contesto; creare feature da `dev`; registrare commit base e stato Git | `ANALYSIS` |
| Analisi | Scrivere brief, criteri e decisioni; handover alla chat supervisore | `PLANNING` |
| Pianificazione | Supervisore genera prompt; nuova chat scrive `plan-rNNN.md` | `PLAN_REVIEW` |
| Review piano | Due nuove chat producono report distinti sullo stesso piano | `PLAN_ARBITRATION` |
| Arbitrato piano | Supervisore valuta entrambi; correzioni locali come errata verificate, nuovo piano per cambi sostanziali | `GO` → `IMPLEMENTATION`; `NO_GO` → correzioni del piano o `PLANNING` motivata |
| Implementazione | Chat implementatrice completa il piano, corregge i problemi e ripete le prove autonomamente; consegna report, prove e riepilogo | `IMPLEMENTATION_REVIEW` |
| Review implementazione | Supervisore legge report e riepilogo, produce due prompt; due nuove chat scrivono le review | `IMPLEMENTATION_ARBITRATION` |
| Arbitrato implementazione | Supervisore decide e consegna insieme eventuale prompt di fix | `GO` → `READY_FOR_MANUAL_INTEGRATION`; `NO_GO` → `IMPLEMENTATION` per fix locali, `FIX_PLANNING` per cambi sostanziali |
| Fix locali | Implementatore applica rilievi accolti nella stessa run/chat, verifica e consegna report nuovo | Due nuove review mirate e arbitrato |
| Pianificazione fix sostanziali | Solo quando cambiano metodo, perimetro o assunzioni decisive; registra il motivo | Riesame del piano sulle parti invalidate |
| Integrazione manuale | Fornire all'utente comandi basati sullo stato Git reale; registrare risultati | `INTEGRATED` |
| Deploy e chiusura | Redeploy se pertinente e autorizzato, archivio verificato e pulizia selettiva | `CLOSED` |

`PREPARATION` è lo stato iniziale se il branch della run non esiste ancora o manca
una dipendenza. Un'interruzione non equivale a `NO_GO`: registrare cosa manca e il
prossimo passo, senza trasformare un'assenza di report in un giudizio tecnico.

## Review, arbitrato e versioni

### Mandati operativi e prove preliminari

Dal 2026-10-05, su adozione esplicita dell'utente e confronto con un processo
esterno, il mandato copre un risultato completo: codice, strumenti necessari,
correzioni, preparazione e prove pertinenti. Il supervisore fissa input protetti,
criteri, perimetro, directory, costi e arresti sostanziali; l'implementatore sceglie
le soluzioni ordinarie. L'autorizzazione già data copre anche i fix locali nella
stessa run. Non richiedere nuove conferme per scelte già decise o attività previste.

La chat implementatrice può continuare fra consegne e fix. Ogni prompt deve essere
comunque autosufficiente per una chat fresca. Si cambia chat per esigenze di
contesto o separazione dei ruoli, non perché termina un singolo comando. I reviewer
sono sempre due chat nuove e indipendenti a ogni giro. Nessuna percentuale rigida
della finestra di contesto né numero promesso di giri.

Errori di sintassi, generazione, attese, fixture, test, lint e propri strumenti si
correggono nella stessa chat. Conservare lo stato necessario a riprodurre il difetto
quando serve, non una copia integrale della storia a ogni modifica. Ripetere le
verifiche invalidate; un negativo previsto è PASS se rifiuta l'input alterato.
Un FAIL di sviluppo non è un NO_GO di arbitrato. Un'istruzione tecnicamente errata
può essere adattata dichiarando motivo e prova, senza ridurre requisiti o garanzie.

**Arresti sostanziali:** base/input protetti difformi o mancanti senza recupero
locale, assunzione decisiva smentita, scope/requisito nuovo, costo o download non
coperto, risorse esaurite, operazione irreversibile non autorizzata, impossibilità
di rispettare confinamento/privacy/integrità, rifiuto effettivo del sistema di
approvazione o pausa esplicita. Un errore di permessi del processo segue la
procedura [sandbox e approvazione](#restrizioni-del-sandbox-e-approvazione-degli-strumenti).
Bloccare la sola attività dipendente e continuare il lavoro indipendente coperto.
Un target di confronto è immutabile; un ambiente nuovo di sviluppo può essere
corretto con lo strumento standard quando il mandato lo ammette. Nessun retry
cieco, nessun PASS trasferito dal target fallito e nessun allentamento dei gate.

Le prove esplorative identificano input/comandi/output/exit e non sostituiscono
l'accettazione. Dal 2026-10-06, per richiesta esplicita dell'utente, l'implementatore
completa implementazione, fix e verifiche senza consegne intermedie per freeze.
Per S/B/I/E o altri vincoli di input stabili identifica autonomamente gli input
effettivi e crea gli eventuali snapshot di esecuzione prima delle prove.
Un errore ordinario invalida le prove dipendenti: correggere, identificare la nuova
versione con una label nuova quando necessaria e ripetere le verifiche nella stessa
chat. Non attendere il supervisore, un nuovo prompt o una review preventiva del fix.

Il supervisore riceve il risultato completo, crea lo snapshot comune per le due
review e prepara i loro prompt. Gli snapshot di esecuzione attribuiscono le prove;
non sono GO né sostituiscono le review. Non congelare output futuri o checkpoint
vivi aggiornabili come input della stessa prova. Gli snapshot precedenti restano
storici, non sovrascritti. Un contesto d'ingresso diventa storico dopo modifiche
autorizzate: registrare i delta, non pretendere MATCH del vecchio worktree contro
codice nuovo. Le prove del risultato consegnato devono riferirsi agli input finali
pertinenti; risultati antecedenti valgono solo con equivalenza verificata e motivata.

Una regola specifica di piano/stage resta vincolante finché un addendum versionato
ne dispone esplicitamente la variazione. Non reinterpretare retroattivamente le
ricevute. Per la run a001, l'addendum r004 raggruppa sviluppo e input del core;
conserva R/D4, S→B→I→E, criteri V0–V9 e mandati distinti per costi pesanti.
L'addendum r016 supera D1 e i vincoli successivi di attesa del freeze del
supervisore: gli snapshot di esecuzione sono delegati all'implementatore, quello
comune delle review e lo stato condiviso restano del supervisore.

### Delega per obiettivo e decisioni non previste

Dal 2026-10-06 l'autonomia non è una lista chiusa di eccezioni. Il mandato
approva un risultato con criteri e confini: l'implementatore decide anche mezzi
tecnici reversibili non anticipati dal supervisore. Prima di fermarsi verifica
se la scelta cambia un requisito, un limite di costo o un confine esplicito;
la sola assenza di un comando, nome di host o caso nel prompt non lo dimostra.
Questo criterio prevale sulle formule generiche «qualsiasi differenza richiede
nuova autorizzazione». Una limitazione specifica motivata resta vincolante;
per stage storici va superata una volta con addendum, non ignorata in silenzio.

Dipendenze previste, acquisizioni ammesse dalle fonti pubbliche già configurate,
endpoint metadata e CDN/redirect con catena di provenienza verificabile sono
mezzi del medesimo obiettivo. Un host di trasporto nuovo non equivale a cambiare
registry, pin o politica delle fonti. Verificare TLS/origine e non trasferire
credenziali o dati privati; un redirect non dimostrabile non è autorizzazione a
fidarsi di qualunque host. Se lo strumento non espone la catena, verificare con
lettura pubblica mirata oppure usare un metodo verificabile, senza dichiarare
osservazioni inesistenti. Restano costi e divieti di payload esplicitamente esclusi.

L'implementatore registra in una sezione del report le decisioni significative:
problema, scelta, alternative rilevanti, effetto sui requisiti, prove invalidate
e verifiche ripetute. Non creare una nuova cerimonia o un report per ogni fix.
I due reviewer controllano queste decisioni insieme al risultato; possono
richiedere correzioni. La supervisione arbitra i rilievi, non approva
preventivamente ogni soluzione tecnica. Nessun PASS di prove invalidate.

L'escalation riguarda cambio di obiettivo/requisiti/decisione architetturale,
nuovi costi o risorse oltre il limite autorizzato, privacy/confinamento non
rispettabili, input di confronto alterati, azioni irreversibili, rifiuto effettivo
del sistema di approvazione
o impossibilità dimostrata di soddisfare i criteri. Diagnosi e tentativi entro
perimetro non sono scostamenti sostanziali. Una contraddizione operativa può
essere risolta preservando lo scopo e documentando il motivo; non indebolire
un criterio di accettazione per rendere verde un risultato.

Snapshot e hash attribuiscono le prove agli input effettivi, non autorizzano
singoli comandi. Dopo una variazione l'implementatore identifica i nuovi input,
aggiorna gli eventuali snapshot di esecuzione e ripete le prove dipendenti,
distinguendo preliminari e ufficiali. Lo snapshot comune delle review e lo stato
condiviso restano del supervisore; due review indipendenti e Git manuale restano
obbligatori. L'autonomia sulle prove non autorizza modifiche agli arbitrati.

### Restrizioni del sandbox e approvazione degli strumenti

Dal 2026-10-06 distinguere il fallimento di una syscall o di un comando
(EPERM/EACCES, socket locale o namespace negato) dal rifiuto della richiesta
di approvazione. Il primo dimostra una restrizione nel contesto corrente;
non dimostra che l'azione già autorizzata sia impossibile in ogni contesto
ammesso. Conservare errore e comando, diagnosticare quanto basta a identificare
il vincolo e usare il meccanismo di escalation dello strumento, se disponibile
e consentito dalle istruzioni della sessione. Nessun passaggio al supervisore
per il solo tentativo di ottenere tale approvazione entro il mandato.

La richiesta deve identificare comando, dati sintetici, directory, costi e
controlli che rimangono attivi. Nell'ambiente Codex che espone questa opzione,
usare `exec_command` con `sandbox_permissions="require_escalated"` e una
giustificazione concreta; decide il sistema di approvazione configurato.
Un prompt di repository non concede permessi al sistema né garantisce il
successo dell'escalation. Non estendere automaticamente rete, filesystem,
budget o operazioni previste dal mandato.

Quando l'isolamento della prova è fornito da R/Firejail, la richiesta può
riguardare il launcher esterno che crea il socket sintetico e avvia il runner;
R, namespace senza egress, controlli daemon e medesima invocazione dei figli
restano obbligatori prima del workload. Non rimuovere i gate per fare riuscire
il comando, non modificare sysctl/AppArmor, socket operativi, setuid o profili
persistenti e non usare sudo come sostituto dell'approvazione dello strumento.

Se l'approvazione viene negata, conservarne il motivo e non aggirarla con
altri strumenti o varianti equivalenti. Continuare il lavoro indipendente
coperto e riferire l'azione bloccata e la decisione effettiva. Se l'escalation
non è disponibile o il recupero richiede nuovi costi/privilegi/confini,
segnalare quel limite concreto. I vecchi esiti IMPEDITA restano storici;
un tentativo approvato richiede nuove ricevute, non un PASS retroattivo.

### Parametri operativi e correzioni dopo un freeze

Dal 2026-10-06 distinguere tre categorie in ogni mandato:

- **Input di confronto protetti:** baseline, sorgenti di riferimento, prove e
  ricevute precedenti. Non modificarli né sovrascriverli.
- **Input che determinano la prova:** codice, helper effettivamente eseguito,
  backend, lock, interprete, configurazione e fixture. Identificare la versione
  consumata; modifiche autorizzate invalidano le sole prove dipendenti. Correggere
  e verificare nella stessa chat; identificare i nuovi input e creare autonomamente
  l'eventuale snapshot di esecuzione, senza una richiesta di freeze al supervisore.
- **Parametri operativi adattabili:** deadline entro il tetto ammesso e i limiti
  del tool, livello di log senza sopprimere gli errori, suffissi esclusivi degli
  output nelle directory ammesse, launcher e
  reader propri. L'implementatore li corregge e riesegue senza nuovo mandato o
  snapshot se gli input della prova e i suoi requisiti restano invariati.

I mandati indicano intervalli e vincoli per i parametri adattabili; un valore
esempio non diventa automaticamente un requisito. Per delegare una deadline
considerare anche il limite del helper e il tempo delle guardie: una stima non è
un cap e una correzione non deve aumentare costi o indebolire il confinamento.
L'implementatore verifica compatibilità fra CLI del tool, parametri scelti e budget
prima della chiamata; il supervisore non riceve una proposta per ogni invocazione.

L'immutabilità riguarda i byte degli artefatti consegnati e l'attribuzione delle
prove. Un comando corretto si registra in una nuova ricevuta; non si riscrive
l'argv storico per farlo risultare eseguito. Conservare errore, diagnosi, delta,
identità e risultato del nuovo tentativo. Un errore CLI prima del workload può
essere corretto e seguito dall'esecuzione; un timeout reale richiede prima di
accertare stato dei parziali, fine dei processi propri, ripetibilità e budget.
Non ripetere alla cieca il medesimo errore; non introdurre un limite universale
di un solo tentativo che trasformi ogni difetto nuovo in una consegna.

Un parametro resta protetto soltanto se il mandato ne spiega il motivo concreto
(per esempio un limite di risorse o una configurazione consumata). La formula
«argv immutabile» da sola non basta a vietare le correzioni operative delegate.
Per stage preesistenti con vincoli incompatibili, il supervisore dispone una
variazione esplicita una volta; non rigenera l'autorizzazione per ogni adattamento.
Nella run a001 l'addendum r010 supera i vincoli di argv del mandato41 nei punti
operativi indicati, conservando la policy sorgenti e gli arresti sostanziali.

Non creare stage di sola preparazione per correggere questi parametri. Si torna
al supervisore per i soli arresti sostanziali sopra elencati, per una consegna
completa o per una pausa esplicita dell'utente. Un freeze tecnico necessario alla
prova successiva si esegue autonomamente. Restano due review indipendenti del
risultato e Git manuale dell'utente.

### Prove e strumenti proporzionati

Usare gli strumenti standard del progetto per ambiente, lock, build e installazione.
Uno strumento riutilizzabile va nel repository con istruzioni, input espliciti e
verifiche adeguate; un output specifico resta nelle evidenze della run. Non costruire
un installer, resolver, monitor o comparatore alternativo senza necessità dimostrata.
La prova conserva comando/argv, cwd, ambiente pubblico pertinente, output, exit,
versioni e identità degli input consumati. Un riepilogo in chat non è evidenza.

Controllare il delta reale rispetto alla base, compresi file nuovi non ignorati:
con commit manuali, il solo `git diff base...HEAD` può essere vuoto pur con lavoro
esistente. Usare diff del worktree, inventario dei nuovi file e snapshot comune.
Git non trasferisce file ignorati né lavoro non committato: temp/ non è un backup.

Prima di una prova controllare sorgenti/build input/fixture/lock/interprete e
config effettivamente consumati. Le identità di moduli installati/archivi sono
necessarie quando si afferma S/B/I/E. Non ricalcolare per ogni giro l'intero albero
dell'interprete o tutti gli artefatti antecedenti: rifare solo i controlli invalidati
o richiesti da un'anomalia concreta. Impedire scritture spurie con scratch della
run, cache/config esplicite e -B sui nuovi processi Python -I; il solo environment
PYTHONDONTWRITEBYTECODE non basta con -I. Non cancellare parziali per far passare gate.

Suite e lint proporzionati al cambiamento e agli strumenti configurati. Per soli
documenti: link/coerenza e diff-check. Per codice: test dei percorsi coinvolti e
regressioni pertinenti; ampliare quando l'impatto lo richiede. La suite legacy
selezionata non è discovery completo; fallimenti ereditati vanno confrontati,
non occultati. Benchmark/API/modelli reali e hardware sono prove distinte, con
mandato e costi espliciti. Non dichiarare PASS una verifica essenziale IMPEDITA.

La baseline di confronto si acquisisce una volta e si conserva su percorsi stabili
della run. Se la variabilità conta, misurarla con esecuzioni ripetute identificate;
non normalizzare contenuto, formule, numeri, ordine o asset per ottenere equivalenza.
Una normalizzazione richiede regola motivata, limiti e verifica del contenuto.

### Review, triage e uscita

Il supervisore scrive prompt con oggetto e criteri tecnici comuni ai due reviewer;
sono ammesse soltanto differenze di identità e file di output. Primo giro sul delta
completo, fix su rilievi aperti, dipendenze toccate e regressioni pertinenti, con
riferimento anche alla base iniziale. Non riaprire ogni volta tutta la roadmap.

Per ogni rilievo verificare: (1) il percorso corrente può produrlo su input ammessi?
(2) quale impatto o requisito viola? (3) è introdotto dalla run, ereditato o causato
dal mandato? Citare codice/prova e motivare gate, backlog o rigetto. I sintetici
sono validi per casi di fedeltà, confini, errori e sicurezza: assenza nel corpus
osservato non dimostra irraggiungibilità. Rare vulnerabilità o violazioni del
contratto non si scartano per frequenza; un difetto preesistente può bloccare un
requisito esplicito della run. Migliorie fuori scope vanno nel debito tecnico o
nelle questioni aperte, non in cicli di fix illimitati.

L'arbitro legge entrambi i report e verifica le premesse dei rilievi sul codice,
non decide a maggioranza. Una correzione locale del piano diventa errata verificata;
nuovo piano solo per cambi decisivi. NO_GO del codice con fix locali produce nella
stessa consegna il prompt eseguibile di fix, ereditando il perimetro autorizzato.
Dopo i fix: due review nuove mirate, senza nuova doppia review del piano.

Uscita: entrambi i report validi sullo stesso oggetto, nessun blocco valido aperto,
criteri essenziali dimostrati e limiti espliciti. Il numero di giri e l'assenza di
rilievi osservati non sostituiscono criteri mancanti. Se ricorre la stessa classe
di difetto, riesaminare contratto, produttori e prova richiesta prima di aggiungere
altro apparato; non indebolire il requisito di fedeltà per chiudere il ciclo.

STATE è il registro breve autorevole; HANDOVER indica prossima azione e letture;
eventi append-only conservano la storia. documentation/ registra decisioni,
traguardi, risultati e debito tecnico; docs/ resta la documentazione d'uso. Aggiornare
changelog e indici per avanzamenti significativi, non per ogni comando. Conservare
report di giro separati e riferimenti, senza incorporare ogni volta i report
antecedenti integralmente. Nessun vault esterno, cambio di nomi obbligato o copia
delle memorie personali del progetto sorgente.

### Esiti delle review

Ogni review dichiara `GO` o `NO_GO`, oggetto, revisione, snapshot, prove, limiti e
rilievi identificati (per esempio `GPT-P001`, `CLA-I002`). I rilievi riportano severità,
posizione, impatto, evidenza e criterio per considerarli risolti. Distinguere un
blocco di correttezza o requisito da un suggerimento opzionale.

L'arbitro richiede entrambi i report validi, verifica che riguardino lo stesso oggetto
e decide su ogni rilievo: accolto, respinto con evidenza, risolto con verifica oppure
rinviato come non bloccante con motivazione. Un `NO_GO` del revisore può derivare da
un rilievo infondato: l'arbitro può respingerlo spiegandolo. Non può ignorare un blocco
valido o una verifica essenziale assente per ottenere un `GO`.

Il `GO` è valido solo per il piano e lo stato del codice esaminati. Conservare revisioni
precedenti e scrivere report nuovi per ogni ciclo. Modifiche sostanziali dopo un `GO`
richiedono una nuova revisione e il relativo arbitrato; anche i cambiamenti di base
dovuti a un merge vanno valutati prima di promuovere il risultato. Per semplici
spostamenti di evidenze in archivio verificare hash e collegamenti, senza attribuirli
artificiosamente a una nuova modifica del codice.

Il GO sul piano autorizza le modifiche previste dal piano stesso: quelle modifiche
non richiedono di riapprovare il piano a ogni file, ma devono passare la review
dell'implementazione. Dopo il GO sul codice, un commit o cambio di branch eseguito
dall'utente modifica HEAD/branch: se contenuti e artefatti restano identici, registrare
la corrispondenza con il nuovo commit senza rifare le review. Se cambiano contenuti
o base effettiva, il supervisore ne valuta l'impatto e riapre le verifiche necessarie.

## Snapshot e strumenti

Lo strumento usa soltanto la libreria standard Python: è disponibile anche prima
della migrazione a uv. Dalla radice del repository:

```bash
python3 scripts/run_context.py bootstrap
python3 scripts/run_context.py init run-a001-fase0-uv --phase 0.1 \
  --title "Migrazione Python a uv" --branch feature/run-a001-uv --base dev
```

`init` crea solo i file locali, registra il branch corrente e quello previsto e non
crea/switcha branch, non committa e non chiama modelli. Rifiuta run già esistenti;
`bootstrap` non sovrascrive file presenti. Lo stato rimane `PREPARATION` finché il
supervisore non ha verificato la preparazione. Il nome della cartella è una convenzione;
il campo fase in `STATE.md` è il riferimento alla roadmap.

Prima di distribuire i prompt di review, congelare il piano o il codice e creare uno
snapshot comune come supervisore. Durante l'implementazione lo stesso strumento
può produrre snapshot di esecuzione a cura dell'implementatore: label nuove,
input reali, verifica prima/dopo e nessuna modifica dello stato condiviso. Non
richiedono una consegna o un'approvazione del supervisore. Esempio da eseguire
solo quando il piano indicato esiste:

```bash
python3 scripts/run_context.py snapshot run-a001-fase0-uv --label plan-r001 \
  --artifact temp/run-a001-fase0-uv/plans/plan-r001.md
python3 scripts/run_context.py verify run-a001-fase0-uv --label plan-r001
```

Lo snapshot contiene branch, HEAD, hash di file tracciati e nuovi non ignorati,
eliminazioni e link simbolici, più gli artefatti esplicitamente indicati. `temp/` e
`documentation/runs/` sono esclusi dall'impronta del working tree perché ospitano
contesto e archivio dei report; gli artefatti della run da revisionare sono inclusi
esplicitamente con `--artifact`, ripetibile per piano, report o evidenze essenziali.
L'esclusione dell'archivio non consente di nascondervi codice eseguibile del prodotto.

Durante l'integrazione, `verify` resta rigoroso anche su HEAD/branch: un risultato
STALE limitato a quei campi richiede il confronto dell'impronta dei contenuti e la
registrazione del passaggio Git, non equivale automaticamente a una regressione.

Per l'implementazione includere anche piano approvato, arbitrato e report finale.
I revisori controllano lo snapshot prima e dopo la review. `verify` segnala variazioni,
ma non giudica la qualità né produce un `GO`; lo snapshot non è un backup del codice.
Gli hash identificano le versioni, non certificano autore o veridicità dei report.

## Recupero in una chat fresca

Aggiornare checkpoint a ogni passaggio di ruolo, decisione sostanziale e prima di
interrompere un lavoro lungo; non attendere che la chat abbia già perso contesto.
Non assumere di conoscere una percentuale esatta di saturazione della finestra.
Mantenere il checkpoint breve e rimandare alle evidenze dettagliate.

Il supervisore mantiene `temp/HANDOVER.md` come prompt copiabile e il file della run
come riepilogo corrente. Ogni altro ruolo mantiene `handovers/<ruolo>.md`, indicando
prompt di origine, output scritto, risultati, comandi ancora in corso e prossimo passo.
La nuova chat legge: `AGENTS.md`, handover globale, `STATE.md`, handover del ruolo,
brief e soli artefatti pertinenti. Verifica subito branch, HEAD e modifiche locali.

Se non ha accesso ai file, si trasferisce la cartella della run con il contesto globale
necessario. Per cambiare macchina, trasferire anche il lavoro non committato e gli
eventuali file nuovi, verificandoli contro lo snapshot. Non usare `git stash -u` o
`git clean` come backup implicito: `temp/` è ignorata. Includere sempre nel riepilogo
in chat i percorsi dei report e il prossimo passo richiesto.

## Archivio e pulizia

| Materiale da preservare | Destinazione permanente |
| --- | --- |
| Piani finali, review, arbitrati, sintesi e manifest | `documentation/runs/<run-id>/reports/` e `manifest.md` |
| Evidenze e note utili, prive di segreti/documenti privati | `documentation/runs/<run-id>/evidence/` e `notes/` |
| Probe riutilizzabile a fini diagnostici | `scripts/diagnostics/<run-id>/` con istruzioni e dipendenze |
| Regressione automatizzabile | `tests/regression/<run-id>/` o suite pertinente, con provenienza delle fixture |
| Decisione architetturale | ADR in `documentation/decisions/`, collegato alla run |

Prima della chiusura selezionare ciò che serve, copiare/promuovere, adattare riferimenti,
verificare integrità e comandi delle prove promosse. Aggiornare manifest e indice
permanente con file di origine, destinazione, hash, motivo di conservazione e limiti.
Istruzioni documentali per utenti e sviluppatori confluiscono in `docs/` secondo Diátaxis.

Il supervisore elenca esattamente gli elementi eliminabili e li pulisce soltanto
dopo aver verificato l'archivio e l'assenza di riferimenti necessari. La richiesta
dell'utente autorizza questa pulizia di fine run; se proprietà o utilità di un file
sono dubbie, conservarlo e segnalarlo. Non eliminare l'intera `temp/`, altre run attive,
il registro globale o l'unico esemplare di un report. Aggiornare l'handover affinché
punti all'archivio o alla run successiva. Lo strumento non implementa cancellazioni.

A fine sprint applicare la stessa selezione e pulizia alle run chiuse, preservando
il contesto delle run ancora attive e le dipendenze tra run.

## Template

- [Stato della run](templates/run-state.md)
- [Piano](templates/plan.md)
- [Review indipendente](templates/review.md)
- [Arbitrato](templates/arbitration.md)
- [Report implementazione](templates/implementation-report.md)
- [Handover](templates/handover.md)
- [Prompt per ruolo](templates/role-prompt.md)
- [Manifest di archivio](templates/archive-manifest.md)
