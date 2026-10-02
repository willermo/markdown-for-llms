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
| Arbitrato piano | Supervisore valuta entrambi e ogni rilievo | `GO` → `IMPLEMENTATION`; `NO_GO` → `PLANNING` con nuova revisione |
| Implementazione | Nuova chat esegue il piano approvato, salva report, prove e riepilogo | `IMPLEMENTATION_REVIEW` |
| Review implementazione | Supervisore legge report e riepilogo, produce due prompt; due nuove chat scrivono le review | `IMPLEMENTATION_ARBITRATION` |
| Arbitrato implementazione | Supervisore decide sui rilievi e sulle verifiche mancanti | `GO` → `READY_FOR_MANUAL_INTEGRATION`; `NO_GO` → `FIX_PLANNING` |
| Pianificazione fix | Nuova chat riceve prompt con arbitrato e rilievi; prepara un piano di fix versionato | Ripete pianificazione, doppia review, implementazione e arbitrato |
| Integrazione manuale | Fornire all'utente comandi basati sullo stato Git reale; registrare risultati | `INTEGRATED` |
| Deploy e chiusura | Redeploy se pertinente e autorizzato, archivio verificato e pulizia selettiva | `CLOSED` |

`PREPARATION` è lo stato iniziale se il branch della run non esiste ancora o manca
una dipendenza. Un'interruzione non equivale a `NO_GO`: registrare cosa manca e il
prossimo passo, senza trasformare un'assenza di report in un giudizio tecnico.

## Review, arbitrato e versioni

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
snapshot. Esempio da eseguire solo quando il piano indicato esiste:

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
