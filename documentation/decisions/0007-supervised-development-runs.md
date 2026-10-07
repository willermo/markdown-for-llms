# ADR 0007 — Run supervisionate e handover persistenti

- Data: 2026-10-02
- Stato decisione: accettata
- Stato implementazione: Protocollo iniziale integrato a66ba822; ciclo continuo richiesto dall'utente adottato nel working tree il 2026-10-06: implementazione, fix e prove senza freeze intermedi del supervisore; snapshot comune alla consegna per le due review; efficacia da verificare
- Origine: paradigma di sviluppo richiesto esplicitamente dall'utente
- Integra: ADR 0005 sulla governance del repository

## Decisione

Organizzare ogni run da un feature branch di `dev` con analisi, supervisione,
pianificazione, due review indipendenti del piano, arbitrato, implementazione,
due review indipendenti dell'implementazione e arbitrato finale. Gli esiti sono
`GO` o `NO_GO`. Dal 2026-10-05 i fix locali proseguono nella stessa run con prompt
di fix e due review mirate; la pianificazione si riapre per cambi sostanziali.
Il supervisore produce prompt e rende esplicite le decisioni sui rilievi.

Usare chat distinte e revisori ChatGPT + Claude. Un eventuale altro provider/modello
richiede disponibilità verificata e sostituzione registrata; non assumere disponibile
un modello annunciato. Il protocollo non avvia automaticamente altre chat e non
autorizza un singolo agente a simulare i due revisori indipendenti.

Conservare contesto globale in `temp/` e contesto per run in una sottocartella dedicata.
Versionare skill, protocollo, template e strumento di inizializzazione. Conservare
evidenze utili in posizioni permanenti prima della pulizia selettiva. Commit e merge,
eventuale push e promozione sono eseguiti dall'utente con comandi forniti dall'agente.

## Miglioramenti operativi adottati

Legare review e arbitrati a revisioni e impronte degli artefatti, mantenere report
precedenti, assegnare un solo proprietario allo stato condiviso e aggiornare checkpoint
a ogni passaggio. Una review mancante o riferita a una versione precedente non è un
`GO`. L'arbitrato motiva l'accoglimento o il rigetto di ciascun rilievo, senza votazioni
a maggioranza o accettazione basata sulla sola autorità del modello.

## Conseguenze e verifica

### Implementazione continua e freeze per le review — 2026-10-06

L'utente ribadisce il ciclo: supervisore, piano, due review del piano, arbitrato,
implementazione completa con correzioni autonome, due review del risultato,
arbitrato e chiusura o fix. Report-r014 della run a001 ha conservato cinque PASS
S/B/installazione prima del FAIL I su uv_cache.json; reader corretto e tredici
controlli locali dichiarati PASS, ma consegna ancora in attesa del freeze s018.
L'arresto rispettava il mandato49: la dipendenza dal supervisore per ogni cambio
agli input delle prove contraddiceva la delega per risultato.

Il [protocollo](../development/run-lifecycle.md#mandati-operativi-e-prove-preliminari)
ora delega all'implementatore anche gli snapshot di esecuzione richiesti dai tool.
Una correzione identifica nuovi input e ripete le prove dipendenti nella stessa
chat. Il supervisore congela il risultato consegnato per le due review; mantiene
stato condiviso, arbitrati e prompt. Errori e decisioni si valutano nelle review,
senza un passaggio preventivo per ciascun fix. Questa regola supera le precedenti
prescrizioni operative di attesa del freeze prima delle prove riportate sotto,
che restano cronologia delle decisioni, non istruzioni correnti.

R016 e prompt50 applicano il cambio alla run a001 senza riscrivere piano, review,
report o ricevute storiche. Conservati criteri tecnici, attribuzione S/B/I/E,
isolamento, costi già autorizzati e Git manuale. Il perimetro del mandato torna
al risultato del piano; un limite reale di risorse o autorizzazione blocca solo
le attività dipendenti. Adozione documentale verificata, risultato operativo
e GO finale ancora aperti; nessuna integrazione dichiarata.

### Delega dei parametri operativi, 2026-10-06

La consegna del mandato41 si è arrestata prima del workload: il supervisore aveva
prescritto timeout900 a un wrapper con limite120, congelando anche l'argv. La
delega generica delle correzioni era contraddetta dal vincolo specifico; arrestarsi
era conforme al mandato ricevuto. Il difetto è nell'autorizzazione operativa.

Il [protocollo](../development/run-lifecycle.md#parametri-operativi-e-correzioni-dopo-un-freeze)
ora distingue input di confronto, input della prova e parametri adattabili.
Deadline compatibili con tool e budget, output esclusivi e strumenti propri si
correggono nella stessa chat; la ricevuta registra il comando reale senza mutare
gli artefatti storici. Cambi agli input invalidano le prove dipendenti e vengono
raggruppati per il freeze ufficiale quando necessario. Nessun passaggio di sola
preparazione per questi fix, nessun aumento implicito di budget o privilegio.

R010 applica la distinzione alla ripresa lock/check della run a001. I controlli
documentali verificano coerenza e compatibilità della chiamata corretta; non
attestano ancora esito del lock né miglioramento misurato dei tempi. Ruoli,
snapshot significativi, due review, memoria documentation/temp e Git manuale
restano quelli adottati.

### Adattamento dal processo esterno, 2026-10-05

L'utente autorizza prendere ciò che è utile dall'handover Claude, adattandolo senza
stravolgere l'impianto. La [valutazione](../development/process-adaptation-2026-10-05.md)
registra provenienza, adozioni ed esclusioni. Manteniamo nomi/struttura temp/, skill
esistente, pianificatore/supervisore separati, due review di piano e codice, snapshot
comuni del supervisore, S/B/I/E quando richiesti, Git manuale e archivio permanente.

Adottiamo mandati per risultato, correzioni locali, errata del piano, NO_GO con
prompt di fix nella stessa consegna, review successive mirate e verifiche del delta.
Lo stato non cresce come trascrizione; documentazione e tool hanno un proprietario.
Non importiamo vault, commit automatici, working tree necessariamente pulito,
eliminazione totale degli hash, suite/lint indiscriminati, normalizzazioni per
far passare equivalenza o esclusione dei sintetici. Restano gate i requisiti
mancanti e i rischi concreti anche se non introdotti dalla run.

La modifica è operativa, non una nuova scelta del prodotto o autorizzazione a
modelli/servizi/spese. Il piano r003 resta approvato, con addendum r004 per il solo
mandato corrente; vecchie prove rimangono legate agli input originali. Non si
riesegue l'installer legacy riuscito né la storia per adottare il processo.
Misurare passaggi di ruolo, tempo al risultato e blocchi motivati nelle prossime
consegne; la validazione documentale non dimostra maggiore velocità o affidabilità.

### Evoluzione operativa adottata il 2026-10-05

Su richiesta esplicita dell'utente, adottare mandati per obiettivi verificabili e
consentire preparazione, correzioni ordinarie e prove preliminari delimitate nella
stessa chat. Congelare gli input pronti prima delle prove ufficiali. I risultati
preliminari non valgono come accettazione; versioni, errori e parziali restano
identificati, con numero di tentativi e costi cumulativi limitati dal mandato.

La granularità precedente rendeva ogni difetto degli strumenti ausiliari un ciclo
di passaggi di ruolo. Mantenere solo preparazione statica non ha intercettato due
problemi reali del recupero baseline. Eliminare i freeze o trasferire PASS fra
versioni comprometterebbe invece l'attribuzione delle prove. La soluzione adottata
anticipa le verifiche operative circoscritte e conserva il freeze per l'accettazione.

Restano responsabilità separate, doppie review indipendenti, arbitrati e Git
manuale. Le politiche di sicurezza/budget richiedono disposizione esplicita; i
mandati non possono estenderle autonomamente. Un FAIL preliminare o operativo
non costituisce un NO_GO di arbitrato e non riapre automaticamente il piano.

Il pilota è il recupero offline della baseline della run a001: directories/venv/seed
sono riusciti preliminarmente. Il 2026-10-05
la ricezione di un SyntaxError nella preparazione c001 ha evidenziato un mandato
troppo restrittivo: «qualsiasi FAIL» aveva imposto una consegna anche per un difetto
ordinario, contrariamente alla semplificazione adottata. Precisata la distinzione
fra correzioni locali di preparazione e arresti operativi nel protocollo e nel
nuovo mandato; input protetti e limiti restano invariati. Il pilota non attesta
ancora maggiore velocità o affidabilità comparativa. Dopo la correzione locale,
c001 è stata completata e ricevuta con install/check/inventory PASS preliminari;
s007 ufficiale è stata ricevuta con cinque operazioni PASS e venv preservata.
Il mandato successivo accorpa preparazione baseline e sonde R preliminari, con
correzioni ordinarie locali, fino a un solo traguardo di consegna. Verificare
tracciabilità, arresti e rifiuto di prove obsolete oltre a tempi, consumo e passaggi.
La [procedura dei mandati](../development/run-lifecycle.md#mandati-operativi-e-prove-preliminari)
contiene le regole; l'addendum locale conserva l'arbitrato r003 originale.

Il ciclo aumenta il lavoro di coordinamento ma consente revisioni indipendenti e
recupero tra chat. Si dimensionano report e prove alla run mantenendo i passaggi
richiesti. `temp/` ignorata da Git non costituisce backup: per un'altra macchina o
una chat senza filesystem condiviso serve un trasferimento esplicito dei file.

Il bootstrap di questa governance avviene sul feature branch già aperto; non vengono
inventate review retroattive. La prima run del nuovo ciclo sarà la migrazione uv,
dopo l'integrazione manuale del bootstrap in `dev` e la creazione del suo feature branch.

Procedura: [ciclo delle run](../development/run-lifecycle.md).

### Delega tecnica per obiettivo — 2026-10-06

Dopo ulteriori arresti ordinari nella run a001, l'utente chiede autonomia reale
per l'implementatore e valutazione delle scelte tecniche in review. Adottato nel
working tree: i mezzi reversibili entro requisiti e confini sono delegati anche
quando non enumerati; fonti già configurate e redirect verificati sono coperte
dall'acquisizione pertinente. Il report raccoglie decisioni e verifiche, le due
review le valutano e il supervisore arbitra. Costi, privacy, confinamento e
irreversibilità restano confini preventivi. I freeze identificano le prove, non
sono autorizzazioni per ogni comando. Protocollo e template contengono la regola;
non si dichiara già misurato un miglioramento dei tempi né GO sul prodotto.

### Restrizioni del processo e approvazione degli strumenti — 2026-10-06

Mandato48 della run a001: bind del socket sintetico R/D4 negato con EPERM,
prima di Firejail/S. L'autore ha rispettato la precedente clausola generica
di arresto; nessuna richiesta al sistema di approvazione era stata effettuata.
Precisato il [protocollo](../development/run-lifecycle.md#restrizioni-del-sandbox-e-approvazione-degli-strumenti):
diagnosi e richiesta di escalation per un'azione già autorizzata rientrano nel
mandato, mantenendo i gate della prova; il rifiuto effettivo o un nuovo confine
restano arresti. R015 applica la distinzione senza nuovi privilegi persistenti,
costi o riduzione R/D4. Regola adottata nel worktree, recupero operativo ancora
da eseguire: nessun successo dell'approvazione o miglioramento misurato dichiarati.
