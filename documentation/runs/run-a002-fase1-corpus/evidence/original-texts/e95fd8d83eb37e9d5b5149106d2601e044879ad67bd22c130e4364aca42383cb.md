# Fix locale nella stessa chat implementatrice — confronto umano utilizzabile

Agisci esclusivamente come **implementatore della run a002**, continuando la
chat attuale. Il lavoro indipendente dichiarato è concluso, A03 è aperto.
L'utente ha respinto la verifica a215righe/blocchi come non eseguibile; accetta
come modalità possibile un confronto tra originale e Markdown. Il supervisore
cambia il mandato umano con addendum esplicito; non è un errore dell'utente.

## Input e autorità

Leggi AGENTS, temp/HANDOVER, STATE/brief della run e il nuovo checkpoint
handovers/supervisor-v-workflow-r001.md; skill manage-implementation-run e
verify-conversion-fidelity, protocollo. Leggi integralmente
arbitrations/addendum-v-human-r001.md, che prevale sul vecchio workflow V di
E02/prompt05. Il restante mandato05, piano+errata e arbitrato piano r002 restano
vigenti. Recupera report/checkpoint propri, corpus rev3, M005/controls003,
packet006 e receipt del budget attuale senza ricaricare tutta a001.

Feature/run-a002-corpus, HEAD/dev80c31da46e5caab990d3596c79c44e524ec1dc95 osservati
dal supervisore, indice vuoto. Il worktree contiene ormai codice/test/fixture/
guide/report a002, non soltanto documenti ereditati: verifica il delta effettivo.
Snapshot tecnico d'ingresso `snapshots/v-workflow-context-r001.json`, hash e
transizione in `evidence/supervisor-v-workflow-r001/context-transition.json`:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label v-workflow-context-r001
```

Contesto d'ingresso, non prova del codice. Correggi nel perimetro, identifica
nuovi input/stage/snapshot autonomamente e ripeti prove dipendenti come nel05.
Non attendere un freeze/mandato per ciascun helper. Non modificare registri
comuni, prompt/arbitrati del supervisore, report review o packet storici.

## Risultato richiesto

Prepara nuovo packet in una directory esclusiva, massimo13coppie documentali:
10documenti corpus e3controlli. Primo collegamento utile **C02**, così l'utente
vede immediatamente il confronto su cui si è bloccato. Una pagina iniziale
semplice e una vista per documento, utilizzabile senza istruzioni tecniche:

- **A sinistra: Originale**, documento effettivo o sue pagine renderizzate in
  ordine, leggibili con zoom/apertura immagine. Nessuna confusione fra originale
  e asset generatore. Per EPUB capitoli actual identificati; conserva provenienza
  e limiti renderer R senza sanarli. Per TXT/TeX/controlli source leggibile e
  asset reali, TeX non compilato chiaramente dichiarato.
- **A destra: Markdown di riferimento**, un documento completo leggibile come
  documento: titolo/paragrafi, numeri, formule in notazione matematica visibile,
  tabella in celle vere, figure e didascalie, note/bibliografia/link e ordine.
  Affianca il file.md scaricabile per chi vuole aprirlo nel proprio viewer.
  Il riferimento deriva da gold/specifiche indipendenti, non da un output
  Pandoc ritenuto corretto. Non usare observations/scorer per completare il gold.
- La frase di scopo: «Confronta l'originale con il Markdown di riferimento e
  segnala ciò che manca o cambia». Spiegazione breve per differenze lecite
  di impaginazione/colonne; preservazione del contenuto e ordine richiesta.
  Niente blocchi b01, AST, locator, code point, hash o controlli M nell'interfaccia
  principale. Questi restano allegati tecnici per strumenti/reviewer.
- Nessuna sezione “trascrizione attesa” vuota per formule/tabelle; tutti i
  contenuti attesi resi leggibili. Non sostituire una formula con la grafia
  LaTeX da interpretare o l'immagine indistinta che nasconde una perdita.
  Usa mezzi locali disponibili/reversibili, niente installazioni/pesi/CDN/JS.
  Un renderer del riferimento è solo mezzo di presentazione, non fonte del
  gold o prova semantica. M convalida codice/sorgenti indipendentemente.

La vista a destra **non è l'output del motore valutato**: chiamala riferimento,
non risultato Pandoc. Gli output e FAIL dei motori sono conservati e rendicontati
separatamente; L04 resta non assegnata. Il feedback dell'utente qualifica gold/
canonical, non autorizza a dichiarare fedeli le32conversioni reali.

## Risposte umane e stato

Accetta un giudizio per documento in linguaggio naturale, per esempio:
«C02: corrisponde», «C02: manca la didascalia sotto la tabella»,
«C02: non riesco a confrontare la formula». Nessun TSV a215righe né timestamp
per riga richiesto all'utente. Una risposta testuale per i documenti già visti
è sufficiente; altri restano PENDING. Se offri scelte visive usa tre etichette
semplici “Corrisponde”, “Ho trovato differenze”, “Non riesco a valutarlo”;
nota eventuale e risposta in chat sempre disponibili. Interfaccia passiva,
nessuna rete/fetch/esecuzione di contenuti importati.

Aggiorna importer/verification e receipt a **granularità documentale dichiarata**:
risposta originale immutata/hash, packet/coppia hash e revisione, autore/data
ricezione reali con limiti, ambito completo della vista esaminata, testo della
risposta, decisione e eventuali differenze. Non inventare firme per le singole
proprietà, non convertire un giudizio in215ACCEPTED artificiali. Una nota vaga
non equivale a completo ACCEPTED: chiedi solo chiarimento localizzato necessario,
senza imporre un nuovo schema all'utente. Un documento respinto non invalida i
giudizi ricevuti sugli altri; A03 richiede tutti i documenti dovuti verificati.

M rimane indipendente e required PASS; V documentale esplicita/completa richiesta
per VERIFIED. Integrità/hash differente→STALE, assenza risposta→PENDING,
differenze→REJECTED o AMBIGUOUS secondo risposta reale. La vista gold deve
coprire **tutto** il contenuto e le proprietà visibili applicabili del documento,
non un estratto comodo che esclude formule/celle/numeri/note/assets. Annotazioni
tecniche/meccaniche hanno receipt proprie, non sono nascostamente firmate dall'utente.

Non chiedere ora una nuova designazione del verificatore o di confermare singoli
comandi. L'utente non deve più lavorare su packet005/006. Prepara il nuovo
confronto C02 e presentane il link primario pronto, insieme al resto del packet
e a una breve stima del lavoro umano.260min resta accordo storico del vecchio
workflow, nessun tempo già svolto attribuito né nuova durata accettata inventata.
La verifica umana vera avverrà quando il materiale è usabile; intanto completa
fix/presentazione/M/test. Non consegnare un mockup o solo un piano del viewer.

## Discrepanza da correggere e controlli

Prima di richiedere V risolvi C02/b04: HTMLpacket006 dichiara caption
“Celle misurate”; la scritta non compare in word/document.xml del DOCX copiato
nel packet né nei render esaminati dal supervisore. Diagnostica la specifica
manuale e correggi generatore/sorgente o annotazione erronea con motivo/revisione;
nessuna taratura su un output Pandoc e nessun taglio della copertura richiesta.
Non chiedere all'utente di certificare un gold già noto incoerente.

Test discriminante M sul canonico: caption attesa assente deve essere FAIL
anche se tutte le celle sono corrette. Controlla gli altri campi caption/figure/
note/target che il nuovo riferimento presenta, senza assumere che M291PASS
garantisca proprietà mai confrontate. Convalida completa dei contenuti del
gold su input effettivo, non validatore che ricalcola lo stesso dato del viewer.

Prove mirate del nuovo workflow: riferimento non perde blocchi/formule/celle/
figure/notes/ordine; positivo completo e omissione intenzionale falliscono
appropriatamente nei controlli di completezza; escaping/risorse locali sicure;
giudizi documentali veri, incomplete/rejected/ambiguous/stale e integrità
risposta; un documento problematico non dà firme agli altri né cancella quelle
valide. Vecchie API e format receipt versionati/coerenti, niente all-green.

Usa stesso managed/wheel/S-B-I/preflight/inventory esterno/R secondo05 per i
nuovi input. Rete/isolamento/capR invariati, errori ordinari e strumenti propri
correggibili nella stessa chat. Nessuna nuova campagna a001 o modifica dei criteriR.
Se cambia solo packet/import non ripetere tutte32conversioni; se cambia gold,
riscore solo dipendente con nuova identità; se cambia inputC02, ripeti i percorsi
C02 (diretto/legacy) e generazioni/render/M pertinenti. Prove anteriori restano
attribuite ai loro input, non rinominate come PASS del nuovo stato.

## Risorse e consegna

Stessa quota unica52hwall/21600CPU-s/6GiB con riserva3h/1800CPU-s/1GiB e stop
49h/19800CPU-s/5GiB; entry reale2026-10-08T07:06:54.369068Z e ledger conservati.
Nessun reset/cap aggiuntivo o nuova acquisizione/servizio/privilegio persistente.
Costi nuovi aggiunti, attese wall distinte da tempo umano attivo non misurato.

Scrivi note di fix/versione, manifest del nuovo packet e prove in nuove directory
sotto evidence/implementation-r001, preservando packet006 e campagne/report
storici. Aggiorna report/checkpoint propri indicando nuovo contratto documentale,
discrepanza/fix/attribuzione prove, risposta umana pendente oppure davvero ricevuta.
Una volta pronto mostra in chat **prima il collegamento C02 affiancato**, poi
il solo indice degli altri documenti e una frase su come rispondere. Non
riproporre un manuale tecnico. Nessun completamento A03 senza risposta reale.

Alla consegna completa supervisore riceve, congela snapshot comune e prepara
due nuove review ChatGPT/Claude. Questo fix non apre nuova pianificazione o
review simulata. Niente modifiche STATE/HANDOVER/changelog del supervisore,
Git manuale utente, niente fase2. Il mandato05 più questo fix coprono risultato
intero e correzioni, non una catena di autorizzazioni per ogni comando.
