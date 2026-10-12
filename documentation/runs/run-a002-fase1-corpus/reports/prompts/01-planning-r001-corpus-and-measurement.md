# Pianificazione r001 — run a002, corpus e misurazione della fedeltà

Agisci esclusivamente come **pianificatore** della run `run-a002-fase1-corpus`.
Il supervisore ti incarica di produrre un piano completo, implementabile e
verificabile per il primo incremento della fase 1. L'utente ha autorizzato la
preparazione e il ciclo supervisionato; questo prompt **non autorizza ancora
l'implementazione**. Usa la skill `manage-implementation-run` e il protocollo
del repository. Non impersonare supervisore, implementatore o reviewer;
le due review saranno svolte in nuove chat indipendenti ChatGPT e Claude.

## Ingresso e stato verificato

Leggi nell'ordine, senza caricare tutta la storia a001:

1. `AGENTS.md`, `temp/HANDOVER.md` e la sola sezione corrente iniziale di
   `temp/PROJECT-CONTEXT.md` (i blocchi successivi sono storia).
2. `temp/run-a002-fase1-corpus/STATE.md`, `brief.md`, `HANDOVER.md`, questo prompt
   e `handovers/supervisor-planning-r001.md` nella medesima run.
3. `.agents/skills/manage-implementation-run/SKILL.md`,
   `documentation/development/run-lifecycle.md` e
   `documentation/development/templates/plan.md`.
4. `documentation/development/next-run-a002-corpus.md`,
   `documentation/README.md`, `documentation/decisions/README.md`, fase 1 della
   `documentation/roadmap.md` e `documentation/open-questions.md`.
5. `documentation/decisions/0001-document-fidelity.md`,
   `documentation/decisions/0004-execution-and-configuration.md`,
   `.agents/skills/verify-conversion-fidelity/SKILL.md` e
   `.agents/workflows/converter-benchmark.md` per i criteri di contenuto.
6. `docs/how-to/ambiente-uv.md`, `docs/reference/toolchain-legacy.md` e soltanto
   il codice/configurazione pertinente per identificare percorsi nativi,
   generazione delle fixture, dipendenze e harness effettivamente disponibili.
7. Per la base ereditata: `documentation/runs/run-a001-fase0-uv/manifest.md`,
   `evidence/integration-push-2026-10-07.json`,
   `notes/cleanup-assessment-2026-10-07.md` e
   `notes/cleanup-completion-2026-10-07.md` sotto lo stesso archivio.

Verifica branch, HEAD, `dev`, indice e delta locali, inclusi nuovi file non ignorati.
Alla consegna di questo prompt, branch attivo `dev`, HEAD/dev e riferimento locale
`origin/dev` sono `80c31da46e5caab990d3596c79c44e524ec1dc95`;
`feature/run-a002-corpus` non esiste. La receipt storica verifica il remoto a
quello SHA; questa supervisione non ha effettuato una nuova query di rete Git.
Se il contesto cambia, registra il delta e la base effettiva; niente reset o
snapshot ricreati per nascondere differenze.

Il comando manuale previsto per l'utente è:

```bash
git switch -c feature/run-a002-corpus dev
```

La pianificazione può procedere su `dev`; prima dell'implementazione serve la
feature corretta. Non eseguire operazioni Git di scrittura. I delta documentali
locali di preparazione/handover/pulizia sono attesi e possono accompagnare la
feature: non imporre un nuovo commit di chiusura a001.

A001 è integrata/pubblicata, GO r003 dopo GO/GO, con limiti e quote temporali
NON PASS conservati. Questi risultati non attestano codice nuovo, inferenza o
hardware diversi. Non riconvalidare a001 per il solo cambio di chat.
A002 resta PREPARATION finché manca il branch; il prompt di pianificazione è
pronto. Nessun piano approvato, review a002, benchmark o scelta motori esiste.

Le receipt reali della pulizia sono state ricevute: 193 file rimossi, keeper
verificati e tre tag Docker r004–r006 rimossi; r007 conservato. Non richiedere di
ripetere la pulizia né usare vecchi path rimossi come input ancora disponibili.
La mappa permanente indica le copie conservate. Nessuna pulizia generale è
necessaria alla pianificazione.

## Risultato da pianificare

Consegnare **corpus sintetico redistribuibile, riferimenti controllabili
indipendentemente, protocollo e runner di misura, prove negative e primo
esperimento reale sui percorsi nativi leggeri**. Il piano copre il risultato
completo e i fix ordinari nella stessa run; non una serie di mandati per sonda.
Prepara il confronto successivo Docling/Marker/remoto senza scegliere un
vincitore, concludere l'intera fase 1 o introdurre il dominio di fase 2.

Il risultato principale resta il documento Markdown completo con asset;
chunking/RAG sono opzionali e non sostituiscono la fedeltà. Definisci contratti
minimi del benchmark senza anticipare FastAPI, worker, DB, UI o schema interno
definitivo. La nuova applicazione è progettata, non implementata.

## Corpus e riferimenti: decisioni richieste

Proponi il numero minimo sufficiente di documenti e pagine, combinando fenomeni
senza perderne la verificabilità. Fornisci una tabella con ID proposto, formato,
lingua, pagine/sezioni, fenomeni, sorgente/generatore, gold, asset, percorso di
conversione previsto e stato della prova (reale prevista oppure non misurata).
Copri TXT, DOCX, EPUB, RTF, progetto LaTeX con inclusioni/asset, PDF testuale a
una e due colonne, scansioni IT/EN e pagine miste. Comprendi testo/Unicode,
numeri sensibili, formule inline/multilinea/matrici, celle significative, figura
e didascalia, note, bibliografia e riferimenti interni/esterni. Un caso deve
esporre perdite di pagina oltre alle perdite di blocco.

Per ogni classe indica una fixture e un riferimento oppure una lacuna
circostanziata proposta al supervisore e alle review. Non omettere silenziosamente
PDF/OCR perché non verranno convertiti in questa run: costruzione del corpus e
misurazione dei motori sono coperture distinte. Spiega come produrre e ispezionare
PDF/scansioni con strumenti leggeri disponibili, senza scaricare modelli o
installare una distribuzione TeX pesante implicitamente.

Definisci manifest e provenienza: ID stabili, hash, formato/classe, lingua,
origine/licenza esplicita anche dei sintetici, pagine, asset, generatori/versioni,
parametri e percorsi dei riferimenti. Richiedi inventario offline e sorgenti
leggibili, generazione deterministica o distinzione motivata fra identità byte
e contenuto riproducibile (ad esempio timestamp negli archivi). Non includere
dati privati o materiali con diritti incerti.

Il gold contiene trascrizioni e annotazioni controllabili sul sorgente:
blocchi/ordine, testo, numeri, formule/simboli, celle, asset e collegamenti.
Progetta una verifica separata dei riferimenti sul documento effettivamente
renderizzato, compreso overflow/troncamento/layout delle pagine. Dichiarare
un generatore deterministico non dimostra da solo che il gold corrisponda al PDF.
Condividere una specifica sintetica leggibile è ammissibile, ma il percorso di
controllo del gold non deve dipendere dal converter o dal comparatore valutati.
Gli output dei motori non diventano gold. Descrivi responsabilità, checklist
concreta e prove della verifica umana, senza inventare una verifica già avvenuta.

## Misurazione e runner: decisioni richieste

Per ciascun fenomeno specifica unità, denominatore, allineamento, metrica,
normalizzazioni ammesse, casi ambigui, risultato atteso e soglia motivata.
Misura testo/Unicode e omissioni, numeri e segni, ordine dei blocchi, formule e
simboli, celle con coordinate, asset/didascalie e riferimenti/target.
Distinguere confronto tipografico, sintattico e semantico delle formule:
renderizzabilità e stringhe normalizzate non provano equivalenza semantica.
Non usare una media unica per autorizzare perdita di pagine, numeri o formule.

Spiega come ricavare osservazioni dall'output reale senza iniettare i valori
attesi dal gold: blocchi mancanti, duplicati, contenuti inattesi e allineamenti
ambigui devono emergere. Definisci anche il positivo del misuratore, così un
runner che rifiuta tutto non possa superare i negativi. Per metriche non
applicabili/non disponibili usa uno stato esplicito distinto da PASS e da zero;
non escluderle silenziosamente dal denominatore.

Definisci identità di esperimento/documento/tentativo e motore/configurazione,
hash degli input e output, versioni/hardware, comandi effettivi, output grezzo
immutato, osservazioni derivate e errori. Non affidarti al solo filename;
includi omonimia, riesecuzione e risultati parziali nelle verifiche pertinenti.
Il runner deve distinguere esito operativo della conversione, qualità misurata,
supporto non misurato e stato del batch; non dichiarare completo un batch con
un errore, output vuoto o artefatti richiesti mancanti. Definisci exit/status e
aggregati per documento/fenomeno con fallimenti visibili.

Dichiara separatamente normalizzazioni conservative per metrica e conservazione
dei byte originali. Niente regex globali che alterino codice, LaTeX, numeri o URL
per far passare il confronto. Indica protezioni proporzionate per asset/link,
archivi, HTML e inclusioni LaTeX sintetici: path confinati, niente esecuzione di
comandi del documento o acquisizioni di risorse remote implicite.

Proponi prove negative con matrice mutazione → rilevatore → esito atteso:
pagina/blocco rimosso, numero/segno alterato, simbolo di formula cambiato,
ordine invertito, cella mancante, asset mancante, riferimento rotto, output vuoto,
batch con un errore. Modifica copie dedicate, non gold o baseline protetti;
preserva le evidenze. Le mutazioni devono verificare il contenuto e il contratto
del runner, non soltanto che venga scritto un JSON.

## Primo esperimento reale e compatibilità

Ispeziona il codice e le dipendenze effettive prima di proporre percorsi
nativi/Pandoc/parser leggeri. Specifica quali formati saranno eseguiti davvero,
come evitare chiamate ML/cloud/fallback e come verificare il Markdown completo
con asset. Almeno un percorso nativo reale completo deve essere ripetibile.
Se un formato richiesto resta non misurato, motiva la lacuna e la prova futura;
non dichiarare OCR, Docling o Marker misurati con mock. Non correggere
indiscriminatamente il prodotto legacy per ottenere un benchmark verde:
perdite osservate sono risultati validi da preservare. Un fallimento del motore
non equivale automaticamente a fallimento del runner o della consegna.

Specifica versioni, configurazione, hardware osservato, misurazione di tempo e
memoria (inclusi figli, unità, metodo e limiti), preparazione distinta dal workload,
cache fredda/calda e numero motivato di ripetizioni. Non importare tutti i motori
durante discovery né predisporre download come effetto dei test.

Il progetto usa managed locale CPython 3.12.13 con origine verificata e venv
con package installato nel medesimo interprete dei figli. Il piano deve prevedere
nuovi input effettivi e preflight secondo la guida uv: dopo modifiche ricostruire
sdist/wheel, reinstallare la wheel canonica, confrontare dieci hash e S/B/I prima
di `--no-sync`; il solo sync non basta. Identifica il confine degli script esterni
alla wheel e come ne saranno attestati gli input. Le prove ufficiali richiedono
R PASS e isolamento del progetto; non trasferire receipt a001 a nuovi input.

Proponi verifiche mirate del nuovo codice, del misuratore e delle regressioni
pertinenti, spiegando quando serve la suite veloce selezionata della reference.
Prima della collection sono necessari installazione/preflight, tokenizer,
Pandoc e R pronti; non installare/scaricare per riparare una collection fallita.
API/packaging/Docker/discovery sono strati distinti: non rieseguire campagne
storiche se il delta non le invalida. Per soli documenti bastano link, coerenza
degli stati e `git diff --check`.

## Risorse, autonomia e limiti

La pianificazione autorizza letture, ispezioni statiche e scrittura del piano,
note proprie e checkpoint. Eventuali sonde di disponibilità/versione devono
essere leggere, senza import applicativi/ML, rete di acquisizione, installazioni
o workload di conversione; registra comando, output, exit e limiti. Non creare
ora corpus, runner, benchmark o report di risultato. Se servono capacità nuove,
proponile nel piano, non acquisirle durante la pianificazione.

Hardware osservato all'ingresso: Intel i7-1165G7, 8 CPU logiche, circa 62,5 GiB
RAM totale; nessun device NVIDIA esposto alla sonda. Circa 117 GiB liberi alla
sonda precedente, non un cap né una misura attuale. Sono osservazioni locali,
non certificazioni GPU/Windows/WSL/macOS. Non serve conoscere altre macchine
per pianificare synthetic/gold, strumenti, negativi e baseline nativa.

Fornisci una stima numerica per preparazione, generazione, build/installazione,
test, workload e registri, con totale, margine e riserva di consegna. Separa
tempo trascorso/CPU, picco disco/RAM, traffico e costo monetario; esplicita
assunzioni e metodo di contabilizzazione/stop. Proponi una **quota unica adeguata
per l'intero obiettivo**, non piccoli cap per sonda o solo tempi dei subprocess.
Le quote a001 non si ereditano né si azzerano in a002. Il supervisore disporrà i
limiti nel mandato dopo piano/review: le stime qui non autorizzano nuove spese.

Privati, acquisizioni di modelli pesanti, inferenza ML, provider remoti a pagamento,
invio di documenti e altre macchine richiedono un perimetro esplicito e restano
fuori dal mandato corrente. Non aggiornare pin/lock per usare l'ultima versione.
Se valuti versioni/capacità/licenze di candidati, consulta fonti primarie aggiornate
e registra URL/data; non acquisire pesi. Non occorre un sondaggio dei candidati
per chiudere questo piano locale.

Il piano deve lasciare all'implementatore i mezzi ordinari reversibili entro
requisiti/costi/confini: timeout compatibili con i tool, log, path esclusivi,
launcher/reader propri e fix dei test nella stessa chat. Distingui input protetti
di confronto, input della prova identificabili e parametri adattabili.
L'implementatore crea autonomamente snapshot tecnici e ripete solo prove invalidate;
il supervisore congela il risultato completo per le due review. Niente nuova
autorizzazione per ogni comando/freeze. Per EPERM/EACCES si diagnostica e, se
disponibile, si richiede escalation del tool mantenendo isolamento; un effettivo
rifiuto non si aggira. I soli arresti sostanziali sono quelli del protocollo:
cambio di obiettivo/assunzione decisiva, costo/confine non coperto, input protetti
irrecuperabili o impossibilità concreta di rispettare criteri/isolamento.

## Consegna del piano

Scrivi `temp/run-a002-fase1-corpus/plans/plan-r001.md` secondo il template,
in italiano, con autore/provider/modello effettivi (non inventare il modello
se non noto), prompt di origine, branch/HEAD/base/delta osservati. Includi passi
ordinati, alternative significative, file/componenti proposti, matrice dei
requisiti/prove/esiti/costi, rischi, recupero e lacune proposte. Destinazioni
indicative da confermare con motivazione:

- `tests/fixtures/conversion-corpus/`: fixture, sorgenti/generatori e riferimenti;
- `scripts/benchmarks/`: strumenti riutilizzabili e protocollo del benchmark;
- `documentation/benchmarks/`: report datato soltanto dopo esperimento reale;
- `docs/`: istruzioni d'uso verificate secondo Diátaxis;
- `temp/run-a002-fase1-corpus/`: report/evidenze/checkpoint dei rispettivi ruoli.

Le note/sonde proprie, se necessarie, vanno in
`temp/run-a002-fase1-corpus/evidence/planning-r001/`. Scrivi il checkpoint
`temp/run-a002-fase1-corpus/handovers/planning-r001.md` con input letti, output,
verifiche, limiti, processi pendenti e prossima azione. Non aggiornare STATE,
eventi, indici condivisi, changelog o checkpoint del supervisore. Non sovrascrivere
revisioni precedenti o artefatti di altri ruoli.

Consegna il piano completo al supervisore, senza etichettarlo GO e senza creare
snapshot di review, report dei reviewer o arbitrati. Il supervisore riceverà il
piano, creerà/verificherà lo snapshot comune e preparerà due prompt con criteri
identici e output distinti per nuove chat ChatGPT/Claude. L'arbitrato reale
precede il mandato di implementazione. Termina indicando percorsi, verifiche,
limiti e prossimo ruolo: **supervisore per ricezione e avvio delle review piano**.
