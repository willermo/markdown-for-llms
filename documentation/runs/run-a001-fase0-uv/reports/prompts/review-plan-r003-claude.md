# Prompt — run-a001-fase0-uv — Review del piano r003 — Claude

Agisci come revisore Claude in una **nuova chat indipendente**, distinta
sia dai pianificatori r001/r002/r003 sia dal supervisore, dai revisori dei giri
precedenti e dall'altro revisore r003. Revisiona il piano; questo ruolo esplicito
prevale sull'handover globale di supervisione. Non implementare la migrazione.

Provider previsto: Claude. Registra provider, modello e interfaccia effettivi
e riferimento chat se disponibile, senza inventare identificatori non esposti.
Le differenze d'interfaccia o sostituzioni devono essere dichiarate e registrate;
non attribuire a Codex un'interfaccia ChatGPT Web né a un agente OpenAI il provider
Anthropic. Se il ruolo non è disponibile o questa chat ha già svolto un ruolo
incompatibile, segnala il problema al supervisore senza simulare il revisore.
Non creare agenti, delegare la review o coordinare i rilievi con la chat concorrente.
Non leggere report, checkpoint o evidenze dell'altro revisore **r003**, né
l'arbitrato r003, anche dopo la consegna per cambiare giudizio. I report/arbitrati
r001/r002 sono invece antecedenti comuni, consultabili per le disposizioni citate.

## Oggetto e identità comuni ai due revisori

- Repository: `/home/davide/workarea/markdown-for-llms`.
- Run: `run-a001-fase0-uv`, fase roadmap 0.1; revisione del piano: **r003**.
- Piano: `temp/run-a001-fase0-uv/plans/plan-r003.md`.
- SHA-256 piano: `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`.
- Snapshot comune: `temp/run-a001-fase0-uv/snapshots/plan-r003.json`.
- Branch: `feature/run-a001-uv`.
- HEAD/base dev/merge-base verificati dal supervisore:
  `66ba82200e5def5a4db76f9bafccb0731b506091`.
- Sei modifiche documentali non committate di supervisione sono previste nel
  manifest: CHANGELOG, indice documentation, roadmap, metadata ADR 0006/0007
  e indice decisioni. Nessun codice della migrazione uv è implementato.
- Bootstrap integrato manualmente in dev; adozione uv già approvata. Arbitrati
  r001 e r002 **NO_GO**, conservati. Nessun GO precedente si trasferisce a r003.
- I quattro file consegnati dichiarano il ruolo di **pianificatore**; il nome
  «implementatore» nel riepilogo dell'utente non cambia oggetto o fase.

Dalla radice verifica Git e snapshot **prima della lettura tecnica e alla fine**:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
python3 scripts/run_context.py verify run-a001-fase0-uv --label plan-r003
```

Atteso MATCH. Leggi il manifest e registra HEAD, branch, impronta worktree,
SHA-256 del manifest e del piano, risultati pre/post ed evidenze dei comandi.
Ricalcola gli hash degli input congelati pertinenti, incluso checks.json del
pianificatore, e verifica la corrispondenza con il manifest. Non ricreare snapshot.
`planning-context-r003` è ora storico dopo i soli sei aggiornamenti documentali:
non è l'oggetto della review; ricevuta e transizione identificano il MATCH in ingresso.

Se plan-r003 è STALE, Git diverge o mancano file, consegna un riscontro d'identità
non valido per l'arbitrato e informa il supervisore; non emettere GO su un oggetto
differente. Il manifest non contiene i sorgenti. Git non trasferisce temp/ o lavoro
non committato: occorrono i file identificati, senza ricostruirli dalla sola chat.

## Input espliciti e ordine di lettura

1. Questo prompt; `AGENTS.md`; `.agents/skills/manage-implementation-run/SKILL.md`;
   `documentation/development/run-lifecycle.md` e template
   `documentation/development/templates/review.md`.
2. `temp/run-a001-fase0-uv/STATE.md`, `brief.md`,
   `prompts/02-planning-r003.md`, `arbitrations/arbitration-plan-r002.md`
   e `arbitrations/arbitration-plan-r001.md`. Brief A1–A7 e disposizioni degli
   arbitrati sono vincolanti; non ripartire dal solo riepilogo in chat.
3. `documentation/README.md`, `documentation/decisions/README.md`, ADR 0001,
   0006/0007, fase 0.1 della roadmap e changelog. Il prodotto rimane legacy;
   approvazione uv e consegna del piano non equivalgono a GO o implementazione.
4. Leggi **integralmente** `plans/plan-r003.md`; poi
   `handovers/planning-r003.md`, `evidence/planning-r003/findings.md` e
   `evidence/planning-r003/checks.json`. Sono i quattro output congelati.
   Inventario completo README, fonti/help e riscontri sono incorporati nei due
   file di evidenze: non presupporre un inventario o sidecar r003 separato.
   I checks dichiarano 183 input invariati e MATCH; il supervisore ha riconfermato
   dimensioni/hash e calcolato l'hash dei checks, omesso dal file stesso.
   Letture e help non dimostrano lock, build, installazioni o prove runtime.
5. `evidence/supervisor-plan-review-r003/receipt.json` per identità/provenienza e
   transizione. Antecedenti essenziali r002: `reviews/review-plan-r002-chatgpt.md`,
   `reviews/review-plan-r002-claude.md`, `evidence/supervisor-arbitration-plan-r002/`
   (receipt, static-findings, sources e transition). Consulta checkpoint/evidenze
   dei revisori r002 solo per i rilievi citati, inclusi estratti cache/settings uv.
   I due report r001 e relative evidenze, vecchi piani, help e contesti sono
   congelati come tracciabilità: leggili se necessari a una disposizione.
6. Sorgenti pertinenti: setup.py, requirements.txt, .gitignore, Dockerfile,
   docker-compose.yml, **docker-compose-build.yml**, README e guide docs interessate,
   i dieci moduli elencati in P2, tests/conftest.py, unit/integration/governance e
   i due smoke alla radice. Gli script diagnostici e test nuovi proposti non
   esistono ancora: valuta la loro specifica, senza crearli o inventarne l'esito.

Tutti i percorsi relativi dei punti 2 e 4–5 sono nella run indicata. Il manifest
include gli antecedenti comuni per integrità, ma non richiede di caricare tutta
temp/. Entrambi i revisori ricevono gli stessi input, criteri e snapshot; cambiano
solo identità del revisore, prefisso dei nuovi ID e destinazioni dei suoi output.
Il giudizio r003 è autonomo anche rispetto ai GO individuali r001/r002.

## Compito e criteri della review

Valuta implementabilità, perimetro e verificabilità di r003 rispetto ad **A1–A7**.
Motiva ogni criterio e verifica coerenza di §R, P0–P7, V0–V12, comandi esatti,
prerequisiti/costi, risultati osservabili, condizioni di uscita e recupero.

Esamina **tutte e tre le matrici** finali del piano: undici rilievi r001
(CLA-P001…P010 e GPT-P001), sei rilievi r002 (CLA-P001…P005 e GPT-P001) e sette
disposizioni CLA-S1…S7 r002. Per ogni voce indica adeguato/parziale/blocco nel
**disegno**, con sezione/evidenza e prove future richieste; presenza della voce
non equivale a chiusura operativa. Il blocco vigente è CLA-P001 r002 A3/A4;
i tre blocchi r001 CLA-P001/P002/P005 e le altre azioni restano nella specifica.

Controlli comuni richiesti:

- **A3/A4, CLA-P001 r002 — corrispondenza del codice.** Valuta reinstall-package
  persistente e flag espliciti uv0.10.10, ricostruzione a metadata invariati,
  installazione finale della wheel canonica in ogni ambiente host d'uso/prova
  e loro ordine. Una nuova venv, un'origine site-packages o una risincronizzazione
  senza confronto non dimostrano freschezza. Verifica la catena **S/B/I/E**:
  impronta/input sorgenti e backend, sdist → wheel dalla sdist, dieci moduli byte
  per byte, RECORD/metadata/entry point, installato e immagine /opt/venv; receipt
  prima di C-fast/C-api/C-package e V3/V4/V5/V7/V8. Server base: spec/hash senza
  import FastAPI implicito. Valuta input README/licenze consumati dal backend,
  sicurezza archivi/path e confronto con i sorgenti realmente esaminati.
  Controlla probe discriminante in copia isolata con cache popolata, modifica
  di solo modulo, negativo prima del sync, verifica rebuild **prima** della
  wheel canonica, rifiuto receipt vecchia e ripristino; nessuna cache clean globale.
  Valuta snapshots di stage affidati al supervisore, handoff concreti e matrice
  di invalidazione/ripetizione: un esito deve identificare sorgenti e derivati
  dello stage corretto, senza approvazione del piano a ogni file o PASS riciclati.
- **A1/A2, CLA-P001 r001 e CLA-P003 r002 — Python e dipendenze.** Pin motivato,
  percorso managed locale unico, matrice shim/bin con/senza PYENV_VERSION,
  sys.executable/stdlib/base_prefix e stop su origine diversa. Valuta guardia
  persistente python-downloads=manual, acquisizione esplicita locale, caso senza
  variabili UV/PYENV_VERSION e con/senza venv. Il test non deve mascherare la
  guardia con --offline/--no-python-downloads; il runner deve impedire egress.
  manual da solo non vincola il percorso né protegge da override deliberati.
  Comandi/help 0.10.10, lock universale, gruppi/extra CPU/cu126 e indici; niente
  modifiche globali. **GPT-P001 r002:** build-constraints in ogni pip che può
  costruire oppure sole wheel/--no-build e stop su sdist; hash runtime e backend
  hanno funzioni diverse, tutte le build devono identificare backend effettivi.
  --no-sync non certifica coerenza del lock: serve check separato (CLA-S5 r002).
- **A4, CLA-P002 r002 — runner prima di P0.** Valuta preflight innocuo §R,
  unshare primario e Firejail alternativa candidata, argv/perimetro equivalenti,
  namespace diverso, sola lo/assenza route, no egress e isolamento dei figli,
  prova AF_UNIX socketpair positiva (CLA-S4), accesso Python/Pandoc/cache/temp.
  AppArmor è indizio storico, non rifiuto dimostrato. Se entrambi impediti, le
  prove dipendenti restano IMPEDITE; nessuna modifica di sysctl/profili o fallback
  non confinato. La previsione di una sonda dopo GO non ne autorizza l'esecuzione
  durante questa review. Daemon Docker locale e container --network none sono
  un percorso distinto, senza concedere il socket agli altri strati.
- **A2/A3, CLA-P004/P005 r001 e CLA-P005 r002.** Dieci moduli flat/cinque CLI,
  config.main, wheel non editable esterna; figli sys.executable -I -m nello stesso
  interprete installato, cwd dati esplicito, console/script clone sicuri. Quattordici
  sentinelle (dieci applicative, incluse master_workflow/unified_converter, e quattro
  transitive), PYTHONPATH avverso, origini padre/figli e contenuti/override reali;
  no marcatore eseguito. Server: spec/hash base, runtime solo profilo API mock.
  .env nel workspace **prima** degli override; shell prevalente, metadata e
  pipeline_state osservabili, processi separati e nessuna credenziale.
- **A4, CLA-P002 r001 e CLA-S2/S6 r002.** Matrice per ogni test nuovo/mutato,
  dev per fast, dev+marker-server per API, wheel-env per packaging e driver Docker
  distinto; preflight S/B/I in tutti gli interpreti pertinenti. C-fast/API/package/
  docker/governance/discovery, ignores prima degli import, raccolta inerte,
  conteggi/partizione ed esclusioni nominate temp/tmp/documentation/runs e ambienti.
  Nessuno skip essenziale, build/install/fetch implicito da collection/test o
  refactoring globale. Cache tokenizer/Pandoc sono preparazione dichiarata;
  versioni locked API/warning/lifespan caratterizzati senza pin alle ultime versioni.
- **A4, CLA-P009/GPT-P001 r001 — fedeltà.** Baseline F1–F6 originale identificata,
  medesimi input validated/argomenti dell'orchestratore, >=2 chunk, sequenza/
  contenuto/metadati/confini/overlap e complemento del metodo sliding esistente.
  Confronti incrociati circoscritti se diverge, interprete/dipendenze/tokenizer/cache
  dichiarati; niente normalizzazione di perdite o correzioni degli algoritmi legacy.
  Consulta `.agents/skills/verify-conversion-fidelity/SKILL.md` per giudicare testo,
  formule, numeri, asset, riferimenti e ordine senza eseguire conversioni.
- **A5, CLA-P008 r001 e CLA-P004 r002 — istruzioni.** Inventario statico completo
  di 74 fence e nove inline, incluse cinque bash indentate/lista e inline Git;
  quattro classi e undici blocchi non operativi, coordinate/hash e riconciliazione
  pre/post senza imporre il totale futuro uguale. Controlla README/help/script
  inesistenti, --chunking-strategy, guide/AGENTS e responsabilità dei ruoli.
  Prove V9 consentite, esempi illustrativi e startup/servizi rinviati distinti;
  il piano non deve presentare nuova app o server host come già disponibili/collaudati.
- **A6, CLA-P003/P006/P007 r001 e CLA-S3 r002 — Marker/Docker.** Wheel PyPI
  Marker1.10.2 scelta, SHA/provenienza/metadata e confronto Git/contratto senza
  assumerne equivalenza. Torch/Surya/native sono candidati da provare. Lock e
  constraints, digest nativi, bookworm/patch storica/manutenzione, apt congelato,
  contesto filtrato reale e input backend ammessi, risorse/mount/cache di **entrambi
  i Compose** e override GPU. Catena S/B/I anche nel builder/runtime.
  **V7/V8 obbligatorie**: build CPU e probe offline import/contratti/rendering native
  senza startup/font/pesi impliciti, immagine identificata, --pull=never/network none.
  Mock constructor prima dei download; config, health o sole sentinelle non bastano.
- **A6, CLA-P010 r001 e CLA-S7 r002 — limiti operativi.** V10/V11 non autorizzate;
  eventuale futuro mandato pesi/font/rete/disco/hardware/costi distinto. local_marker
  non verificato dopo migrazione, server host privo di prova dedicata non eredita
  PASS Docker. Valuta quando incompatibilità renda V10 essenziale, da riportare
  al supervisore; nessun criterio passato soltanto rinviando la prova.
- **A7 e CLA-S1 r002.** Quattro report reali, due arbitrati su snapshot correnti,
  gestione scostamenti e manualità Git dell'utente. S1 rinviato: non restringere
  platforms del lock implicitamente. Un GO di questa review non avvia l'implementazione.

Le scelte sono proposte: usa help congelati, documentazione ufficiale e sorgenti
primarie per verificare affermazioni aggiornabili/incerte; registra URL/data,
accessi falliti e fatti/inferenze. Fonti mobili non provano automaticamente il tag
0.10.10. Una prova futura non svolta non è di per sé difetto del piano, ma deve
essere fattibile e sufficiente per il criterio. Non attribuire PASS runtime a letture.

Sono ammessi letture, hash, controlli statici e help/versioni senza download.
Non eseguire uv lock/sync/run/build, installazioni, suite/collection, conversioni,
Docker config/manifest/build/run/up, startup Marker, namespace, Firejail o probe
runtime/cache/rete. Nessun download di pacchetti/modelli/font, invio documenti a
provider o costo implicito. Formula nuove prove necessarie come rilievi con
criteri verificabili; non correggere il piano o creare codice diagnostico ora.

## Output obbligatori

- Report: `temp/run-a001-fase0-uv/reviews/review-plan-r003-claude.md`.
- Evidenze proprie: `temp/run-a001-fase0-uv/evidence/review-plan-r003-claude/`.
- Checkpoint: `temp/run-a001-fase0-uv/handovers/review-plan-r003-claude.md`.

Usa il template review. Dichiara autore/provider/modello/interfaccia effettivi,
prompt di origine, oggetto r003, snapshot/hash/HEAD, nuova chat/indipendenza,
input/controlli, esiti pre/post, limiti e **GO oppure NO_GO motivato su r003**.
Includi riscontro A1–A7 e disposizione di tutte le voci delle tre matrici.
Assegna ai **nuovi rilievi r003** ID `CLA-P001`, `CLA-P002`, ecc.;
quando citi vecchi ID aggiungi r001/r002. Non confondere GPT-P001 multi-chunk
r001 con GPT-P001 pip/constraints r002. Per ogni nuovo rilievo: severità, blocco
sì/no, posizione, impatto, evidenza e criterio di risoluzione; separa suggerimenti
opzionali. Se non trovi rilievi, dichiaralo senza inventarli.

Scrivi soltanto i tuoi report, evidenze e checkpoint. Non sovrascrivere output
esistenti, input, piano, codice, manifest, arbitrati o registri condivisi.
Nessun commit, merge, push, promozione o deploy. Aggiorna il checkpoint con
processi ancora attivi o loro assenza e percorsi necessari alla ripresa.

Consegna al supervisore percorsi, esito e limiti. Il supervisore attende entrambi
i report validi sullo stesso **plan-r003**, arbitra tutti i rilievi e soltanto con
il nuovo arbitrato GO prepara il prompt per una nuova chat implementatrice.
