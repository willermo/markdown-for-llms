# Prompt — run-a001-fase0-uv — Pianificazione r003 dopo NO_GO r002

Agisci come pianificatore in una **nuova chat distinta dal supervisore e dalle
review r002**. Produci il piano completo r003, senza implementare la migrazione.
Repository `/home/davide/workarea/markdown-for-llms`, fase roadmap **0.1**.
Adozione uv approvata; il NO_GO riguarda il piano r002, non l'obiettivo.
Il ruolo assegnato da questo prompt prevale sul recupero globale del supervisore.

## Recupero e identità

Leggi nell'ordine:

1. AGENTS.md; temp/HANDOVER.md e PROJECT-CONTEXT.md; STATE.md e HANDOVER.md della run.
2. `.agents/skills/manage-implementation-run/SKILL.md`, protocollo
   `documentation/development/run-lifecycle.md` e template
   `documentation/development/templates/plan.md`.
3. Brief A1–A7, `prompts/02-planning-r002.md`, piano r002 **integrale**,
   checkpoint `handovers/planning-r002.md`, findings/checks/sidecar e inventario/help
   in `evidence/planning-r002/`.
4. **`arbitrations/arbitration-plan-r002.md`, specifica vincolante del giro**;
   entrambi i report `reviews/review-plan-r002-chatgpt.md` e
   `reviews/review-plan-r002-claude.md`, i loro checkpoint e le evidenze pertinenti.
   Il pianificatore può leggere entrambe le review antecedenti.
5. `evidence/supervisor-arbitration-plan-r002/receipt.json`, `sources.md`,
   `static-findings.json`, `transition.json` e `checks.json`;
   arbitrato r001 e relativi antecedenti solo per la specifica conservata.
6. Indice architetturale, changelog, indice ADR, ADR 0001/0006/0007, fase0.1 della
   roadmap; sorgenti/test interessati, entrambi i Compose, README/help/guide e
   file di packaging. Consulta altri file solo per un rilievo pertinente.

I percorsi della run senza prefisso sono relativi a temp/run-a001-fase0-uv/.
Input comuni sono identificati dal manifest **snapshots/planning-context-r003.json**.
Verifica prima della lettura tecnica e alla consegna:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
python3 scripts/run_context.py verify run-a001-fase0-uv --label planning-context-r003
```

Branch atteso feature/run-a001-uv; HEAD/dev/merge-base
`66ba82200e5def5a4db76f9bafccb0731b506091`. Sono previste soltanto sei modifiche
tracciate documentali di supervisione, identificate nel nuovo contesto.
Risultato atteso **MATCH**, senza GO. Registra impronta e hash dal manifest;
non ricreare o sovrascrivere snapshot. Piano r002 ha SHA-256
`df23588de1247820a797e033ba7e93f60ed7d6b5f1f64e1948f012646039ca71`.
Gli snapshot plan-r002 e precedenti sono storici dopo i nuovi metadata di stato;
piano, review e loro evidenze sono invariati. Bootstrap già integrato/pubblicato:
nessuna operazione Git da ripetere o autorizzazione uv da chiedere.

Se cambia l'identità segnala differenze al supervisore, senza consegnare come
MATCH input diversi. Se manca accesso ai file, recupera il trasferimento necessario:
manifest e Git non contengono temp/ e modifiche non committate.

## Obiettivo e vincoli conservati

R003 è un piano completo autosufficiente, non una lista di errata. Conserva A1–A7,
P0–P7/V0–V12 e la specifica dei **tutti undici rilievi r001**. I tre blocchi r001
sono affrontati nel disegno r002; non sono chiusi operativamente. Mantieni managed
locale unico, pyenv globale invariato, dipendenze base/dev/API/CPU/cu126 separate,
wheel installata fuori clone, isolamento -I/override .env, F1–F6 con >=2 chunk e
confronto overlap/incrociato, Marker wheel PyPI con provenienza/contratto, entrambi
i Compose e i limiti legacy. CLA-P003 r001 viene precisato da GPT-P001 r002;
CLA-P008 r001 da CLA-P004 r002. Nessun esito GO precedente si trasferisce.

Nessuna nuova app/Web UI, fase2 applicativa, revisione dei parser/algoritmi o scelta
OCR definitiva. Non cambiare versioni/piattaforme/candidati implicitamente per
chiudere i rilievi: scostamenti sostanziali vanno motivati al supervisore.

## Blocco da chiudere: CLA-P001 r002

Definisci una catena verificabile **sorgenti → distribuzione → installazione → esito**.
La risincronizzazione non editable senza regola di invalidazione non basta.

- Scegli un meccanismo di ricostruzione/reinstallazione per **ogni** ambiente host
  previsto (uso normale, base/dev/api, server opzionale) dopo modifiche al codice.
  Possibili cache-keys con input predefiniti/dieci moduli e metadata realmente
  consumati, oppure reinstallazione mirata che forzi effettivamente rebuild,
  con semantica uv0.10.10 verificata nelle fonti. Una sola venv nuova, --refresh
  o hash della wheel senza confronto col sorgente non sono una dimostrazione.
  Non pulire la cache globale o installazioni altrui.
- Prima delle prove pertinenti (C-fast/C-api/C-package, V3/V4/V5 e immagine V7/V8)
  confronta SHA-256 dei dieci moduli con il sorgente identificato; includi sdist,
  wheel/RECORD e copia installata in site-packages e /opt/venv. Per i profili base
  leggi/hash i file senza importare Marker/FastAPI. RECORD attesta i file wheel,
  ma da solo non dimostra che derivino dal clone corrente. Identifica metadata,
  backend/build constraints/lock e input necessari alla build con manifest.
- Ogni esito registra l'impronta run_context del worktree della prova e la catena
  degli artefatti/input. Distingui baseline originale e codice nuovo. Definisci
  quali prove invalidare dopo modifiche ai loro input e quali ripetere; non
  attribuire prove di uno stage a un worktree diverso senza confronto motivato.
  Report e log stanno negli spazi esclusi, senza codice prodotto nascosto lì.
- Aggiungi una futura verifica discriminante: modifica del solo modulo in copia
  isolata a pyproject/metadata invariati, sync con cache già popolata, confronto
  della nuova copia installata e controllo negativo del preflight. È preparazione
  dopo GO, non test veloce con build/rete né modifica da compiere ora. Ripristino
  e identità devono essere provati prima delle verifiche finali.
- Riallinea P2, comandi V1–V6, aggiornamenti futuri di guida/README/AGENTS. Non
  sostituire la prova esterna con editable, conftest o copia manuale dei moduli.

## Altri cinque rilievi accolti r002

- **CLA-P002, runner:** preflight innocuo dopo il futuro GO e **prima di P0/V0**,
  con target/perimetro/comandi/risultati. Predefinisci un'alternativa concreta
  equivalente: Firejail locale è solo candidato disponibile nella review, da
  verificare, oppure un runner Linux identificato. Prova rete senza egress e
  isolamento ereditato dai figli, disponibilità di interpreti/Pandoc/cache/temp e
  limiti del daemon Docker separato. Nessun cambio di sysctl/AppArmor/rete host o
  privilegio persistente. Se manca il runner, V0 resta IMPEDITA; non considerare
  proxy, uv --offline o sola guardia pytest equivalenti. La negazione unshare è
  ancora inferenza statica. **Non creare namespace o avviare Firejail ora**.
- **CLA-P003, download Python:** guardia persistente nel progetto coerente con
  uv0.10.10, acquisizione esplicita nel path managed unico e controllo dell'origine.
  python-downloads=manual da solo non garantisce il path se un'altra installazione
  è già disponibile. V1 include shell senza variabili: errore esplicito oppure
  solo ambiente/origine previsti, nessun download altrove; registra prima/dopo,
  senza modificare directory managed globali. Guide/AGENTS/IDE dichiarano
  prerequisiti e limiti; non promettere resistenza a override deliberati.
- **CLA-P004, inventario:** correggi l'estrazione dei fence indentati/lista. Baseline
  README attuale: 74 blocchi, 63 shell/Python/YAML, cinque bash omessi e inline
  Git a L1538. Classifica tutti, incluse le undici configurazioni/output/alberi/
  Markdown, con regola esplicita verificabile o illustrativa; riconcilia pre/post
  senza aspettarti lo stesso numero dopo le modifiche. Usa static-findings.json;
  non correggere i report r002 o cancellare il primo riscontro ChatGPT incompleto.
- **CLA-P005, sentinelle:** V5 copre dieci nomi applicativi e quattro transitive,
  padre e figli nelle modalità sicure. Include master_workflow/unified_converter.
  Nessun import server nel base per provare una sentinella: spec/origini oppure
  profilo API appropriato. Marcatori ineseguiti e contenuto/override conservati.
- **GPT-P001, vincoli pip:** ogni pip che può costruire applica il file constraints,
  oppure usa solo wheel con stop su sdist; riga V3 coerente con P1. Hash runtime
  non vincolano backend. Conserva manifest di input/output/backend effettivi e
  stop se non identificati. Non chiamare GPT-P001 r001 il nuovo dettaglio pip.

## Suggerimenti S1–S7 e autorizzazioni

Traccia esplicitamente le sette disposizioni dell'arbitrato, senza trasformarle
in nuovi blocchi o ampliamenti del prodotto: S1 rinviato, nessuna restrizione
implicita delle piattaforme; S2 esclusioni nominate e conteggi; S3 input metadata
README/licenze ammessi nel contesto Docker se necessari; S4 socketpair locale
AF_UNIX per API senza allentare egress; S5 lock --check separato da --no-sync;
S6 versioni locked e warning osservati, nessun refactoring on_event; S7 percorso
server host etichettato non verificato se privo di prova.

V7/V8 restano **obbligatorie future** per A6 e la preparazione pesante va distinta
dai test veloci. **V10/V11, pesi/font e inferenza non autorizzati** nel mandato
corrente. local_marker non verificato dopo migrazione; V10 torna essenziale se
un'incompatibilità lo richiede e il bisogno va al supervisore. Non chiedere ora
permessi per scrivere il piano. Operazioni privilegiate o nuovo perimetro operativo
non diventano autorizzati per il solo fatto di apparire nel piano.

## Metodo e consegna

Consulta sorgenti primarie/help congelati per flag/configurazioni; registra URL,
data, esiti e inferenze. Solo letture statiche, hash, AST e controlli documentali.
Nessun uv lock/sync/run/build, installazione, suite/collection, conversione, Docker,
namespace/Firejail o download di interpreti/pacchetti/tokenizer/pesi/font. Non
creare codice, fixture implementative, pyproject/lock, ambienti o guide uv ora.

Consegna esclusivamente:

- `temp/run-a001-fase0-uv/plans/plan-r003.md`, completo secondo il template, con
  autore/provider/modello effettivi, origine e identità, P/A/V, rischi/recupero e
  tre matrici: undici ID r001, sei ID r002, sette suggerimenti/disposizioni r002.
- `temp/run-a001-fase0-uv/evidence/planning-r003/findings.md` e `checks.json`:
  controlli reali, hash output/input, MATCH finale, collegamenti/whitespace e limiti.
  Se usi un sidecar, evita autoreferenzialità e identifica tutti gli output nuovi.
- `temp/run-a001-fase0-uv/handovers/planning-r003.md`, checkpoint del ruolo,
  processi attivi o loro assenza e materiali per ripresa.

Non sovrascrivere output già presenti; segnala il conflitto. Non modificare
registri/handover comuni, indici/eventi, snapshot, arbitrati, piani/report r001/r002
né codice. Nessun commit, merge, push, deploy o GO del pianificatore.

Destinatario: supervisore. Dopo la consegna serviranno snapshot **plan-r003**, due
nuove chat di review indipendenti e nuovo arbitrato. Nessuna implementazione prima
di entrambi i nuovi report validi e GO riferito al piano/snapshot identificati.
