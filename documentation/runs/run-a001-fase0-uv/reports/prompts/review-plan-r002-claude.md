# Prompt — run-a001-fase0-uv — Review del piano r002 — Claude

Agisci come revisore Claude in una **nuova chat indipendente**, distinta
sia dal pianificatore r002 sia dal supervisore, dai revisori r001 e dall'altro
revisore r002. Revisiona il piano; il ruolo assegnato da questo prompt prevale
su un handover globale che parla di supervisione. Non implementare la migrazione.

Provider previsto: Claude. Registra provider, modello e interfaccia effettivi
e riferimento chat se disponibile, senza inventare identificatori non esposti.
Registra eventuali sostituzioni; se il provider richiesto non è disponibile o
questa chat ha già svolto uno dei ruoli incompatibili, segnala il problema al
supervisore senza impersonare la chat mancante. Non creare agenti o review fittizie.
Non leggere report, checkpoint o evidenze dell'altro revisore **r002**, né
l'arbitrato r002, e non coordinare i rilievi con la chat concorrente.

## Oggetto e identità comuni ai due revisori

- Repository: `/home/davide/workarea/markdown-for-llms`.
- Run: `run-a001-fase0-uv`, fase roadmap 0.1; revisione del piano: **r002**.
- Piano: `temp/run-a001-fase0-uv/plans/plan-r002.md`.
- SHA-256 piano: `df23588de1247820a797e033ba7e93f60ed7d6b5f1f64e1948f012646039ca71`.
- Snapshot comune: `temp/run-a001-fase0-uv/snapshots/plan-r002.json`.
- Branch: `feature/run-a001-uv`.
- HEAD/base dev/merge-base verificati dal supervisore:
  `66ba82200e5def5a4db76f9bafccb0731b506091`.
- Sei modifiche documentali non committate di supervisione sono previste e
  identificate nel manifest comune: CHANGELOG, indice documentation, roadmap,
  metadata ADR 0006/0007 e indice decisioni. Nessun codice uv implementato.
- Bootstrap già integrato manualmente in dev; adozione uv approvata. Arbitrato
  r001 **NO_GO**, conservato. Nessun esito r001 si trasferisce al piano r002.

Dalla radice verifica Git e snapshot **prima della lettura tecnica e alla fine**:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD dev
git merge-base HEAD dev
python3 scripts/run_context.py verify run-a001-fase0-uv --label plan-r002
```

Il risultato atteso è MATCH. Leggi il manifest e registra nel report HEAD,
impronta worktree, SHA-256 piano e risultati pre/post, con evidenze dei comandi.
Ricontrolla che l'hash del piano corrisponda alla voce del manifest. Non ricreare
né sovrascrivere snapshot o input. `planning-context-r002` è ora storico dopo gli
aggiornamenti documentali: non è l'oggetto della review. Gli snapshot r001 sono
antecedenti, con identità verificata all'epoca e cronologia conservata.

Se plan-r002 è STALE, Git diverge o mancano file, descrivi il problema al
supervisore e consegna la verifica d'identità non valida per l'arbitrato;
non emettere un GO sul contenuto differente. Il manifest non contiene i sorgenti.
Git non trasferisce temp/ o modifiche non committate: recupera i file necessari,
senza ricostruire input/esiti dalla sola chat.

## Input espliciti e ordine di lettura

1. Questo prompt; `AGENTS.md`; `.agents/skills/manage-implementation-run/SKILL.md`;
   `documentation/development/run-lifecycle.md` e template
   `documentation/development/templates/review.md`.
2. `temp/run-a001-fase0-uv/STATE.md`, `brief.md`,
   `prompts/02-planning-r002.md` e `arbitrations/arbitration-plan-r001.md`.
   Il brief con A1–A7 e l'arbitrato di ripianificazione sono vincolanti.
3. `documentation/README.md`, `documentation/decisions/README.md`, ADR 0001,
   0006 e 0007, fase 0.1 della roadmap e changelog. Il prodotto rimane legacy;
   l'approvazione uv non equivale a GO sul piano o a implementazione realizzata.
4. Leggi **integralmente** `plans/plan-r002.md`; poi
   `handovers/planning-r002.md` e `evidence/planning-r002/findings.md`,
   `checks.json`, `checks.sha256`, `static-inventory.json`, `local-help.json`.
   Questi sette file sono congelati nel manifest. I checks dichiarano MATCH e
   controlli statici/documentali: non dimostrano lock, installazioni o prove runtime.
5. `evidence/supervisor-plan-review-r002/receipt.json` per identità/provenienza.
   Le due review **precedenti r001**, disponibili a entrambi come antecedenti,
   sono `reviews/review-plan-r001-chatgpt.md` e `review-plan-r001-claude.md`.
   Consulta i loro checkpoint/evidenze e `evidence/supervisor-arbitration-plan-r001/`
   solo per i rilievi/probe citati. Piano r001 e contesto di ingresso r002 restano
   congelati per tracciabilità. Il manifest comprende gli input del contesto,
   ma ciò non richiede di caricare tutta temp/ o tutti gli help storici.
6. Sorgenti pertinenti: setup.py, requirements.txt, .gitignore, Dockerfile,
   docker-compose.yml, **docker-compose-build.yml**, README e pagine docs interessate,
   i dieci moduli del piano, tests/conftest.py, unit/integration/governance e i
   due smoke alla radice. Leggi altri file solo se necessari a un rilievo.

Tutti i percorsi relativi del punto 2 e dei punti 4–5 sono nella run indicata.
Gli help uv r001 e della review Claude r001 richiamati dal pianificatore sono
input storici identificati nel manifest: consulta quelli pertinenti. Entrambi
i revisori r002 ricevono gli stessi input, criteri e snapshot. Cambiano soltanto
l'identità del revisore, il prefisso dei nuovi ID e i percorsi dei suoi output.
Il giudizio r002 deve essere autonomo, anche rispetto agli esiti r001.

## Criteri e compito della review

Valuta implementabilità, perimetro e verificabilità di r002 rispetto ad A1–A7.
Motiva il risultato per ogni criterio e controlla la coerenza di P0–P7 e V0–V12,
prerequisiti/costi, risultati osservabili, condizioni di uscita e recupero.
Controlla **tutti gli undici rilievi accolti**: CLA-P001…CLA-P010 e GPT-P001.
La presenza della matrice non basta a chiuderli: indica per ciascuno se il piano
lo affronta adeguatamente, parzialmente o lascia un blocco, con sezione/evidenza.
I blocchi dell'arbitrato r001 sono CLA-P001, CLA-P002 e CLA-P005. Le altre otto
azioni accolte restano nella specifica. Non presumere che siano risolte né che
il precedente GO sia ancora applicabile.

Controlli comuni richiesti:

- **A1/A2, CLA-P001:** percorso managed unico, pin/versioni, rapporto con pyenv,
  matrice shim/bin con e senza PYENV_VERSION, origini sys.executable/stdlib.
  Controlla fattibilità dei comandi uv0.10.10 e assenza di cambi globali; separa
  help statico dalla selezione reale ancora da provare. Valuta lock, gruppi/extra,
  CPU/cu126, indici, vincoli dei backend e di tutte le sdist, non solo Marker.
- **A2/A3, CLA-P005/P004:** dieci moduli flat e cinque entry point, config.main,
  wheel non editable fuori clone; avvio figli sys.executable -I -m e preflight
  nello stesso interprete installato; origini applicative/transitive e sentinelle
  con PYTHONPATH avverso. Valuta console/script clone e pulizia PYTHONPATH/PYTHONHOME
  nei contesti prescritti, senza fallback ai dati. Controlla .env esplicito nel
  workspace **prima** degli override, precedenza shell e osservabili del test.
- **A4, CLA-P002:** matrice di ogni test nuovo/mutato, base+dev e dev+marker-server
  per API mock, progetto installato nello stesso interprete dei figli, wheel-env
  esterna. Valuta C-fast/API/package/docker/governance/discovery, ignores prima
  degli import, raccolta inerte, conteggi/partizione e zero skip essenziale.
  Build/install/cache appartengono alla preparazione: nessun fetch o build da
  collection/test. Controlla il runner Linux offline, prerequisiti di unshare e
  Docker e condizione IMPEDITA se mancano; uv --offline da solo non prova egress.
- **A4, CLA-P009/GPT-P001:** baseline F1–F6 identificata, stessi input validated
  e argomenti orchestratore, >=2 chunk reali, sequenza/contenuto/metadati/overlap
  e confronto complementare del metodo sliding esistente. Valuta i confronti
  incrociati se diverge l'output, con Python/dipendenze/tokenizer/cache dichiarati.
  Consulta `.agents/skills/verify-conversion-fidelity/SKILL.md` per giudicare
  testo, formule, numeri, riferimenti, asset e ordine senza normalizzare perdite.
- **A5, CLA-P008:** inventario README/help completo, opzioni e script inesistenti,
  guide uv e responsabilità sull'istruzione operativa AGENTS. Verifica che
  pseudocodice e nuova applicazione siano distinti dalle funzionalità disponibili,
  comandi consentiti da eseguire in V9 e istruzioni rinviate siano identificati.
- **A6, CLA-P003/P006/P007:** scelta **wheel PyPI Marker1.10.2** al posto di Git,
  SHA/provenienza/metadata, confronto file e contratto installato; non assumere
  equivalenza wheel/tag. Torch/Surya e native restano candidati. Controlla Docker
  dal lock, digest nativi, bookworm/patch storica/manutenzione, apt congelato,
  contesto filtrato, risorse/mount/cache di **entrambi i Compose** e override GPU.
  Valuta V7/V8 obbligatorie: build CPU, import/contratti e native/rendering offline
  senza font/pesi/constructor reali impliciti. Config o health non bastano.
- **A6, CLA-P010:** V10/V11 non autorizzate nel mandato corrente, futuro perimetro
  pesi/font/rete/disco/hardware/costi distinto, local_marker non verificato dopo
  la migrazione. Valuta quando V10 diventa essenziale in caso di incompatibilità:
  nessun criterio richiesto deve risultare passato per un test semplicemente
  rinviato. V7/V8 restano obbligatorie anche se V10/V11 non vengono autorizzate.
- **A7:** quattro report reali, due arbitrati e snapshot identificati, nuove chat
  indipendenti e gestione scostamenti; commit/merge/push/promozioni manuali.
  Un GO di questa singola review non autorizza l'implementazione.

Le scelte del piano sono proposte. Verifica affermazioni aggiornabili o incerte
con help congelato, documentazione ufficiale e sorgenti primarie; registra URL/data,
accessi falliti e distingue fatti da inferenze. Valuta le incertezze prima del GO
finale: prove future non eseguite non costituiscono automaticamente un difetto
in una review del piano, ma devono essere fattibili e sufficienti per il requisito.

Sono ammessi letture, controlli statici, hash e help/versioni senza download.
Non eseguire uv lock/sync/run/build, installazioni, suite/collection, conversioni,
Docker config/manifest/build/run/up, startup Marker, namespace o prove runtime.
Formula le ulteriori prove necessarie come rilievi con criteri di risoluzione;
non cambiare l'oggetto. Non inviare documenti a provider, scaricare modelli/font
né introdurre benchmark o costi impliciti.

## Output obbligatori

- Report: `temp/run-a001-fase0-uv/reviews/review-plan-r002-claude.md`.
- Evidenze proprie: `temp/run-a001-fase0-uv/evidence/review-plan-r002-claude/`.
- Checkpoint: `temp/run-a001-fase0-uv/handovers/review-plan-r002-claude.md`.

Usa il template review. Dichiara autore/provider/modello/interfaccia effettivi,
prompt di origine, oggetto r002, snapshot/hash/HEAD, indipendenza, input/controlli,
esiti pre/post, limiti e **GO oppure NO_GO motivato sul piano r002**.
Includi il riscontro per A1–A7 e la disposizione degli undici rilievi precedenti.
Assegna ai **nuovi rilievi r002** ID `CLA-P001`, `CLA-P002`, ecc.; il report
r002 distingue il giro. Quando citi un vecchio ID aggiungi «r001», senza riscrivere
la sua storia. Per ogni nuovo rilievo indica severità, blocco sì/no, posizione,
impatto, evidenza e criterio verificabile di risoluzione. Distingui blocchi da
suggerimenti opzionali. Se non hai rilievi, dichiaralo senza inventarli; non
approvare prove runtime come già svolte.

Scrivi soltanto report, evidenze e checkpoint propri. Non sovrascrivere output
già presenti, input, piano, codice, snapshot, arbitrati o registri condivisi.
Nessun commit, merge, push o deploy. Il checkpoint dichiara processi ancora
attivi o la loro assenza e i file necessari alla ripresa.

Consegna al supervisore percorsi, esito e limiti. Non leggere la review concorrente
neppure dopo la consegna per cambiare giudizio. Il supervisore attende entrambi
i report validi sullo snapshot **plan-r002**, arbitra tutti i rilievi e solo con
il nuovo arbitrato GO prepara il prompt per una nuova chat implementatrice.
