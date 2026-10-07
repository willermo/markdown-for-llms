# Implementazione r001 — Concludere il piano con fix e prove autonomi

Agisci esclusivamente come implementatore della run a001, nella stessa chat di
report-r014 se disponibile. **Completa il piano approvato nel perimetro autorizzato.**
Correggi i problemi che incontri e ripeti le prove nella stessa chat. Il traguardo
è la consegna del risultato, non la preparazione di un altro freeze al supervisore.

Repository `/home/davide/workarea/markdown-for-llms`, branch `feature/run-a001-uv`,
HEAD/dev/base `66ba82200e5def5a4db76f9bafccb0731b506091`; worktree modificato
e nuovi file autorizzati, indice vuoto alla ricezione. Verifica lo stato corrente
e preserva il lavoro esistente. Git manuale dell'utente.

## Fonti correnti

Leggi AGENTS, skill manage-implementation-run, protocollo corrente e, relativamente
a `temp/run-a001-fase0-uv/`:

1. `STATE.md`, `HANDOVER.md`, `handovers/implementation-r001.md` e
   `arbitrations/addendum-operational-protocol-r016.md`.
2. `brief.md`, `plans/plan-r003.md` e `arbitrations/arbitration-plan-r003.md`.
   Se già letti nella stessa chat, recupera requisiti e parti pertinenti senza
   ricaricare la storia. In una chat fresca leggi il piano completo.
3. `implementation/report-r014.md`,
   `implementation/stages/impl-r001-stage-product-s018/request.json`,
   `evidence/implementation-r001/product-sbi-s017-r001/delivery.json` e
   `evidence/implementation-r001/product-sbi-s018-r001/static-input-checks.json`.
4. Scope di costo già ricevuto:
   `evidence/supervisor-implementation-r001/product-input-reception-r001/authorized-scope.json`.
   R013/R014/R015 restano pertinenti per acquisizioni e isolamento, con le
   clausole di attesa del freeze superate da R016.

Piano SHA `462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b`;
arbitrato SHA `f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d`.
Il GO del piano resta valido; nessun GO finale del codice è ancora stato emesso.
Questo prompt sostituisce i mandati intermedi incompatibili, non va concatenato
ai quarantanove prompt storici.

## Riparti dal reader corretto e prosegui

S017 è storico: cinque PASS, I root FAIL uv_cache.json, sedici passi non eseguiti.
Il reader è stato corretto con tredici controlli locali; questi non sostituiscono
I ufficiale. **Non attendere il supervisore per s018.** Il vecchio checkpoint
WAITING_FOR_STAGE_SNAPSHOT e l'owner della request sono superati da R016.

Usa gli input preparati in `work/product-sbi-r002/` e
`evidence/implementation-r001/product-sbi-s018-r001/`. Controlla i binding
pertinenti, correggi i difetti ordinari e crea tu lo snapshot di esecuzione:

- `scripts/run_context.py snapshot run-a001-fase0-uv --label impl-r001-stage-product-s018`;
  aggiungi un `--artifact` per ogni input della lista `artifacts_to_freeze`
  della request realmente esistente e pertinente, più R016 e questo prompt.
  Gli `extra_files_to_freeze` del repository sono raccolti automaticamente
  in `files`: non passarli come artefatti confinati a temp/.
- Identifica nuovi inventari/driver effettivi prima dello snapshot. Se servono
  nuove versioni, usa file/label nuovi e includili nella lista; conserva request,
  driver e ricevute originali. Escludi checkpoint vivi e output futuri.
- Esegui `verify` prima/dopo; allinea snapshot, inventario e argv S nei driver.
  La nuova cattura deve includere i file di governance ora modificati. Non
  imporre MATCH degli snapshot precedenti sul worktree corrente.

Un errore già visibile nella request: `changed_trial_inputs.before` e `.after`
identificano entrambi il reader corretto. Ricava il prima dalla copia
`product-sbi-s017-r001/verify_distribution-frozen-s017.py` e dal manifest s017,
registrando la rettifica in una nuova evidenza. Non riscrivere la request ricevuta
e non fermarti per questa correzione di binding.

L'adapter `product-sbi-s018-r001/resume_product_s018.py` è un punto di partenza,
non un requisito immutabile: verificane import, guardie, label e contabilità;
correggilo in una nuova versione se necessario, senza disattivare i controlli.
Esegui la nuova catena S → build nativa offline sdist/wheel dalla sdist → audit B
→ sync/runtime e canonica root/base-a/base-b → I/E → esterno hash-locked,
pip-check e I/E. Root .venv esiste già: fresh sync/reinstall, non falsa pretesa
di assenza. Ricava nomi/hash degli archivi dalla build reale. Conserva i fallimenti.

Se una prova fallisce, diagnostica, correggi, identifica i nuovi input e ripeti
le dipendenze invalidate. Se il driver termina al primo FAIL, il tuo lavoro
continua: quel return code non è un ordine di consegna al supervisore.
Non trasferire PASS vecchi a codice nuovo e non bypassare S/B/I/E per proseguire.

## Concludi il lavoro restante del piano

La sequenza base non esaurisce il mandato. Verifica i criteri del piano con una
matrice sintetica criterio → evidenza → esito/mancanza. Completa il lavoro
autorizzato pertinente: V1 config-only/rebuild/stale/ripristino, import e origini
fuori sorgenti, cinque CLI, fasi e override/workspace, confronti con la baseline,
negativi, packaging, suite nei profili effettivamente pronti, discovery e guide.
V7/V8 restano essenziali per Docker; V10/V11 restano escluse.

Decidi i mezzi reversibili necessari entro i requisiti del piano. Errori nel
codice nuovo, test e diagnostici si correggono qui; non chiedere permesso per
ogni fix, tool, timeout compatibile o snapshot. Conserva baseline e prove
storiche; non correggere le perdite degli algoritmi legacy fuori scope.
Il report raccoglie problemi, soluzioni significative, prove invalidate e
risultati effettivi: saranno valutati dalle due review.

## Confini già autorizzati

La fase base conserva 112MiB cumulativi/Hentry553541632, guardie e directory
dello scope ricevuto, 7200s complessivi con 487.34544921084307s già addebitati
prima della ripresa. Conta anche preparazione e tentativi successivi, senza
reset né duplicazione degli intervalli annidati. Deadline compatibili con tool
e residuo; nessun aumento implicito della quota. Nuovi output esclusivi.

Per i workload offline conserva R/D4 reale nella medesima invocazione,
Firejail net=none, guardie padre/figli/pre-startup/backend, prima delle prove.
Se il sandbox nega una syscall, diagnostica e richiedi l'escalation ufficiale
degli strumenti già delegata da R015; non servono una nuova chat o un prompt
del supervisore per richiederla. Rispetta un rifiuto effettivo dell'approvazione.

Le risorse della fase base non autorizzano download dev/API/ML o build pesanti
Docker esclusi dagli scope precedenti. Per il lavoro restante usa risorse già
ammesse; accerta concretamente quelle mancanti. Un limite reale blocca solo
le prove dipendenti: completa intanto codice, verifiche e documenti indipendenti.
Se occorre un costo nuovo, consegna un'unica proposta concreta con artefatti,
dimensioni, tempo, spazio disponibile e prove mancanti, insieme al risultato
del lavoro autorizzato. Non chiamare PASS un criterio essenziale non verificato.

Solo scostamenti sostanziali reali richiedono supervisione anticipata: nuovo
obiettivo/requisito, costo escluso, input di confronto irrecuperabile,
isolamento impossibile, rifiuto effettivo degli strumenti o azione irreversibile.
Un FAIL ordinario, helper cambiato o nuovo snapshot tecnico non sono tali casi.
Nessun commit/merge/push/deploy, modifica persistente dell'host, cleanup globale,
scrittura dei registri comuni o simulazione delle review.

## Una consegna, al risultato

Scrivi `implementation/report-r015.md`, evidenze sotto
`evidence/implementation-r001/completion-r001/` e aggiorna il tuo checkpoint
`handovers/implementation-r001.md`. Non aspettare un ulteriore mandato per
continuare fix e verifiche. Il report deve distinguere criteri dimostrati,
falliti e non eseguiti, con comandi/input/output/costi e limiti reali.

Quando il lavoro autorizzato è concluso, consegna al supervisore il risultato
e la matrice dei criteri. Se un vincolo sostanziale resta irrisolto, descrivilo
concretamente dopo aver completato il lavoro indipendente. Non consegnare una
sola request di freeze. Il supervisore preparerà snapshot comune e due prompt
di review quando il risultato è revisionabile; le review indipendenti e
l'arbitrato decideranno GO/chiusura oppure fix nella stessa run.
