# Checkpoint reviewer Claude — implementazione r012

2026-10-11 Europe/Rome. Reviewer Claude/Anthropic, Claude Opus 5.5 (`claude-opus-5-5[1m]`), Claude
Code nell'estensione VS Code. Effort e ID della chat non esposti. Chat nuova e indipendente: nessun
accesso a review, evidenze o checkpoint ChatGPT r012 (la cartella è comparsa solo come nome in un
elenco), né a un arbitrato r012. Nessuna delega.

## Stato

**Review completata: NO_GO.** Report in
[review-implementation-r012-claude.md](../reviews/review-implementation-r012-claude.md); evidenze in
`evidence/review-implementation-r012-claude/`:

- `entry.json`, `freeze-receipt.json`, `cases.py`, `probe.py`, `rrun.py`, `check.py`;
- `check-005.json` e `check-005-corrections.json`;
- `out-probe-005/` con `summary.json`, che include gli AST Pandoc;
- `R-probe-00{1..5}/`;
- `reader-v11-v12.diff`, `verify-final.json`, `budget-final.json`.

**Rilievi bloccanti.**

- **CLA-R012-I001 (media).** Sessione della prima unsafe: `attrs()` solleva `Unsafe` direttamente,
  scavalcando la prima violazione già differita. Per heading, div, fence e figure con `onclick`
  successivi a un'img evasiva il detail diventa `Active markdown attribute` invece di `../first.png`
  (A1–A4, v12 4/4 errati, v11 4/4 corretti). È una regressione r012. Nella stessa classe c'è un caso
  ereditato: in A7 l'attributo span viene deciso dopo la guardia HTML.
- **CLA-R012-I002 (media).** Il walk dei container accetta il fence tilde dopo una riga di testo e il
  fence non chiuso. Pandoc 3.1.3 non li apre e produce invece RawBlock con lo script. Esito v12
  COMPLETE con lo script nascosto (B1–B3, quote/list/note) e asset sicuro perso con COMPLETE (B4),
  dove v11 dava UNSAFE o proiettava l'asset. È una regressione r012. Al livello superiore (B7) lo
  stesso caso è ereditato; il fence non chiuso (B9) dà UNSUPPORTED invece di UNSAFE.

**Osservazioni non bloccanti.**

- **CLA-R012-O001:** `_guard_coverage` serializzato perché `located()` non lo rimuove; è l'origine
  delle 640 occorrenze in 56 osservazioni.
- **CLA-R012-O002:** ordine tra reader e audit (A8), ereditato.

**Chiusure confermate dalle sonde:**

- F50: C1/C3 con formula e coordinate char/UTF8, C2 non chiuso unsafe, C4 token senza prestito;
- F52 sui discriminanti GPT: A9/A10, e A12 con visite una volta in ordine sorgente;
- F51 per il raw `{=html}` esatto: C5–C7, D1, C8, C9.

## Modalità, prove e limiti

- **Attesi.** Fissati prima dell'esecuzione: `cases.py` SHA256 `9a715a2d…d284`, freeze 13:24:34Z.
- **Esecuzione.** probe-005 sotto R con interprete stage-s005 (managed 3.12.13).
  - Gate: origin PASS, inventory s005 e propria PASS, snapshot tecnico proprio MATCH prima e dopo.
  - Isolamento: Firejail netnone, figli reaped, input invariati.
  - Pandoc solo per microparsing AST da stdin.
- **Fallimenti conservati.**
  - probe-001: `--bundle` senza comando.
  - probe-002: inventory s004 STALE sul PROTOCOL, per costruzione.
  - probe-003: mio errore di `sed`.
  - probe-004: mio `mkdir`.
  - Tre errori dei miei oracoli, corretti in un file separato: A13 atteso sbagliato, somma delle visite
    di due `observe` in A12, forma RawInline invece di RawBlock nella premessa Pandoc.
- **Non rieseguiti.** Suite, build, rescore, CLI, H01–H12, S/B/I, 64 catene. Prove autore ricevute e
  lette nei campi riassuntivi.

## Identità e costi

- **Comune.** `implementation-r012`, SHA256 `a8ad87d3…8625`, impronta `c5c7eb56…8378`.
- **Verify d'ingresso.** 13:07:31Z, MATCH, exit 0, 46,42 CPU-s.
- **Verify finale.** Dopo report e evidenze, 13:30:30–13:31:07Z: **MATCH**, exit 0, 36,53 CPU-s
  (`verify-final.json`).
- **Git.** `git diff --check` exit 0; indice vuoto; HEAD/dev `80c31da4…`.
- **CPU.** Costo proprio misurato 102,6 CPU-s; visibile lower 16783,64 = ledger del supervisore
  16681,04 + costi propri, senza i costi concorrenti del reviewer ChatGPT.
- **Disco e memoria.** Evidenze circa 10,1 MB allocati più snapshot tecnici 0,3 MB. Picco RSS
  campionato 389615616 B.
- **Rete, acquisizioni, invii.** 0; €0.
- **Limiti.** Nessun processo pendente. Stop CPU/disco/RAM non raggiunti; nessun upper certificato.
- **File scritti fuori dalla cartella evidence.** Snapshot tecnici propri
  `snapshots/review-r012-claude-probe-00{2..5}.json`. Non sono approvazioni né un nuovo comune.

## Prossimo ruolo

Il supervisore riceve entrambi i report reali e arbitra. Non ho modificato prodotto, comune, stato,
arbitrati o output di altri ruoli. Nessun Git, cleanup, fase 2, motore o nuova V.
