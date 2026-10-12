# Checkpoint reviewer Claude — review implementazione r004

2026-10-09 UTC. Reviewer Claude (Anthropic, Claude Opus 5.5, Claude Code VS Code; ID chat non
esposto), chat nuova e indipendente, prompt17 SHA256 `64a1517c…6129c0`. Nessuna sostituzione o
delega. Review ChatGPT r004 e relative evidenze non lette: il report è comparso durante la review e non è stato aperto.

## Input

- Snapshot comune `snapshots/implementation-r004.json` (SHA256 `7105ec0e…d768c7`, impronta
  `3976b269…064edf`, 313 file/41997 artefatti): verify prima e dopo MATCH/exit 0.
- Branch `feature/run-a002-corpus`, HEAD/dev/origin-dev/merge-base `80c31da4…`, indice vuoto.
- Delta: 8 file autore (observations, metrics, PROTOCOL, test r004, due guide, report e dati
  r004) più 4 documenti del supervisore; 38018 artefatti precedenti invariati.
- Report/CP autore r004, arbitrato r003, prompt15, report r003 propri e ChatGPT ammessi, prove
  autore (rescore-001, final-checks, contracts, receipt test), stage s008.

## Output

- Report [review-implementation-r004-claude.md](../reviews/review-implementation-r004-claude.md).
- Evidenze `evidence/review-implementation-r004-claude/`: `commands.md`, `verify-entry.txt`,
  `verify-after.txt`, `delta-check.txt`, `own_rescore_check_r004.py` e risultati, `gate/`
  (hash gate prima/dopo, preflight s008 con S STALE diagnosticato), `probe/` (sonda v1/v2,
  builder, target, inventari, preflight), `R-probe-001` (FAIL conservato), `R-probe-002`
  (PASS), `probe-results-002.json`, `snapshot-probe-00{1,2}.txt`,
  `own-snapshots-verify-after.txt`, `work/`.
- Snapshot tecnici propri `snapshots/review-r004-claude-probe-001.json` e `-002.json`
  (impronta = comune). Nessuno snapshot ricevuto modificato.

## Esito

**NO_GO**. Rilievi:

- CLA-R004-I001, media, bloccante: heading HTML `<h1>`–`<h6>` (block, in wrapper, in fence
  `{=html}`) → TypeError in `observations.node` (prefisso Markdown `#` obbligatorio), crash di
  observe e audit, exit 4. Introdotto in r003 (F13), ereditato in r004. Nessun falso PASS.
- CLA-R004-I002, bassa, da chiudere con I001 (statico): h1–h6 e `code` non sono host di
  html-inline-v1 → table/figure in heading appiattite senza unsupported.
- CLA-R004-I003, bassa, non bloccante: `text_lower_bound` può superare la misura completa o
  condizionata (DOCX 62>60, EPUB 54>53, D1 0,667 contro 0). Serve chiarirne il dominio in
  PROTOCOL/guide/report/dati.

Stato dei fix: F17, F18, F19, F20 (nel dominio dichiarato) e F21 chiusi. F16 preservato
(64/64 identici a r003, bound ricalcolato in modo indipendente). Ramo HTML F09/F13 aperto
per I001.

## Processi e risorse

Nessun processo proprio pendente. Due invocazioni R terminate (circa 7 s di wall, CPU dei
workload 0,45 s). Checker stdlib 55 s di wall. Wall della review circa 20 minuti, costo €0,
nessuna rete né acquisizione. Directory tmp/socket `/tmp/claude-1000/a004cla` vuota.

## Prossimo ruolo

**Supervisore**: ricezione dei due report r004 (ChatGPT e Claude) e arbitrato reale. Questa
review non è un arbitrato né un GO. Nessun fix, Git o aggiornamento dei registri comuni
eseguito.
