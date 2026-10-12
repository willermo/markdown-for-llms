# Checkpoint reviewer Claude — implementazione r008

2026-10-10 Europe/Rome. Reviewer Claude, Anthropic, Claude Opus 5.5 (`claude-opus-5-5[1m]`),
Claude Code in VS Code. L'ID chat non è esposto. Chat nuova e indipendente, senza deleghe.
Prompt `prompts/29-review-implementation-r008-claude.md` (`b3f8d7ba…07e8`).

## Input

- Comune `snapshots/implementation-r008.json` (`2f6d690a…4152`), impronta `448b854a…7eda`, 325 file
  e 76333 artefatti. Verify prima (13:22:36Z) e dopo le prove (13:44:46Z): MATCH, exit 0.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev/merge-base `80c31da4…dc95`; indice vuoto.
  Piano ed errata coincidono. Reader v8 `817852a8…`; reader r007 ricevuto `5d443387…`.
- Letti: arbitrato r007, prompt27, addendum risorse con la nota utente, review/checkpoint/tracce
  r007, report/checkpoint r008, PROTOCOL, guide, report/dati r008, receipt finali. Codice: reader
  integrale con il diff r007→r008, audit, observe_bundle, PassiveHTML e test r008.
- Non letti: la review ChatGPT r008 e le sue evidenze; nessun arbitrato r008.

## Output

- Report: [review-implementation-r008-claude.md](../reviews/review-implementation-r008-claude.md).
- Evidenze in `evidence/review-implementation-r008-claude/`:
  - `entry.json`, `static-checks.json`, `reader-r007-r008.diff`;
  - sonde `probe*.py` con attesi congelati (`expectations-frozen*.json`) e runner `rrun.py`;
  - receipt `R-*`, output `out-*`, `input-bytes-*`, `benchmark-inputs-*`, `preflight-*`, `target-*`;
  - verify prima e dopo.
- Snapshot tecnici propri: `snapshots/review-r008-claude-{f35-001,f35-002,main-001,r007-html-001,
  extra-r007-001,extra-r008-001}.json`. Attribuiscono le prove, non sono GO.
- Non ho modificato input ricevuti, registri comuni, snapshot comuni o storici, checkpoint del
  supervisore o file di altri ruoli.

## Esito

**NO_GO.**

Chiusi:

- F35 (C01): terminazione dimostrata dalla monotonia delle transizioni e da 8 sonde R;
- F36 (C02) e F37 (C03): le mie varianti indipendenti, inclusi T1/T2/T3 esatti, sono conformi;
- C07: 64/64 coerenti, osservazioni e metriche r008 = r007, 4/28 per set, numeri per formato.

Due rilievi C04, entrambi regressioni r008 misurate a runtime contro i byte r007:

- **CLA-R008-I001**, media, bloccante. `html_nodes` non chiama più `html(raw)`. Dopo un abort del
  tree (attributi duplicati, profondità >16) script, `onerror` e `../` danno `UNSUPPORTED_RESOURCE_SYNTAX`
  invece di `UNSAFE_REFERENCE`. Casi D1/D2/D3/D6; con r007 tutti UNSAFE.
- **CLA-R008-I002**, media, bloccante. Nella nuova `incomplete_tag_token` ogni apice apre uno stato
  quotato. `<p title=it's><img …>` seguito da un commento diventa un `incomplete-start` inventato:
  l'img non è validato né proiettato. Caso D7 UNSUPPORTED invece di UNSAFE; con `p.png` si perde il
  riferimento immagine reale.

F38 (C06) è parziale: la documentazione afferma che ogni apertura passa da PassiveHTML, ed è da
rettificare insieme al fix. Nessun processo, file o fetch aperto in modo indebito; status ed exit
restano INCOMPLETE/2.

FAIL conservati: `s007-conserve-001` (S007 STALE per i 4 registri del supervisore) e `f35-001`
(mio errore di preflight, corretto). Non ho trasferito PASS e non ho ripristinato input.

## Costi e processi

- Review dalle 13:22:31Z a circa 13:47Z.
- CPU misurata 81,67 s (lower bound) prima del verify finale. RSS campionato massimo 291812 KiB.
  Disco circa 7,7 MB tra evidenze e snapshot. Rete, invii e servizi: €0.
- Wall cumulativo circa 54,63 h. Lo stop 2026-10-11T06:37:31Z è lontano (margine circa 16,9 h).
  CPU in contabilità circa 11737 s, sotto lo stop di 19800 s.
- Processi propri pendenti: 0.
- Verify finale dopo la scrittura di report e checkpoint: 13:48:31Z → 13:48:54Z, MATCH, exit 0
  (`verify-after-final.*`), 23,14 CPU-s. CPU propria totale 104,81 s (lower bound). Le evidenze
  finali occupano 7443208 B; indice ancora vuoto.

## Prossimo ruolo

**Supervisore**: ricezione di entrambi i report r008 e arbitrato reale. Git resta manuale, solo
dopo un eventuale GO finale.
