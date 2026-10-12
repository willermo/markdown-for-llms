# Checkpoint reviewer Claude — implementazione r010

2026-10-10 Europe/Rome. Reviewer Claude, Anthropic, Claude Opus 5.5 (`claude-opus-5-5[1m]`),
Claude Code in VS Code. Effort e ID della chat non sono esposti. Chat nuova e indipendente, senza
deleghe. Prompt `prompts/36-review-implementation-r010-claude.md` (`f8407128…a250`).

## Input

- Comune `snapshots/implementation-r010.json` (`cfc1e56f…3945`), impronta `b36a7ec2…f9bf4`,
  331 file e 107203 artefatti. Verify prima (18:54:02Z): MATCH, exit 0.
- Branch `feature/run-a002-corpus`; HEAD, dev, origin/dev e merge-base `80c31da4…dc95`; indice
  vuoto. Coincidono piano, errata, fix-context-r009 e delivery-001. Reader v10 `382d6be9…`, v9
  fisico `a8ffa9eb…`.
- Letti: arbitrato r009, prompt34, addendum risorse e nota utente, review e checkpoint r009 di
  entrambi i reviewer, report e checkpoint r010 dell'autore, PROTOCOL, guide e report r010.
  Codice: reader v10 integrale e diff, audit, observe_bundle, safety, metrics, test r010, matrix.py,
  CLI dell'autore.
- Non letti: review, evidenze o checkpoint ChatGPT r010; nessun arbitrato r010. Gli snapshot
  `review-r010-chatgpt-*` sono comparsi solo in un elenco e non li ho aperti.

## Output

- Report: [review-implementation-r010-claude.md](../reviews/review-implementation-r010-claude.md).
- Evidenze in `evidence/review-implementation-r010-claude/`:
  - gate e controlli host: `entry.json`, `static-checks.json`/`-002`, `derived-check.json`,
    `stage-equivalence-002.txt`, `reader-v9-v10.diff`;
  - casi e attesi: `probe_cases.py`, `expectations-frozen.json`, `expectations-frozen-002.json`;
  - collector e runner: `probe.py`, `cause.py`, `cli.py`, `perf.py`, `rrun.py` (versione
    precedente `rrun-001.py`), `cmd.sh`;
  - receipt `R-*`, output `out-*`, `probe-check.json`, `findings-detail.md`, `costs.json`;
  - verify prima e dopo.
- Snapshot tecnici propri: `snapshots/review-r010-claude-{v10-001,v9-001,cause-001,cli-001,perf-v10-001,perf-v9-001}.json`.
  Attribuiscono le prove, non sono GO.
- Non ho modificato input ricevuti, registri comuni, snapshot comuni o storici, checkpoint del
  supervisore o file di altri ruoli.

## Esito

**NO_GO.**

Chiusi nelle mie prove indipendenti:

- F44: falsi unsafe GPT COMPLETE/PASS/5; varianti F1–F5 literal COMPLETE; F6 con prefisso F36
  UNSUPPORTED invece del falso unsafe;
- F43 nei sei host della matrice e in heading, caption, didascalia di figura, enfasi e label
  di link (T1–T5);
- F36: B1–B4 UNSUPPORTED, documentato;
- C07: 64/64 coerenti, catena verificata, 4/28 per set, 56/8 bound.

Rilievo **CLA-R010-I001**, media, bloccante. In modalità inline `markdown_lexemes` applica la regola
dei metadata heading (`observations.py:215–217`) a ogni riga logica `# … {…}`. Il tag reale dopo un
backtick o un `$` non chiusi non diventa evento e il fallback lo nasconde con COMPLETE. La CLI reale
dà run 0 / evaluate 5 per list, cella e risorsa sicura persa. Coinvolge list, quote, note, cella,
caption e heading. È ereditato (v9 identico); la regressione di classificazione UNSAFE →
UNSUPPORTED in H2 è introdotta da r010. Viola C01/F43, C03/F45 e le dichiarazioni del PROTOCOL r010.

Osservazioni non bloccanti:

- K1: lunghezza dei run di backtick rispetto a CommonMark;
- `{=html5}` e `{=html }`: Pandoc non eseguito;
- costo del walk inline: v10 circa 5 volte v9 a k=4000;
- readerID `observations-v1` nel fallback unsafe.

FAIL ed errori propri conservati:

- nome B4 nel checker statico;
- `audit_sha256` non escluso in `derived_check`;
- prima equivalenza degli stage con percorsi collassati;
- KeyError del mio gold `figure` sul v9.

Nessun PASS trasferito, nessun input ripristinato.

## Costi e processi

- Review dalle 18:54:02Z. CPU misurata 112,23 s lower bound; maxrss esterno 324756 KiB.
- Disco nuovo: evidenze circa 10,6 MB apparenti (13,2 MB allocati) più 6 snapshot propri per
  444308 B. Rete, acquisizioni, invii: 0, €0.
- Wall cumulativo 216435,3 s alle 19:14:09Z; stop 2026-10-11T06:37:31Z, deadline 09:37:31Z.
- Contabilità CPU visibile circa 14114,89 s lower (finale del supervisore più la mia), senza il
  reviewer ChatGPT r010.
- Nessun processo pendente.

## Verify finale

Dopo la scrittura di report, checkpoint, `findings-detail.md` e `costs.json`:
`python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r010`,
cwd radice del repository, 19:17:28Z → 19:18:00Z: **MATCH**, exit 0, stderr vuoto, CPU 31,70 s.
Non è inclusa nei 112,23 s sopra: totale proprio misurato 143,93 s lower bound. HEAD `80c31da4…dc95`,
indice vuoto, `git diff --check` exit 0, link locali del report e del checkpoint presenti. Output in
`verify-after.*` nella cartella delle evidenze. Nessun processo pendente.

Questa sezione è stata aggiunta dopo il verify: modifica solo questo checkpoint, che è escluso dal comune.

## Prossimo ruolo

**Supervisore**, per la ricezione di entrambi i report r010 e l'arbitrato. Git resta manuale,
solo dopo un GO finale.
