# Tracce statiche — review Claude r014

Reader `scripts/benchmarks/observations.py` SHA256 `613aaa99…5cefd`, adapter
`safety.py` `974656f6…2b2`, audit `audit.py` letti dal working tree che il
verify del comune r002 dà MATCH. Nessun import del prodotto da parte mia:
la sonda R `probe-001` si è fermata al preflight (sotto). Le premesse Pandoc
qui sotto sono inferenze dalla semantica del reader Markdown pinned e dagli
AST **ricevuti** citati; nessun AST nuovo è stato eseguito da me.

## I001 — host lista forza la chiusura del paragrafo

Input minimo: `- item\n# H {k="<img src=../x.png>"}\n`.

1. Scanner radice, riga `- item`: `block['container']='list'` (404–406),
   `physical_host='list'` (407), `paragraph_open=True` poi forzato `False` (412).
2. Riga `# H {…}`: `before=False` → `block['heading']=True` (399);
   `trailing_metadata(line)` accetta il gruppo (414–416), intervallo saltato.
3. Producer radice: l'item lista consuma solo righe indentate a 4 spazi
   (1249–1252); la riga successiva usa `block['heading']` (1264–1266),
   `peel_attrs` (577–580) → `trailing_metadata` → `attrs` (554–575): valore
   `k='<img src=../x.png>'`. `read_inline('H')` non produce eventi.
4. Nessun `guard_event`; nessun nodo unknown; `audit` (40–43) → COMPLETE.

Varianti dallo stesso stato (412): `- item\n::: {k="<img src=../x.png>"}\nBody\n:::`
(div con metadata accettati), `- item\n[^n]: ~~~\n    <img src="../x.png">\n    ~~~`
e `- item\n> ~~~\n> <img src="../x.png">\n> ~~~` (container ammesso con fence
iniziale, corpo code), `- item\n# H {k='<img src="p.png">'}` (asset sicuro
perso), `- item\n# H {onclick="x"}` (falso UNSAFE, classe O001). Lo stesso vale
per le righe dopo continuazioni indentate a quattro spazi (407–408).

Premessa Pandoc 3.1.3 (non eseguita da me): l'item lista continua in modo
lazy con ogni riga non vuota che non sia nuovo item o fenced code; il testo
dell'item è poi letto con `+blank_before_header`/`+blank_before_blockquote`.
AST ricevuti a supporto, `evidence/implementation-r014/baseline-001.json`
(SHA256 `b2506f5f…f0d1e`, legato nel comune): K2c = `BulletList`, poi
`CodeBlock` radice (l'unica eccezione, motivo della regola 412);
`heading-attached`/K1h = riga `# …` dopo testo resa `Str "#"` nel `Para`;
HC2 overlay/K4 = HTML in graffe non metadata reso `RawInline`.
Nessuno dei 157+2 casi ha una lista seguita da heading/div/quote/note.

## I002 (condizionato) — opener div non chiuso

Input: `::: x\n~~~\n<script>x</script>\n~~~\n\nEnd.\n` (U1), con img sicura (U2),
heading con valore quoted (U3); controllo chiuso (C1).
`div_at` (142–150) accetta l'opener dalla sola riga; 401 lo registra,
`paragraph_open=False` (409); la fence successiva è accettata (381) e il
producer crea `code` (1246). A fine documento 1313 aggiunge
`node('unknown',…,reason='unclosed div')`: raw vuoto, nessuna lista `unknown`,
nessun fallback di guardia (1189–1192), quindi `audit` non vede
`resource_syntax` → COMPLETE. In un container (`- ::: x\n    ~~~…`) manca
anche quel residuo. Premessa: in Pandoc un fenced div senza chiusura non è
Div (manuale: il Div termina con una riga di due punti; parser con chiusura
obbligatoria), quindi `::: x` resta paragrafo e `~~~` non lo interrompe.
Non verificata da me: gli AST ricevuti `grammar-21–23` coprono solo div chiusi.

## O001/O002 (condizionati)

`ATTRIBUTE_TOKEN` (42) accetta `+` in id/classi e classe con `_` iniziale;
`grammar-001` prova `#1`, `.1`, `_x=x`, `a=`, `#a.b`, `.a.b`, non `+`/`._c`.
Heading ATX con 1–3 spazi iniziali accettata da 399/1264 (` {0,3}#`),
ereditata da V13; nessun AST ricevuto la copre.

## Sonda R non riuscita (errore mio, conservato)

`launch.py` → `run_offline.py` con preflight `stage-s002/preflight.json`:
`check_preflight` exit 2, `Stage non MATCH: STALE: snapshot diverso;
campi modificati: worktree_sha256, files`. È lo STALE generale S002 dopo i
registri del supervisore, già indicato dal prompt: avrei dovuto creare un
preflight corrente. Il comando non è partito, nessun import del prodotto,
binding MATCH prima/dopo, `inputs_unchanged` true, socket pulito. Una
ripetizione (nuovo preflight o R solo Pandoc, circa 3 CPU-s più il verify
finale) supererebbe prevedibilmente il piano di 10 CPU-s: non eseguita.
