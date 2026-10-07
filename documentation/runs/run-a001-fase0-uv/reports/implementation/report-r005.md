# Implementazione r001 — due build native EbookLib, prompt39

2026-10-06. **WAITING_FOR_SUPERVISOR_RECEPTION**. Implementatore esclusivo.
Risultato della sola dipendenza: entrambe le build native offline exit0,
R/wrapper/namespace e wheel PASS. Riproducibilità byte **FAIL**: hash esterni
diversi, timestamp ZIP diversi; tutti i15 payload sono byte-identici, METADATA,
WHEEL e RECORD inclusi. Nessuna normalizzazione degli archivi o della versione.

## Identità e perimetro

Manifest83329byte SHA85649c577a0715100f19304a48b660a0054efd65ad7fe66f5ef4889167ed2009;
request103249byte SHAdbb4fe8d9aa077d439a0670da0e6f1eb42e98b07f52a3771d653b4813cd86365;
scope27842byte SHAc5923c454a398b91bf05d719b2486816d7292582ac0a443e20bc20da33525037.
Piano/arbitrato r003 SHA462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b /
f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d.
Verifica s009 MATCH ingresso/pre-post di entrambe le chiamate; worktree
518749e5472ae9a69384b3a2255f659c218a441c84a7773fdad8e2b39428b00e,123file/211artefatti.
Addenda r004–r007 applicati; quota precedente conclusa. Baseline-s007 storica
per cinque metadata ricevuti; nessun rerun o richiesta MATCH contro quella storia.
Operazioni request/scope identiche, eseguite strutturate senza shell/merge env.
Git feature/run-a001-uv HEAD/dev/base66ba822, indice vuoto, nessun delta tracciato
introdotto; root uv.lock/.venv assenti. Report-r004, checkpoint e delivery di
ingresso conservati byte-identici in resume-package-s009/before/.

## Operazioni realmente raccolte

Una chiamata per a/sessione61022 e b/sessione62602, entrambe raccolte exit0,
nessuna ripetizione. Bootstrap assoluto pyenv3.12.3 -I -B; R baseline padre/figlio
target nuovo, Firejail noprofile/net=none/D4. Inline execve congelato esegue uv
0.10.10 nello stesso namespace, ambiente nativo chiuso19variabili.
R/inside/wrapper PASS, comandi/descendenti osservati nelle netns
a net:[4026533883], b net:[4026533882], distinte dall'host.
I processi effettivamente osservati includono uv, probe interprete -I -B,
hook Python -c get_requires e build_wheel, senza flag -I/-B inventati sugli hook.
Python3.12.13 managed/BUILD20260310, origine/startup, uv, constraints19byte,
archivio115484byte e wheel backend818216byte verificati prima/dopo ogni build.
Find-links offre soltanto setuptools84 come wheel top-level; gli altri file
non-archivio sono ignorati dal nativo e riportati nel trace.
Trace: DEFAULT_BACKEND effettivamente setuptools.build_meta:__legacy__,
requisito standard setuptools>=40.8.0 con constraints84; unica versione risolta
e installata nell'isolamento temporaneo: setuptools84.0.0 dalla wheel locale.
La dicitura generica «Downloading and building requirement» non è evidenza
di rete: offline/no-index e net=none, fonte locale e supply chiusa.
Native reflink fallback e warning byte-compiling disabled conservati nei log.
Nessuna lxml/six/import EbookLib o installazione nel prodotto.

Raw get_requires_for_build_wheel.txt rimosso dal lifecycle uv e **non osservato**.
Nessun ulteriore requisito selezionato nel trace; non dichiaro raw list=[].
Receipt degli hook non fabbricate: le prove sono trace/inside native con recipe
Python -c e wheel autentica. uv ha rimosso venv/hook tmp propri; inventario
residuo conservato, nessuna cleanup agente. Stato dei tempi tra campioni ignoto.

## Wheel e contenuti

Entrambe38882byte: a SHA1b519cfb43c77aa1ccdbcac0761a6076812328bfcb82f38fbc371c43ddd7ec01,
b SHA133d01e16e1e74451a1a645f057f04752ccb9327d0a39d4bf29eea60e1dc7717.15membri ZIP sicuri, niente traversal,
link/speciali/duplicati/cifratura. RECORD completo e hash/size dei14membri
non-RECORD verificati; RECORD self-reference vuota conforme.
Name EbookLib, Version0.18, METADATA2.4, Requires-Dist lxml/six;
Generator setuptools(84.0.0), tag py3-none-any/purelib true, nessun entrypoint.
I9moduli package e2licenze coincidono esattamente col sorgente auditato.
Costante interna0.18.1 conservata; nessun pyc nel payload.
Tutti15membri a/b sono byte-identici; date_time varia su15membri.
Non viene dichiarata riproducibilità byte o equivalenza runtime/ABI/prodotto.

## Guardie e costi

Ledger r007 run intera+.venv-python, storico/cache/tmp/log/directory/linkmetadata
inclusi con lstat senza follow, hardlink contati perpathname. Formula H/Delta/
residuo32MiB/riserva16MiB applicata identica a entrambe le build; attività24MiB
controllata conservativamente sul Delta intero, oltre agli output, registri8MiB.
H ingresso513622016byte; incremento massimo **campionato**
13275136byte, sotto24MiB e32MiB. Non è un picco
istantaneo certificato. Gap massimi a0.500096778s,b0.500096721s; guardie tutti
i campioni ammesse. Child120s/outer300s, durate outer3.0825/3.0933s; limite
RLIMIT_FSIZE32MiB, stream1MiB e JSON8MiB rispettati sugli output conservati.
Guardie finali e costi di report/consegna in final-checks.json.
Zero rete/download/costi remoti. Managed/backend non reinstallati nell'ambiente
persistente; installazione della sola wheel backend nella venv temporanea nativa
esplicitamente ammessa. Target+lock1892file/211directory preservati prima/dopo.
Processi osservati propri raccolti, nessun processo altrui terminato; nessun
cambio host, privilegi, profilo persistente, daemon, AppArmor o sysctl.

## Seguito concreto del lock, non autorizzato

Help lungo/versione del binario pinned acquisiti localmente in sola lettura;
sorgente setuptools84 locale letto staticamente e recipe hook uv realmente
osservata. Riferimento Rust pinned ricevuto conservato: non disponibile checkout
Rust locale, nessuna nuova acquisizione di sorgente/rete. Ordine cache lookup
non indipendentemente attestato dal Rust: resta da provare con operazione nativa.
Cache a/b: backend locale/interprete, nessun payload sdists-v9 EbookLib.
Cache originale: simple-v20/pypi/ebooklib.rkyv già presente, nessun sdist/metadata
nativo EbookLib; nessuno spostamento o forgery. Le due build uv build locali
non popolano una cache riusabile per quella fonte registry.

[Richiesta concreta — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md):
uv pip install --dry-run --no-deps --require-hashes su requisito registry
EbookLib==0.18/hash auditato, backend-env84 esistente, no-build-isolation-package
ebooklib, constraints, stessa cache originale. Nessun prodotto installato.
Solo quella chiamata chiede eccezione al divieto backend e HTTPS pubblico per
fonte originale; wrapper corrente net=none non può scaricare. Il metodo uv
supportato non separa fetch e hook offline: il nuovo perimetro rete/isolation/D4
deve essere ricevuto e congelato, nessun aggiramento dichiarato pronto.
Costo proposto32MiB+16MiB esterni, runtime/download backend zero, archivio115484byte,
metadata registry stimati≤262144byte, body non atomicamente limitato dal CLI,
wire non misurato. Questo limite è esplicito nella richiesta e da arbitrare.
Segue tentativo originale universale uv lock --offline --no-build e lock-check
solo dopo exit0. Help ammette riuso di wheel cached; dry-run può produrre metadata
ma non garantisce wheel cached/consumabilità con no-build. Un FAIL di cache o
altro metadata rimane FAIL, niente fallback di fonte/build o nuove acquisizioni.
Input/argv/env/cwd/target/output/constraints e incertezze sono nel JSON.
**NOT_AUTHORIZED, ready_for_execution=false**, nessuna operazione proposta avviata.

## Consegna

[Completion — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[delivery — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[checkpoint — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md). Evidenze reali ignorate e
work a/b/cache/managed devono essere trasferiti insieme al working tree.
Baseline originale ricevuta resta: S.id/67nodeid/201eventi/62pass5fail, perdite
legacy, caratterizzazione PASS ma suite FAIL; nessun test nuovo questa tranche.
Lock storico FAIL, lock-check NOT_EXECUTED, tests-s001 non pronto senza lock/S/B/I;
S/B/I/E/ABI prodotto esclusi. V7/V8 obbligatorie con mandato pesante distinto,
V10/V11/pesi/font/inferenza esclusi, local_marker non convalidato. Due review
indipendenti reali e arbitrato finale ancora necessari. Nessun GO finale,
commit/merge/push/promozione/deploy/cleanup o servizio da attendere.
Prossimo supervisore prompt05+r004–r007 con checkpoint corrente.

## Limite documentale della guardia bootstrap

Alla consegna site pyenv3.12.3 verificato: zero .pth e nessun site/usercustomize.
Non è stata salvata un'enumerazione separata di quello site prima di ciascuna
build: non attesto retroattivamente una coppia completa pre/post del bootstrap.
Controlli pre/post managed/baseline e prerequisiti congelati risultano invece
registrati. Questa lacuna di attestazione resta in completion/final-checks;
non viene sanata rieseguendo build già avviate né trasformata in PASS del gate
completo. PASS delle wheel/backend/namespace effettivamente osservati resta
distinto. Il ricevente deve considerare il limite prima di qualunque seguito.
