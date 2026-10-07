# Implementazione r001 — lock no-build con metadati EbookLib, prompt40

2026-10-06. **WAITING_FOR_SUPERVISOR_RECEPTION**. Implementatore esclusivo.
Il risultato ammesso è **FAIL sostanziale del lock**: uv0.10.10 richiede wheel
utilizzabili per EbookLib0.18 sotto --no-build. La dichiarazione statica ricevuta
non rende quella sdist ammissibile nella chiamata osservata. Nessun lock prodotto,
ramo HTTPS e check **NOT_EXECUTED** secondo le condizioni congelate.

## Identità e input

Request126813byte SHAb8f84310c594b6659d6d95d6bab5e72c38696be38bb019f5c03584d45e76dded;
scope17544byte SHAa0a92ad475d189bc7c053eec8750ce21f2985c25ef301a29ffb667efceb17ccc;
manifest109116byte SHA f8a87df3212b28ae25b03044f2ce55bc9cb90230e4f3043670c69a58f1199b9b.
Piano/arbitrato r003 SHA462dc1f4bca35d15df1a1005c60d8418d8c25540569fc6ab35e0277a2066f38b /
f14ebe2fe15957ce3e8754bbc3f1111659875760e3666afeb10b562a4540df0d invariati;
r008 aa835d8f2b4fcecee8a4cebe65710984a553476d6e0da82ef78ec328e0122275.
Stage s010 MATCH ingresso, prima/dopo comando e consegna; worktree
0e6c17aeff05d9f6887bac753fe5658fedd830730b0570616075d6606d11927d,123file/310artefatti.
Request.proposed_operations==scope.operations, confrontati strutturati.
Nuova copia supervisor project esistente non ricreata:16input preservati,
15byte-identici, solo TOML metadata ebooklib0.18/lxml/six; confronto semantico
indipendente conferma assenza di altri delta. Root/vecchia copia/constraints/pin/
Marker/indici/extras/conflicts immutati; nessun root uv.lock/.venv.
S009 storico solo per cinque metadata; mai verificato contro nuovo worktree.
Checkpoint/report-r005/delivery d'ingresso salvati byte-identici in before/.

## Chiamata nativa e causa

Una chiamata tool distinta lock-offline, sessione7493 raccolta, exit1,
durata0.10781s, stop_reason null. Launcher assoluto pyenv3.12.3 -I -B, ambiente
esterno chiuso pubblico; uv riceve esattamente19variabili scope, shell=False,
close_fds=True, no-build/offline/managed3.12.13 assoluto/no-python-downloads,
keyring-provider disabled. argv/cwd/env/tempi/PID/log nelle receipt operative.
Receipt del launcher d'autore distinta dai log raw uv; uv non produce una
receipt strutturata propria, nessuna inventata.

Il trace trova static pyproject della copia corretta, pre-fork universale in3
risoluzioni, runtime+dev+marker-server+CPU/cu126. Il FAIL è nel ramo darwin,
python_full_version>=3.12, marker-cu126 incluso/marker-cpu escluso:
«ebooklib==0.18 has no usable wheels»; requisito Marker-full1.10.2
ebooklib>=0.18,<0.19; hint nativo «Wheels are required for `ebooklib` because
building from source is disabled ... --no-build». Non seguo l'hint di restringere
environments, né modifico il grafo per far passare il resolver.

[Classificazione — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
con righe/esatto excerpt: conflitto della policy sorgente, non cache metadata
mancante. La condizione ONLY_IF_OFFLINE_FAIL_IS_SOLELY_MISSING_CACHED_METADATA
è falsa; quindi nessun lock-metadata-online né lock-check-offline avviato,
directory di quelle receipt non create. Nessun retry, nuovo backend, acquisizione
o input alterato. Il FAIL resta autentico, non attribuito a timeout/sandbox.
Non dichiaro che TOML sia malformato o ignorato: il limite osservato è l'ammissione
della distribuzione. L'assenza di lock impedisce audit package/fonti/marker/hash
sdist reali: registrati NOT_EXECUTED, nessuna struttura lock fabbricata.

## Guardie e costi

Startup bootstrap e managed prima/dopo: enumerati site-packages e stdlib noti,
zero .pth/sitecustomize/usercustomize, receipt pre scritta prima del probe managed.
Prima e dopo ogni avvio uv/managed binary/BUILD20260310/constraints/origine
3.12.13 verificati, -I -B probe controllato, nessun import applicativo/backend.
Config e credential scope solo assenza, nessun contenuto letto/stampato;
nessun proxy/auth/PYTHONPATH/coverage o override ereditato. Keyring disabled.
Baseline target/lock1892identità/211directory PASS via sola verify helper, non
main storico. Nessuna reinstallazione o build della dipendenza/prodotto.
Cache originale riutilizzata nativamente: confronto prima/dopo zero aggiunte,
rimozioni o modifiche hash/size/type/allocated. Atime/mtime non inventariati;
«Found fresh/stale response» sono letture cache in modalità offline, non HTTP.
Nessuna nuova rete/body, wire non misurato, costo remoto zero.
Processo uv osservato proprio raccolto; nessun figlio applicativo osservato,
probe native ammesso ma non osservato in quel brevissimo campione. Nessuna prova
R/net=none o D4 del prodotto attribuita a questo resolver; quelle restano future.

Ledger r008 run+.venv-python incluso storico/cache/tmp/log/dir/linkmetadata
lstat senza follow, hardlink perpathname. Hentry523460608byte,32MiB tranche,
24attività8registri+16esterni, riserva residua contata una volta, max1GiB/stop896,
libero1GiB. Gate attività conservativo usa Delta intero; registri separatamente
8MiB. Primo comando dura meno del periodo0.5s: un campione iniziale, nessun
picco istantaneo certificato. Guardie finali/costi consegna in final-checks.json.
RLIMIT_FSIZE32MiB/file, log1MiB/stream (cattura pipe con cap), JSON8MiB.
Cap HTTP atomico non attestato; ramo online non eseguito. Nessun cleanup agente,
nessun cambio host/profili/rete/daemon/privilegi/sysctl/AppArmor.

## Soluzione proporzionata da ricevere

Letti in sola lettura documento risoluzione uv0.10.10 ricevuto e help lungo pinned.
Il meccanismo dependency-metadata evita build metadata; non dimostra che una
sdist registry sia eleggibile con global no-build. Il risultato reale smentisce
quell'assunzione per input/scope correnti. Non estendo metadata ad altri package.
Con cache/fonte/policy attuali non è dimostrato un metodo supportato che produca
il lock richiesto mantenendo tutte le condizioni: questo scostamento torna al
supervisore con esito concreto, non un altro mandato di sola preparazione.

[Nuova richiesta non autorizzata — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
formula la possibile preparazione nativa della **wheel cache registry** della
sola EbookLib0.18: uv pip install no-deps/require-hashes, target esclusivo soltanto
dipendenza, backend-env84 esistente/no-build-isolation-package ebooklib,
constraints e cache originale, poi stesso lock offline --no-build/check.
È distinta dal prime dry-run metadata-only precedentemente rifiutato; quel
JSON resta immutato/NOT_AUTHORIZED. Nessuna di queste operazioni avviata.
Il tentativo proposto installerebbe esclusivamente in scratch della dipendenza,
non baseline/prodotto, e popolerebbe cache soltanto con uv nativo, mai manuale.
Riuso registry cache con no-build/grafo universale da provare, non garantito dal
solo help che menziona cached built wheels. Fonte PyPI/hash sdist invariati.
Eccezione backend+HTTPS assente da r008: supervisore deve riceverla o rifiutarla,
con guardie/privacy/D4 e costo distinto32MiB+16esterni; comando/input/env/target/
output/hash/limiti/incertezze nel JSON. Non aggiro wrapper offline o sandbox.
Se mantiene divieto di quell'esecuzione, il blocco resta; nessuna rimozione
generale no-build, cambio Marker/pin/indici/ABI o wheel locale promossa a PyPI.

## Stato e consegna

[Completion — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[delivery — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[checkpoint — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md). File reali ignored/managed/
cache/work e working tree devono essere trasferiti; manifest non è backup.
Git feature/run-a001-uv HEAD/dev/base66ba822, indice vuoto; tracciati invariati,
nessun commit/merge/push/promozione/deploy. Nessun registro comune/snapshot.
S009 byte reproducibilityFAIL/raw hooknonosservato/lacuna bootstrap prebuild
conservati; nessuna ripetizione retroattiva. Baseline completa caratterizzazione
PASS e suiteFAIL62pass5fail/perdite legacy ricevute, non ripetute.
Lock attuale FAIL, check NOT_EXECUTED; product S/B/I/E e V0–V9 pertinenti aperti,
V7/V8 mandate/costi pesanti distinti, V10/V11 escluse, local_marker non convalidato.
Tests-s001 non pronto, nessun PASS applicativo/ABI/multi-ABI/GO finale;
due review reali indipendenti ChatGPT/Claude e arbitrato finale ancora necessari.
Prossimo supervisore prompt05+r008/checkpoint corrente. Nessun servizio da attendere.
