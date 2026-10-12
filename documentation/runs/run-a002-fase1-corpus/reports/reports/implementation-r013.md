# Implementazione r013 — consegna completa F54–F57

2026-10-11, ruolo esclusivo implementatore, mandato43. Il fix locale è
implementato e le prove richieste sono completate sui byte identificati
qui. Non è un GO, una review o un arbitrato; nessuna delega o impersonazione.

Reader `observations-v13-ordered-guards-fences`, SHA256 `27d1b5809bc2dca51b75e50222aaab1eef86b2c907732c33537fbb5df20c43ad`.
Feature branch `feature/run-a002-corpus`, HEAD/dev/origin-dev locale e
merge-base ricevuti `80c31da46e5caab990d3596c79c44e524ec1dc95`; HEAD e branch
ricontrollati, indice vuoto. Nessuna mutazione Git, cleanup, fase2, engine,
modello o nuova V. Le modifiche già presenti all’ingresso sono preservate.

## Risultato e prove

| Gate eseguito | Esito e identità |
| --- | --- |
| Ingresso comune corrente | `entry-verify.json`: MATCH, exit0, CPU35.513515s, argv/cwd/stdout conservati |
| Matrice principale285 | V12 132/285 su `baseline-004`, V13 finale285/285 su `after-006`; attesi `cases-005.json`, `check-after-006.json` |
| Attributi LF supplementari16 | `mapping-v12`0/16, `mapping-previous`0/16, `mapping-current`16/16, posizioni char/UTF8 esatte e coda HTML visitata |
| Oracolo sintattico | Pandoc3.1.3 locale:199 AST principali,139 proprietà code ordinarie,60 raw con profilo dichiarato;16 AST attrs LF e un AST storico F29 aggiuntivo |
| Due mutazioni private correnti | `mutation-inputs-002.json`, diff/hashes/copie/output:0/3 ciascuna, detail/visite e asset/unsafe discriminanti |
| Regressioni principali | r012350/350, r011153/153, r010303/303, r009308/308, directory `regression-*-002` |
| Suite veloce prescritta | `R-r013-fast-003`:2997 passed,310 subtests passed; API/packaging/Docker e due smoke radice esclusi |
| CLI OS effettiva | `cli-002` e `cli-metadata-001`:43 casi principali+8 LF/102 run/evaluate, exit/detail/raw/artefatti tutti attesi; writer sintetico dichiarato, gold PROVISIONAL |
| Contratti | `contracts-002`:H01–H08 R PASS;H12± harness1 contro R expected0, wrapper2 e R FAIL deliberati |
| Rescore64 | `rescore-002`:zero engine/newV; ciascuno dei due set32 resta4PASS/28FAIL, osservato e non imposto |
| Confronto indipendente | `derived-check-002.json`:19 campi metrici/gate invariati, catene/hash/raw/aritmetica/distanza a matrice completa PASS |
| JSON distinto dalle metriche | `observation-delta-002.json`:640 privati rimossi in56 risultati, ID reader; altri delta0 |
| Binding consegna | `delivery-check-003.json`, R PASS:dieci moduli source/wheel/installed uguali tra s008/s010;37 helper Python uguali |
| Protezioni | 338 file d’ingresso e108577 artefatti controllati;129 file corpus invariati, fixture350 originale e r012 permanente preservati |
| Documenti | 50 link locali/anchor, whitespace dei nuovi/modificati, `git diff --check` PASS |
| Processi e risorse | 60 ricevute controllate, nessun workload pendente, figli raccolti e socket diagnostici puliti |

La [tabella completa fonte/callsite → proprietà → prove](../../../../benchmarks/2026-10-11-a002-r013-guardia-e-fence.md#inventario--proprietà--prove)
chiude il [censimento iniziale](../notes/implementation-r013-guard-inventory.md).
I [dati permanenti](../../../../benchmarks/2026-10-11-a002-r013-guardia-e-fence-data.json)
contengono raw, attesi, risultati prima/dopo, AST,16 righe di inventario,
controprove, CLI, metriche19, identità e sorgenti dei checker pertinenti.

## Cambiamento consegnato

Tutte le fonti Unsafe del reader producono eventi sorgente, deduplicati
per tipo/posizione/snippet e drenati in ordine indipendente dalla proiezione.
Attrs esclude valori attivi, riconosce LF/CRLF fuori valori quotati e riceve
la mappa carattere per carattere del producer; HTMLGuard visita ogni apertura
reale una volta. La prima causa reader viene rilanciata dopo la coda; reset
sempre in finally. Gli errori d’albero gestiti non interrompono le guardie.

La decisione fence è condivisa da scanner/radice/container/boundary/consumer.
Tilde non interrompe paragrafo, ordinario EOF non possiede code, veri owner
code/math e token opachi mantengono inertness. La proprietà è verificata sul
corpo discriminante AST, non su un qualsiasi Code altrove. Conservati raw
HTML/LaTeX e LF label/source-map, F36 deliberato e F44; non introdotto un
parser universale. Asset sicuro preservato o residuo esplicito con posizione.

La coverage privata resta fino al consumo node/located/source-map e viene
rimossa ricorsivamente dal solo JSON finale. Scorer/safety/audit non modificati.
La patch comprende reader, test r006/r012/r013 e tre fixture r013, protocollo,
due guide, nuovo report permanente/dati. Nessun nuovo modulo o schema pubblico.

## Oracoli, errori e recuperi conservati

Generatore senza import prodotto, attesi manuali, seed13013, limiti400/2048B.
301 casi finali identificati (285+16);297 raw unici fra tutte le revisioni e
probe esplorativi, massimo186B. Non si afferma che297 siano tutti PASS: le
versioni intermedie fallite restano tali. L’autore non è un reviewer; gli
oracoli indipendenti sono Pandoc per la sintassi e checker senza import
prodotto per confronto/aritmetica. Helper di runtime non definiscono il gold.

| Traccia | Diagnosi e recupero |
| --- | --- |
| `before-001` exit1 | EPERM socket diagnostico nel sandbox; escalation autorizzata, stesso R Firejail netnone, `before-002`PASS. Nessun rifiuto auto-review aggirato |
| Primo after237/242 | Quattro contesti lista con riga vuota e una premessa tree-error; trasporto corretto, raw iniziale preservato, prefisso `È ` seleziona il producer voluto |
| Premesse AST | List/backtick Code inline, raw LaTeX radice ricevuto, first-note-heading Plain; correzioni tracciate nei freeze, nessun gold copiato dal reader |
| Targeted00320fail/1854pass | Venti attesi EOF r012 incompatibili con F55; overlay separato, fixture350 preservata; targeted0041884pass prima dei fix finali |
| Fast0011fail/2980pass | Vecchio F29 assegnava code a EOF; AST aggiuntivo conferma assenza code; mantiene contratto commento opaco, atteso corretto e fast ripetuto |
| Tre checker amministrativi | Nome risultato uguale a receipt host: calcolo concluso, receipt exclusive-create fallita; output/log conservati, CPU di quelle amministrazioni non misurata |
| Probe finale metadata | Offset logico invece di fisico e LF non separatrice;16 raw congelati/AST, fix e ripetizione completa sotto s008 |

I documenti provvisori pre-mapping sono conservati in `evidence/implementation-r013/pre-mapping-*`.
Le campagne final001/remaining001 e stage005/006 non sono attribuite al nuovo
SHA. La campagna corrente `final-002` include tutta la matrice, suite,
controprove, CLI, rescore, contratti, regressioni e checker; CLI supplementare LF in s009 con gli stessi byte; prove autonome
LF in `mapping-proof-001`. Nessun difetto ordinario rinviato al supervisore.

## Identità eseguibili e comandi

Stage runtime finale s008; stage s010 ricostruito dopo soli documenti/dati.
In ciascuno: sdist/wheel, reinstallazione canonica, package-inputs/target/
benchmark-inputs/startup/preflight e S/B/I. Managed locale CPython3.12.13
e interprete dei figli coerenti; dieci hash e origini controllati. Nessun
`--no-sync` usato per sostituire questa prova. I manifest prima/dopo e
`input-store/` conservano le copie dei Python effettivi; verifica byte per
ogni input della campagna finale e delle prove LF.

Comandi effettivi, cwd, timestamp, exit e costi sono nelle receipt host e
nelle directory R. Forma amministrativa realmente usata (percorsi relativi
al repository, espansi nelle receipt figlie):

```bash
python3 -I -B temp/run-a002-fase1-corpus/work/implementation-r013/host.py command final-002 temp/run-a002-fase1-corpus/evidence/implementation-r013/stage-s008/dev/bin/python -I -B temp/run-a002-fase1-corpus/work/implementation-r013/operations.py temp/run-a002-fase1-corpus/work/implementation-r013/plan-final-002.json
```

Il comando suite dentro R è `python -I -B -m pytest tests/ --ignore=tests/api
--ignore=tests/packaging --ignore=tests/docker -q`, con Python assoluto s008.
AST `/usr/bin/pandoc -f markdown -t json`, stdin raw, readerformat/versione/
hash/exit e AST salvati. CLI run/evaluate102 ha argv e receipt per comando;
writer copia fixture, non effettua conversioni native. `run_context.py`
crea snapshot tecnici prima dei workload, verificati prima/dopo. Non sono
approvazioni. General snapshot storici STALE non trasferiscono PASS: il
checker finale lega byte correnti, modulo installato e oracoli pertinenti.

Lo snapshot tecnico di consegna è `implementation-r013-delivery-001`,
con receipt `delivery-snapshot-001.json` e verifica `delivery-verify-001.json`.
Identifica report/checkpoint e binding, senza sostituire il comune di review.

## Budget e confini rimasti

Base effettiva ricevuta17421.058308CPU-s lower dal ledger finale supervisore,
non17301.553741 pre-arbitrato; costi propri sommati una volta tramite host,
senza risommare wrapper R e figli. 45sClaude r007approx distinti, amministrazione
in parte non misurata. Contabilità finale in
[budget-consegna](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r013/budget-consegna.json),
processi/RSS in [receipt risorse](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r013/process-and-resources-final-001.json).
Il controllo dello scope deduplica inode e non somma nuovamente lo storico.
Quote invariate:CPU stop19800/cap21600,disco stop5/cap6GiB,RAM sampled
stop6/cap8GiB,riserva1800CPU-s/1GiB,R≤900s,stream1MiB/file32MiB. Nessuna
deadline; entry08ott07:06:54.369068Z e contatori senza reset. CPU lower/RSS
sampled non certificano upper. Rete/acquisizioni/invii/servizi0,€0benchmark.

A8/CLA-O002 resta limite reader/audit: il path Markdown è verificato dopo
observe, nessuna prima risorsa unsafe globale promessa. Raw finite profile
ha differenze esplicite da Pandoc. Restano F36, run backtick/format exclusions/
comment cost/fallback IDv1 r010, L04outputUNAVAILABLE distinta da V13/A03 sul
riferimento, L01/L02NOT_MEASURED,L05 e gli altri NONPASS/helper12gap/Chrome/
H15/E9. La suite selezionata non è collaudo completo. Nessun benchmark motore,
download, nuova verifica umana, semantica formule o performance universale.

## Passaggio di ruolo

Report e checkpoint propri completi; STATE/HANDOVER condivisi, arbitrati,
prompt reviewer e CHANGELOG non modificati perché riservati al supervisore.
Il prossimo ruolo è supervisore: ricevere questa consegna, creare il comune
per due NUOVE review reali ChatGPT/Claude e arbitrare. Nessuna promessa di
ultimo ciclo. Commit/merge/push/promozione rimangono manuali dopo eventuale GO.
