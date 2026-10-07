# R014 — lock ricevuto, promozione prodotto e input S/B/I/E

2026-10-06, supervisore. Ricevuto45/report-r011: lock universale e check nella
copia exit0 con nuovi R/D4/wrapper PASS, SHA lock invariato.143package/141nomi,
CPU/cu126/extra/conflitto, metadata EbookLib0.18 e sdist/hash preservati.
88file autore e882record cache ricalcolati; s015 MATCH alla ricezione.
Confermato semanticamente: pyproject della copia differisce dal root soltanto
per tool.uv.dependency-metadata di EbookLib. Nessun GO finale/installabilità
multiABI/prova prodotto trasferiti. I fix ordinari del reader sono già stati
risolti nella stessa chat, nessun riesame del piano per quei fix.

## Mandato e dipendenze

R013/autonomia per obiettivo resta: mezzi reversibili non enumerati, fonti già
configurate ed endpoint verificati; decisioni nel report per le due review.
Autorizzato l'implementatore a portare la sola tabella ricevuta in pyproject e
copiare byte identici del lock nel root. Non rigenerare il lock né mutare fonti/
pin/extra/grafo per promuoverlo. Root lock SHA
b3288f1d51b880d9c683a43607d9eedc5b8c87465c8a1d357818c6b61d12f05f.

Eseguire acquisizione runtime **base** locked senza progetto/gruppi/extra,
no-build, stesso uv/managed e cache, origini reali verificate. Target scratch
runtime-acquisition, non root .venv; nessun entrypoint/import applicativo.
Sono ora ricevute le wheel del piccolo runtime base selezionato dal lock:
requests/tiktoken/tqdm/dotenv e transitive effettive, non wheel ML/extra/dev/API.
Se uv conserva cache espansa anziché whl, inventaria i file reali e distinguine
hash dai digest upstream. Archivi whl necessari possono essere acquisiti da
URL/hash esatti del lock via lettura HTTPS pubblica limitata nel budget,
senza resolver/installer alternativo o digest fittizi.

Predisporre ingresso package reale: lock, input prodotto, backend84 già
acquisito, origine/interprete e runtime/copie pertinenti identificati. Attese
future di wheel/dist/receipt restano attese senza hash. I mezzi/helper necessari
si preparano e correggono nella stessa chat; fix diagnostici locali ammessi
prima del freeze senza indebolire gate, attribuendo le prove al codice eseguito.
Non modificare baseline/receipt storiche o moduli prodotto estranei all'obiettivo.

Consegna **request product-s016 WAITING_FOR_STAGE_SNAPSHOT** con lista di soli
file esistenti senza symlink/escape, package-inputs realmente presente, template
R/D4/build/sync/install/preflight completi e scope ricevuto. Non congelare per
conto del supervisore; nessun S ufficiale né build/install progetto prima del
freeze prodotto. Nessun nuovo snapshot in questa ricezione: input di prodotto
e inventory non sono ancora presenti. S015 storico dopo i delta di governance
qui dichiarati e le modifiche autorizzate, non MATCH del vecchio worktree.
Non è stop per un difetto: è l'unica dipendenza input→snapshot→S di D1/P2.

L'intera sequenza successiva **S→B→I→E base** è ricevuta condizionatamente al
freeze prodotto con input e comandi effettivi: S ufficiale, build offline
sdist e wheel dalla sdist tramite setuptools84 locale/constraints84, audit B,
root/base-a/base-b con sync locked offline e reinstall della stessa wheel
canonica, I/diecihash/RECORD/origini, E corrente; venv esterna con runtime
esportato hash-locked e wheel canonica/pip-check/I/E. R/D4 reale per i workload
offline; bootstrap e backend realmente attestati prima della build, non solo
post. S, B, I ed E mai inclusi nel medesimo freeze d'ingresso come output futuri.
Non serve ripianificare né ricevere di nuovo il costo ad ogni comando:
al freeze il supervisore lega gli input concreti e consegna ripresa diretta.

## Costo distinto del runtime base

Ricevuta stima112MiB per **tutta** questa fase, compresa acquisizione scratch,
root/base-a/base-b/esterno, cache/build/tmp/dist e registri. Stime8wheel/64env/
24backend/8registri più8contingenza non quote autonome; unico pool condiviso.
Nuovo Hentry reale in authorized-scope; tutto il pregresso resta incluso nel
pool complessivo, vecchia tranche metadata32MiB chiusa con residuo non sommato.
Non112MiB nuovi a ogni freeze/tentativo; costi prefix→prove restano cumulativi.
Ledger run/.venv-python/.venv e /tmp/run-a001-product-wheel-env-r001, con lstat
senza seguire link; nessun cache/tmp fuori ledger o cleanup per passare gate.
Pool1GiB/stop896/libero1GiB/riserva esterna16MiB invariati. Formula
Delta=max(0,H-Hentry)<112MiB; H+max(0,112MiB-Delta)+16MiB<896MiB.
Gate ricevuto con oltre230MiB di margine sullo stop, stima non garanzia.

Monitor0,5s/gap target1s non quota atomica; stream1MiB/JSON8MiB/file32MiB/log
normali invariati. Root e target esterno devono essere inizialmente assenti,
nuovi output esclusivi, no cleanup/reset. Workload totale7200s cumulativi
prefix+prove; deadline singola adattabile fino900s entro residuo, outer+180.
Stime del JSON sono template, non nuovi tetti per ogni errore. Costi paid0,
nessun download Python/extra pesanti, pesi/font/MLpayload. ABI iniziale managed
Linux x86_64, non universal installabilità. V1–V9/dev/API/suite/V0comparativo
ancora da legare a costo/input pertinenti; V7/V8 pesanti distinti, V10/V11 esclusi.

Consegna report-r012, checkpoint e request pronta senza sola-preparazione
astratta; prossimo supervisore nella nuova chat handover47. Registri comuni e
snapshot del supervisore; due review indipendenti/finale futuro, Git manuale.
Nessun commit/merge/push/deploy o servizio automatico. Supervisore non promuove
né installa il prodotto al posto dell'implementatore.

### Spazio libero sui filesystem distinti

Verifica finale: repository oltre1GiB libero, /tmp circa517MiB su filesystem
separato. Il requisito1GiB di spazio libero è del filesystem repository;
non si applica identico a un tmpfs più piccolo. Per /tmp ricevo un controllo
separato almeno128MiB liberi (intero budgetfase112+riserva16, conservativo
rispetto alla sola venv esterna), oltre al medesimo gate cumulativo112MiB.
Controllare entrambi periodicamente, STOP dipendente se insufficiente;
nessun aumento costo/pool o scrittura fuori ledger. Non bloccare il mandato
per un controllo uniforme1GiB introdotto dal checker supervisore su /tmp.
