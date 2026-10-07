# Disposizione IMPL-V0-004 — preparazione baseline s005

Il supervisore **accoglie IMPL-V0-004** come difetto del diagnostico ammesso
nel freeze s004: il confronto integrale di inicfg rifiuta differenze richieste
dal mandato. La preparazione precedente e i suoi test non coprivano la forma
reale di quel campo. **V0 s004 resta FAIL/exit2; suite FAIL/exit1**, P1–P3 in
attesa. Si dispone preparazione s005 in nuova chat, non retry dello stage
congelato né promozione retroattiva. Nessuna review codice o GO finale.

[Ricezione — riferimento locale non archiviato](../../../notes/materiale-locale-conservato.md), [osservazioni raw della suite verificate — riferimento locale non archiviato](../../../notes/materiale-locale-conservato.md),
[controlli contenuti — riferimento locale non archiviato](../../../notes/materiale-locale-conservato.md) e [fonti — riferimento locale non archiviato](../../../notes/materiale-locale-conservato.md).
Riconfermati 3.340 file nello scope autore e 228 riferimenti input/output
dell'impedimento; 455 artefatti s004 e antecedenti immutati. All'ingresso
s004 STALE solo per CHANGELOG dichiarato, senza delta tecnico. Copie ricevute
di report/checkpoint/driver/changelog conservate qui; receipt originali intatti.
Autore/chat/esecuzione sono dichiarazioni dell'implementatore accompagnate dai
dati tool; gli hash identificano versioni, non autore o motivazione della review.

## Esiti ammessi e limiti

Quattro receipt R complete PASS, bootstrap/baseline/S/V0; wrapper S exit0 e
S canonico verificato, IDsha256:65dc74bf1517b5a93764d40cfbe01bcd4b9fb0907396c1f9fac281c711f54768,
fileSHA85a3b5b9b621c4817ba2921bd2bba48ab8bb0e732e4dcc0ea586edae07cc06a5.
28 record S e dieci originali/quattro metadata HEAD/copie/correnti verificati.
Runtime dell'autore, nessuna R/conversione/tokenizzazione o suite rieseguita
dal supervisore. Stesso host/Firejail/TMPDIR/gate; otto daemon negati e tre
assenti senza prova blacklist, warning overlay e /proc campionato espliciti.

Collection exit0 e run exit1, 67 nodeid in entrambi e 201 eventi osservati,
setup/teardown PASS, 62 call PASS e cinque call FAIL ENOENT/AssertionError
individuali. Cache nuove separate, argv/selection/versioni osservati; i pass
non sono un complemento inferito. Limite specifico dei nodeid s003 superato
nei dati s004, ma requisito complessivo SUP-V0-003 resta aperto per il confronto
configurazione e la receipt finale incompleta. Non ricostruire quei campi nella
receipt fallita: inventory/copie workspace sono evidenze aggiuntive distinte.

F4 e sliding diagnosticati positivamente sui nuovi output conservati: due
marker nei contesti corretti; 16 finestre sliding, 15 intersezioni di 100 ID
e 326–349 caratteri, unica encode osservata. 51 chunk semantic/slice/metadati
e 160 paragrafi in ordine per tre config, 84 Markdown/report byte-identici s003,
120 workspace byte-preservati. PNG RGBA2x2 decodificato, asset downstream perso;
perdite F2/F5/anchor/separatori legacy inventariate, bundle incompleto. Il limite
ASCII resta; nessuna equivalenza con wheel/prodotto o conversione end-to-end.
Nessun PASS di questi dati chiude V0 globale; nuovo driver richiede prove nuove.

## Evidenza e causa

Tra gli otto campi comuni confrontati dal driver, solo inicfg differisce:
collection = cache_dir della collection; run = cache_dir del run più
verbosity_test_cases="1". Invocation argv effettivi corrispondono esattamente
alle due fasi prescritte. Rootdir/configfile/hash/addopts/effective_args/
pytest/plugin distributions coincidono; configfile e suo record sono null.
Verbosity globale/test_cases -1/-1 in collection e 1/1 nel run.

La sorgente installata pytest9.1.1 mostra il proxy e l'aggiunta degli override
a _inicfg; comportamento coerente con il
[tag ufficiale 9.1.1](https://raw.githubusercontent.com/pytest-dev/pytest/9.1.1/src/_pytest/config/__init__.py).
Questo è riscontro statico più dati raw, non una nuova esecuzione pytest.
I test TestBaselineSuite usavano solo plugins/pytest_version e mancavano
inicfg, quindi non potevano rilevare il conflitto. Non basta togliere il
campo dal confronto o accettare qualsiasi differenza tra le fasi.

## Azione assegnata e criterio

Cambiare soltanto run_baseline.py e tests/unit/test_uv_diagnostics.py per il
confronto config/relativi test. Helper stdlib puri consentiti nello stesso
driver. Niente modifica del prodotto, wrapper, producer, algoritmi F4/sliding,
test legacy, fixture, deps/cache o host. Raw observed JSON sempre conservati.

Separare controllo comune e controllo per fase, verificando prima presenza/
tipo dei campi obbligatori: mancante non equivale a null in entrambe le fasi.
Restano confrontati rootdir/configfile/record/hash/addopts/effective_args/
pytest_version/plugin_distributions e ogni voce comune di inicfg. I due soli
override ammessi sono cache_dir (entrambe le fasi) e verbosity_test_cases
(solo run). Rimuoverli da una copia per confrontare il comune **solo dopo**
aver verificato esattamente chiave, valore, fase, argv e path attesi.

| Controllo | Collection | Run |
| --- | --- | --- |
| cache_dir raw/effective/cache_before/cache_after | cwd/pytest-cache-collection assoluto | cwd/pytest-cache-run assoluto |
| verbosity_test_cases override | assente negli argv e inicfg attuali | una volta, stringa "1" in inicfg |
| verbosity globale/test_cases | -1/-1 | 1/1 |
| flag display | --collect-only -q --verbosity=-1 | -v -rA --verbosity=1 |
| argv | quattro file + display + solo -o cache_dir | quattro file + display + -o verbosity_test_cases=1 + -o cache_dir |
| config comune baseline attuale | configfile/record null, addopts [], inicfg comune vuoto | identica |

Path/fasi devono derivare dal CWD/repo attesi della propria invocation; due
path diversi qualsiasi non sono sufficienti. Rifiutare override extra,
duplicati/confliggenti, cache uguali/scambiate/stale, valore mancante o errato,
argv/invocation/effective config discordanti. Non introdurre eccezioni per
env/addopts che cambino la selezione. I nomi plugin numerici legati a ID di
processo restano raw: non inventare uguaglianza di quegli ID tra figli distinti;
preservare controllo delle distribuzioni/versioni e i dati plugin già richiesti.

Test stdlib con forma completa e realistica dei due observed JSON s004:
positivo per le differenze esatte prescritte, test delle evidenze reali in
lettura senza modificare gli originali; negativi per override/valori/path/fase/
config comune/versioni/record mancanti e incoerenti. Preservare negativi per
nodeid/report/skip/errori/cache/cause e test F4/sliding precedenti. Nessun pytest
import/collection o invocazione applicativa ora; replay puro non è V0 PASS.

Poi inventari/target/baseline-inputs/argv/costi e request reale s005
WAITING_FOR_STAGE_SNAPSHOT. Il supervisore congelerà **solo dopo** quella
consegna; un nuovo prompt ammetterà R-bootstrap completa → R-baseline completa
→ S s005 nuovo → V0/audit completo. Config corretta deve rispettare tutti gli
altri gate invariati e mantenere suite FAIL/exit1 62/5 osservati. Dati s004
non riutilizzati come PASS del driver cambiato; fallimenti preservati.

Questo concretizza D1/D2 e V0, non cambia piano/A1–A7/24 disposizioni né richiede
una r004 del piano. Il NO_GO codice richiederebbe due review reali; qui si
dispone un impedimento interno alla IMPLEMENTATION già autorizzata. Nessun
arbitrato finale simulato, nessun GO condizionale che aggiri V0 FAIL.

Durate wrapper autore 1,467/1,467/2,168/7,531s, audit separato; spazio finale
288.968.704 byte alle15:57:24UTC, da ricontrollare prima di prove future.
Stime 50 MB/output, due minuti R/S, 35 minuti V0/audit, rete/acquisizioni zero.
Preservare temp e /tmp/a001-uv-baseline-pi8cvs6x; nessuna pulizia implicita.
Ogni comando host require_escalated richiede review automatica propria,
justification specifica e niente prefix_rule ampia. Rifiuto/gate incompleto
→ IMPEDITA e supervisore, senza host/terminale alternativo o modifica policy.
V7/V8 future obbligatorie e costo separato; V10/V11/pesi/font/inferenza esclusi.
P1–P3 attendono baseline adeguata e mandato successivo, nessuna fase roadmap,
acquisizione/deploy/invio remoto. Git manuale utente, nessun commit/merge/push.
