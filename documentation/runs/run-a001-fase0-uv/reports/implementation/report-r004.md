# Report implementatore r004 — baseline completa e audit EbookLib

2026-10-06 Europe/Rome. Codex/OpenAI, famiglia GPT-6; modello specifico/ID chat
non esposti. Ruolo esclusivo implementatore, mandato38; nessuna delega,
supervisione o review. **WAITING_FOR_SUPERVISOR_RECEPTION**.

## Risultati

Una chiamata ufficiale baseline r002 conclusa: **R/S/wrapper PASS**,
**V0 PASS di caratterizzazione**, **suite legacy FAIL/exit1,62pass5fail**.
Audit EbookLib **PASS_STATIC_SOURCE_AUDIT_ONLY**. Backend setuptools tramite
setup.py osservato; fallback `setuptools.build_meta:__legacy__` inferito,
nessun backend eseguito. Lock precedente FAIL e lock-check non eseguito preservati.

## Autorità e ingresso

Letti AGENTS, skill manage-implementation-run, protocollo, STATE/HANDOVER,
indici architettura/decisioni/roadmap, brief A1–A7, piano r003 integrale,
arbitrato D1–D5, r004/r005/r006, checkpoint implementatore/report-r003 e
checkpoint supervisore baseline-lock-reception-r001. Applicata anche la skill
verify-conversion-fidelity e ADR0001 ai controlli F1–F6.

Branch feature/run-a001-uv; HEAD/dev/base66ba82200e5def5a4db76f9bafccb0731b506091,
indice vuoto. Nessuna modifica tracciata o nuovo sorgente prodotto in questa tranche.
Nessun root uv.lock/.venv creato. Modifiche preesistenti preservate;
registri comuni/changelog/snapshot/arbitrati non modificati dall'implementatore.

Baseline-s007: manifest598215byte,
SHA52f2948014b6b8a26c078cf789dc8edd42deb33c77fe6fb5386fa83cc7121a79;
request847523byte/SHA014db0ac7fe27ccc6590e2056f99b7e0b91af8c9844c91fdcf95c16d4a45bfce;
scope12242byte/SHAb4ec9e2f89c9e44d1923927e42050556bf4f439d7d69e3acfe877b7f70cbf975.
Impronta f5f1f111d5f961580e5f131b95c7581c4b3e0c20a0f7a9a27ce7ca5e224bc81b,
123file/1982artefatti. Byte/hash/piano/arbitrato verificati; MATCH all'ingresso,
prima/dopo baseline/audit e alla consegna. S006/s008 storici per i soli cinque
metadata di ricezione: non preteso MATCH contro i metadata nuovi.

Report-r003/delivery precedente intatti; checkpoint d'ingresso copiato prima
dell'aggiornamento. Copie byte-identiche in resume-baseline-s007/before/.
Target s007+lock1892identità/211directory verificati prima/dopo, senza avviarlo
tramite installer/seed/sync. Fixture/originali/tokenizer/input protetti immutati.

## Chiamata ufficiale e contenuti

[Completion — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md),
[evidenze — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md).
Sessione tool65643 raccolta exit0, durata launcher21,103s, nessun stop/retry;
eccezione r003 non usata. Bootstrap assoluto3.12.3 -I -B; operation[0] della
request uguale alla scope, argv/cwd/env esatti, shell=False/close_fds=True,
nessun merge os.environ. Host require_escalated ammesso senza rifiuti.

Native receipt wrapper/inside/R/V0 e S effettivamente prodotte in
core-completion-r001/official-baseline-r002 e S-baseline-official-r002.json;
nessuna receipt ricostruita. R padre/figlio PASS, stessa netns isolata;
Internet/daemon noti/socket sintetico negati, socketpair positivo. S generato
dopo R nel namespace, V0 dopo S; comando/namespace figli osservati concordanti.
Wrapper controlla input e socket/daemon prima/dopo, chiude il proprio socket.

S.id sha256:3dd2fd09a7637f2cc089d0becc202a55b2a4d33a272b19a6908ecd1cf0980ebe;
SHA file3df9d431facdaac8b423a5047cb16a5f0e8e4b113a1d8d1ad3feeaf88009387c.
Id payload ricalcolato, binding V0/S/snapshot confrontati.

Controlli read-only ulteriori sui file reali in content-verification.json:

- F1/F6 cleaning = sorgente meno la sola newline finale, confronto byte esatto.
- F2 conserva le perdite legacy di graffe/indici/LaTeX, codice, link e immagine;
  diff integrale ricalcolato senza normalizzazione. Numeri/riferimento inventariati.
- Validation-chain4 e F3 due documenti: copie Markdown byte-identiche.
- F4 Pandoc failed0; Unicode/numeri/formula testuale/tabella e due riferimenti
  nei contesti/ordine verificati; anchor HTML omesso resta perdita legacy.
- F5 PNG source/converted identici; cleaner non copia asset e perde riferimento.
- F6 tre configurazioni1000/100,1200/80,1600/120:17chunk ciascuna, metadata/frontmatter/
  index identificati, corpi uguali alle slice sorgente, paragrafi1–160 in ordine.
  Overlap semantic0 preservato. Sliding16chunk/15intersezioni,100IDtoken per
  intersezione; corpi/suffix-prefix rispetto al sorgente e copertura verificati.

La suite selezionata è realmente eseguita, non soltanto raccolta:67nodeid,
62callPASS/5callFAIL,zero skip/error;67setup e67teardown PASS. Eventi e cache/
config osservati conservati nel JSON nativo. Cinque integrazioni falliscono
per ricerca legacy di clean_markdown.py nel workspace esterno; stderr/longrepr
e difetto orchestratore exit1 preservati. Il PASS V0 caratterizza questo
comportamento originale, non promuove la suite né il codice migrato.
62/5 sono osservazioni nuove, non trasferimento di numeri storici.

## Audit e gate successivo

Sessione24731 raccolta exit0; acquisizione/lettura0,605s di launcher, gap massimo
0,5001s, nessun stop. [Report audit — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
con raw JSON/sdist/header/URL/digest/copie. Due HTTP200/no redirect, body120577byte,
wire non misurato.66membri/271811byte dichiarati,14153byte metadata letti.
Nessuna extractall/import/exec/hooks/backend.

setup.py importa setuptools.setup; nessun pyproject/build-system. Backend legacy
PEP517 e requisito iniziale setuptools>=40.8.0 sono inferenze; pin84 proposto,
già acquisito. Dipendenze runtime literal lxml/six; nessun setup_requires osservato,
requisiti dinamici reali ignoti. Lettura README UTF-8/re.sub long_description;
nessun import runtime nel setup. Costante sorgente VERSION0.18.1 distinta da
distribuzione/setup0.18, senza reinterpretare l'identità.

[Richiesta concreta package — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
non autorizzata: due build native isolate/offline della sola sdist EbookLib con
setuptools84 da wheel locale/hash e constraint corrente, cache/tmp/output distinti,
R/D4 e nuovo freeze prima dell'esecuzione. Costi/input/argv/env/output e criteri
METADATA/RECORD/payload/ripetibilità definiti. Nessun pin/grafo/fonte modificato.
Lock universale non prodotto: no-build non rimosso; build locale non autorizza
un resolver con nuovi backend ignoti. Tests-s001 non pronto senza lock/S/B/I.

## Guardie, preservazione e limiti

Ledger comune run+.venv-python, directory/link metadata inclusi senza seguirli,
formula r006 con quota residua74MiB e16MiB esterni. Nessun main storico
check_preserved né baseline=True usato per ammissione.43campioni baseline,
gap massimo0,500190s,zero entry scomparse nei campioni. Incremento baseline
campionato18.067.456byte allocati, conservativo su tutta la run, sotto64MiB.
Include fixture temporanea legacy15.000.000byte in tmp/pytest-1 oltre agli output
principali: conservata/inventariata, nessuna pulizia o spostamento.

Il contatore activity del launcher eseguito misurava i tre output ufficiali;
il ledger globale campionava anche tmp/log/storico. La verifica conclusiva usa
il delta conservativo dell'intero ledger per dimostrare il cap64MiB anche per
quei temporanei. Conservato lo strumento eseguito e corretto il contatore futuro
per includere quel delta nel gate d'attività. Nessun errore ha interrotto la
chiamata, nessuna ripetizione ammessa usata e nessuna receipt reinterpretata.

Audit/report/request sotto2MiB, registri/strumenti sotto8MiB; misure finali e
formula/costi in delivery-checks-final.json. Max1GiB/stop896MiB/free1GiB/file32MiB/
log1MiB/JSON8MiB rispettati nei campioni/pre/post. Monitor periodico, non quota
atomica; misure/tool overhead esterno non attestano un picco istantaneo o wire.

Verificati freeze/hash, target, contenuti/output/S/receipt/events, archivio/AST,
link locali documenti d'autore, Git/indice e diff-check. Processi propri raccolti;
nessun processo altrui terminato. Non eseguiti backend/prodotto build/install/sync/
tests nuovi/Docker/inferenza. Managed/checksum inferito e setuptools acquisiti
non attestano il prodotto. V7/V8 obbligatorie con mandato pesante distinto;
V10/V11/pesi/font esclusi, local_marker non convalidato. Due review indipendenti
e arbitrato finale mancanti; nessun GO finale/commit/merge/push/promozione/deploy.

Prossimo supervisore prompt05+r004/r005/r006 e checkpoint autore corrente.
Trasferire file reali ignorati/cache/tmp/work/managed e lavoro non committato,
non solo manifest. Nessun servizio da attendere, medesimo obiettivo36.
