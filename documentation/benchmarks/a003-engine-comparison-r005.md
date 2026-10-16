# A003 r005 — Protezioni, completezza e numeratori corretti

Nota di conservazione aggiornata il 2026-10-15: le prove citate sono locali e
non incluse in Git; risultati e attribuzioni di questo giro restano storici.
[Disponibilità e limiti](../runs/run-a003-fase1-engine-comparison/manifest.md#disponibilità-delle-prove).

2026-10-14 Europe/Rome. Consegna dell’implementatore per mandato15 e arbitrato
r004, senza delega, review impersonata o GO autore. Ingresso
implementation-r005-fix-context-r001 MATCH prima dei fix; branch
feature/run-a003-engine-comparison, HEAD/dev9f702a2, indice vuoto. Human PENDING,
fase1 aperta e debito a002/ADR0008 invariati. I risultati storici non diventano
funzionalità della nuova applicazione.

Indice di lettura (prova locale non versionata),
matrice finale (prova locale non versionata),
invalidazioni (prova locale non versionata),
attribuzione (prova locale non versionata).
La [consegna r004](a003-engine-comparison-r004.md) è preservata; aggregati e ledger
sono corretti in nuovi derivati, senza riscrivere report o prove precedenti.

## Dieci ID del giro r004

| ID r004 | Chiusura autoriale e prova | Limite |
| --- | --- | --- |
| GPT-I001 | Fence distinto dallo span inline; chiusure esterne su riga, run interni e chiusure corte preservati. Positivi con/senza info e negativi incompleti nelle quattro partizioni. | Protezione conservativa delle forme non qualificabili; nessun parser generale. |
| GPT-I002 | T03 qualifica flanking, ruolo open/close, punteggiatura/whitespace e parità degli escape sul raw, oltre a coppie/run/nesting già mantenuti. | Supporto finito a delimiter indivisi di uno/due caratteri; ambiguità respinta, KEEP resta possibile. |
| GPT-I003 | Ledger collega figli e parent reali con argv/output/manifest/identità, intervallo e semantica wait4. | Subset corretto, totale CPU/wire/energia UNKNOWN; nessuna nuova suite per soli conti. |
| CLA-I001 | Passate distinte e unione monotona delle protezioni: il dollaro di prosa non consuma i backtick. T00/T02/T03/T06 e rimozioni sintattiche nel payload respinte. | Non introduce una grammatica math o Markdown completa. |
| CLA-I002 | Code indentato dopo blank/all’inizio protetto; URI con schema maiuscolo qualificato senza cambiare il contenuto. Positivi di prosa esterna. | Regola finita, nessun fetch/casefold del documento o caso reale nuovo attribuito al corpus. |
| CLA-I003 | Chiuso insieme a GPT-I002: marker escapati e flanking di punteggiatura non autorizzano l’eliminazione. | Sovrapposizione, senza duplicare il problema. |
| CLA-I004 | T01 richiede posizione reale di blocco; T08 distingue blank/terminale da singolo soft-wrap interno. Numeri, negazione e sillabazione nei negativi restano. | 48T01/332T08 reali qualificati; gli ID delle unità non provano la posizione. |
| CLA-I005 | Delivery operativa r005, staging finito/hashato e stato new-only prima del boundary. Quattro leaf storici respinti solo in prove pure; nuova negativa live su r005 concluso. | Manifest/caller non dispongono nuove delivery né raw ufficiali; nessun vecchio attempt avviato. |
| CLA-I006 | Attraversamento fallibile esplicito, no-follow/identità, RuntimeError nel driver/runner, final nonverde e causa host; sentinella sostituita conservata prima del cleanup. | Mock distinti da R autentico; cleanup non provato resta False/UNKNOWN, final non scrivibile ha canale host. |
| CLA-I007 | EXTRA attribuiti dai due vicini MATCHED concordi e dai locatori raw; inserzioni sommate ai numeratori, denominatori gold invariati. | Ambiguità rende aggregati pertinenti UNAVAILABLE; zero-denominator locale non è PASS. |

Questa tabella è una dichiarazione autoriale da sottoporre alle due nuove review.
TargetT04/unicità/memfd/seals/no-follow/errno/RSS/timeout riconosciuti restano nelle
regressioni; gli ID omonimi dei giri precedenti non sono rinumerati.

## Contratto e prove finite

Versione `a003-text-projection-v4`: copertura raw/UTF-8/hash e relazioni precedono
l’emissione. Ogni famiglia aggiunge protezioni; contesti prose e partizioni non
le revocano. I fence qualificano wrapper esterni separati dal payload. Il profilo
protegge anche code indentato, URI, math e literal dichiarati. Costrutti incompleti
o non qualificabili consentono KEEP o rifiuto, senza trasformarsi in falso zero.

T03 considera i caratteri raw adiacenti e gli escape, senza usare il gold per
scegliere il dominio. Le regole finite di flanking seguono la
[fonte primaria CommonMark0.31.2](https://spec.commonmark.org/0.31.2/#emphasis-and-strong-emphasis).
Non è una dichiarazione di conformità completa ai dialetti. T01 verifica il
prefisso reale di riga e i wrapper precedenti; T08 richiede un separatore raw
completo, oppure il terminale/anchor qualificato. Un LF interno tra frammenti
non può essere eliminato come metadata.

1.060 interazioni (prova locale non versionata)
coprono fence3/4/tilde, info/no-info, chiusure corte/midline/incomplete, famiglie
concorrenti, punteggiatura, escape pari/dispari, indentazione, URI e posizione,
con quattro partizioni e KEEP byte-identico. Restano rigiocati
244 domini (prova locale non versionata)
e 102 target (prova locale non versionata).
I 16 negativi sintattici precedenti e il fence positivo sono nelle regressioni
ufficiali correnti. Le matrici conservano raw hex/UTF-8, annotazioni, esiti baseline
v3 e v4, errori e identità del producer.

## Rigioco e misure

Replay finale (prova locale non versionata):
23 mappe con emissioni identiche a r004, 17 assessment ricostruiti su 67 unità
attese da sette gold indipendenti, 14 destinazioni native qualificate. Tutti i
raw/gold/IR/asset e i canali critici restano intatti. C01 conserva il canale
letterale separato: CER raw/WER diagnostico e payload esatto sotto il fence.
La somma dei denominatori delle classi è verificata contro l’atteso di ogni
assessment, incluso C01.

Attribuzioni EXTRA e somme (prova locale non versionata)
ricalcolate dai raw/annotazioni/assessment, senza copiare i controcalcoli:

| Motore / campione / classe | CER corretto |
| --- | ---: |
| Marker C06 p2 nativa it | 2/629 |
| Docling C06 p2 nativa it | 6/629 |
| Docling C10 p1 nativa it | 9/632 |
| Docling C10 p2 raster en | 36/600 |
| Docling C10 p3 raster it | 7/175 |

Cinque spazi EXTRA aggiungono cinque inserzioni CER, senza token WER nuovi e
senza aumentare denominatori. La classe richiede locatori raw ordinati e due
vicini MATCHED di prosa con classe completa identica. Classe assente/discordante
mantiene l’extra e rende UNAVAILABLE gli aggregati pertinenti. Ogni extra non
vuoto ha FAIL_EXTRA locale e ratio UNAVAILABLE_ZERO_DENOMINATOR. I sintetici
native0 verificano FAIL con extra, PASS senza extra, UNAVAILABLE con attribuzione
ambigua. Nessun canale formula/cella/asset/numero/ordine/note viene sostituito
con OCR o prosa inferita.

TXT/DOCX preservati nel campione; EPUB perde nota/backlink e nessun PDF supera
tutti i gate. Le soglie native0 e raster1%CER/2%WER sono invariate. I sintetici
scanner non sono perdite attribuite ai 23 raw. Human PENDING, nessuna firma,
scelta definitiva del motore o chiusura della fase1.

## Prove operative e completezza

67 casi policy (prova locale non versionata)
verificano prima del boundary root/parent/sibling/link/traversal/env, quattro leaf
storici, scratch/evidenza/R già usati e destinazioni setup. Le fixture proprie
sono eliminate dai loro context manager; directory vuote non sono attempt misurati.
Il runner ammette solo r005 con staging finito, input/config RO e cache/home/tmp
sotto scratch. Non basta un nome o una collisione del monitor.

TXT finale002 (prova locale non versionata):
SUCCESS, R/confinement PASS, 44 sonde PASS, payload letterale byte-identico,
input invarianti e figli raccolti. Postcheck (prova locale non versionata)
verifica anche che i live negativi non abbiano modificato gli artefatti. Due
negative correnti, clone e r005 concluso, falliscono prima di bwrap/consumer/motore;
nessuna invocation/admission nuova. Il consumer-process del positivo concluso
resta storico: “no engine process” nei proof originali significa nessun nuovo
figlio avviato. Le cinque famiglie (prova locale non versionata)
sono ammesse in nuovi staging identificati, senza cinque nuove inferenze.

Mock post-run (prova locale non versionata):
nove regressioni precedenti, più directory assets illeggibile e RuntimeError
del monitor sul tentativo iniziato. Inventario attraversa tramite dirfd/scandir
fallibile, controllando no-follow/identità prima di leggere. EACCES non perde
asset in silenzio e non lascia SUCCESS. Una sentinella sostituita da file regolare
conserva byte/inode e copia passiva; directory/link/special conservano descrittore
ed errore, senza dereferenziare target o cleanup ricorsivo. Parziali e receipt
rimangono disponibili. Final non scrivibile: causa nel canale host, nessun file
inventato. RuntimeError del runner (prova locale non versionata)
produce HOST_RUN_ERROR/FAIL e cleanup socket, children_reaped UNKNOWN; la causa
è propagata dal driver. Questi mock non sono R autentici o nuovi motori.

## Installazione e suite attuali

S5/B5/I5 (prova locale non versionata):
managed locale3.12.13, uv0.10.10, backend84 invariati. Nuove sdist/wheel, canonica
reinstallata nei quattro prefix, 40 confronti source/wheel/installed e startup/
closure riusati per identità. I due scratch disposti sono container di più prefix.
Le nuove scritture setup sono limitate a setup-r005, output della canonica e
singolo prefix selezionato; il resto del container è montato RO. Modelli/font/
ambienti non sono ricreati, nessun download o cambio pin/profilo.

Suite corrente (prova locale non versionata):
**72 regressioni +1.495subtest PASS; 3.228test +1.805subtest della suite veloce
ufficiale PASS, zero skip**, exit0/R PASS/preflight/postcheck e input invarianti.
Regressioni: timeout900 interno/900 monitor esterno; fast-suite:690 interno/900
esterno. Il wrapper standard distingue ancora la propria attesa dal comando
figlio. Nel r004 storico fast-suite-launch-result-r005 aveva690 e regression-
launch-result-r005900: nessuna frase universale “tutti690/entrambi900”.
API/packaging/Docker/smoke radice esclusi; suite selezionata non certifica
migrazione, fedeltà generale o la nuova applicazione.

Fallimenti ordinari conservati: NETLINK_ROUTE EPERM del sandbox esterno,
poi escalation dello strumento approvata mantenendo Bubblewrap; ripresa iniziale
delle directory setup, SyntaxError del helper, campo monotonic_end assente e
candidate parent negativo con vecchi output, primo riepilogo con path errato.
Il primo collector ometteva il canale letterale C01: derivato precedente
conservato, collector finale e replay correggono/verificano i denominatori.
Il primo postcheck confrontava l’intero Markdown TXT alla fonte invece del
payload delimitato: fallimento corretto usando wrapper raw comprovati.
Nessun esito dei tentativi ordinari è un rifiuto dell’approvazione. S4/I4,
primo TXT/regressioni e prove precedenti restano attribuiti ai loro input;
S5 e le prove finali sono distinti. Nessun vecchio PASS globale trasferito. S5 è attribuito alle prove e diventa
storico dopo i nuovi documenti finali; nessun futuro --no-sync usa questo esito.

## Ledger e consegna

Ledger r004 corretto (prova locale non versionata):
stesso subset **1086,484211CPU-s/128outer/9inner esclusi**, cinque duplicati
229,309660CPU-s rimossi. Originale immutato; alias/hash/argv/relazioni wait4
conservati. Una semplice sovrapposizione temporale non prova il nesting.
Ledger r005 (prova locale non versionata):
**411,133918CPU-s/53outer/5inner esclusi**, attività misurate effettive, senza
sommare i vecchi ledger. TotaleCPU/wire/energia UNKNOWN; attività host non
monitorate non sono chiamate zero. Costo€0 e documenti remoti0, risorse gratuite
disposte senza nuove quote amministrative.

16.581 file storici verificati byte-identici dopo le prove, inclusi raw/gold/
review/archivi e registri condivisi. Nuovo comparison-r005 con originali passivi,
producer/hash/argv, matrici e dipendenze comparison-r004/r003/r002/references.
Portabilità di lettura verificata; path assoluti JSON LOCAL_EXECUTION_REFERENCE,
nessuna promessa di esecuzione portabile o copie di runtime/pesi.

Snapshot tecnico a003-r005-implementation-delivery-001 e indice integrativo dei
path esclusi da run_context; copie/hash/link/JSON/whitespace verificati. Questo
snapshot identifica la consegna autoriale, senza comune di review o GO.
Prossimo ruolo: supervisore, nuovo comune, due review nuove mirate indipendenti
ChatGPT/Claude e arbitrato. Nessuna nuova pianificazione/fase/app/UI/deploy/main,
modifica di STATE/HANDOVER/arbitrati/roadmap/changelog o Git mutativo. Git manuale
soltanto dopo GO.
