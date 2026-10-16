# A003 — Confronto motori r001: prerequisito di confinamento non superato

Nota di conservazione aggiornata il 2026-10-15: le prove citate sono locali e
non incluse in Git; risultati e attribuzioni di questo giro restano storici.
[Disponibilità e limiti](../runs/run-a003-fase1-engine-comparison/manifest.md#disponibilità-delle-prove).

2026-10-13 Europe/Rome. Consegna dell'implementatore, **senza GO** e senza
chiusura fase1. Piano r001+errata approvati; risorse gratuite disposte. Nessuna
spesa monetaria o trasmissione di documenti a servizi remoti.

**Il confronto non è stato eseguito.** Nessun motore documento o modello ML è
stato avviato. Le 13 coppie del minimo (cinque native, quattro PDF × due candidati)
e i quattro piloti sono NOT_MEASURED: fallisce il prerequisito E04, non una
misura di fedeltà dei motori. Nessun CER/WER, latenza inferenziale o ranking.

Indice passivo con i sette riferimenti completi (prova locale non versionata),
matrice e ammissioni (prova locale non versionata),
gate corrente (prova locale non versionata).

## Prerequisito reale

Bubblewrap standalone funziona dopo escalation approvata. Il runner unshare
standard isola IP, ma le sonde rilevano AF_UNIX/sentinella e daemon raggiungibili;
il suo R non passa. Il fallback Firejail standard passa R e preflight I nello
stesso namespace prima del consumer. Bubblewrap nel suo figlio fallisce al mount
del procfs privato: `Operation not permitted`, prima di Python/sonde/import.
Anche la sonda primitiva con `--noroot` fallisce nello stesso punto.

E04 richiede sospensione senza esecuzione non confinata. Non sono stati usati
proc host/parent, rimozione del namespace PID, cap aggiuntive, root/sudo,
modifiche sysctl/AppArmor/setuid o runner alternativi. **Non è un rifiuto
dell'approvazione degli strumenti:** l'escalation è stata approvata, il vincolo
è tecnico nella composizione provata. Nessuna impossibilità universale dedotta;
il supervisore deve valutare un recupero compatibile della composizione R e
consumer prima dei cinque nativi e dei piloti/PDF. Non manca un budget gratuito.

Sei preflight C01 conservati, zero conversioni: errore R unshare, snapshot
storico correttamente negato dopo delta, poi procfs EPERM su input correnti e
verifica del fix exit. C02/C03 e PDF hanno ammissioni sospese con exit null,
senza tentativi di inferenza inventati.

## Lavoro realizzato e verificato

Driver minimo e proiezione finita T00–T08 in
`scripts/diagnostics/run-a003-fase1-engine-comparison/`, riusando R, monitor,
profili e child nativi standard. Staging leaf protetto/parent scrivibile,
started/final, raw preservati, inventario asset inerte, missing/extra/unknown
e cap senza falsi zero. Le 67 unità attese E01 sono versionate dai gold qualificati;
senza output non sono state create mappe raw o distanze. Adapter PDF presente
come ramo non convalidato, senza import/inferenza o attestazione di readiness.

Profilo I Docling puntuale nel verificatore standard: manifest separato legato a
S, closure/archivi/METADATA/versioni/startup/RECORD identificati, invarianti dei
dieci moduli e gate CPU Marker conservati. Non è stato eseguito in un runtime
Docling e non sostituisce I Marker.

Ultime catene base/dev S-r007 → sdist → wheel canonica → B-r007 → reinstallazione
finale → I-r009, managed locale CPython3.12.13, Setuptools84 e uv0.10.10,
startup ispezionato prima dell'avvio, setup offline con sonde Bubblewrap PASS.
Runtime identificati riusati nel refresh finale. Parametri/calls/failed receipts
precedenti conservati, senza trasferire i PASS ai delta successivi.

Suite veloce ufficiale finale (prova locale non versionata):
**PASS, 3.182 test raccolti e 310 subtest, nessuno skip, input invariati**.
Include 26 contratti sintetici nuovi: TX01–TX15, Unicode/intervalli, prosa falsa
code/math, missing/extra/zero, cella vuota/assente, caption duplicata, note/figure,
raw/omonimi/partial/timeout, asset escape/symlink/URI e gate negati. Sono mock,
non prove OCR/crop/formule/ordine reali. Sonde Pandoc nel consumer non raggiunte.
API, packaging e Docker non eseguiti come suite separate; non collaudo completo.

## Metadata e acquisizioni gratuite

Marker1.10.2/Surya0.17.1: cinque checkpoint esatti, senza riuso incompatibile
della cache storica. Recognition/foundation e layout2025_09_23; detection2025_05_07;
table/OCR-error2025_02_18. Recuperati manifest/HEAD sullo stesso host primario
dopo403 urllib, usando curl standard. Acquisiti **50 file / 3.480.589.456 byte**
(checkpoint, manifest e due font GoNoto v7.0); SHA osservati e header/provenienza
registrati. Header safetensors/intervalli e JSON verificati staticamente;
training_args opachi mai deserializzati. Nessuna prova di caricamento/ABI.

Il [README Marker del pin](https://raw.githubusercontent.com/datalab-to/marker/v1.10.2/README.md)
separa GPL del codice e termini dei pesi gratuiti per ricerca/uso personale;
card dei checkpoint incomplete, nessuna decisione commerciale/redistributiva.
Il [README GoNoto v7.0](https://raw.githubusercontent.com/satbyy/go-noto-universal/v7.0/README.md)
indica SIL OFL1.1 per i font. Pesi/font runtime non copiati nell'archivio Git.

Docling candidato: rilascio2.138.0, distribuzione ufficiale
`docling-slim[format-pdf,models-local,convert-core]`, manifest uv separato dal lock
prodotto, CPU torch2.7.1+cpu/torchvision0.22.1+cpu, OCR Tesseract CLI ita+eng,
formula enrichment disabilitato/dichiarato. **72 wheel / 309.783.381 byte compressi**,
1.096.143.597 byte espansi dichiarati nei ZIP: hash-locked, path/duplicati/RECORD
verificati con audit standard. Metadata/licenze/versioni transitive preservati;
nessun import. Acquisizione mediante pip download standard con Python sistema3.12
per ABI cp312, senza installazione; esecuzioni prodotto sempre sul managed3.12.13.
Runtime/pesi Docling e I ML non preparati dopo sospensione E04.

## Limiti, portabilità e seguito

Ricognizione ML iniziale parziale prima dei setup nativi: closure/byte/checkpoint
completati dopo; la sequenza E08 iniziale non è stata integralmente soddisfatta.
Nessuna installazione ML anticipata. Errori operativi ordinari corretti e
conservati; un exit bwrap inizialmente attribuito al TXT ha corrigenda separati
e fix verificato con nuova receipt. Ricreazioni precedenti di venv/inventari
restano costo amministrativo reale, non procedura da imporre per ogni comando.

Ledger (prova locale non versionata):
misure dei soli intervalli monitorati, receipt annidate non sommate due volte;
CPU amministrativa totale UNKNOWN, RSS campionata senza upper hard, bytes e
parziali conservati. €0, documenti remoti0, processi propri pendenti0.

126 copie di riferimento verificate (prova locale non versionata)
per4.328.626 byte; fonte/gold/render/figure e ricevute storiche intatti. Copie
testuali separate e chiusura della lettura (prova locale non versionata)
dei sette riferimenti, relativi asset/frammenti e indice. HTML/JSON/argv collettivi
storici restano dati LOCAL_HISTORY_REFERENCE; runtime, S/I e riproduzione locali.
Portabilità attestata solo per il bundle passivo di lettura, non dell'esecuzione.
Temp a002 intatta; nessuna nuova M/V. Human_assessment PENDING per assenza di
output: nessun giudizio utente inventato, L04/fase1 ancora aperti.

Proposte per formato sospese: TXT literal come controllo; DOCX/EPUB stessi
Pandoc3.1.3 e wrapper, senza ranking di motori; PDF/OCR senza scelta. Ipotesi
documentale da requisiti/riferimenti: completo, blocchi/provenienza, asset e
raw/revisioni separati, unknown conservati; non convalidata da IR/output nuovi.
Nessun ADR di motore, debito ADR0008 invariato, nessuna fase2/3/UI.

Consegna al supervisore per limite E04 e due nuove review reali ChatGPT/Claude
sul delta, poi arbitrato e recupero compatibile. L'autore non produce GO e non
aggiorna stato/changelog condivisi. Git e deploy non eseguiti.
