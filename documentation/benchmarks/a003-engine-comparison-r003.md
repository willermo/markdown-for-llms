# A003 r003 — Fix locali e confronto rimisurato

Nota di conservazione aggiornata il 2026-10-15: le prove citate sono locali e
non incluse in Git; risultati e attribuzioni di questo giro restano storici.
[Disponibilità e limiti](../runs/run-a003-fase1-engine-comparison/manifest.md#disponibilità-delle-prove).

2026-10-14 Europe/Rome. Implementatore Codex/OpenAI, stessa run/chat, nessuna
delega. **Nove rilievi accolti corretti e verificati.** Nuove guardie dimostrate
su cinque consumer effettivi; 17 campioni qualità r002 rimisurati senza ripetere
il confronto completo. Nessun GO autore, giudizio umano o chiusura della fase1.
R002 resta storia dopo il NO_GO delle due review indipendenti e dell’arbitrato.

Indice passivo completo (prova locale non versionata),
matrice r003 (prova locale non versionata),
invalidazioni (prova locale non versionata),
prove operative (prova locale non versionata).

## Risultati corretti del campione

I 13 confronti ufficiali e quattro piloti identificano i raw **storici r002**:
cinque nativi, quattro PDF interi per Marker e quattro per Docling, due piloti
per ciascuno. Proiezione `a003-text-projection-v2`, 23 mappe qualificate (17correnti
più sei raw Docling anteriori al replay), 17 assessment ricostruiti. Nessun nuovo
PASS runtime attribuito ai campioni storici. Le soglie approvate restano native0,
raster CER≤1%/WER≤2%, zero perdite critiche. Human PENDING in tutte le righe.

| Campione | CER e WER corretti (distanza/denominatore) | Altri contenuti osservati AI |
| --- | --- | --- |
| C01 TXT | 0/624; 0/82 | Payload631byte esatto, nove numeri. |
| C02 DOCX, diretto/legacy | 0/1270; 0/193, entrambi | Due formule, nove celle incluso vuoto,13numeri, PNG byte-identica e tre riferimenti; campione preservato. |
| C03 EPUB, diretto/legacy | 53/1223; 8/196, entrambi | FAIL: nota b11/backlink mancanti,11/13numeri; formule/celle/figura conservate. |
| Marker C06 | p1 0/586;0/92, p2 **1/629;1/85** | FAIL: URI contiene backslash; frazione/matrice e limiti delle somme perdono struttura, header introdotto rispetto a gold0. |
| Marker C07 | p1 0/544;0/87, p2 **1/574;1/85** | FAIL: URI frammentato conserva backslash; limiti della somma staccati, gruppo aligned UNKNOWN. |
| Marker C08 | 5/472;6/65 | FAIL: segni/formule/URI deteriorati,7/9numeri. Pilota identico sul caso a pagina unica. |
| Marker C10 | nativeITp1 **1/632;1/93**, rasterENp2 **3/600;3/91**, nativeITp3 0/457;0/70, rasterITp3 175/175;23/23 | FAIL: quattro prose rasterp3 assenti, matricep2 mancante,25/29numeri; figura p3 non contiene il testo perso. |
| Docling C06 | nativep1 3/586;3/92, nativep2 **5/629;1/85** | FAIL: segni alterati, entity/backslash URI conservati;8/9celle,9/12numeri, formule mancanti/placeholder. |
| Docling C07 | nativep1 3/544;3/87, nativep2 **5/574;1/85** | FAIL: segni/URI alterati; frazione/matrice leggibili in PNG, matematica strutturata UNAVAILABLE. |
| Docling C08 | **13/472;9/65** | FAIL: underscore perso, n1→nl, intercalazione fonte/nota;7/9numeri e due formule mancanti. Pilota identico sul caso a pagina unica. |
| Docling C10 | nativeITp1 **8/632;4/93**, rasterENp2 **35/600;7/91**, nativeITp3 1/457;1/70, rasterITp3 **6/175;3/23** | FAIL: dodici celle assenti,20/29numeri, didascalia EN riordinata, E=mc? errato; matrice PNG leggibile. |

Le cifre in grassetto sostituiscono le conclusioni pertinenti r002; le altre
sono riprodotte con v2. CER/WER sommano distanze e denominatori per blocchi della
stessa pagina/lingua/native-raster, senza media di rapporti o perdita dei missing.
Nessun profilo PDF supera il gate completo. I fallimenti scientifici restano
risultati utili; non erano la causa del NO_GO tecnico r002. C02/C03 confrontano
lo stesso Pandoc3.1.3 attraverso due wrapper, non due algoritmi indipendenti.
Non scegliere un vincitore o inferire fedeltà universale da questo corpus sintetico.

## Regola URI e proiezione finita

Nel Markdown ordinario i URI visibili rimangono nella prosa qualificata dal gold:
`&amp;` e i backslash interni sono **KEEP byte-identici**. Anche la continuazione
osservata di un URI frammentato da spazio, C07, mantiene questi byte; niente
ricomposizione, fetch o correzione del raw. I target nascosti sono conservati nel
registro riferimenti, con label/destinazione/dichiarazione e span verificati.
T02 opera solo su escape di prosa ordinaria comprovata fuori URI/code/math;
T07 richiede il contenuto tra una coppia finita di tag HTML supportati verificati,
mai un mero campo `domain=html-text`. Entity del Markdown ordinario non diventano
HTML per ottenere zero. Costrutti ambigui restano UNRESOLVED/UNAVAILABLE, con
contenuto e denominatori conservati. Nessuna estensione CommonMark post hoc.

La migrazione conserva la copertura UTF8 esaustiva, cambia24span a KEEP e registra
le versioni/hash e il prima/dopo: sette T07 in cinque mappe correnti più sette in
cinque originali, oltre gli escape URI pertinenti. Il primo derivato interno r003
non copriva la continuazione frammentata C07: invalidato e preservato in temp.
Il derivato corrente è `assessment-r003`, non quel primo tentativo.

ID osservati duplicati intra/fra righe sono rifiutati prima di concatenare testo;
frammenti distinti sono leciti. Le 25 mappe di allineamento r002 controllate non
contenevano duplicati: nessun errore reale inventato dalla controprova sintetica.
T03 verifica ordine/annidamento; coppie incrociate sono rifiutate. T04/T05 hanno
un registro finito verificato, non stringhe dichiarative che possono eliminare
`non `. Copie native link/note/anchor e tutte23mappe riprodotte; attesi ricostruiti
indipendentemente dai67gold, nessun AST candidato come riferimento.

Le osservazioni su formule/celle/figure/numeri/ordine/copertura riusano i domini
r002 identificati sullo stesso raw/IR/asset, con nuova precisazione URI nei
riferimenti. Le14immagini uniche già osservate AI rimangono byte-identiche; non
è una nuova firma umana o verifica AST formale. Aligned/legami logici UNKNOWN e
matematica strutturata UNAVAILABLE restano tali. Nessun raw/gold modificato.

## Chiusura dei nove rilievi

| ID | Correzione e verifica |
| --- | --- |
| GPT-I001 | Unicità intra/fra righe prima del testo; negativi AA e fra righe, positivo frammenti distinti; audit17assessment e25allineamenti storici. |
| GPT-I002 | Registro finito T04/T05 con ruoli/delimitatori/span/label/target/canale; note collegate a definizioni effettive; nesting T03. Negazione/relazione inventata/incroci rifiutati,23mappe positive reali. |
| GPT-I003 | Apertura dirfd no-follow degli antenati e leaf, file regolare/nlink1/limiti/identità verificati sull’FD letto. Ammissione autentica nel memfd sigillato prima exec, poi chiuso; inode e byte della copia controllati separatamente. Positivo e sostituzioni live senza falso R PASS. |
| CLA-I001 | T07 solo HTML finito comprovato, URI/non-HTML KEEP;23mappe e17misure rigenerate. |
| CLA-I002 | URI visibili, anche frammentati, conservano backslash/entity e restano prosa; target nascosti nel registro riferimenti, no fetch. |
| CLA-I003 | Regressioni probe/preflight/R/receipt/integrità/wrapper non verdi; unit/mock distinti dalle prove live e dalla fedeltà. |
| CLA-I004 | Errno ammessi per ciascun contratto, ETXTBSY/EIO/EEXIST non attestano RO. Create-negative O_EXCL per runtime/modelli/font, input leaf protetto e positivo scratch; RW inatteso FAIL, sentinella raccolta. |
| CLA-I005 | Allowlist finita di bind minimi runtime/venv/helper/I/modelli/font; clone/gold/home/temp/antenati scratch/link/traversal rifiutati prima del lancio. Manifest minimi reali accettati, nessuna fuga retroattiva imputata a r002. |
| CLA-I006 | Figlio≤690s e wrapper=figlio+210≤900, preflight respinge combinazioni non rappresentabili. Confini120/450/690/900 verificati; default RSS6GiB invariato, soglia configurata e invalidi verificati. |

## Nuove prove operative e toolchain

Cinque esecuzioni reali correnti: TXT, Pandoc diretto e legacy su C02, Marker e
Docling su C08p1. SUCCESS, R/confinement PASS, input invarianti e figli raccolti.
Due errori di selezione pagina del launcher sono conservati come ERROR operativi,
non nuovi confronti: il driver riceve indici zero-based per entrambi e traduce
solo all’ingresso API Docling. Rerun dei soli rami invalidati, stessi profili.
I cinque raw nuovi sono byte-identici ai campioni r002 **consegnati**, compreso
Docling dopo replay nativo degli URI asset; non è una prova su tutti i casi.

13casi live finiti nel runner standard: autentico PASS; dodici negativi FAIL,
con engine non eseguito dopo sonde/preflight/bind FAIL. Motore sintetico dichiarato
per le sostituzioni, veri bwrap/consumer/monitor e namespace. Symlink leaf/antenato,
hardlink/FIFO/oversize/JSON assente/malformato/PASS regolare e copia byte-identica
con inode diverso non diventano R PASS. Nessun target esterno dereferenziato
nel percorso di ammissione; sentinelle esclusivamente sintetiche. API validate/argv
rifiuta sette forme vietate, include link/traversal; positivi sui bind effettivi.

Il memfd è creato dall’host, ereditato solo dal consumer, scritto e sigillato con
SEAL_WRITE/GROW/SHRINK/SEAL, chiuso prima exec; il motore non riceve FD aggiuntivi.
Il runner legge quella sorgente sigillata, non un JSON selezionato dal motore.
La copia nello scratch serve all’audit e deve conservare inode/byte. La primitiva
`--bind-fd` falliva la risoluzione proc/self/fd locale: conservata, sostituita
con eredità FD provata live. Nessun bind proc host, root/sudo o privilegio
persistente. Managed omette wrapper/constanti memfd: libc Linux e ABI verificata
nei suoi header, come già per pidfd nel monitor; nessun package aggiunto.

Rete/egress/daemon/PID/proc/mount privati, env/startup e runtime sono verificati
nel consumer prima del motore, con namespace freschi. Pandoc reale ha positive/
negative e confronto dei namespace; Tesseract figlio reale ha positive/negative
su risorse/lingue e namespace nel health Docling corrente. Monitor standard
esterno alla barriera, cleanup/subreaper/pidfd e watchdog conservati. Le negative
file non vengono usate come socket lunghi: AF_UNIX path-too-long non prova privacy.
La prova daemon/socket sintetico vivo usa percorsi effettivi brevi del runner.

S/B/I finali `preflight-r007`: managedCPython3.12.13, uv0.10.10, backend84, nuova sdist/wheel
canonica, reinstallazione nei quattro prefix riusati e confronto dei dieci hash
nei medesimi interpreti dei figli. Closure105Marker e73Docling separata, pin e
lock prodotto invariati; stessi pesi/font/data/runtime identificati. Source/I
delle preparazioni precedenti invalidati dai cambi successivi, nessun PASS
trasferito. I comandi di progetto hanno preflight corrente; i nuovi documenti
successivi rendono S storico per comandi futuri, non autorizzano --no-sync libero.

**48regressioni mirate PASS; suite ufficiale3204test+310subtest PASS,0skip**, R PASS,
input invarianti e postcheckI PASS; esclusi API/packaging/Docker e smoke radice.
Non collaudo completo della migrazione e non prova di fedeltà. Monitor live:
default6GiB PASS, soglia32MiB produce RSS_STOP, timeout produce TIMEOUT, figli raccolti.
ML mantiene16GiB/450s, quattro CPU e getterTorch4/1; RSS campionato non hard upper.

## Consumi, archivio e limiti

Ledger r003 (prova locale non versionata):
1469,477709CPU-s in122intervalli misurati, inclusi fallimenti/preparazioni invalidate,
senza sommare nuovamente consumer annidati. Il primo collector includeva55receipt
inner del monitor: esclusi dal ledger finale e conservati come storia. Totale run UNKNOWN; hash/scoring/copie/
editoria/orchestrazione fuori dagli intervalli non chiamati zero. Ledger r002
8496,364446CPU-s resta storico e non viene contato di nuovo. €0 spese/documenti a
servizi remoti di conversione/revisione0; wire/energia/consumi non osservati UNKNOWN.

Bundle `comparison-r003` distinto e passivo. Dipendenze di lettura dichiarate:
`comparison-r003`, `comparison-r002` e `evidence/references` del vecchio a003.
Raw/asset/IR storici riusati per link/hash, senza ricopiare runtime/pesi o l’intero
archivio r002. Nuove evidenze e generatori sono conservati con argv/versione/hash
 e copie passive; path assoluti nei JSON sono LOCAL_EXECUTION_REFERENCE, non
promessa di portabilità runtime. Originali r001/r002 e review/arbitrati protetti.

Annotazioni/visione AI, corpus piccolo, host condiviso e cache non controllata:
nessuna firma utente, capacità massima dei motori, throughput statistico o
qualità universale. Formula enrichment Docling resta disabilitato. A002/debito
ADR0008 e fase1 aperta invariati; nessuna nuova app/parser/motore/fase/UI/deploy.
Prossimo ruolo: supervisore, nuovo comune e due review nuove indipendenti mirate
ai fix/dipendenze/regressioni, poi arbitrato. Git manuale dell’utente dopo GO.

## Ultimo controllo locale e attribuzione finale

Il negativo aggiunto prima della consegna mostrava che una dichiarazione anchor
diversa dal raw e un escape code/math qualificato come prosa venivano ammessi.
Corretto con verifica delle dichiarazioni effettive e rifiuto conservativo di
contesti con delimitatori code/math o continuazioni URI dubbie; nessuna grammatica
nuova o riscrittura del raw.48regressioni PASS,23proiezioni finali riprodotte
identiche, metriche17campioni invariate. Suite precedente3203+310 resta storica;
nuova suite **3204test+310subtest PASS**,0skip, S7/I7/R/postcheck correnti.

Le sette sorgenti operative del delta (consumer/confinement/driver/pdf-child,
runner/monitor/verificatore) restano byte-identiche agli input delle cinque prove
reali, Tesseract e negative su S6/I6. Dipendenze, origini e startup dei quattro
prefix identici confrontati con I7, canonical40hash verificati; quei R restano
attribuiti a S6/I6 e non rinominati R7. Nuova ammissione bwrap TXT reale su S7/I7
PASS e suite R7 corrente. Cambiati solo comparatore/test finali, senza nuova
inferenza necessaria per E01. Ledger finale1469,477709CPU-s/122outer comprende
anche quest’ultima qualifica e la suite precedente, senza doppi inner.
