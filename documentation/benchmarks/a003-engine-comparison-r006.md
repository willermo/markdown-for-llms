# A003 r006 — Domini concorrenti e marker raw

Nota di conservazione aggiornata il 2026-10-15: le prove citate sono locali e
non incluse in Git; risultati e attribuzioni di questo giro restano storici.
[Disponibilità e limiti](../runs/run-a003-fase1-engine-comparison/manifest.md#disponibilità-delle-prove).

2026-10-15 Europe/Rome. Consegna dell’implementatore per mandato18/arbitrato
r005. Ingresso `implementation-r006-fix-context-r001` MATCH, branch
`feature/run-a003-engine-comparison`, HEAD=dev/base9f702a2, indice vuoto.
Ripresa dal contesto su disco; nessuna delega, review impersonata o GO autore.
Quattro chiusure autoriali verificate nel profilo finito, da sottoporre alle due
nuove review indipendenti. Human PENDING, fase1 aperta e debito a002/ADR0008 invariati.

Indice completo (prova locale non versionata),
matrice (prova locale non versionata),
protocollo v5 (prova locale non versionata),
invalidazioni (prova locale non versionata),
attribuzione (prova locale non versionata).
La [consegna r005](a003-engine-comparison-r005.md), i suoi NO_GO e tutti gli originali
restano storia identificata. La nuova applicazione non è implementata da questi fix.

## Quattro ID del giro r005

| ID r005 | Chiusura autoriale | Scelta e limite |
| --- | --- | --- |
| CLA-I001 | Domini fissi opachi; letture code/math indipendenti e concorrenti aggiunte fino a stabilità, senza revocare span. A1–A7 e controlli riprodotti nelle quattro partizioni. | Unione conservativa finita, con KEEP possibile. Un delimiter interno non sposta il pairing esponendo il payload. Non enumera ogni grammatica/dialetto. |
| GPT-I001 | Coppie T03 ricavate dai run massimali raw, inclusi opener in KEEP o frammentati. La coppia annotata deve coincidere con quella qualificata sul raw. | Opener più recente, flanking/escape e nesting finito; run parziali/lunghi/incroci non qualificati. Il primo marker di `**foo **bar baz**` non è rimovibile; apertura successiva e coppie semplici/nested restano positive. |
| CLA-I002 | Fence dopo prefissi quote/list/indent riconosciuti come blocchi; closer qualificato solo su riga, stessa famiglia/lunghezza sufficiente/prefisso finito. | Coordinate del raw originale. Midline/closer corto non revocano protezione; senza closer qualificato si protegge fino a EOF. Wrapper esterno rimuovibile solo dove comprovato; contenitori non qualificati restano KEEP/rifiuto. |
| CLA-I003 | Marker ordinato1–9 cifre e gate sul contesto del blocco; numero≠1 non interrompe un paragrafo. | Inizio/blank/heading finito o catena di liste qualificata. Numero/anno ambiguo preservato tramite KEEP/rifiuto, marker raw e span nei record strutturali; nessuna normalizzazione delle cifre. |

I cinque residui parziali r004 sono coperti: GPT-I001→fence, GPT-I002 e CLA-I003
sovrapposti→T03, CLA-I001→domini, CLA-I004→numero/lista. Restano mantenuti i cinque
ID r004 chiusi: GPT-I003 ledger, CLA-I002 profilo circoscritto, CLA-I005 RW,
CLA-I006 finalizzazione, CLA-I007 EXTRA. Gli omonimi di giri diversi non sono
rinumerati. TargetT04/unicità/memfd/seals/no-follow/errno/RSS/timeout restano
nelle regressioni pertinenti; nessun vecchio R trasferito all’intero modulo cambiato.

## Prove finite e limiti

11.812 varianti (prova locale non versionata):
10.508 r006 (412 positive e10.096 respinte),1.060 regressioni r005 (220/840),
244 domini r004 (36/208), tutte nelle quattro partizioni. Includono i due versi
code/math, delimiter interni/esterni, code/pre/indent/URI, T00/T02/T03/T06/T07 e
altre rimozioni ammesse, apertura/chiusura, fence3/4/tilde/info/no-info,
quote/list/indent, numeri/negazione, competizione marker e positivi esterni.
KEEP preserva i byte in ogni negativo. Non si usano unit_id, gold o pair_id
per provare dominio o coppia raw.

30 controprove ricevute/120 varianti (prova locale non versionata)
sono rieseguite contro baseline r005 e v5, compresi i32 casi marker richiesti.
102 target T04 (prova locale non versionata)
sono rigiocati senza fetch. I test conservano semplici/nested, opener successivo,
escape pari/dispari, flanking/punteggiatura, domini protetti e vecchi negativi.
I sintetici sono controprove di contratto, non nuove perdite attribuite al corpus.

Tre osservazioni Claude restano esplicite, come nell’arbitrato:

- Code indentato dopo heading senza blank resta fuori dal profilo circoscritto
  inizio/dopo blank. La controprova registra ancora l’accettazione di T02 con
  annotazione prose: nessuna promessa di protezione universale; KEEP conserva
  i byte e la qualificazione code di quella forma resta indisponibile.
- T08 doppioLF può concatenare frammenti nella stessa unità; unità distinte
  restano distinte. LF singolo interno respinto. Questo non prova sillabazione
  o fedeltà generale, non vieta ogni blank e non elimina contenuto nonblank.
- La fixture RuntimeError temporanea non eseguita da Claude r005 non è una sua
  prova. I mock autoriali correnti sono eseguiti in fixture proprie eliminabili;
  non attestano R autentico o inferenze.

## Misure rigiocate

Replay finale (prova locale non versionata):
23mappe/proiezioni nuove,17assessment su67unità da sette gold indipendenti e14target.
Emissioni23/23 identiche. Il confronto integrale trova negli assessment il solo
campo version v4→v5; classi, edit, numeratori, denominatori, EXTRA, allineamenti
logici e canali critici restano identici. Gli allineamenti nuovi identificano
la precedente versione. La matrice cambia soltanto attribuzione/versione e i
campi derivati r006; il resto dei campi è verificato per uguaglianza.

Censimento reale (prova locale non versionata):
48T01 qualificati (42heading/2fence/2note/2caption), zero liste;332T08
(322blank-layout/10pagine). Nessuna mappa è stata forzata al vecchio PASS.
Somme EXTRA (prova locale non versionata)
ricalcolate dai raw: MarkerC06p2 CER2/629, DoclingC06p2 6/629, DoclingC10p1
9/632, p2rasterEN36/600, p3rasterIT7/175. Cinque inserzioni senza aumento dei
denominatori; ratio locale zero-denominator/FAIL_EXTRA e ambiguità UNAVAILABLE
restano coperti. C01 usa il canale letterale separato.

TXT/DOCX preservati nel campione; EPUB perde nota/backlink, tutti i PDF fullFAIL.
Soglie native0 e raster1%CER/2%WER invariate. Raw/gold/IR/asset intatti.
I percorsi dei motori non importano text_projection:13conversioni ufficiali e
4piloti storici riusati per byte/dipendenze, nessuna nuova inferenza PDF.

## Ambiente, boundary e suite

S2/B2/I2 (prova locale non versionata):
nuove sdist/wheel, canonica reinstallata nei quattro prefix,40hash
source/wheel/installed uguali. Managed locale3.12.13 verificato, uv0.10.10 e
backend84 invariati; closure/startup/pesi/font/profili riusati per identità.
Solo due file diagnostici cambiano rispetto a r005: text_projection e confinement;
quattro test adattano le fixture a DELIVERY corrente, test_contract_r006 è nuovo.
Consumer/driver/pdf_child/monitor/runner/verificatore sono byte-identici alla
ricezione; l’identità di quattro funzioni non trasferisce il PASS del modulo.

L’adeguamento operativo necessario autorizza soltanto DELIVERY r006 e
setup-r006-002/dist-fix-r006-002 nei due container multiprefix già disposti.
La allowlist RW esclude anche setup-r005/dist005 e il primo setup r006 concluso.
Container RO, RW soltanto temporanei/output nuovi e singolo prefix selezionato;
input/config/runtime RO e tentativi new-only. Per il modulo cambiato sono
ripetuti 72policy (prova locale non versionata),
cinque staging minimi (prova locale non versionata),
TXT003 reale (prova locale non versionata)
SUCCESS/R PASS/44sonde PASS e
due negative live (prova locale non versionata)
clone/concluso prima del boundary. Nessuna nuova invocation/admission nelle
negative; la consumer-process del positivo è storia preservata.
Postcheck (prova locale non versionata)
conferma payload byte-esatto, identità artefatti/input e figli raccolti.

Suite ufficiali (prova locale non versionata):
78test+12.007subtest regressioni e3.234test+12.317subtest suite veloce PASS,
zero skip/collection errors, exit0, R/preflight/postcheck PASS e input invarianti.
Selezione veloce `tests/ --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q`;
smoke radice esclusi. Regressioni timeout900 interno/900monitor, fast690/900;
non collaudo completo API/packaging/Docker, migrazione o fedeltà generale.
I13file codice/test finali coincidono nei tre snapshot delle prove principali.
S2 e questi snapshot identificano le prove eseguite; dopo l’editoria restano
storici, senza riuso come preflight per nuovi comandi --no-sync.

## Fallimenti, risorse e consegna

Chiamate e tentativi (prova locale non versionata)
conservano fixture URI malformata, timeout launcher1800 incompatibile con900,
fixture su delivery vecchia/collisione summary, EPERM NETLINK_ROUTE, path
errato del launcher finale (corretto prima del workload) e assert
editoriale sulla versione assessment. Diagnosi/fix/ripetizioni avvengono nella
stessa chat; nessuna collection riparata con install/download. Escalation
approvata per la medesima build/runner confinati; nessun rifiuto aggirato.
Una controprova pura finale ha rilevato che il primo adattamento r006 ammetteva
ancora RW sui vecchi output setup r005. Corretta la whitelist ai soli nuovi
setup-r006-002/dist-fix-r006-002 e prefix selezionati; aggiunti quattro negativi
e ripetuti build/canonica4prefix/40hash, TXT003/negative/policy/staging e suite
sugli input S2. S1/TXT002/prime suite e primo bundle sono preservati in
work/implementation-r006/failed-comparison-r001, senza PASS trasferiti.
La prima stampa postcheck diceva due TXT ma verificava una riga: versione e
receipt conservate, r002 ripete il controllo e corregge la descrizione.

Ledger r006 separato (prova locale non versionata):
711,650599CPU-s nel subset misurato,43outer/39inner esclusi. Alias per identità,
parent/argv/output/manifest/intervalli/wait4 verificati;33step setup sono già
compresi nei tre producer monitorati. Non si sommano i subset r004
1086,484211/128outer9inner e r005411,133918/53outer5inner. TotaliCPU/wire/energia
UNKNOWN; editing/hash/copia/scoring non integralmente monitorati. €0/documenti
remoti0. Nessuna pulizia globale:19.238file storici verificati byte-identici.

Bundle passivo comparison-r006 con copie/hash, producer e argv reali, dipendenze
di lettura comparison-r005/r004/r003/r002 e references. JSON con path assoluti
**LOCAL_EXECUTION_REFERENCE**, nessun runtime/peso/viewer attivo. Snapshot tecnico
finale `a003-r006-implementation-delivery-001` e indice integrativo temp/archive,
verificati separatamente senza selfhash circolari. Link/JSON/whitespace/diff-check
verificati alla consegna; limiti scientifici e operativi restano espliciti.

Prossimo ruolo: supervisore crea nuovo comune, due NUOVE review mirate indipendenti
ChatGPT/Claude e arbitrato. Nessun GO autore/firma/L04/scelta motore/nuova fase/UI/
deploy/main o Git mutativo; commit/merge/push restano manuali dopo GO.
