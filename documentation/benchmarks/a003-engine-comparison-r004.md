# A003 r004 — Contratto finito e policy dello scratch

Nota di conservazione aggiornata il 2026-10-15: le prove citate sono locali e
non incluse in Git; risultati e attribuzioni di questo giro restano storici.
[Disponibilità e limiti](../runs/run-a003-fase1-engine-comparison/manifest.md#disponibilità-delle-prove).

2026-10-14 Europe/Rome. Consegna autoriale del mandato12 e dell’arbitrato r003:
sei rilievi corretti e verificati nel perimetro finito. Ruolo implementatore,
skill manage-implementation-run e verify-conversion-fidelity; nessuna delega,
review impersonata o GO autore. Human PENDING e fase1 aperta.

Indice completo di lettura (prova locale non versionata),
matrice (prova locale non versionata),
invalidazioni (prova locale non versionata),
attribuzione (prova locale non versionata).
La [consegna r003](a003-engine-comparison-r003.md) rimane storia identificata:
il successivo NO_GO riguarda il contratto, senza imputare i sintetici al corpus.

## Chiusura della classe del comparatore

`a003-text-projection-v3` verifica prima la copertura raw, i confini UTF-8,
le identità e le relazioni finite. Costruisce poi intervalli protetti sul raw
completo, prima dell’emissione di T00/T02/T03/T07. Nessun matching con il gold
determina il dominio. Anche T01/T04/T05/T06/T08 sono respinte nel payload
code/math/literal; l’unica eccezione è il wrapper esterno di un fence chiuso,
con apertura/chiusura comprovate sul raw e payload byte-identico. `unit_id` raggruppa le emissioni per l’allineamento e
non rende prosa il payload tra delimitatori code/math o all’interno di un URI.

Il profilo conservativo comprende run backtick e dollaro, paren/bracket math,
fence tilde, tag code/pre e URI con schema o bare www. Le aperture incomplete
proteggono la regione non qualificabile; KEEP conserva i byte, una trasformazione
in quella regione è respinta. Non è un parser Markdown/LaTeX generale e non
introduce normalizzazioni. Un contesto prose sovrapposto non revoca le protezioni;
un URI dichiarato frammentato aggiunge un vincolo, senza rimuovere byte.

T03 verifica flanking, underscore intraword, run indivisi, contenuto non vuoto,
coppie e nesting. T00 rimuove solo padding a confine raw di riga oppure davanti
a un suffisso strutturale già comprovato e conclusivo. Un confine inventato
tra unità non qualifica uno spazio interno. Soft-wrap, CRLF e hard-break restano
distinti; niente collapse. T02 richiede prosa fuori dai domini protetti; T07
richiede la coppia finita di tag HTML e non opera nei URI.

T04 determina dal raw la prima conclusione valida del target nudo con parentesi
bilanciate, oppure di `<...>`. Non accetta whitespace, controlli, escape/dialetti
non qualificati o titoli; la forma angle non ammette angle interni/newline.
La chiusura deve coincidere con lo span annotato. Un target che ingloba la prosa
successiva viene respinto anche cambiando label/unità. Label, target e dichiarazioni
restano nel registro riferimenti, senza fetch o riscrittura URI.

La prova finita contiene **244 casi di dominio/partizione**, con stessa regione
raw e quattro variazioni dei segmenti/ID, positivi di prosa fuori code/math nello
stesso documento e negativi di dichiarazione artificiosa; **102 casi T04** con
label differenti, controlli/whitespace/parentesi/prosa assorbita e positivi;
le 14 destinazioni native reali restano qualificate. Altri16negativi di rimozione
sintattica nel literal e un positivo di fence esterno coprono la stessa classe. Raw, annotazioni complete,
byte hex, hash, risultati v2/v3 e rifiuti sono conservati. Il rifiuto non viene
convertito in uno zero: le prove trattengono raw e il registro degli attesi.

## Rigioco del campione e producer

Producer `replay.py`, versione/hash/argv e copia passiva identificati nel bundle.
Migrazione v2→v3 senza gold nel passo di qualificazione: nove spazi T00 interni
privi di prova raw diventano KEEP in cinque mappe, comprese due dei serializer
Docling originali. Nessun raw, gold, IR o asset viene modificato.

Tutte le **23 mappe** sono rigiocate con il helper finale; **18 emissioni** sono
identiche a r003. I **17 assessment** sono ricostruiti dai **67 gold indipendenti**:
14 restano identici salvo il campo versione; tre conservano nuove unità EXTRA,
prima vuote, senza attribuire una pagina/classe inventata. Docling C06 aggiunge
uno spazio, Docling C10 tre e Marker C06 uno. Le altre quattro modifiche riguardano
le due mappe originali Docling, senza nuovi assessment ufficiali.

Gli aggregati per pagina/lingua/native-raster e tutti i denominatori attesi sono
identici a r003. Ogni spazio EXTRA ha distanza CER1 e denominatore0, esplicitamente
`UNAVAILABLE_ZERO_DENOMINATOR`; WER `NOT_APPLICABLE`. Non è un nuovo testo PASS.
Tracce, extra e allineamenti nuovi rimangono distinti dagli originali. Dopo gli
ultimi fix della qualificazione, il rigioco finale riproduce esattamente questi
derivati r004: nessun altro ricalcolo o inferenza è necessario.

TXT/DOCX preservati nel campione; EPUB perde nota/backlink; nessun PDF fullPASS.
Le osservazioni critiche su formule/celle/figure/numeri/ordine/copertura e i FAIL
restano quelli attribuiti ai raw storici. UNKNOWN/UNAVAILABLE e human PENDING
rimangono visibili. Le soglie native0 e raster1%CER/2%WER sono invariate.

## Sei rilievi r003 e chiusure precedenti

| ID del giro r003 | Correzione e prova autoriale |
| --- | --- |
| GPT-I001 | Protezioni su intervalli raw indipendenti da `unit_id`; split dei delimitatori/payload e ID rinominati non autorizzano T02. Matrice244 e positivi fuori regione,23mappe qualificate. |
| GPT-I002 | T03 fuori URI/code/math, underscore intraword e run ambigui respinti, coppie/nesting verificati prima delle emissioni; positivi annidati e negativi incrociati conservati. |
| CLA-I001 | T04 controlla la conclusione sul raw, non due stringhe annotate uguali;102varianti e14target reali, nessuna prosa assorbita nel canale riferimenti. |
| CLA-I002 | RW limitato al leaf di attempt della run e della delivery, legato al percorso di evidenza/config; setup soltanto nei due scratch già disposti.40casi policy, overlay input, cinque famiglie ammesse e nuova negativa live. |
| CLA-I003 | Guardie comuni T00/T02/T03/T07; padding interno/URI ed enfasi letterale respinti, aperture incomplete KEEP/rifiuto conservativo. N6 resta respinto sui byte reali.9spazi storici preservati, nessun falso zero occultato. |
| CLA-I004 | Cleanup sentinella via dirfd no-follow e solo leaf; errori inventory/copia/identità registrati, finali non verdi e parziali conservati.9casi mock espliciti e completamento TXT autentico. |

Gli ID omonimi r002 restano separati. Le sette chiusure riconosciute dall’arbitrato
sono mantenute: unicità, memfd/no-follow, URI T02/T07, regressioni, errno e
RSS/timeout. I residui del vecchio GPT-I002/E01 e CLA-I005/RW sono trattati qui.
Consumer, pdf_child, runner e monitor sono byte-identici alla ricezione; le funzioni
memfd/safe_bytes/safe_json/sealed_admission del confinement hanno sorgenti identiche.
Non si trasferisce il PASS dell’intero script cambiato né si riaprono vecchie classi
senza una dipendenza toccata. Le regressioni precedenti passano nella suite attuale.

La frase r003 «sette forme respinte» riguardava i **bind RO** storici. La policy
**RW nuova** è verificata separatamente; nessuna fuga viene imputata ai manifest
r003 reali che già usavano un singolo attempt.

## Prove operative, errori e installazione

TXT reale finale `a003-r004-real-txt-004`: SUCCESS, R/confinement PASS, input invarianti,
figli raccolti e `final.json` persistito. Manifest minimi TXT, Pandoc diretto/legacy,
Marker e Docling riqualificati dalla policy attuale: ammissibili tutti e cinque.
Questa è validazione delle famiglie, non cinque nuove conversioni o inferenze ML.
La negativa live RW clone avvia il runner host e produce R FAIL prima di bwrap,
consumer e motore; ammissione assente, figli raccolti. Nessun bind vietato eseguito.

I nove casi di finalizzazione sono **mock**, senza R autentico: normale, sentinella
assente/directory/leaf/link cambiato, copia fallita, artifact con stessi byte e nuovo
inode, lettura fallita e final non scrivibile. I primi otto producono un record finale
coerente; l’ultimo conserva causa ed errori nel canale host senza inventare final.json.
Le directory sostitutive e i loro parziali restano intatti; il target esterno non
viene seguito/eliminato. Le receipt mock sono marcate come tali.

Preparazione finale **preflight-r007**: nuova sdist/wheel offline, canonica reinstallata
nei quattro prefix già preparati,40confronti source/wheel/installed dei dieci moduli.
Managed locale3.12.13, uv0.10.10 e backend84 invariati; I cpu Marker e I Docling separati,
dipendenze/startup confrontati per identità. Pesi/font/runtime riusati selettivamente;
niente download, nuovi pin o ricreazione universale.

Fallimenti conservati: NETLINK_ROUTE EPERM del sandbox esterno, poi escalation
approvata con bwrap isolato; snapshot S2 stale durante una modifica ai test;
deadline1080 non ammessa dal monitor; positivo fence con chiusura scambiata per nuova
apertura; fixture tempfile con underscore nell’ID. Correzioni e nuovi tentativi
nella stessa run, senza indebolire guardie o cambiare codici per ottenere verde.
Le versioni S3/S4/S5/S6 e le prove precedenti restano attribuite ai loro input.
Consegna tecnica001 e archivio iniziale rimangono storici; la002 aggiorna la guardia
sintattica e ripete le prove dipendenti. Conservati anche la collisione iniziale
report/checkpoint nel bundle e un errore di path nel launcher, corretti localmente.

**61 regressioni e428subtest PASS; suite veloce ufficiale3217test e738subtest PASS,
zero skip**, R PASS, preflight/postcheckI, invarianti e stream/exit conservati.
Il launcher finale usa690s interni e900s esterni. API/packaging/Docker/smoke radice
sono esclusi; questa suite non dimostra fedeltà o collaudo completo della migrazione.
S7 è storico dopo i nuovi report: nessun futuro `--no-sync` autorizzato da questi esiti.

## Consumi, portabilità e passaggio

Ledger r004: **1315,793871 CPU-s misurati in133intervalli outer**, quattro intervalli
consumer annidati esclusi e alias deduplicati. Include fallimenti/preparazioni
invalidati; nessuna somma dei ledger r002/r003. Totale CPU/wire/energia UNKNOWN,
attività host di editoria/hash/copie non chiamate zero. Costo monetario€0,
documenti inviati a servizi remoti di conversione/revisione0.

Bundle nuovo `comparison-r004`, copie passive originali e producer/hash/argv.
Dipendenze di lettura dichiarate e verificate: `comparison-r003`, `comparison-r002`
ed `evidence/references`; raw, gold, trascrizioni, render, asset e IR storici rimangono
accessibili. JSON con path assoluti sono LOCAL_EXECUTION_REFERENCE, non una promessa
di esecuzione portabile. Nessun peso/runtime incluso nell’archivio.

Snapshot tecnico di consegna `a003-r004-implementation-delivery-002` e indice integrativo
dei path esclusi da run_context; verifiche finali di copie/hash/link/JSON/whitespace
nel checkpoint autoriale. Originali e registri condivisi preservati. Nessuna pulizia,
operazione Git mutativa, nuova app/parser generale/UI/deploy o fase successiva.

Prossimo ruolo: supervisore, ricezione completa, nuovo comune e due review nuove
indipendenti ChatGPT/Claude mirate ai fix e alle dipendenze, poi arbitrato.
A002/debito ADR0008 e fase1 aperta invariati; nessuna firma umana o scelta motore.
Commit/merge/push rimangono manuali dell’utente dopo GO.
