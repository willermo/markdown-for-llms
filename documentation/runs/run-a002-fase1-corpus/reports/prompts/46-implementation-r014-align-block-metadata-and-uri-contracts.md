# Implementazione a002 r014 — contratti URI,blocchi e metadata

Agisci esclusivamente come implementatore in una **NUOVA chat con contesto fresco**.
Subentri alla chat autrice r013 per disposizione dell’utente: stessa run,
storia, arbitrato e contatori, senza reset. Fix locale completo **F58–F62** dopo NO_GO r013 reale,stessa run/incremento.
Non implementare soltanto gli esempi,non delegare/impersonare reviewer o
supervisore,non modificare stato condiviso/arbitrati/output altrui. Nessun
nuovo piano,parser universale,motore,ML,nuova conversione/V o cleanup.

## Decisione,identità e budget

Leggi integralmente [arbitrato r013](../arbitrations/arbitration-implementation-r013.md)
e [metodo per contratti](../notes/recovery-method-r014.md). Quattro cause
media/bloccanti accolte,non nuove regressioni V13 attribuite genericamente.
Gli originali F54/F55 funzionano nei domini dimostrati,F57 chiuso salvo O003
non bloccante;le invarianti globali restano parziali. I difetti sono nel nostro
reader/adapter,non in Pandoc o nei modelli di conversione.

Repo /home/davide/workarea/markdown-for-llms,run run-a002-fase1-corpus,
feature/run-a002-corpus,HEAD/dev/origin-dev locale/merge-base
80c31da46e5caab990d3596c79c44e524ec1dc95,indice vuoto all'arbitrato.
Controlla delta esistente,niente reset/checkout/query remota/Git mutativo.
Comune revisionato r013 SHA256ac928bb37a34e637b352013241652132e03daa86412c32e2705655e92549bccc,
impronta062e6fd310234f0e3953f0c48b3c9478f94b5ac2223a8d2f430534ec4d141685,
344file/102949artefatti/20PDFgenerali. Storico dopo quattro registri amministrativi,
non rigenerare perMATCH. Reader V13 observations-v13-ordered-guards-fences,
79328B,SHA25627d1b5809bc2dca51b75e50222aaab1eef86b2c907732c33537fbb5df20c43ad.
V12 fisico work/implementation-r013/r012-observations.py,
SHA2565cabe6778ed53149ad02e1f50aaea6f334fc1c7ff3f9ffd38dd7f87c010abc22.
Piano3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8,
errataa09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9.
Ingresso corrente **snapshots/implementation-fix-context-r013-fresh-author-r001.json**,
identità in evidence/supervisor-author-transition-r014-r001/snapshot-receipt.json.
Il precedente fix-context-r013 resta storico e immutato nei byte: il suo
STALE dopo questa transizione è attribuito al mandato e ai registri aggiornati.
[Disposizione di nuova chat](../notes/user-fresh-author-r014-2026-10-11.md).
È compatto:byte comune r013/storia legata/prove reviewer esplicite,non pulizia
o nuovo comune di review,non GO. Conserva i byte di V13 eadapter prima dei cambi.

**Nessuna deadline della run**,addendum-resources-r012-r001 e fonte utente
no-deadline prevalgono sui gate temporali storici. Clock per accounting,
entry08ott07:06:54.369068Z e contatori senza reset. Altri limiti invariati:
CPU21600/**stop19800s**,disco6/**stop5GiB**,RAM8/stop sampled6GiB,
riserva1800CPU-s/1GiB,R≤900s/comando/stream1MiB/file32MiB,
rete/acquisizioni/invii0/€0benchmark,isolamento/canonicalSBI invariati.

Ledger corrente **evidence/supervisor-author-transition-r014-r001/budget-final.json**.
Base due review ricevute19366,705447CPU-s lower comprende GPT110,188566,
Claude87,33 eadminGPT0,193093 una volta,poi costi supervisore nel ledgerfinale.
Riserva120s per interruzione GPT non misurata è **prudenziale**,non costo
certificato;le vecchie200s riservate all'altro reviewer sono sostituite dai
costi reali,non aggiunte. 45sClaude r007approx separati,admin in parte non
misurata,CPUlower/RSSsampled,no uppercertificato. Retainedscope5dedup,
non somma storico oCPU R/base annidate. Non presume nuova quota dalla
rimozione della deadline né promette spazio per ogni futuro ciclo.

Pianifica **circa180CPU-s propri misurati o meno**,comprese preparazione,
build/install/preflight/snapshot/R/errori/prove finali;non è una quota nuova.
Proteggi margine per nuova ricezione/due review/arbitrato e120s prudenziali.
Usa residuo effettivo del ledger,non399s iniziali prima dell'amministrazione.
Non avviare un comando che prevedibilmente supera il residuo disponibile.
Al primo stop non temporale sostanziale reale conserva CP e chiedi disposizione
necessaria;continua lavoro indipendente consentito. Non consegnare un fix
come completo se mancano prove invalidate dal codice. Se il costo previsto
non è compatibile,checkpoint con lavoro/prove eseguibili e disposizione
necessaria,senza inventare consumi,estendere quote o indebolire R/SBI.

## Recupero selettivo e verify

Recupera il contesto dai file sotto elencati, senza presumere la memoria della
chat precedente. Il mandato tecnico F58–F62 e l’arbitrato r013 restano validi.
Non riaprire pianificazione o review già concluse. Registra provider, modello
e effort effettivi se esposti; il cambio di chat non cambia automaticamente
configurazioni o autorizza servizi nuovi. Se una consegna r014 è già presente,
segnalala al supervisore prima di ripetere il lavoro.

Usa manage-implementation-run e verify-conversion-fidelity. Percorsi sotto
temp/run-a002-fase1-corpus salvo indicazione diversa. Nell'ordine:

1. AGENTS,temp/HANDOVER,prima parte correntePROJECT-CONTEXT,STATE/HANDOVER/
   brief,questo prompt/CP supervisor-author-transition-r014-r001;skill,
   run-lifecycle/ADR0001,addendum r012 corrente/no-deadline eledgerfinale.
2. Arbitrato r013,metodo r014 eentrambi report/checkpoint reviewer r013
   **integrali**. GPTcases001/002/probe-results001/002/owncheck001/AST,
   oraclecorrections/gatebinding/Rprobe002–003/accountingfinale;
   Claudecases.py/freeze/check002/outprobe001summary/AST/Rprobe001/budget,
   originali K1/K2/K3/K4/A/P econtrolli corretti. Sono storia ricevuta,
   non riscrivere review o abbassare severità. Recuperi ecorrezioni oracoli
   separati,nessuna nuova CLI attribuita alla traccia statica GPTexit4.
3. Report/checkpoint autore r013,censimento precedente,PROTOCOL/due guide/
   permanente/dati eprovecases005/finalafter006/mapping16/mutation002/
   overlay20/F29/checkbaseline005/S008runtime/S009CLI/S010documentale.
   Arbitrato r012 emandato43 per obblighi ricevuti,non tutti i log storici.
4. Sorgenti pertinenti integralmente nei rami toccati:observations scanner/
   producer/consumer/context/fence/metadata/source_map/recursion/guard/drain/
   fallback/node/public e safety.local_uri/PassiveHTML/observe_bundle/audit/
   CLI. Piano/errata/mandati05–06/guideuv/toolchain/contrattiF31/F36/F40/F41/
   F44/F46/F50–F57 dove decisivi. Non nuovo piano diarchitettura.

All'ingresso conserva argv/cwd/output/exit/costo:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r013-fresh-author-r001
```

STALE attribuito ai delta autorizzati,mai restaurare V13 o rigenerare storico
perMATCH. Nuovi snapshot tecnici puntuali prima delle prove quando richiesti
da R,non freeze per ogni argv/microcaso. Comune review finale resta al supervisore.

## F58 — adapter URI con esito di rifiuto contrattuale

GPT-R013-I001: `È <script>x</script> <img src="http://["> <img src="../last.png">`.
Atteso prima detail script,tre aperture char2/21/42,byte3/22/43,ordine/multiplicità
esatti,prima causa conservata,coda visitata. V13/V12 propaganoValueErrorInvalidIPv6,
audit senza risultato contrattuale;reset passa. Variante malformedURIprima
dello script blocca già alla prima img. Nessun fetch o falsoCOMPLETE dimostrato.

Contratto all'adapter:classificare i prevedibili errori del parsing URI come
rifiuto Unsafe,senza accettare URI invalide o cambiare whitelist. È autorizzato
il fix locale in safety.py oadapter pertinente;non obbligo di limitarlo a
drain. **Non catch generale di Exception/ValueError nell'intera observe**:
non nascondere errori di programmazione o proiezione estranei al parsingURI.
Analizza i punti di errore effettivi diurlsplit/proprietà utilizzate ecallsite
raggiungibili. Prima causa evisite devono continuare per l'intera classe
di rifiutiURI,anche più malformati/safeprima/dopo,href/src e UTF8.
Preserva frammenti,relativi consentiti,path evasivi,schemi vietati/encoded,
valori assenti eURIesterne già consentite come stringhe mai fetched.
Prova a livello adapter eobserve/audit/CLI reale di pochi casi discriminanti:
esitoINCOMPLETE/UNSAFE_REFERENCE eexit2/2 previsti,non erroreinternoexit4.

## F59 — accettazione dell'opener ordinario prima di assegnare code

GPT-R013-I002: `~~~foo bar` chiuso conscript,quote equivalente eimg p.png.
AST3.1.3:HTML reale nel corpo,nonCodeowner;V13/V12 COMPLETE senza visite,
asset perso. Controlli `~~~python`/`~~~{.python}` sono CodeBlock reali.
Backtick info apparentemente invalide possono invece possedere **CodeInline**:
tre premesse errate del reviewer sono conservate ecorrette,non regressioni.

Definisci la grammatica finita dell'opener ricevuto usando AST pinned per
partizioni info vuota/lingua valida/attributi validi/info invalida/rawesatto,
spazi/tab/marker/chiusura pertinenti. Solo accettazione sintattica eowner
dimostrati rendono inerte il corpo. Condividi la decisione scannerproducer
consumer eparagrafo;non introdurre altre regex indipendenti sugli stessi byte.
Fuori dalla forma accettata,HTML reale riceve guardia easset/testo sono
preservati oppure residuo esplicito conraw/risorsa/posizione,maiCOMPLETEvuoto.
Preserva rawesatto{=html}/{=latex},F36,codeInline reale ebodyletterale,
EOFoverlay ricevuto;non imporre secondofeedHTML indiscriminato ohtml5/space.

## F60 — stato del blocco attraverso i confini dei producer

CLA-R013-I001: K1a–d container senza blank dopo paragrafo,K2a/b/d lazyquote/
note eK2e figura→tilderadice. ASTdel corpo è HTML reale,V13/V12 COMPLETE.
Pinned+blank_before_blockquote/-lists_without_preceding_blankline e
+blank_before_header vanno rispettati nei domini modificati. K2clistlazy è
CodeBlock reale,non finding;K1ebacktick/K1frawhtml/K2fbacktick controlli corretti.

La stessa funzione con inputstato diverso non è una decisione condivisa.
Produci/consuma uno stato fisico orecord di decisione di blocco comune,
oppure controlli equivalenti espliciti,che conservi inizio/paragrafo aperto/
separazione/host/continuazione eposizioni. Non azzerare lo stato all'ingresso
di quote/list/note/figura né chiamarefence_decision al root condefault
incoerente conledger. Rivedi PARAGRAPH_BOUNDARY e i chiamanti reali:
un marker non termina automaticamente il paragrafo se quel blocco non è ammesso.

Non serve parsernesteduniversale. Per contesti non proiettabili serve guardia
dei tag reali e residuo esplicito,non proprietàcode ottenuta per fallback.
Inclusa O001nonbloccante comecontrollo:heading non separata chePandoclegge
testo conserva quel testo eonclickletterale,non falso attributo attivo.
Mantieni heading realmente separata eattrs attivi rifiutati. ASTdel corpo/
tagmarcato discrimina anche figure/lazy,non presenza diCodetrovataltrove.

## F61 — metadata opachi soltanto dopo accettazione della grammatica

CLA-R013-I002: K3a/d codeopenerdiv,K4a–e HTML dentrograffe dopo span/link/
code/heading. Esempi `[a]{<img src="../x.png">}`, `# Title {<script>…}`,
`::: <img …>`:ASTHTMLreale,V13/V12COMPLETE,guardia assente/path nonsegnalato/
asset e testoletterale persi. Attributovero K4f va preservato.

Stabilisci un solo contratto di accettazione metadata nel profilo ricevuto,
con intervallo consumato/posizioni/decisione accettato o no. `[^{}]*` non è
un certificato di grammatica. Collega metadata_at/span/peel_attrs/fenceinfo/
heading/div/code/link/figure escannercheoggi salta l'intervallo.
Contenuto non valido comemetadata conserva caratteriletterali eHTMLreale
passa alla guardia;seproiezione fuori profilo,diagnosi esplicita conposizione,
maiCompletevuoto. **Valori quoted validi contenenti `<…>` restano valori**:
non reinterpretarli come tagHTML per un feed di sicurezza indiscriminato.
Preserva id/class/key=value/speciali giàammessi,LF/CRLF nelmapping,posizioni
charUTF8,attrs attivi rifiutati ecode/math reali inerti. Rifiuto sintattico
del gruppo non elimina automaticamente il contenuto né inventa nuovi blocchi.
Div chiusura/nesting/metadatafinali ebranchunsupported vanno tracciati;
non imporre nuovo schema pubblico o politicaHTML.

## F62 — contratti e prove che possano smentire le premesse

Prima del codice nota tecnica breve con:

1. Tipi di uscita dell'adapterURI ecallsite,esito di ciascuna classe di errore.
2. Regole diaccettazione blocco/fence emetadata,positivi e negativi,record
   tipizzato interno o meccanismoequivalente e intervallo/owner/source-map.
3. Tabella **ogni callsite → stato/decisione ricevuta → proprietà → caso**,
   inclusi confini tra producer;non solo lista di funzioni che condividonohelper.

Refactoring locale reversibile ammesso,anche non enumerato;nessuna nuova
architettura/toolchain/dipendenza/download. Integra ricerca dei controesempi
nella stessa chatprima della consegna. Non aumentare soltanto il numero dei test.

- Attesi/raw nuovi congelati prima del runtime,max160microcasi/2048B ciascuno,
  selezione finita motivata e seed solo se usato;nessun target numerico minimo.
  Includi originali accolti econtrolli V13corretti. Partizioni esterno×interno,
  validitàinfo×host,metadata validi/invalidi/quotedHTML×producer,erroriURI×ordine.
  Oracolo sintattico AST Pandoc3.1.3 **obbligatorio per decisioni modificate**,
  input/hash/argv/reader/exit salvati sottoR/netnone;tag/bodydiscriminante,
  non inferenza diowner daCodeinqualunquealtro ramo. Testo/asset/URI/posizioni
  e sicurezza restano attesi manuali indipendenti,non outputproductcomegold.
- BaselineV13 sui medesimi casi,prima dei cambi. **safety.py potrebbe cambiare**:
  non attribuire un V13 importato poi conadapter nuovo alla base originale.
  Usa before subyteoriginali archiviati eidentificati,oistanzia esplicitamente
  vecchio reader+vecchioadapter/copie sottoR. V12nonnecessario in questo giro.
- Quattro controprove su **copie private finali**,mai prodotto/corpus protetti:
  URIparseerror sfugge;qualsiasiinfofence accettata;statochiamante azzerato;
  qualsiasigraffe metadata. Uno/due discriminanti perassunzione,tutte devono
  fallire sucontratto/visite/asset/preservazione pertinente. Diff/hash/base e
  risultati;noninteracampagnamutation néreview simulata. Non basta mutare
  soltanto l'interno delfencehelper per dimostrare stato dei chiamanti.
- Spyuna observe:identità/snippet/char/byte/molteplicità/sequenza tutteaperture
  reali,eventi eguardie **effettivamente validate**,non solo raccolte. Reset
  dopo errori;asset safe proiettato oppure residuoesplicito,script unsafeanche
  dietro erroremetadata/proiezione. Proveend-to-end reali distinguono writer
  sintetico/goldPROVISIONAL,nessun nuovo motore o Vumana.

Documenti PROTOCOL/due guide/permanente precisi suaccettazione econfini,
non promessa universale «tutte lefonti» senza prova dei tipi diuscita ecaller.
Baseline132 cita check005/attesi005 eoutputbaseline004;preserva133/check004/
attesi004 efreeze005,nessuna riscrittura di prove storiche.
La precisazione va nel report r014 e nei riferimenti correnti,con collegamento
alla citazione storica:non modificare il permanente r013 o le sue evidenze.
O003namespaceutente nonbloccante:documentare o correggere localmente nella
stessa serializzazione seappropriato,con preservazione valore utente;non
ampliare campagne o condizionare GO a quella sola osservazione.

## Verifiche finali e organizzazione delle risorse

Nessun importprima delgate/provenienzamanaged3.12.13/copie correnti/RPASS.
Dopo modifica codice ricostruisci sdist/wheel e reinstalla canonica offline
secondoguidaAGENTS;diecihashsource/install/archive einputS/B/I prima --no-sync,
37helper/Pythonesterni o nuovocensimento esplicito. VecchioS010runtime/generale
storicoSTALE nonPASStrasferito. Nessunainstallazione/download percollectionfallita.

Snapshot tecnici **puntuali** dei byte effettivi/copiedescriptor/pinnedPython,
storia legata al comune r013 intatto. Non includere100kfile in ogni snapshot,
non rifare la build per un commento dei registri o congelare ogni argv.
Accorpa verifiche compatibili sui byte finali,conservando stdout/exit/hashes
separati eR/processreap/isolamento. Non modificare harness R oallargare limiti
per rendere economicamente fattibile una prova. Recuperi ordinari nella chat,
fallimenti conservati;escalationtool autorizzata mantieneisolamento e si ferma
al rifiuto effettivo,nessun approval per ogni comando.

Richiesti: test delle classi e regressioni301r013/350r012overlay/153r011/
303r010/308r009,casiF31/F36/F40/F41/F44/F46/F50–F57 pertinenti;fast corrente
secondoAGENTS API/packaging/Docker/smokeradice esclusi. Se un vecchioatteso
contraddice AST/requisito corretto,nonderivarlo dal prodotto:motivazione e
overlay separato,originaleraw/gold/atteso preservato,controlli diassenza
regressioni validi dichiarati;non cambiare goldcanonico per alzare qualità.
PocheCLI reali negative/safe/reset delle quattroclassi,poiH01–08/H12solo se
prove invalidate dahelper/reader effettivo;FAILH12 deliberati restanoFAIL.
64ricalcoli su raw immutati senza motori/nuovaV quandoinvalidati dai byte:
confronto19campi/gate/catene/distanzearitmetica,0delta dove atteso;qualsiasi
delta reale attribuito egiustificato,nonforzare4PASS/28FAIL. Reader/canonical
assets/text/coordinate osservati effettivamente,unknownnonwildcard.

Non ripetere tutta la vecchia amministrazione né trasferire prove a byte
diversi. Se costo/residuo incompatibili,**checkpoint esplicito incompleto**
prima di superare limiti e disposizione necessaria,non consegna «completata».
Nessuna nuova Vgenerale,L04outputUNAVAILABLE distinta,V13/A03 chiusi,
L01/L02/L05,F36B1–B4/A8/quattroOsr010/ChromeNONPASS/helper12gap/H15/E9/
FAIL/STALE/erroriintermedi preservati,nessuna sanatoria o cleanup.

## Consegna e passaggio

Scrivi reports/implementation-r014.md,handovers/implementation-r014.md,
evidence/implementation-r014/,work/implementation-r014/,nota tecnica e
reportpermanente documentation/benchmarks/2026-10-11-a002-r014-contratti-sintassi.md
con dati/provenienza collegati. Nuovefixture/test sintetiche identificati,
nessuna riscrittura di artefatti protetti,registroregole/callsite econtroprove
pubblicato prima delreportfinale. Descrivi scelte/rischi/limiti,comandi reali,
hashbytefinali/gates/Python/copie/proveinvalidateripetute/CPUlower/discoscoped/
RSSsampled/net0/€0/pending0 eclocksenza deadline. Snapshot delivery tecnico
opzionale puntuale/verify,nonnuovocomune oGO.
Consegna completa al supervisore→nuovo comune→due NUOVEchatChatGPT/Claude
indipendenti→arbitrato. Nessun GOautore o promessa ultimo giro. Git commit/
merge/push/promotion manuali utente dopo GO finale,nessunGitautomatico.
