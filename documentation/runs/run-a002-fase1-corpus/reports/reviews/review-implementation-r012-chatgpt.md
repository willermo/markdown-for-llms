# Review implementazione a002 r012 — ChatGPT/OpenAI

**NO_GO.** Le classi originali F50/F51 e raw inline della label F52 sono
sostenute dalle prove ricevute e dalle sonde mirate. Rimane un requisito
essenziale delle interazioni F52/F53: attributi Markdown attivi in un blocco
successivo sostituiscono la prima unsafe HTML e interrompono le guardie dei
domini seguenti. Finding **GPT-R012-I001, media/bloccante**, riprodotto sotto R
sui byte del comune. L'audit resta unsafe; nessun falso PASS o fetch dimostrato.

## Ruolo e oggetto

11 ottobre 2026. Reviewer ChatGPT/OpenAI in nuova chat indipendente, estranea
all'autore e al supervisore. Provider OpenAI; agente Codex basato su GPT-6
secondo le istruzioni esposte. Deployment preciso, effort e ID piattaforma
non esposti. Nessuna delega, impersonificazione o lettura di materiale reviewer
concorrente r012 o arbitrato futuro. Applicate
[manage-implementation-run](../../../../../.agents/skills/manage-implementation-run/SKILL.md)
e [verify-conversion-fidelity](../../../../../.agents/skills/verify-conversion-fidelity/SKILL.md).
[Prompt41](../prompts/41-review-implementation-r012-chatgpt.md) e
[mandato40](../prompts/40-implementation-r012-align-label-atoms-and-container-raw.md)
dopo NO_GO r011 reale; nessun nuovo piano.

| Identità | Oggetto verificato |
| --- | --- |
| Comune | `snapshots/implementation-r012.json` |
| SHA256 comune | `a8ad87d33d3d5ea6fecf40557cb0e470b1282bd043c312b005fd7ad17b048625` |
| Impronta | `c5c7eb56c0933f8653a6a71354b224d0c885cfe6105a4ffb99dc982be9108378` |
| Perimetro | 338 file generali/107888 artefatti/20 PDF generali/29921684 B snapshot |
| Branch | `feature/run-a002-corpus`; indice vuoto all'ingresso |
| HEAD/dev/origin-dev locale/merge-base | `80c31da46e5caab990d3596c79c44e524ec1dc95` |
| Reader | `observations-v12-shared-atoms`,74476 B,SHA256 `5cabe6778ed53149ad02e1f50aaea6f334fc1c7ff3f9ffd38dd7f87c010abc22` |
| V11 fisico | `work/implementation-r012/r011-observations.py`,67260 B,SHA256 `af5299bb0c52bd4a0634a1aebb8a28743e2793c315eec5bbaf9b6424a98d8482` |
| Piano/errata | `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8` / `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9` |
| Delivery autore001 | SHA256 `fd15440f45ba8dcc106fa097edfcbe8691ea52d408d69bedefae028f1fe9dbc5`, distinto dal comune e da GO |

Clock reale iniziale **2026-10-11 13:07:31Z**. Fonte utente no-deadline e
addendum r012 letti: nessuna deadline/finestra24h/riserva temporale o richiesta
temporale pendente. Entry cumulativa `2026-10-08T07:06:54.369068Z` invariata.
Altri limiti e riserve non aumentati.

Verify amministrativo host realmente eseguito all'ingresso, cwd repository:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r012
```

Exit0 **MATCH**,wall49.334160s/CPU49.245357s, in
[verify-entry.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r012-chatgpt/verify-entry.json).
Binding R prima/dopo sonda riuscita MATCH sul medesimo SHA. Verify d'uscita
dopo report/checkpoint nel record separato `verify-exit.json` della stessa
cartella evidence; esito nella nota finale di chiusura.

## Letture e metodo

Letti AGENTS,temp/HANDOVER,prima parte corrente PROJECT-CONTEXT,STATE/HANDOVER/
brief,prompt e checkpoint supervisore r012;skill,ciclo delle run,ADR0001.
Arbitrato r011,mandato40,recovery-method-r012,entrambi i report/checkpoint
reviewer r011 e tracce originali GPT/Claude T1–T4 letti integralmente. Sono
storia ricevuta, non nuove prove runtime. Report/checkpoint autore r012
integrali e inventario atomi anteriore al codice; nota permanente/dati,
final-delivery-inputs001,delivery-check001,budget-delivery-final001 e process
check consultati.

Reader integrale/delta V11,test/fixture r012,generate002,freeze-cases002,
matrix,CLI,check-derived letti. Consultati scanner/producer/consumer,
ricorsioni/source-map/node/PassiveHTML/safety/audit/observe_bundle/metrics/
alignment e launcher/gate. PROTOCOL e due guide, con rilettura delle sezioni
r012 decisive; arbitrati r009/r010,parti pertinenti r008/F40/F41/F36 e F31r007,
contesto architetturale/roadmap fase1,guida uv e toolchain legacy. Letture
selettive: non dichiaro lettura integrale dell'inventario storico,di ogni log
precedente o di tutti i documenti architetturali.

[Static-check003](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r012-chatgpt/static-check-003.json)
verifica1565descriptor,archivi senza estrazione,origini S/B/I,record/catene/
aritmetica, senza import prodotto. Non è rescore,nuova suite o ricertificazione
indipendente dell'alignment. Due errori propri preservati: confronto erroneo
PROTOCOL storico S004/corrente,poi percorso sbagliato della copia storica;
corretti usando received-bytes.json,senza finding prodotto o PASS dei fallimenti.

Attesi manuali propri in
[cases001](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r012-chatgpt/cases-001.json),
fissati prima di ogni runtime prodotto:32casi,limite40/2048B,nessun generatore
casuale o helper prodotto come oracolo. Trasformazioni finite:quattro producer,
math escape-LF chiusa/nonchiusa,raw metadata,owner HTML→Markdown,duplicati
class passivi con errore d'albero,nestedquote quote/list/note backtick/tilde,
metadata tra domini,reset ed enfasi/code. Nessuna grammatica universale richiesta.

## GPT-R012-I001 — attrs di blocco interrompe la sessione globale di guardia

**Provider OpenAI; severità media,bloccante.** Mandato40 righe73,132–133 e
criteri C03/C04/C05,C09: prima unsafe in ordine sorgente,continuazione delle
visite dopo violazione/errore di proiezione,copertura effettiva dei domini seguenti.

Caso `metadata-after-first-26`,153B,SHA256
`14ea6e563477368ab3bde76fbed88c12d2dd522b890f9cffea8e9380be4b02b3`:

```markdown
È `<script data-review="first">x</script>`{=html}

# Heading {onclick="x"}

<img src="../last.png" data-review="last">

# Visible heading

After words.
```

Atteso fissato prima:detail **script**,aperture sorgente `first,last` in
quest'ordine,una volta ciascuna. Posizioni char/UTF8: `first`3/4,`last`76/77.
Osservato realmente: `Unsafe('Active markdown attribute')`,sole visite
`['first']`;l'immagine seguente non è guardata. Audit non strumentato dello
stesso bundle: `INCOMPLETE/UNSAFE_REFERENCE`,detail `Active markdown attribute`.

Trasformazione `metadata-after-first-27`:heading sostituito da
`::: {onclick="x"}`,147B,SHA256
`1f075f8db02b860cdc5402d7c54615751486c21b98feb297392cb04bc9ac2059`.
Opener `first`3/4 e `last`70/71;medesimo risultato. Controllo con
`[word]{onclick="x"}` in un dominio inline successivo conserva invece script
ed entrambe le visite: inline guarda il dominio e salta la proiezione quando
la sessione è già unsafe. Disaccordo circoscritto ai rami di blocco provati.

Percorso raggiungibile: inline_html_events/validate_html registrano la prima
unsafe con report_unsafe;observe differisce il rilancio fino al ritorno di
_observe. Ma [attrs442](../../../../../scripts/benchmarks/observations.py#L442)
solleva direttamente Unsafe. [Heading1120](../../../../../scripts/benchmarks/observations.py#L1120)
chiama peel_attrs prima di read_inline;[div1077](../../../../../scripts/benchmarks/observations.py#L1077)
chiama attrs nel loop dei blocchi. L'eccezione attraversa _observe,impedisce
il rilancio della prima unsafe987 e la visita del blocco HTML seguente.
Il finally resetta correttamente ContextVar:il controllo safe successivo passa,
quindi non è contaminazione fra chiamate.

Attribuzione: attrs/raise diretto ereditato;r012 introduce la sessione differita.
Nel V11 fisico la prima raw unsafe ferma subito observe con detail script;
la nuova sostituzione del detail è deducibile dal confronto statico.
**Nessun V11 runtime proprio su questi microcasi.** Mancata continuazione:
invariante essenziale non completamente chiusa,non attribuisco al V11 una
prosecuzione che non aveva.

Prova:[probe-results001](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r012-chatgpt/probe-results-001.json),
[own-results-check](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r012-chatgpt/own-results-check.json),
[R receipt003](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r012-chatgpt/R-probe-003/receipt.json).
Spy conserva tag,attributi completi,identità univoca dall'input e snippet
passato alla guardia;confronta sequenza e molteplicità esatte,non membership.
Checker host lega identità a posizioni e raw hash. Nessun cambio prodotto.

Impatto:nessuna esecuzione HTML,fetch o falso COMPLETE dimostrato. Prima causa
persa e foglia successiva senza guardia violano il requisito esplicito.
Occorre uniformare gli errori di blocco con la sessione,mantenere il rifiuto
attributi attivi e provare detail/visite/reset. La promessa permanente «ogni
dominio successivo riceve guardia» e «detail globale ... prima violazione» non
vale per questi rami. Documenti/prove da riallineare con il fix nel mandato
supervisore;nessun ampliamento di dialetto o nuovo piano richiesto.

## Sonde proprie e recuperi

Probe001 **IMPEDITA**,EPERM socket prima del workload. Escalation strumenti
richiesta e approvata,mantenendo Firejail/netnone/blacklist. Probe002 passa R
preliminare ma preflight FAIL:S005 generale storico STALE su quattro registri
supervisore. Nessun import prodotto nei due tentativi. Argv/cwd/clock/CPU,
receipt/stdout/stderr preservati.

Recovery stessa chat:gate standard corrente sorgenti contro comune e
validazione degli **stessi archivi/installazione** già preparati;nuove receipt
private current-S/B/I,build/reinstall0. Dieci origini canoniche,input build/
backend/toolchain/diagnostici,archivi/install confrontati con S005 e invariati.
Startup managed locale3.12.13,binary SHA256
`8392ec78d090ab7e1b1c058cff454de64a0780e263e6a9e66cdb1256be322fb0`,pth/assenza
customize controllati;copie helper effettivi prima della sonda. Vecchio S005
resta STALE,senza rigenerare comune o trasferire PASS. Queste receipt valgono
soltanto per la sonda propria.

[Current-gate-binding](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r012-chatgpt/current-gate-binding.json)
+input-bytes-before003 legano input al comune. R-probe003 usa target003,
snapshot comune,preflight corrente,venv managed S005,`-I -B probe-001.py`,
timeout120/expected-exit0,cwd repository. Argv completi/ambiente chiuso in
[probe003](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r012-chatgpt/probe-003.json).
R **PASS**,preflight2×exit0 prima import,workloadexit0,wall0.284023s,
netns distinta/netnone,binding prima/dopo MATCH,inputunchanged,reap/socketcleanup.
**Exit0 raccoglie i risultati,non32PASS:**30attesi soddisfatti,2fallimenti reali.

Nei sei casi sicuri verificati link unico,formula dove prevista,raw nodi,
char/UTF8 e heading/paragrafo esterni nell'ordine atteso. Math chiusa con script
opaca;nonchiusa visitata. Owner HTML prevale sul backtick dell'img e torna al
code inerte dopo chiusura. Sei raw fence annidate,quattro raw label e quattro
errori d'albero preservano aperture esatte. Reset/enfasi-code passano.
Nessuna suite/build/reinstall/rescore/CLI generale/Pandoc/conversione/download/
modello/nuovaV propri. Un IndexError iniziale del preparatore gate è preservato
e corretto;anche il primo patch report rifiutato per formato non ha scritto file.

## Criteri comuni

| Criterio | Giudizio sul comune r012 |
| --- | --- |
| C01 F50 | **Chiuso nei casi ammessi revisionati.** Decisione atomi/LF condivisa;30formula ricevute,token/barriere/escape/code/combinazioni/LF libera. Formula/link/raw/charUTF8 coerenti;proprie4LF+reset sostengono il comportamento. Chiusura lessicale distinta da math strutturale unsupported. |
| C02 F51 | **Chiuso nei casi revisionati.** Quote/list/note decidono fence dopo stripping,raw body guardato/residuo unknown-resource_syntax;48rawblock/48literalblock ricevuti. Backtick/tilde/EOF/nestedquote finita;6sonde unsafe proprie. Raw asset sicuro resta unsupported,code ordinario inerte. |
| C03 F52 | **Originali chiusi,invariante parziale/bloccante.** Raw metadata label preservato;13GPT/135owner ricevuti,sonde ordine/identità/ritorno/errore d'albero passano. GPT-R012-I001 perde prima diagnosi/visite via attrs di blocco. |
| C04 F53 | **Parziale/bloccante.** literal_atom/InlineDecision/slice/source-map,progresso e limiti coerenti nei rami revisionati;reset proprio passa. Sessione non media raise di attrs;due rami di blocco interrompono la prosecuzione. |
| C05 coverage | **Sostenuta nei producer normali,invariante globale incompleta.** Guard precede proiezione/slice,raw fence ha evento body,node usa coverage/residui;segments non basta. Range/hash non prova guardia di un dominio mai raggiunto. Compatibilità privata separata sotto. |
| C06 regressioni | **Ricevute sostenute,chiusura complessiva impedita dal finding.**153r011/303r010/308r009 PASS;F31,F35/F36,F40/F41,F43/F44,F46/F48 e contratti precedenti preservati nei casi/delta consultati. Scorer/safety/alignment invariati;non PASS di tutte le interazioni. |
| C07 oracoli | **Adeguati nel perimetro finito.** Inventario anteriore,350attesi/generatore seed12012 senza producthelper;matrix riusa test,dichiarato. Vecchi input/attesi errati preservati;before/after byte uguali. Proprie32trasformazioni anteriori/visite esatte/2fallimenti. |
| C08 evidenze | **Attribuzione sostenuta.** S004runtime/S005documentale distinti,10hash/37Python/copie/gate/R/reap/binding verificati nei record;ownR corrente separato.134CLI/64catene19campi/aritmetica/56delta privati attribuiti;nessuna qualità/V nuova o PASS dei FAIL storici. |
| C09 documenti | **Limiti rispettati,promessa globale parziale.** Atomi/owner/LF/fence/residui descritti per classi chiuse;promessa prima diagnosi/prosecuzione da correggere con finding. F36/quattro osservazioni r010/O001microcosto/O002container/no-deadline preservati. |

## Evidenze ricevute e limiti

350raw/gold/attesi identici nel confronto V11→V12:233/350→350/350.
Classi30formula/61unsafe/48rawblock/48literalblock/135owner/10token/5freeLF/
13GPT;13raw GPT confrontati con fonti. Math score5/INDETERMINATE,run0/evaluate5;
unsafe2/2,rawsafe nonproiettato UNSUPPORTED2/2.67fixtureCLI/134exitOS/detail/
final coerenti,writer sintetico/gold PROVISIONAL. Regressioni153/303/308 e
stdout fast **2688pass+310subtest** suS004/R ricevuti,non rieseguiti.
API/packaging/Docker e smoke radice esclusi,non collaudo completo.

20receipt finali:18PASS+2FAIL H12± deliberati;H01–H08PASS. ExpectedR0/
H12harness1/wrapper2 non promossi aPASS. Processcheckautore pending0 ricevuto,
sampled664543232B/storico749760512B. Before231/after340/targeted1315pass11fail,
EPERM/preH01/golderrati conservati e attribuiti;finali non riscrivono preliminari.

64derived:19campi numerici/gate invariati,4PASS/28FAIL per set32;
56EXACT_GLOBAL_DISTANCE/8TRIVIAL_ZERO. Original conversion_final r001,
previousderivedr011,finalir012 distinti. Checker autore distanza/aritmetica
indipendente reader/scorer letto/PASS ricevuto;ownhost descriptor/catene/
aritmetica,nessuna nuova distanza o alignment ricertificati. Condizionato/
globalbound/full separati,TRIVIAL_ZERO non fullPASS.

**56osservazioni cambiano realmente:**640 _guard_coverage in caption_content/
cell_contents,oltre readerID. Diff depurato solo di queste aggiunte e ID
coincide,confermando observation-delta supervisore. Top-level node rimuove
campo;located conserva nelle foglie JSON. Nessun nuovo schema pubblico
dichiarato;consumer in-repo consultati leggono campi tipizzati pertinenti e
tollerano chiavi private aggiuntive. Compatibilità sostenuta per quei consumer,
non certificazione di consumer esterni rigidi o osservazioni byteidentiche.
Range/hash identifica il dominio guardato,non sostituisce la visita effettiva;
continuazione globale distinta dall'invarianza numerica64derived.

O001:4microcosti N4/16/64/256 ~0.0006–0.005s,hardwarelocale/cachecalde;
non upper dell'intero reader o timeout900 osservato. O002 include list/note,
nessun parser nested universale. F36B1–B4 deliberate UNSUPPORTED senza secondo
feed. Quattro r010:backtick-run,html5/htmlspace/Pandoc nonconfermato,timing
storico commenti,fallbackIDv1 preservate. L01PDF/OCR,L02compileNOT_MEASURED,
L04outputUNAVAILABLE distinta,V13/A03riferimenti chiusi,L05lower/sampled,
ChromeNONPASS/helper12gap/H15/E9/errori intermedi rimangono limiti.

## Accounting e passaggio

Ledger corrente supervisore **16681.037756CPUlower/4020101120B allocated scoped
dedup**;aggiunti soli costi propri,senza risommare CPU R annidate,baseautore,
retained storico o45sClaude r007approx. Argv/cwd/clock/output/exit/costi nei
record per comando;accounting finale proprio in `accounting-final.json`.
Letture e alcuni errori amministrativi non misurati separatamente:nessun upper
CPU/RSS certificato. Sharedbudget:supervisore riconcilia anche costi altro
reviewer,non letto qui. CPU21600/stop19800,disco6/stop5GiB,RAM8/stop sampled6GiB,
riserva1800CPU-s/1GiB,R≤900/stream1MiB/file32MiB invariati. Nessun reset/quota
nuova. Network/acquisizioni/invii remoti0/servizi benchmark€0;sole sonde
socket/netns diagnostiche prescritte daR.

Nessun lavoro proprio pendente,cleanup,stato comune/arbitrati/prodotto,
commit/merge/push,fase2/motore/nuovaV modificati. Prossimo ruolo **supervisore**
dopo entrambi i report reali indipendenti,arbitrato e mandato fix completo.
Questo NO_GO non è arbitrato finale della run.

Chiusura: verify uscita **MATCH/exit0** dopo report/checkpoint; argv/output/clock/CPU in verify-exit.json. Accounting finale: accounting-final.json. Nessun workload proprio pendente.

CPU proprio misurato lower517.915985s;base+proprio17198.953741s,senza risommare costi annidati. RSS workload sampled28102656B. Dopo il verify sono stati aggiunti soltanto note finali/accounting/manifest privati,esclusi dal comune.
