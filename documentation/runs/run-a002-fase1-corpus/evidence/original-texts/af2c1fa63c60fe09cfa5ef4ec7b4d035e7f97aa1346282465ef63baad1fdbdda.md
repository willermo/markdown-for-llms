# Implementazione a002 r012 — label ammesse e raw nei container

Agisci esclusivamente come implementatore nella chat autrice r009/r010/r011.
Stessa run, stesso incremento, nessun nuovo piano. Usa manage-implementation-run
e verify-conversion-fidelity. Consegna completa, nessun GO autore.

## Disposizione corrente: nessuna deadline della run

L'utente ha rimosso ogni deadline e stop temporale della run: disposizione
registrata in [addendum-resources-r012-r001](../arbitrations/addendum-resources-r012-r001.md),
fonte [messaggio originale](../notes/user-no-deadline-r012-2026-10-11.md).
**ESECUZIONE r012 AUTORIZZATA NEL PERIMETRO DEL MANDATO**, nessun gate temporale
pendente o nuova finestra24h. Leggi clock per accounting, non per stop wall.
Nessuna riserva3h/deadline globale, entry e contatori non azzerati. Limiti
CPU/disco/RAM/rete/isolamento e timeout dei singoli comandi R invariati.
Le review r011 restano statiche: non inventare nuove prove retroattive.
Nessuna nuova conferma temporale per test o recuperi ordinari del mandato.

## Identità e letture

Run run-a002-fase1-corpus, feature feature/run-a002-corpus; HEAD/dev/origin-dev
locale/merge-base80c31da46e5caab990d3596c79c44e524ec1dc95, indice vuoto alla
ricezione. Verifica delta non committato; niente checkout/reset o query remota.
Reader v11 observations-v11-inline-ownership, SHA256
af5299bb0c52bd4a0634a1aebb8a28743e2793c315eec5bbaf9b6424a98d8482.
Comune storico r011 SHA29efea9167ad4809fc08e1ec9a1923998c6434d2f518a92c9c663811f53c16f5,
impronta525b853f32493abe8c3637bcb455219f557442cc032ac69d6b4358dcb370b680.
Nuovo ingresso snapshots/implementation-fix-context-r011-unlimited-r001.json; identità
nella receipt evidence/supervisor-resources-r012-r001/.
Il contesto fix-r011 precedente resta storico: questo mandato aggiornato e
quattro registri generali producono delta amministrativi autorizzati, non un
prodotto diverso o una rigenerazione dei vecchi comuni.
Verify host amministrativo, non approvazione:
```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r011-unlimited-r001
```
Registra argv/cwd/output/exit/costo, attribuisci STALE senza forzare MATCH.

Leggi nell'ordine AGENTS,temp/HANDOVER,prima parte PROJECT-CONTEXT,runSTATE/
HANDOVER/brief,skill/protocollo/ADR0001/addendum risorse r012 corrente e
notes/recovery-method-r012.md integrali;
arbitrato r011 e questo mandato integrali; entrambi i report/checkpoint r011,
Claude static-traces/diff, GPT finding-static-trace/static-cases001 e receipt
pertinenti; report/CP autore r011, nota tecnica/permanente, raw/gold/attesi
e after-final002 metadata-closed-quote-1; mandato37/arbitrato r010 e finding
r010 originali per regressioni; reader integrale/delta/test/matrix/CLI,
PROTOCOL/due guide e percorsi scanner/producer/consumer/node/guardia/ledger/
audit/observe_bundle/score/alignment/CLI pertinenti. S/B/I e uv/toolchain per
contratti di prove, r009/F36/r008F40-F41/r007F31 dove decisivi. Non tutti i log.
Nuovi finding sono **statici**: nessuna prova eseguita dei payload T1/T2/GPT
da inventare. Preserva v11 fisico prima dei cambiamenti, input storici intatti.

## Metodo obbligatorio: chiudere le classi, eliminare le decisioni duplicate

Il consolidamento chiesto in r009 non è stato sufficiente. Prima delle
modifiche scrivi una nota tecnica breve con un inventario di **tutti** i punti
che decidono le classi toccate: scanner/markdown_label/producer/guardia/
consumer/ricorsioni/node-fallback, dominio fisico o proiettato, intervalli e
source-map. Per ciascuno indica chi decide e quali dati ricevono i consumatori.

Rendi unica e riusabile la decisione sul tipo/intervallo/owner di un atomo
nel dominio corretto: code inerte, math lessicalmente chiusa, token opaco,
raw inline esatto con metadata, raw block di un container, sintassi incompleta.
È ammesso refactoring interno circoscritto, anche mediante descrittori/stream
di eventi tipizzati; scegli i mezzi. Guardie e proiezione devono consumare la
stessa decisione e il mapping al sorgente. Usare una stessa funzione su
limiti diversi o riscoprire confini dalla stringa strippata non basta.
Spiega quali duplicazioni elimini e prova le invarianti con casi che le
avrebbero fatte divergere prima. Non imporre un nuovo schema pubblico,
libreria/parser universale, cambiare dialetto o restringere forme ammesse.

Guardia e costruzione dell'albero rimangono responsabilità distinte:
errore di proiezione non tronca le visite, prima unsafe in ordine sorgente.
La presenza di segments non certifica guardia completa: deve risultare
dimostrabile quale dominio/range ha ricevuto la guardia; quello residuo
non rappresentato resta esplicitamente incompleto. Unknown non wildcard.

Prima della consegna usa un generatore finito di microcasi, separato dal
prodotto, con trasformazioni motivate: via di LF ammessa × tipo atomo/raw
× container × delimitatore aperto/chiuso × owner × ordine safe/unsafe.
Fissa attesi di contenuto/coordinate/proprietà prima di eseguire; non usare
helpers del reader nel generatore/oracolo. Esplora anche combinazioni non
ricavate dai tre payload attuali, conserva seed/limiti e riduci ogni
controesempio a fixture indipendente. Correggi nella stessa chat prima della
consegna completa. Numero di test non è criterio di copertura: pubblica
mappa decisione/classe → proprietà → casi e risultati effettivi.
Non serve testare ogni combinazione sintatticamente invalida o tutta Pandoc.

Se una premessa raw/code/math dipende dal dialetto Pandoc, sono ammessi
microcasi di parsing AST con il Pandoc già installato, sotto preflight/R e
senza rete: oracolo sintattico distinto dal gold manuale, non nuova campagna
conversione/corpus/V. Nessun download/installazione; non presentare parsing
Pandoc come prova di fedeltà, sicurezza o semantica universale.
Regressioni e criteri restano quelli F50–F53 qui sotto.

## F50 — CLA-R011-I001, bound delle label accettate

Label con backslash seguito da LF è accettata da markdown_label ma il bound
guardia math si ferma alla LF mentre il consumer chiude oltre. Riproduci T1
e varianti (2 payload × paragrafo/quote/list/note) con attesi manuali formula/
link/testo/blocchi/ordine/charUTF8: math chiusa resta opaca, nessun falso unsafe
nel corpo. Aggiungi controllo delimitatore non chiuso con tag reale: deve
restare unsafe con detail. Copri LF in code già chiuso, escape+LF e combinazioni
finite ammesse, LF libera fuori profilo, token opaco/comment/CDATA e barriere/
limite/ricorsioni F31r007. Stessi confini per producer/guardia/consumer.
Non coprire solo code-LF o risolvere rifiutando label prima ammesse.

## F51 — CLA-R011-I002, raw block nei container

Fence raw esatto {=html} dentro quote complessa non può essere certificata
COMPLETE come code inline con unknown=[] e nessuna diagnosi risorsa.
Visita corpo HTML attivo ammesso, priorità unsafe/detail corretta; se una
risorsa sicura non viene proiettata, audit almeno INCOMPLETE/UNSUPPORTED.
Non serve implementare tutti i blocchi nested; serve conservare il contratto
di guardia e dichiarare contenuto non rappresentato. Includi trasformazioni
finite list/note (spazi/tab) della stessa causa CLA-R011-O002, mantenuta non
bloccante/ereditata e non promossa a finding autonomo. Controlli script,
escaping img, asset sicuro, normale fence/code chiuso letterale, blocchi
esterni e quote già coperte. Raw html5/htmlspace restano fuori mandato;
Pandoc non confermato resta limite storico.
Non ripristinare il feed raw indiscriminato che rompe F44: riconosci dominio
e ownership effettivi prima di guardia/fallback. segments deve significare
guardia completa del dominio pertinente, non esenzione automatica.
La prova autore metadata-closed-quote-1 conferma solo il ramo, non T2 eseguito.

## F52 — GPT-R011-I001, metadata raw inline nella label

Il ledger code salta backtick-script-backtick{=html} nella label e la prima
diagnosi viene dalla seconda violazione esterna. Il formato esatto è già
riconosciuto nel consumer. Usa gli otto discriminanti indipendenti GPT
(2 ordini × paragrafo/quote/list/note) e cinque controlli. Prima unsafe globale
deve essere la reale prima nel sorgente, visite successive continuate dopo
prima violazione/errore d'albero; non basta exit2 o assenza di eccezioni.
Con/senza seconda violazione, ordine script/img, raw dentro/fuori label,
ordinarycode letterale, safe reference e math chiusa. Conserva F48 ownerHTML,
chiusure estranee/bilanciate, source-map F41 e quote, char/UTF8.
Nessun ampliamento dei formati raw fuori subset.

## F53 — interazioni, costo locale, documenti e prove

Matrice finita motivata label-LF × tipo raw × container × owner × ordine
unsafe/safe. Oracoli manuali indipendenti, fissati prima, con posizione e
contenuto gold espliciti; non derivare attesi dal prodotto o duplicare la
stessa implementazione nel checker. Ogni proiezione/foglia/fallback toccato
deve avere guardia sul dominio corretto. F46 metadata, F40 tokenizer, F41
source-map, F43/F44 e F31 tokenlabel preservati. F36B1–B4 resta limite
deliberato INCOMPLETE/UNSUPPORTED, nessun secondo feed/nuova grammatica.
O001 costo del ciclo LF è stima statica, non timing/timeout osservato.
Consentito lookup monotono/indice locale invece di scansioni ripetute;
prova terminazione/costo su trasformazioni piccole sotto R con la disposizione
corrente senza deadline, senza campagna benchmark generale o budget aggiuntivo.
Conserva quattro osservazioni r010 e IDfallbackv1, non fingere chiusure.

Con la disposizione temporale già effettiva, gestisci autonomamente errori ordinari con argv/log/
fallimenti conservati e recuperi nello stesso mandato, senza consegna parziale.
Snapshot tecnici identificano input, non approvazioni. Preflight managed3.12.13,
sdist/wheel/reinstallazione canonica, S/B/I e dieci hash source/wheel/install,
copie helper/Python finale prima dei workload, R PASS/netnone. Se separi un
modulo interno, aggiungi i nuovi file alle origini/package/copie e verifiche
S/B/I; il confronto dei dieci moduli preesistenti non copre nuovi import. Gate storico
S006 STALE per quattro registri non trasferito come PASS. Nuove prove sugli
input reali finali, repeat solo quelle invalidate.
Suite veloce prescritta dopo gate:
```bash
"$RUN_DEV_PY" -I -B -m pytest tests/ --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q
```
153 r011/303 r010/308 r009 come regressioni, H01–H08 e H12±FAIL deliberati
con wrapper/harness/expectedR distinti, CLI reale exitOS e detail primaunsafe.
F50math chiusa: run0/evaluate5 con qualitàINDETERMINATE e gold manuale, non
PASS semantico. F51unsafe run2/evaluate2, sicuro unsupported2/2 salvo effettiva
proiezione completa provata. Per F52 exit da solo non convalida prima diagnosi.
Writer sintetico dichiarato, nessun motore/conversione generale nuova.
64derived con raw/gold/chain/19campi/gate/aritmetica/bound: originale
conversion_finalr001 distinto dalle precedenti derived, zero differenze
numeriche/gate attese salvo reale differenza spiegata, non forzare4/28.
Checker distanza indipendente distinto dal limite di indipendenza alignment.
Preserva originali/corpus/gold/raw/V13/A03/L04outputUNAVAILABLE e limiti
L01/L02/L05, NONPASS Chrome/helper12gap/H15/E9/fallimenti intermedi.

Aggiorna PROTOCOL e due guide, permanente report/dati r012, test pertinenti,
readerID reale e nota tecnica: confini di LF accettate, raw inline/block,
container/fallback, firstdiag/visite e limiti misurati/unmisurati precisi.
Nessuna promessa che segments provi da solo guardia completa. CPU lower/
RSS sampled/hardware/cache dichiarati, nessun upper certificato.

## Consegna e confini

Scrivi report reports/implementation-r012.md e CP handovers/implementation-r012.md,
evidence/work/snapshot tecnici propri, nota iniziale e prova finale attribuita
a byte/interprete/gate reali. Stato/arbitrati/comune riservati al supervisore.
Risorse CPU21600/stop19800s, disco6/stop5GiB, RAM8/stop sampled6GiB,
riserva1800CPU-s/1GiB, R≤900s/stream1MiB/file32MiB, rete/acquisizioni/invii0/€0.
Base/nuovo budget effettivi nella receipt supervisore, contatori invariati,
45sClaudeapprox separati non risommati; clock/wall misure senza limite di
durata della run, addendum r012 corrente prevale sugli stop temporali storici.
Al primo limite sostanziale checkpoint e disposizione necessaria, lavoro
indipendente solo entro permessi/risorse residue. Nessun download/cleanup/
fase2/nuovaV/scelta motore/Git automatico o uso di servizi nuovi.
Alla consegna completa prossimo ruolo supervisore: ricezione, nuovo comune,
due nuove chat indipendenti ChatGPT/Claude e arbitrato, nessun GO autore/futuro.

