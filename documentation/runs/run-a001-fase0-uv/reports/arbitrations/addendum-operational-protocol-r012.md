# R012 — proseguire con log proporzionati e budget condiviso

2026-10-06, supervisore; mandato43 ricevuto, pianoGO r003/r011/r010 invariati
nei requisiti. Tre priming exit0,139nomi/25nuovi con provenienza, R/D4 PASS in
tre lock; nessun lock/check prodotto. Terzo lock interrotto per cap log/registri,
non conflitto del grafo o deadline esaurita. Stage s013 MATCH alla ricezione.

## Ricezione e causa

137file d’autore,844record cache e baseline1892file/211directory ricalcolati.
Stderr nativo1164160byte contro1048576, overshoot115584; registri finali circa
10MiB contro8MiB. Il gate cumulativo32MiB resta ammesso con circa12MiB residui.
Il wrapper interno tracciato ha anch’esso il cap fisso1MiB: alzare soltanto il
monitor esterno a2MiB non risolve. Il campo timed_out nel wrapper significa
«tempo oppure log»: il parziale true a0,206s non dimostra deadline900s esaurita.
Receipt storiche intatte, receipt esterna terzo lock assente/non inventata.

## Disposizione operativa ricevuta

La ripresa usa **livello nativo normale**: rimuovere --verbose/-v/-vv dal priming,
lock/check, senza --quiet o soppressione degli errori. Receipts, argv/cwd/env,
input/hash, esiti, audit cache e prove R/D4 restano completi per quanto prodotto;
assenza del debug non va spacciata per osservazione di URL/body/redirect/hook.
Il livello di log è parametro operativo delegato secondo AGENTS/protocollo,
non requisito semantico del grafo. Non serve nuova autorizzazione per adattarlo
nei limiti ricevuti. Default normale per le risoluzioni ripetute; niente
troncamento, cancellazione o rewriting di log storici. Cap stream **1MiB invariato**,
wrapper immutato; non autorizzata l’opzione2MiB con questo helper.

Il totale **32MiB è unico pool condiviso** per metadata, input, strumenti,
log/registri, nuove evidenze e snapshot. Superati soltanto i sotto-limiti24/8
storici: nessun sotto-cap autonomo dei registri, nessuna riserva interna24MiB
sottratta dal totale. I campi legacy dei due cap categoria sono entrambi32MiB
per compatibilità, ma non sono due quote da sommare. Formula/gate sull’intera
run viva è l’unico limite cumulativo: Hentry525762560byte, Delta=max(0,H-Hentry),
H=max(logical,allocated), Delta<32MiB e H+max(0,32MiB-Delta)+16MiB<896MiB.
Pool1GiB/libero1GiB e riserva esterna16MiB invariati. Vecchi costi inclusi,
zero nuova allowance/cleanup/reset/storage esterno. Monitor periodico non
quota atomica; overshoot storico resta FAIL. File32MiB/JSON8MiB/stream1MiB
restano. Non usare il vecchio sottocap8MiB per bloccare il mandato nuovo.

## Completamento nella stessa chat

Ricevuto il priming universale diagnostico già proposto: fonttools[woff]4.66.1,
brotli/brotlicffi/zopfli, con **solo --only-binary=:all:**; no-build ridondante
è incompatibile nella CLI osservata. Transitive/no-sources/config vuota e
managed/publicenv/host ricevuti invariati. Sono solo metadata diagnostici,
mai extras/pin/constraints del prodotto. Dopo acquisizione: cache/provenienza
sigillate, lock universale originale offline sotto R/Firejail net=none/D4 e
check dopo audit. Niente solo-preparazione intermedia.

Altri miss entro la chiusura r011 sono già ricevuti: massimo64nomi nuovi dai
114 iniziali,25acquisiti contano e non si azzerano. Derivare divieti build da
tutti i nomi reali sigillati salvo EBook0.18, mantenendo progetto/Marker/backend
ricevuti. Pin/grafo/fonti/CPU/cu126/Python/metadata prodotto e baseline invariati.
No --no-deps sul priming metadata, no nuovi override, cache forgiata o lock manuale.
Nessun freeze per ogni nuovo nome o parametro adattabile; ricevuta/seal di ogni
workload identifica i dati reali. Artefatti scope/input di riferimento immutati.

Tempi già consumati inclusi: metadata residuo898,0215938149486s,
lock899,488482683897s, check120s, outer deadline+180. Non riaprire900s ogni volta
che cambia label. Sessioni vive raccolte, retry solo dopo diagnosi/gate/processi
propri conclusi. Usare inventari compressi e riferimenti a sigilli invariati;
non ricopiare ogni volta l’intera cache o la storia. Logs nuovi/actual argv/exit
sono evidenze necessarie; parsimonia non è omissione dei fallimenti.

Rete pubblica solo sul priming e con i limiti r011; non R offline PASS su HTTPS.
Metadati/eventuale fallback wheel nei caps, niente sdist/Git/backend/install/
pesi/font/archivi ML completi. Origine/startup/baseline/namespace/closure verificati.
Conflitto dimostrato, costo o nomi oltre limiti, origine/input protetti alterati,
isolamento non rispettabile e rifiuto sandbox restano arresti sostanziali.

Consegna report-r010, completion s014 e delivery nuovi, errori/receipt vecchi
intatti. Se riuscito, richiesta concreta di seguito promozione/S-B-I-E nella
stessa consegna; nessuna esecuzione implicita di quel seguito. Baseline62/5,
s009 byteFAIL/lacune e altri limiti storici preservati. V7/V8 pesanti distinti,
V10/V11 esclusi, due review reali/arbitrato finale necessari. Nessun GO finale/
commit/merge/push/deploy/cleanup: Git manuale. Supervisore non esegue workload.
