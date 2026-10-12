# Implementazione a002 r007 — contesti Markdown e confini condivisi

Agisci esclusivamente come **implementatore**, nella stessa chat autrice
di r006. Completa F31–F34, regressioni, prove dipendenti e consegna r007
nella stessa chat. Il mandato copre il risultato e i suoi fix; non una
sequenza di approvazioni per comandi. Nessun nuovo piano. Registri comuni,
review, arbitrati e checkpoint supervisore appartengono al supervisore.

## Mandato e identità

Run `run-a002-fase1-corpus`, branch `feature/run-a002-corpus`, HEAD/dev/
origin-dev locale/merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`.
Indice vuoto alla ricezione; Git manuale utente solo dopo GO finale.
Piano r001+errata e GO piano, mandati05/06/09/12/15/18/21 e addendum V
restano validi nei requisiti pertinenti. Prevale ora
`arbitrations/arbitration-implementation-r006.md`, **NO_GO locale**.
F31–F33 bloccanti; F34 aggiornamento documentale/probatorio da completare.

Percorsi relativi della run sotto `temp/run-a002-fase1-corpus/`. Leggi:

1. AGENTS, temp/HANDOVER e parte iniziale PROJECT-CONTEXT, STATE/HANDOVER/
   brief, `handovers/supervisor-implementation-arbitration-r006.md`.
2. Skill manage-implementation-run e verify-conversion-fidelity, protocollo
   run-lifecycle, questo prompt e arbitrato r006 **integrale**.
3. **Entrambi** report/checkpoint r006 ChatGPT/Claude e sonde/output/receipt
   citati. Sei autore dei fix e puoi leggerli entrambi. GPT50casi in tre
   famiglie21/14/15; Claude15 K/Y r006, confronto r00515 e16 E/C/D. R PASS
   distinto da assert: E9 Claude atteso sbagliato conservato, non difetto.
4. Consegna completa/chiusura finale002 report e checkpoint autore r006;
   PROTOCOL, guide, report/dati permanenti
   `documentation/benchmarks/2026-10-09-a002-r006-confini-lessicali-html*`.
5. Reader integrale e contratti scanner/markdown_units/inline/attrs/
   peel_attrs/pipe_row/markdown_link/inline_delimiter/HTMLTree; tutti i
   produttori blocco pertinenti, observe_bundle/audit/safety/scorer e test
   r003–r006. Tokenizer **locale managed3.12.13** nei rami HTML consumati,
   con letture statiche prima di avviare l'interprete. Piano/errata/ADR0001/
   workflow benchmark/guida uv per i requisiti pertinenti, niente tutta a001.

Comune r006 revisionato `snapshots/implementation-r006.json`, SHA256
`633503036862eab4cd21443311aa25cfa5b1e5d73fc970840a2c56bfeed8398d`,
impronta `8eefc61f4380b11ac1f77fd8764c7658778cb40085e83533ac94e81684375158`,
319 file/58212 artefatti. MATCH nelle due review e alla ricezione, storico
dopo soli quattro registri generali. Non rigenerarlo.

Ingresso tecnico corrente `snapshots/implementation-fix-context-r006.json`:
SHA256 `7a0925ec9c87fb20cc1112bc7270aa1c2c22319940f0ff9d8b162972821b95d0`, impronta `c58b760933ea7ab029d809b99064ea821ea3cc82284d097d3cf4c91f7d389747`,
identità/conteggi in `evidence/supervisor-implementation-arbitration-r006/snapshot-receipt.json`.
La transizione distingue i quattro registri generali da prodotto/test/corpus,
che non sono stati modificati dal supervisore. Verifica Git/base/indice/
delta e contesto con argv/cwd/output/exit reali:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r006
```

È manager stdlib, non prova prodotto o GO. Dopo modifiche autorizzate
identifica autonomamente input effettivi e prepara manifest/gate/target/
launcher e snapshot tecnici r007 con label uniche, output vivi esclusi.
Nessun freeze o consenso intermedio richiesto per correzioni ordinarie.

## Contratto finito da completare

I casi ricorrenti dimostrano una divergenza tra scanner globale e consumer.
Prima dei nuovi payload verifica e correggi l'invariante: **i contesti
lessicali che determinano i confini sono quelli realmente consumati dai
produttori**. Delimitatori, escape, fence, metadata, comandi e autolink
devono avere intervalli/domini coerenti; un'apertura HTML presente solo
in una foglia letterale o attributo non può estendere il blocco.

Un token inerte veramente aperto fuori da tali domini attraversa invece
blank/ATX/lista/quote/fence/nota sino alla sua chiusura effettiva o EOF:
intero raw e coordinate char/byte esatti, unsupported visibile secondo
profilo `html-tokens-v1`, audit INCOMPLETE/exit2. Corpo mai trasformato in
prosa/formula/ref/asset, mai usato per soddisfare un blocco mancante.
Uno span falso non può occultare testo esterno o eliminare confini validi.
Non rendere genericamente unsupported ciò che il profilo già consuma.

Scegli i mezzi ordinari reversibili entro questi requisiti: condividere
logica finita, correggere il dominio o riorganizzare il reader è ammesso.
Non è imposto l'algoritmo suggerito da un reviewer. Non bastano patch per
ogni stringa, una massima end o un token diagnostico che lascia il corpo
attivo. Non richiediamo parser HTML/Markdown/TeX universale, nuovo schema
applicativo, normalizzazione globale regex o nuove dipendenze/download.

### F31 — domini code/math/tag e blocchi successivi

GPT-R006-I001 + CLA-R006-I002. Scanner nuovo cerca chiusure sull'intero raw,
salta EOF o chiusura in blocco diverso, mentre inline lavora nel proprio
blocco. Può saltare l'apertura di un commento/CDATA o riprendere dentro
un fence. Difetto ereditato commenti spezzati Y3–Y7 e nuove regressioni Y8/Y9.

Correggi la coerenza dei confini per backtick, `$`, `$$` e scansioni di tag
pertinenti: delim non chiuso non sospende la scansione di blocchi successivi;
mantieni diagnosi/prosa coerenti col dominio del consumer, senza salto EOF
o chiusura in un blocco estraneo. Token inerte realmente aperto conserva
priorità sui marcatori Markdown interni; non spezzarlo per risolvere il primo
problema. Non far riprendere lo scanner a metà di codice/fence valido.

Copri originali GPT otto comment/cdata-open/cross-block-code/math/display,
ClaudeY3–Y9 e K1–K5, varianti delimiter senza chiusura/chiuso in altro blocco
o nel fence; importo `$5`, `$x`/`$$` incompleti, `<`+lettera in prosa,
blank e inizi blocco, Unicode e chiusura/EOF del token inerte. Assert su
intero raw/coordinate, nessuna apertura/lettura/ref/asset dal corpo anche
con immagine esistente/URI unsafe, nessun heading inventato, q/pagina
omessa sempre missing. Testo esterno una volta, blocchi/fence validi e
formula/code bilanciati conservati. Stato negativo non basta da solo.

### F32 — autolink e letterali nello stesso blocco

CLA-R006-I001, regressione r006. `<https://…>` entra in html_stack e sospende
code/math/escape/destination; il consumer lo tratta come link. Un `<!--` o
`<?` nel code successivo genera falso span che assorbe heading/quote/
paragrafo seguenti. COMPLETE senza diagnosi, perdite inventate nella misura.

Autolink finite già supportato deve essere consumato come tale anche nei
confini, senza falso stack HTML. Copri Y1/Y2/Y10, positivi K3/K4 e varianti
http/https/mailto, code con comment/PI/CDATA/declaration, chiusura falsa in
altro blocco/EOF, escape e link destination/title dopo autolink. Stessi
blocchi/tipi/ordine/testo/offset del positivo, COMPLETE senza token spurio;
autolink/ref conservati senza fetch. HTML vero, token inerte fuori codice
e risorse realmente attive restano rilevati, nessuna disabilitazione globale.

### F33 — metadata Markdown e comandi raw TeX

GPT-R006-I002, regressione r006. Scanner non consuma i metadata quotati e
la foglia raw TeX intera come inline/attrs: `<` in title o command genera
span sino EOF, accorpando/occultando blocchi validi.

Copri i quattro host link/span/code/heading con attributi già ammessi,
title `<!--`/CDATA e positivi ordinary; comandi finiti ref/eqref/label/foo
con apertura nel valore e positivi plain. Stringhe quotate HTML rimangono
distinte dai metadata Markdown. Il command non supportato rimane unknown
localizzato nel proprio blocco; non renderlo supportato o occultare i blocchi
seguenti. HTML vero in label/prosa va ancora analizzato, destinazioni e
attributi sicuri conservati, attributi attivi/URI realmente unsafe rifiutati.

Assert su blocchi separati, tipi/ordine/testo/offset Unicode, refs/anchors/
metadata e diagnostica; nessuno span falso o assorbimento del seguito.
Non rifiutare genericamente i valori sicuri per evitare il bug. La variante
esplorativa raw latex con fence interno GPT non fonda un nuovo requisito;
regressioni necessarie sono quelle del contratto finito effettivo.

### F34 — documentazione e attribuzione delle nuove prove

Rettifica PROTOCOL e due guide con dominio/prove r007; nuovo report e dati
permanenti con nome datato alla consegna descrivono prima/dopo e limiti.
Conserva report/dati r006 e tutti gli storici. Distingui sintassi localmente
chiusa da garanzie globali, sintomo ereditato da regressione, sonde con gold
PROVISIONAL da V reale, CLI con stub da motori. Nessun PASS retroattivo E9,
checker, EPERM, preflight STALE o test falliti. Reader/versione attuale e
nuovi ID derivati con legami originali, niente copie di conteggi non verificati.

## Prove e consegna completa

Prima del primo managed: startup/package/origine CPython3.12.13 locale,
archivi/installazione e dieci moduli effettivi. Dopo modifiche: S/B/I,
ricostruzione sdist/wheel canonica reinstallata, dieci hash e preflight
attuale prima di --no-sync. R e figli stesso interprete, Firejail senza
rete, shell=False, registry/PATH/safety/confini preservati. Vecchio S005
integrale STALE per registri supervisore non diventa PASS. Equivalenza
circoscritta ai byte realmente consumati solo se dimostrata; nuovi gate
generali sull'input finale. Non restaurare pyc o input per mascherare delta.

Proteggi tutti i ricevuti: comune/contesto/review/sonde/output/snapshot,
corpus/source/gold/render/asset/packet/V13, raw e run originali. Le copie
di prova e helper r007 hanno byte/versioni/manifest prima-dopo e output
esclusivi. Conserva argv/cwd/config/attesi prefissati/output/exit/receipt/
errori, dati intermedi necessari a ricostruire il prima/dopo. Nessuna nuova
firma agente su gold/V. Se correggi un helper/atteso errato, documenta e
riesegui la prova valida senza riclassificare il vecchio fallimento.

- Prove prima/dopo su originali dei reviewer e varianti/positivi finiti;
  percorso reale observe→audit→score, assert su contenuto/risorse/aperture/
  missing/stati e denominatori. Esplicita la matrice dominio→scanner→
  consumer→confine con casi discriminanti, oltre ai singoli esempi.
- Test mirati e regressioni pertinenti F13–F30, con F27/F30 chiusi e F28
  locale preservati. Suite veloce prescritta tests/ con ignore API/packaging/
  Docker e smoke radice esclusi, preflight/R PASS; nessuna collection
  riparata con install/download offline assente. Nessuna nuova campagna a001.
- Exit **OS reali** run/evaluate con writer sintetico dichiarato su positivi
  e negativi pertinenti, run.json/final/errori tracciabili. Gold PROVISIONAL:
  gate automaticoPASS non è qualitàPASS, evaluate5 può essere atteso.
  H applicabili e H12 FAIL deliberati restano distinti, non rinominati PASS.
- Nuovo scoring o invarianza dei64raw con input finali modificati e checker
  aritmetico indipendente, ID/versioni/catena derivati attuali e campi reali.
  Per il delta previsto **zero motori e nuova V0**, 4PASS/28FAIL per set32
  da verificare su file. Rawtree/source/gold/V e identità conversioni
  originali preservati. `final` corrente distinto da conversion_final/run
  e old_*storici, nessun PASS delle conversioni trasferito al reader nuovo.

Scorer e domini F26 non si modificano per compensare il reader; TRIVIAL_ZERO
non è misura completa0, bound globale non limita il CER/WER strutturale.
F16sui64 preservato. F27/F30 chiusi, F28locale chiuso/globale tramiteF31,
F29 tramiteF31–F33; residui F13/F14/F09 circoscritti. F23/F24/F26 e
F19/F20/F05/F15 da preservare. V13/A03 chiuso e L04output UNAVAILABLE
distinta; L01/L02/L05, ChromeNONPASS/helper12gap/fallimenti non sanati.
Niente nuovi motori, browser/render, M/V generale, API/Docker/ML/pesi/remoto,
nuove macchine o fase2. Nessuna nuova V chiesta all'utente per questi fix.

Output tuo: `reports/implementation-r007.md`, `evidence/implementation-r007/`,
`work/implementation-r007/`, `handovers/implementation-r007.md`, nuovi
report/dati permanenti datati. Matrice F31–F34/requisito/prova/esito/costi/
limiti, delta reale e input finali, decisioni reversibili motivate, processi
propri pendenti e consegna completa. Nessun GO autore. Non modificare STATE,
HANDOVER/eventi/artefatti/changelog/arbitrati/checkpoint supervisore o reviewer.

## Autonomia, risorse e arresti sostanziali

Quota unica52h/187200s da `2026-10-08T07:06:54.369068Z`, tutte le attese
incluse; CPU21600s/disco nuovi6GiB/RAM8GiB, stop sampled6GiB. Stop nuovo
workload49h/19800CPU-s/5GiB; riserva3h/1800CPU-s/1GiB. Stop nuovi
`2026-10-10T08:06:54.369068Z`, deadline `2026-10-10T11:06:54.369068Z`.
Nessun reset/incremento/cap minimo ripetuto. Ledger autore closing002
storico140540,315652s wall/9277,600452CPU-s lower/1861397431B conservati/
RSSsampled740855808B non è saldo corrente. Receipt supervisore aggiorna wall;
integra costi reviewer GPT costs.json (124,47258CPU-s subset) e Claude
receipt/checkpoint (workload1,09CPU-s, wall~25min stimati), senza sommare
wrapper/R già inclusi o wall contemporanei. Costi amministrativi non misurati
restano tali; RAMsampled/disco conservato/CPU lower non sono upper/hardware
certificato. Aggiorna ledger/riserva alla ripresa e alle prove/consegna.

R≤900s/comando,1MiB/stream,32MiB/file; preparazione, workload, log, registri
e consegna compresi. €0 nuovi servizi, niente rete/acquisizioni/invio impliciti.
Conserva riserva al limite effettivo, completa lavoro indipendente coperto e
riporta un impedimento sostanziale; errori ordinari non sono consegne parziali.

Diagnostica/correggi/riprova test, argv, attese, path, permessi, timeout,
log e launcher nella stessa chat. EPERM/EACCES non è rifiuto approvazione:
richiedi eventuale escalation tool per l'azione già autorizzata, mantenendo
R/isolamento; fermati al rifiuto effettivo senza aggirarlo. Non aumentare
budget o ridurre gate. Snapshot tecnici attribuiscono prove e si preparano
autonomamente; nessun freeze intermedio. Alla consegna completa: supervisore,
nuovo comune, due NUOVE review indipendenti mirate e arbitrato; Git manuale.
