# Implementazione a002 r006 — confini lessicali HTML e contenuti inerti

Agisci esclusivamente come **implementatore**, nella stessa chat autrice di
r005. Completa F27–F30, regressioni, prove dipendenti e report r006 nella
stessa chat. Il mandato copre il risultato completo e i suoi fix, non una
sequenza di approvazioni per comandi. Registri comuni/review/arbitrati restano
al supervisore; nessun nuovo piano.

## Mandato, identità e letture

Run `run-a002-fase1-corpus`; feature `feature/run-a002-corpus`, HEAD/dev/
origin-dev locale/merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`,
indice vuoto alla ricezione. Git manuale utente dopo GO finale.
Piano r001+errata/GO piano, prompt05/06/09/12/15/18 e addendum V restano
il mandato; prevale ora `arbitrations/arbitration-implementation-r005.md`.
F27–F29 bloccanti, F30 completamento diagnostico non bloccante accolto.

Percorsi relativi della run sotto `temp/run-a002-fase1-corpus/`. Leggi:

1. AGENTS.md, temp/HANDOVER, parte iniziale PROJECT-CONTEXT, STATE/HANDOVER/
   brief; `handovers/supervisor-implementation-arbitration-r005.md`.
2. Skill manage-implementation-run e verify-conversion-fidelity, protocollo
   run-lifecycle, questo prompt e arbitrato r005 **integrale**.
3. **Entrambi** reviews/review-implementation-r005-chatgpt.md e
   reviews/review-implementation-r005-claude.md, checkpoint e prove citate.
   Sei autore dei fix e puoi leggere entrambi. GPT36casi/28attesi rispettati/
   8unmet, CLI4stub/8exit; Claude15casi/7controlli e8casi X.
4. Report/CP finale autore r005, PROTOCOL e guide, report/dati permanenti
   `documentation/benchmarks/2026-10-09-a002-r005-diagnosi-html-e-domini*`;
   observations/inline/HTMLTree/token/visit/node/observe, splitter dei blocchi,
   safety/audit/observe_bundle/scorer e test/regressioni pertinenti.
5. Tokenizer locale managed3.12.13 pertinente: parse_endtag, dichiarazioni,
   commenti e callback/EOF. Letture statiche prima di import/esecuzione.
   Requisiti piano/errata/ADR0001/workflow benchmark/guida uv nei punti pertinenti.

Comune r005 revisionato `snapshots/implementation-r005.json`: SHA256
`b4fe0240533ad92e0acb65c355c01991c6d15f83abab2f4781e7d28f9653433e`,
impronta `0a99ca81113185b6ed48263b798e72bc2bf542c0a92d64d2bd2fb72033ccdd34`,
316file/47654artefatti. MATCH nelle review e alla ricezione del supervisore,
ora storico dopo soli quattro registri generali; non rigenerarlo.

Ingresso tecnico corrente `snapshots/implementation-fix-context-r005.json`:
SHA256 `56a3ebbbe53ead47325d1f3a3f7e3049ca543172b40a1e2cc7af1a57876e8d49`, impronta `6cc99d3589476913865ddc8480cd854c490a76ddab0f762a3e78dc8cbc142061`, 316file/48306artefatti.
Receipt `evidence/supervisor-implementation-arbitration-r005/snapshot-receipt.json`
e transition.json identificano la sola transizione amministrativa.
Verifica Git/base/indice/delta e contesto, registra argv/cwd/output/exit:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r005
```

È manager stdlib, non GO prodotto. Dopo cambi autorizzati prepara autonomamente
manifest/target/launcher/input e nuovi snapshot tecnici r006 con label uniche,
output vivi esclusi. Non mutare ricevuti né chiedere freeze intermedio.

## Contratto finito da completare

Correggi la classe di difetto **token lessicale → confine Markdown/trasporto →
foglie e consumatori**, oltre ai payload dei reviewer. La ricorrenza rende
necessaria questa verifica del produttore e dei suoi trasporti; un massimo
end non dimostra copertura, una diagnosi sul primo frammento non rende inerte
il resto. Sono ammesse scelte tecniche reversibili non enumerate entro mandato.

- Ogni token riconosciuto dal ramo locale ha un intervallo raw esatto:
  intero costrutto chiuso oppure fino a EOF. Unsupported conservato e
  diagnosticato raggiunge audit/scorer; raw generale da solo non basta.
- Il corpo di commento/CDATA fuori profilo resta inerte. Non viene nuovamente
  interpretato come prosa, Markdown, formula, reference o asset attivo.
  Non può soddisfare un blocco atteso visibile o alterarne i conteggi noti.
- I confini Markdown rispettano un commento aperto fuori dal codice letterale;
  riga vuota e marcatori interni non diventano nuovi blocchi del documento.
- Il ramo finito che consuma `</>` senza callback produce una diagnosi
  localizzata coerente; positivi HTML validi/literal/code rimangono discriminanti.

Mantieni il profilo conservativo r005: comment/declaration/unknown_decl/PI
fuori profilo sono unsupported visibili, audit INCOMPLETE/exit2. Non rimuovere
questo gate per ottenere PASS. Nessun parser HTML universale, schema applicativo,
normalizzazione globale regex o nuovo apparato di installazione richiesti.

### F27 — chiusura vuota consumata senza diagnosi

GPT-R005-I001, residuo F22/F25/C01/C04. Tokenizer parse_endtag consuma `</>`
senza callback; HTMLTree close e covered_end non lo rilevano. Paragrafo/cella/
fence risultano COMPLETE/PASS automatico, CLI run0/evaluate5 con gold provvisorio.

Intercetta il ramo o una copertura lessicale precisa e finita, con raw `</>`,
intervallo/coordinate/tipo e diagnostica consumata da audit/scorer. Prove
prima/dopo su paragrafo, cella, wrapper, raw HTML fence e trasporto inline:
INCOMPLETE/run2/evaluate2, niente COMPLETE/PASS. Prosa/ref/math conservati una
volta. Positivi bilanciati, literal< e `</>` nel codice/commento inerte non
inventano prosa o ulteriori risorse. Nessuna perdita di testo o bypass URI
da attribuire al caso originale.

### F28 — intervallo CDATA e avanzamento inline

GPT-R005-I002 e CLA-R005-I002, unica radice. unknown_decl locale consuma fino
a `]]>` ma token('declaration') tronca al primo `>`. Il trasporto inline
riparte nel corpo e inventa prosa/risorsa (audit.regular su absent.png).
È bloccante per rappresentazione e misure, pur restando INCOMPLETE.

Determina l'intero intervallo CDATA dal ramo effettivo (chiusura `]]>` o EOF),
distinto da commenti bogus/PI/doctype, e avanza oltre il token completo.
Unsupported inerte/localizzato con raw/end/offset e coordinate Unicode corretti.
`Known <![CDATA[secret > hidden]]>words.` lascia solo `Known words.` noto;
variante con img/URI nel corpo: zero reference/asset/aperture tentate da esso.

Copri X1–X3 Claude e varianti GPT inline/blocco/cella/caption/residuo,
`>` interno, più token, Unicode, chiusura e EOF. Mantieni commenti normali/
--!>/abrupt, PI/doctype, literal< e codice/fence Markdown inerti. Non serve
supporto semantico CDATA o leggibilità del suo corpo nei conteggi di fedeltà.

### F29 — commenti interi attraverso i confini Markdown

CLA-R005-I001, splitter ereditato ma requisito F25 corrente. Un commento che
attraversa riga vuota o un inizio di blocco viene spezzato: il corpo diventa
prosa/heading/figura, asset letto/hashato o UNSAFE_REFERENCE; un blocco presente
solo nel commento non risulta missing. Stato INCOMPLETE/2, senza rete/falso PASS.

Rendi la suddivisione consapevole del commento aperto nel contesto lessicale,
estendendo sino alla chiusura effettiva del tokenizer (normale/--!>/abrupt)
o EOF. Un solo token/nodo diagnostico inerte, raw e intervallo esatti; i byte
interni non ritornano ai produttori Markdown. Il codice inline/fence che
contiene `<!--` resta letterale e non estende il blocco. Esamina i produttori
effettivamente interessati, non solo il ramo del paragrafo principale.

Varianti X4–X8: blank, ATX, lista, fence, quote, nota e marker `:::` dentro
commento, top-level e wrapper/blocco HTML, prosa prima/dopo, chiusura/EOF.
Nessun heading/prosa/figura/ref/formula dal corpo,0asset/aperture da immagini
commentate anche se esistenti o con URI non sicura. Reason conforme al profilo
unsupported, non UNSAFE_REFERENCE per un percorso interno al commento.
Il gold con paragrafo assente visibilmente deve restare missing anche se il
suo testo compare nel commento. Unknown non nasconde errori noti già rilevabili.
K4/K5, codice letterale e F22–F25 già chiusi conservano comportamento/coordinate.

### F30 — riepilogo diagnostico coerente

CLA-R005-I003, non bloccante da chiudere insieme. Il nodo sintetico senza foglie
usa sempre unmatched HTML close. Deriva la reason dalle occorrenze o usa una
descrizione neutra; conserva occorrenze, raw, offset, severità e stati corretti.
Prove wrapper vuoto non chiuso, sola chiusura spaiata e diagnostiche multiple.

## Prove e consegna completa

Prima del primo managed: startup/package input/origine locale3.12.13/dieci
moduli e archivi/installazione effettiva. Dopo modifiche, S/B/I/build e wheel
canonica reinstallata/diecihash/preflight attuali secondo AGENTS/guida uv;
helper/test fuori wheel hanno inventario/hash/byte prima-dopo. R e figli nello
stesso interprete, Firejail/no-network, shell=False/registry/confini preservati.
S005 integrale stale per documenti supervisore è storia, non PASS nuovo.
Non ripristinare pyc o riclassificare vecchi fallimenti per nascondere delta.

Proteggi input dello snapshot d'ingresso, corpus/gold/source/render/asset/
packet/V13 e raw originali. Conserva byte dei helper consumati e delle versioni
intermedie prima di sovrascriverli, manifest/input/argv/attesi/receipt/output/
exit e legami prima-dopo. Snapshot tecnici attribuiscono prove, non approvazioni.

- Prove prima/dopo sugli originali, varianti e positivi dei quattro fix nel
  percorso reale observe→audit→score. Assert su contenuto/coordinate/ref/
  aperture/missing/stati/denominatori, oltre a raw presente o file creati.
- Test mirati e regressioni pertinenti di F13–F26; suite veloce prescritta
  tests/ con ignore API/packaging/Docker e smoke radice esclusi, dopo gate
  attuali/R PASS. Nessuna collection riparata con acquisizioni offline mancanti.
- Exit **operativi reali** run/evaluate con writer sintetico dichiarato per
  i percorsi del fix (positivi e negativi), run.json/final/errori tracciabili;
  gold PROVISIONAL distinto da V/qualità PASS. R PASS non è qualità, contratti
  H applicabili e H12FAIL attesi restano distinti e conservati.
- Scoring nuovo o prova d'invarianza dei64raw con gli input cambiati, nuovi
  ID derivati/legami/versioni actuali. Per questo delta zero motori e nuovaV0;
  byte raw/V13/gold e identità conversioni originali preservati. 4PASS/28FAIL
  per set attesi per invarianza, controllare i file invece di copiarne il conteggio.
  Checker aritmetico indipendente e confronto di campi numerici/gate reali.
- Rettifica F25 in PROTOCOL e due guide; nuovo report permanente e dati r006
  chiariscono i casi r005 errati e le prove reali senza riscrivere report/dati
  r005/r004 o fingere conversioni/humanV nuove. F26/domino globale resta chiuso,
  algoritmo/normalizzazioni/denominatori non si cambiano per questi fix.

F23/F24/F26 chiusi, F22 casi originali chiusi con residuo F27, F25 parziale
su F28/F29; F13/F14/F09 residui localizzati. F19/F20/F05/F15 e F16sui64
preservati. V13/A03chiuso, L04outputUNAVAILABLE distinta e L01/L02/L05,
ChromeNONPASS/helper12gap non sanati. Nessun nuovo motore, render/browser,
M/V generale, API/Docker/ML, modello/remoto/nuove macchine o fase2.

Output tuo: `reports/implementation-r006.md`,
`evidence/implementation-r006/`, `work/implementation-r006/`,
`handovers/implementation-r006.md` con sezione finale consegna completa,
matrice F27–F30/requisito/prova/esito/costo/limiti, processi propri pendenti,
identità e inventario del delta, registri permanenti indicati sopra.
Nessun GO autore. Non modificare STATE/HANDOVER/eventi/artefatti/changelog/
arbitrati/checkpoint supervisore o report reviewer ricevuti.

## Autonomia, costi e arresti

Quota unica52h/187200s dalla entry2026-10-08T07:06:54.369068Z, attese incluse;
CPU21600s/disco6GiB/RAM8GiB/stop sampled6GiB; stop nuovi workload49h/
19800CPU-s/5GiB, riserva3h/1800CPU-s/1GiB. Stop nuovi2026-10-10T08:06:54.369068Z,
deadline11:06:54.369068Z. Nessun reset/incremento o cap minimo per ogni sonda.
Ledger autore closing132920,871316s/CPU8211,089263s lower/retained1646699588B/
RSSsampled740855808B è storico. Receipt ricezione aggiorna wall; integra i costi
reviewer (costs.json GPT e checkpoint/receipt Claude) evitando di sommare
wrapper e R già inclusi. CPU non misurata/RSSsampled/disco conservativo/cacheOS
restano limiti, non upper/certificazioni. Aggiorna ledger e riserva alla ripresa.

R≤900s per comando/1MiB per stream/32MiB per file. Preparazione, workload,
registri, attese e consegna rientrano nella quota finita già disposta. €0nuovi
servizi, niente rete/acquisizioni/invio documenti impliciti. In caso di limite
reale preserva prove e riserva, completa lavoro indipendente consentito e
riferisci il confine effettivo al supervisore; errori ordinari non sono consegne.

Diagnostica/fixa/riprova argv/path/permessi/timeout/log/launcher/gate e test
rossi nella stessa chat, conservando fallimenti. Per EPERM/EACCES richiedi
eventuale escalation tool dell'azione già autorizzata mantenendo R; fermati
al rifiuto effettivo, senza aggirarlo. Non aumentare budget o indebolire
isolamento. Snapshot/prove dipendenti si aggiornano autonomamente, nessun
freeze intermedio. Alla consegna completa: supervisore per nuovo comune,
due NUOVE review indipendenti mirate e arbitrato reale; Git manuale dopo GO.
