# Review indipendente implementazione a002 r008 — Claude

Agisci esclusivamente come **reviewer Claude**, in una NUOVA chat
indipendente da autore, supervisore e altro reviewer. Registra provider/modello
effettivo; segnala differenze senza fingere sostituzioni. Nessuna delega.

Output esclusivamente tuoi, sotto `temp/run-a002-fase1-corpus/`:

- Report: `reviews/review-implementation-r008-claude.md`.
- Evidenze: `evidence/review-implementation-r008-claude/`.
- Checkpoint: `handovers/review-implementation-r008-claude.md`.
- ID rilievi: `CLA-R008-I001`, progressivi.
- Prompt origine: `prompts/29-review-implementation-r008-claude.md`.

## Oggetto comune e identità

Run `run-a002-fase1-corpus`, **implementazione r008**, giro di fix F35–F38.
Stato IMPLEMENTATION_REVIEW / FIX_REVIEW_PROMPTS_READY. Consegna completa
ricevuta; ultimo arbitrato tecnico r007 NO_GO reale conservato. La ricezione
non chiude i rilievi. Nessuna review/arbitrato/GO r008 preesistente.

- Snapshot: `temp/run-a002-fase1-corpus/snapshots/implementation-r008.json`.
- SHA256 snapshot: `2f6d690a8f3db7af8f192f78a9e6f5c0e3704b9757256f9dfdf72e35ee3b4152`.
- Impronta worktree: `448b854a88dc3dc883f7337f6fe037fd27846513e3a19104fc88529841e07eda`.
- 325 file generali e 76333 artefatti espliciti;20PDF nella fingerprint generale.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev **locale**/merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`, indice vuoto alla ricezione.
- Piano r001 SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`;
  errata SHA256 `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.
- Mandato locale: prompt27, arbitrato r007 e addendum-resources-r008-r001
  **DISPOSED24h**, non la precedente proposta12h PENDING.

Il comune vincola sorgenti/test/documenti generali e le prove esplicite.
`temp/` e `documentation/runs/` esclusi dall'impronta generale, ma gli input
espliciti restano vincolati. Nuovi output r008 dei reviewer, registri vivi,
prompt28/29 e checkpoint corrente supervisore esclusi; non nascondere codice
nelle esclusioni. Inventario/receipt/transizione in
`evidence/supervisor-implementation-review-r008/`. Non leggere tutte le
76mila voci: seleziona le prove pertinenti, il verify controlla l'identità.

Ingresso storico dell'autore: `implementation-fix-context-r007-resources-r001`,
SHA256 `c43903ff8f3b4155fc643c3ff7e4118f964b8f40d8849a518a75de3f0d6b6889`.
Transizione:7file autore +4registri generali supervisore; nessun altro delta
prodotto. Comune/fix context r007 e altri snapshot storici preservati;
non devono tutti restare MATCH dopo modifiche autorizzate. Non rigenerarli.

## Letture e verifica iniziale

Usa le skill `.agents/skills/manage-implementation-run/SKILL.md` e
`.agents/skills/verify-conversion-fidelity/SKILL.md`. Percorsi della run sotto
`temp/run-a002-fase1-corpus/`, salvo indicazione diversa. Leggi nell'ordine:

1. AGENTS, temp/HANDOVER, solo parte corrente iniziale PROJECT-CONTEXT,
   STATE/HANDOVER/brief e questo prompt; checkpoint
   `handovers/supervisor-implementation-review-r008.md`.
2. Protocollo `documentation/development/run-lifecycle.md`, entrambe le skill,
   template `documentation/development/templates/review.md`;
   `arbitrations/arbitration-implementation-r007.md`, prompt27 integrali e
   `arbitrations/addendum-resources-r008-r001.md` con fonte utente.
3. Entrambi i report/checkpoint **r007** e le tracce statiche pertinenti:
   GPT `static-label-trace.md`, Claude `static-traces.md` nelle loro evidenze.
   Sono input storici comuni, non report concorrenti r008. Le review r007
   furono solo statiche dopo il vecchio stop; non attribuire loro sonde runtime.
4. Report/CP r008 **integrali**, sezioni finali Consegna completa e Chiusura
   amministrativa finale autorevoli; permanent report/dati
   `documentation/benchmarks/2026-10-10-a002-r008-avanzamento-scanner-e-domini-inline*`;
   receipt consegna/final-checks-002/semantic-equivalence-002, budget finale,
   ricezione supervisore e receipt/transizione del comune.
5. Reader `scripts/benchmarks/observations.py` integrale, specialmente
   markdown_lexemes/source_scope/raw_fence_end/tag_end/incomplete_tag_token,
   markdown_label/label_codes/markdown_units/inline/HTMLTree e consumer/producer
   sui trasporti. Nuovo test r008 e regressioni r003–r007 pertinenti, PROTOCOL,
   due guide; audit/observe_bundle/metrics/alignment/safety/PassiveHTML,
   input validator e CLI necessari a verificare i percorsi raggiungibili.
6. Prove autore mirate sotto `evidence/implementation-r008/`: matrix-before-001/
   matrix-after-005, before-loop e after-loop*-005, Unicode/fence-positive/
   edge-002, CLI-after-001/CLI-image-004, contracts-002, rescore-002,
   derived-check-002, gate e input S006/S007, stdout finali. Errori intermedi
   da leggere se incidono sulla conclusione, non tutte le campagne storiche.
7. ADR0001, indice/registro ADR, fase1 roadmap, piano/errata/workflow benchmark
   e guida uv/reference toolchain solo per contratti/limiti pertinenti.
   Nessuna scansione indiscriminata di temp/a001 o nuova campagna a001.

Verifica branch/HEAD/dev/merge-base/indice, modificati e nuovi non ignorati;
non usare soltanto `git diff base...HEAD` per il lavoro non committato.
Verifica hash del comune e **prima e dopo** la review, con argv/cwd/output/exit
nelle tue evidenze:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r008
```

Manager stdlib: non prova del prodotto Python. Se STALE/identità diversa,
conserva risultato, identifica le differenze e segnala l'invalidità al
supervisore; non rigenerare/correggere input né dichiarare GO sul vecchio stato.

## Delta e prove ricevute da verificare

Sette file autore: observations.py, PROTOCOL.md, nuovo
`tests/unit/test_conversion_fix_r008.py`, due guide benchmark-fedelta,
nuovi report/dati permanenti r008. Reader
`observations-v8-progress-and-inline-domains`, SHA256
`817852a8f8b2645a3bbc6212d982aca4546ab1c4b3f128374137313124aa6061`;
scorer v4 invariato. Safety/audit/alignment/adapter/R/resources/PATH/config/
registry/writer/corpus invariati; **nel reader l'applicazione di PassiveHTML
alle aperture reali è nuova**, quindi i suoi percorsi fanno parte della review.

Ricezione solo statica supervisore:67628confronti hash/65763protetti,
130input-copie consegna prima-dopo/77copie ricevute,205runtime S006→S007
identici/124helper-test secondo equivalenza autore/10origini installate
coincidenti/206input esterni S007 coincidenti alla ricezione.
3813package-startup invariati secondo prove autore, equivalenza selettiva
non intera identità stage: gate/shebang/binding nuovi. Tre pyc ricevuti
storici preservati senza attribuzione causale inventata.

1373mirati/16deselected (261nuovi) e1530veloci+310subtest dichiarati PASS,
non eseguiti dal supervisore.25receipt finali selezionate:23PASS+2FAIL H12
voluti; storia47receipt distinta dal set finale.10contratti conformi.
Tre TIMEOUT3s sul reader r007 preservato (observe/audit/measure), tre PASS
after-loop*-005, figli raccolti e binding prima/dopo.128matrice prima
non conformi/128dopo conformi (64F36+64F37),5Unicode/4tracce cursori,
4fence positivi/2edge.57casi CLI sintetici/114exit OS reali:55+2immagini,
writer dichiarato e gold PROVISIONAL senza V/motori. Qualità INDETERMINATE
può accompagnare automatic gate PASS/evaluate5; non significa VERIFIED.

Rescore00264nuoviIDr008b, due set32 con4PASS/28FAIL ciascuno; zero differenze
19campi numerici/gate vsr007. Checker indipendente di distanza/denominatori/
legami, non prova indipendente di allineamento completo. Raw/old_* e
conversion_final/run originali conservati, derivati correnti distinti.
Corpus/gold/V13/A03 chiuso invariati, zero nuove conferme umane;
L04formule output UNAVAILABLE distinta, L01PDF/OCR e L02compile NOT_MEASURED,
L05CPU lower/RSS sampled/hardware locale. ChromeNONPASS/helper12gap/
H15 storico/E9 atteso storico errato e failures intermedi non sanati.

## Criteri tecnici comuni

Valuta l'intero delta e le dipendenze raggiungibili; non un elenco chiuso di
payload. Per ogni rilievo: input ammesso, percorso reale, requisito violato,
impatto, posizione precisa, attribuzione r008/ereditato/mandato, prova
statica o runtime, correzione minima e prova di chiusura. Sintetici/rari
pertinenti se esercitano profilo/contesti/fedeltà; nessun parser universale
richiesto e nessun rilievo inventato per riempire la tabella.

| ID | Controllo sostanziale |
| --- | --- |
| C01 F35 progresso e terminazione | Ogni transizione markdown_lexemes avanza o consuma uno stato finito dimostrabilmente terminante? Limiti aggiornati dopo blank, cursore mai arretra per backslash/backtick/raw fence; chiusura effettiva/EOF e token attraverso falsa chiusura coerenti col consumer? Contenuti esterni/tipi/raw/offset/diagnosi preservati, non semplice timeout o rifiuto di input ammesso. Tre prove prima/dopo con timeout breve/reap e snapshot dei byte r007/r008 reali. |
| C02 F36 stesso blocco | Backtick/$x/$$/<alpha incompleti nella stessa riga e soft-line non saltano veri commenti/CDATA chiusi o EOF? Token intero attraverso blank/heading/list/quote/nota/fence, corpo mai prosa/math/ref/asset attivi, zero aperture/hash/UNSAFE_REFERENCE per immagini inerti anche ../. Omissione q/page2 realmente resta FAIL, unknown non wildcard. Letterali chiusi/code/math/attributi/comandi, valuta/< in prosa e veri asset attivi discriminano; evitare falsi errori su input valido. |
| C03 F37 label e intervalli | markdown_label/label_codes e scanner/consumer usano lo stesso intervallo code soft-line nel dominio finito di label, senza chiusura presa da blocchi estranei? Link/image-label, uno/due backtick, commento/CDATA/PI/declaration letterali, paragraph/quote/list/nota/cella/HTML/raw fence pertinenti preservano struttura/ordine/testo/ref; vero token fuori code resta intero. EOF label e contesti interni non assorbono heading/paragrafi esterni. Offset char/UTF8 Unicode reali, non solo COMPLETE. |
| C04 confini HTML e guardie | Aperture HTML reali passano tutte dalla guardia PassiveHTML prima della proiezione; URI/attributi/risorse/duplicati/eventi e confini path conservati, indipendentemente dal contenitore supportato? Prefisso incompleto prima di un token vero finisce localmente come unsupported/incomplete-start, non apertura fino EOF né attributo attivo inventato. Trasporti, nesting, tabelle/figure e nodi residui non fanno sparire testo/ref/unknown; byte inerti non riattivati. Niente comandi/fetch/fallback da dati o attenuazione safety; verificare raggiungibilità delle modifiche alla collocazione della guardia. |
| C05 regressioni dei domini | F32autolink, F27close vuoto, F30reason, F28CDATA locale, F26 e F16 preservati; F31/F33/F34 e classi F29/F09/F13/F14 chiuse nei domini toccati? Raw fence letterali/HTML/TeX, metadati/attributi e foglie TeX restano inerti, parser limitato/unknown visibili. Positivi discriminano un reader che rifiuta tutto; un falso FAIL/invenzione di perdita su documento valido è pertinente anche senza falso PASS. |
| C06 F38 documentazione e storia | PROTOCOL/due guide/report/dati descrivono il codice/profilo effettivo e le prove finali, con ID/hashes/dipendenze? Prove finite non certificano tutto Markdown/HTML/TeX; TIMEOUT prima, test/helper/edge/CLI-image failures e H12 rimangono fallimenti. Correzioni degli attesi/fixture proprie motivate con gold canonici e scorer invariati; nessun PASS retroattivo o V agente. |
| C07 64misure e V | Due set/64raw, hash/catena/source/gold/V/run/old_* e metriche attuali coerenti, nessun motore rieseguito o qualità forzata? Somme/categorie/numeri/celle/omissioni/unknown/ordine/asset/formule secondo contratto, qualità non media. WER complete/condizionali/bound globale domini distinti: DOCX11/182 didascalie,RTF19/162 condizionale e13/162globale,EPUB8/186,TXT0/89,TeXfullNone/cond1/152/globale0TRIVIAL_ZERO; zero bound non fullPASS. V13 sui riferimenti non verifica formule output L04; L01/L02/L05 precise. |
| C08 toolchain/prove attribuite | Nuovi S/B/I S007/sdist-wheel/installazione canonica/dieci hash/startup/stesso interprete/gate precedono managed/no-sync, R/firejail/rete0/figli reap/binding attuali? Equivalenza S006→S007 limitata a205runtime/124helper-test/3813package e stage/binding diversi; receipt23PASS2FAIL finali e114exit reali, non soli assert Python. Helpers fuori wheel manifest/hash/copie reali prima-dopo, modificati test invalidano le sole prove dipendenti. Non trasferire preflight vecchio o restauro pyc; distinguere prove autore lette da prove proprie. |
| C09 risorse e arresti | Disposizione24h effettiva con entry/contatori originali, wall di ricezione distinta dal ledger autore storico, costi/RSS/lower/approssimazioni dichiarati. Deadline/stop nuovi workload e limiti CPU/disco/RAM/stream/file applicati cumulativamente, timeout breve e reap per loop, niente download per correggere collection. Riserva protegge consegna/report, nessun microcap che richiede nuovo prompt per ogni sonda. |
| C10 ciclo e perimetro | Report completo senza GO autore, ruoli separati, nuovo comune valido e due chat realmente indipendenti. Fix locali non nuovo piano o fase2; residui fuori scope suggerimenti motivati, requisiti essenziali raggiungibili non ignorati. Nessun Git automatico/cleanup/a001/nuova V/motore scelto; esito GO o NO_GO sul comune corretto, non dal numero di test o dall'assenza dell'altro report. |

## Autonomia, input propri e risorse correnti

Consentiti letture/hash/link/aritmetica/manager stdlib e propri output.
Piccole sonde sintetiche reader→audit→score, incluso CLI pertinente con writer
sintetico, consentite se necessarie: copie proprie, attesi prima, gate validi
sui byte reali e R corrente. Non rieseguire suite/test campaign, build,
reinstallazione, contratti H generali, baseline/conversioni/render/browser,
M/V, download/modelli/remoto/pagamento o Git di scrittura. Nessun fix al prodotto,
test/corpus/gold canonici/input altrui. Review statica ammissibile con limiti
espliciti; se una prova decisiva manca, non trasformare l'incertezza in GO.

Prima del primo managed registra origine locale3.12.13/startup/package,
archivi/installazione reale/dieci hash e S/B/I/preflight sulle dipendenze
consumate. Manifest/launcher/target e nuovi snapshot tecnici propri con label
`review-r008-claude-*` autorizzati, output vivi esclusi; anche se l'helper
risolve sotto la directory comune. Nessun freeze intermedio o mutazione di
snapshot ricevuti. S007 integrale dopo quattro documenti supervisore è STALE:
conserva FAIL, diagnostica e identifica nuovi input circoscritti realmente
consumati. Non avviare con gate rossi, trasferire PASS generali o ripristinare
vecchi input. Ripeti soltanto prove invalidate; niente nuova campagna a001.

Operazioni ordinarie reversibili e recuperi argv/percorso/log/launcher restano
nel mandato. EPERM/EACCES non equivale a rifiuto di approvazione: diagnosi ed
eventuale escalation tool della stessa azione mantenendo R/no-network e
confini; fermati al rifiuto effettivo, senza aggirarlo. shell=False/registry/
PATH/safety e stesso interprete dei figli. R≤900s/1MiB per stream/32MiB per
file; per sospetti cicli usa timeout breve predeterminato e raccogli i figli.

Quota temporale24h **già disposta** dalla conferma osservata
2026-10-10T09:37:31Z (non timestamp piattaforma inventato). Entry originale
2026-10-08T07:06:54.369068Z; nessun reset. Stop nuovi workload
**2026-10-11T06:37:31Z =08:37:31 Europe/Rome**;
deadline **2026-10-11T09:37:31Z =11:37:31**, riserva3h.
Limite wall cumulativo268236,630932s/stop257436,630932s. CPU21600s/
stop19800,disco6GiB/stop5GiB,RAM8GiB/stop sampled6GiB;
riserva1800CPU-s/1GiB,rete0/acquisizioni0/document-send0/€0. Stop al primo
limite raggiunto; dopo stop solo consegna/letture/documenti consentiti
nella riserva, niente nuovi workload. Leggi clock reale all'ingresso.

Ledger autore r008 storico:wall185399,827848s (~51,500h),CPU11610,304610s
noti lower-bound;45CPU-s Claude r007 approssimati separati (stop accounting
11655,304610s),retained2384268251B/RSSsampled740855808B.
Wall ricezione supervisore53.109h, non saldo CPU/disco certificato. Registra
costi propri e aggiorna accounting per le prove visibili senza sommare wall
parallele o duplicare CPU già contabilizzata. Vecchia quota52h/stop/r007
NONPASS preservati; estensione prospettica non sanatoria. Nessuna nuova
approvazione per ogni comando/snapshot entro il mandato. Se limiti già
raggiunti, prosegui la review statica entro deadline e dichiarane i limiti.

## Report, esito e consegna

Usa il template review; dichiara provider/modello/chatID reali se esposti,
prompt/revisione/hash/base/delta/indipendenza, verify prima/dopo, criteri
controllati, argv/input/prove/costi e limiti. **GO** o **NO_GO** motivato sul
comune r008; GO non certifica altri motori/L04 o fase1 intera. Identità
invalida: dichiara impedimento e prossimo passo, nessun GO sullo stato vecchio.
Ogni rilievo:ID/severità/blocco,posizione/raggiungibilità/requisito/
attribuzione,problema/impatti,evidenza statica o runtime,correzione minima/
criterio di risoluzione. Chiusure e regressioni F35–F38, non un voto medio.

Non modificare STATE/HANDOVER/eventi/artifacts/indici/changelog/CP supervisore,
snapshot comuni/storici o file concorrenti. Non avviare subagenti/reviewer
delegati; questa è una delle due nuove chat indipendenti richieste.
Non leggere la review concorrente r008 né l'arbitrato corrente r008, anche
se compaiono durante la chat. Nel checkpoint registra input/output/esito,
costi/processi propri pendenti e prossimo ruolo **supervisore per ricezione
di entrambi i report r008 e arbitrato**. Concludi in chat con esito reale e
percorsi, senza implementazione o altre review delegate.
