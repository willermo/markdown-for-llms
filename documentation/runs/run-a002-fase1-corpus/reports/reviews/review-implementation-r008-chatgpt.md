# Review — implementazione — run-a002-fase1-corpus — r008 — ChatGPT

- Autore/provider/modello: reviewer richiesto ChatGPT; esecutore effettivo
  OpenAI Codex, basato su GPT-6 secondo l'identità esposta nella sessione.
  Identificatore preciso di deployment e chat ID non esposti. Nessuna sostituzione
  o delega simulata; nessun subagente.
- Prompt origine: [28](../prompts/28-review-implementation-r008-chatgpt.md),
  SHA256 `f191bf1352dba4808348a2d8aae83c6fae14051b553f5a19ed2924490ae63249`.
- Oggetto: implementazione r008, fix F35–F38, stato ricevuto
  IMPLEMENTATION_REVIEW / FIX_REVIEW_PROMPTS_READY. Esito **NO_GO**.
- Comune [implementation-r008](../../../../../temp/run-a002-fase1-corpus/snapshots/implementation-r008.json), SHA256
  `2f6d690a8f3db7af8f192f78a9e6f5c0e3704b9757256f9dfdf72e35ee3b4152`;
  worktree `448b854a88dc3dc883f7337f6fe037fd27846513e3a19104fc88529841e07eda`.
  325 file generali / 76333 artefatti espliciti. Verify proprio prima e dopo
  **MATCH**, exit 0; nessuno snapshot ricevuto rigenerato.
- Branch `feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
  `80c31da46e5caab990d3596c79c44e524ec1dc95`, indice vuoto. Considerati anche
  modificati e nuovi non ignorati, non soltanto commit/diff base...HEAD.
- Indipendenza: nuova chat reviewer rispetto agli input autore/supervisore;
  letti i due report e checkpoint **r007** come storia comune, nessun report
  concorrente o arbitrato **r008**. Le review r007 erano statiche: non attribuite
  loro prove runtime. Nessuna scrittura fuori dai propri output e snapshot
  tecnici autorizzati, nessun aggiornamento di stato condiviso.
- Giro di fix, arbitrato tecnico r007 NO_GO e prompt27 come mandato; disposizione
  risorse r008-r001 **DISPOSED24h**, non proposta 12h PENDING. Base piano r001
  SHA256 `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8`,
  errata `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9`.
- Delta autore: sette file (reader, PROTOCOL, test r008, due guide, report/dati
  permanenti). Quattro registri generali del supervisore separati. Reader
  `observations-v8-progress-and-inline-domains`, SHA256
  `817852a8f8b2645a3bbc6212d982aca4546ab1c4b3f128374137313124aa6061`.
  Scorer v4 e dipendenze di safety/audit/adapter/R invariati; la nuova
  collocazione della guardia PassiveHTML è inclusa nella review.

## Ambito e prove

Percorsi delle prove proprie relativi a
[evidence/review-implementation-r008-chatgpt](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r008-chatgpt).
Lette le istruzioni, contesto corrente/handover/stato/brief e checkpoint
supervisore; protocollo, skill manage-implementation-run e
verify-conversion-fidelity, template, prompt27 e arbitrato/addendum con fonte
utente. Letti report/checkpoint autore r008 **integrali**, incluse consegna
completa e chiusura amministrativa, report/dati permanenti, receipt/final-checks/
semantic-equivalence/budget e transizione del comune. Consultati ADR0001,
indice/registro ADR, fase1 roadmap, piano/errata/workflow per i contratti
pertinenti, guida uv e toolchain per i limiti delle prove.

Reader letto integralmente, compresi domini producer/label, progresso dei
cursori, fence, prefissi incompleti, trasporti Markdown/HTML e guardia prima
della proiezione. Seguiti audit, safety/PassiveHTML, metrics/allineamento,
inventory/profile/input validator, adapters e CLI raggiungibili. Letti il nuovo
test e regressioni r003–r007 pertinenti, PROTOCOL e due guide. Confronto sorgente
con r007 conservato in [reader-delta.diff](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r008-chatgpt/reader-delta.diff).
Nessuna campagna a001 o lettura indiscriminata delle voci del comune.

### Identità e installazione effettiva

Comando comune, cwd `/home/davide/workarea/markdown-for-llms`:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r008
```

[Prima](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r008-chatgpt/verify-before.json):
ingresso 2026-10-10T13:22:46.836281Z, exit 0, 31,507s, MATCH; stdout/stderr
separati conservati. [Dopo](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r008-chatgpt/verify-after.json):
fine 13:37:51.973274Z, exit 0, 23,921s, MATCH, stdout/stderr nel receipt.
Manager host stdlib, non prova del prodotto Python. Identità Git e stato
non committato in `git-identity.json`, `git-status.json`, `git-index.json`;
`diff-check.json` exit 0. Hash degli input e snapshot propri in `review-inputs.json`.

Il verify del vecchio S007 integrale è correttamente **STALE/exit 2** e resta
in `verify-s007-stale.json`: i quattro delta sono soltanto CHANGELOG, indice
documentation, next-run e roadmap del supervisore, identificati per hash in
[supplement.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r008-chatgpt/supplement.json).
Non trasferito quel PASS generale, non restaurati documenti o pyc.

Prima del primo managed: `host-before-managed.json` verifica origine locale
3.12.13, startup, 3813 package input, installazione I effettiva, archivi
sdist/wheel e dieci hash sorgente=wheel canonica=installato. Poi
`origin-current.json` e `scoped-preflight-host.json` PASS. I preflight propri
consumano 5001/5004 input circoscritti e controllano origine/interprete -I -B,
startup e ten modules; sono un gate della review, non nuova build/installazione
né certificazione di S007 generale. Target/manifest/launcher/attesi vincolati
nei due snapshot tecnici propri, senza mutare i ricevuti. Le dipendenze realmente
consumate hanno gate PASS corrente prima delle sonde; stesso interprete dei figli.

### Prove ricevute controllate, senza riesecuzione

[static-audit.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r008-chatgpt/static-audit.json)
documenta 7843 controlli stdlib su byte, legami e aritmetica. Verificati i 25
receipt finali selezionati: 23 PASS + 2 FAIL H12 reali, scope gate-only distinto
dal workload, binding prima/dopo, inputs_unchanged e reap dei comandi. Le 47
receipt storiche restano una serie diversa. Controllate 4310 copie helper
consumate archiviate per hash/size; equivalenza di 205 input runtime e dieci
origini installate S006/S007, con PROTOCOL unico delta dei 206 input benchmark.
I 124 helper/test e 3813 package/startup sono equivalenza selettiva ricevuta e
verifica dei byte pertinenti, non identità dell'intero stage (gate/shebang/binding
sono nuovi). Non attribuita causalità ai tre pyc storici.

Controllata matrice: prima 128 non conformi, dopo-005 128 conformi, 64 F36 +
64 F37; 5 Unicode/4 trace cursor strettamente crescenti, fence positivi/edge
pertinenti letti. Tre before-loop observe/audit/measure ricevuti sono FAIL,
TIMEOUT 3s, exit -15 e figli reaped; tre after-005 sono PASS/exit 0 senza timeout
con lo stesso limite. Sono prove **autore lette**, non mie sonde dei cicli.
1373 mirati/16 deselected (261 nuovi) e 1530 veloci+310 subtest sono risultati
autore ricevuti, non suite rieseguite dal reviewer. Contratti-002 e stdout
finali letti nei limiti del mandato, senza nuova campagna H.

Il supplemento collega i 57 casi CLI ai **114 exit OS registrati** dai
subprocess run/evaluate, manifest-before e file correnti per hash: 55+2 image,
writer sintetico dichiarato, nessun motore, gold PROVISIONAL/no V. Le immagini
positive hanno asset osservato e hash/dimensioni/ref Unicode nel set autore;
scorer 5 con automatic PASS mantiene qualità INDETERMINATE.

Per tutte le 64 misure r008b verificati legami/hash di raw/gold/V/run/old_*/
metriche/observations/audit, raw copiati identici e final derivati distinti.
Dati permanenti coerenti: `final` corrente, `conversion_final/run` originali;
non confusi con il final storico nel summary di input. Due set32, ciascuno
4 PASS/28 FAIL; zero differenze nei 19 campi numerici/gate contro r007.
Calcolo indipendente Levenshtein su edit registrati e domini globali, somme e
denominatori; **non** ricostruito indipendentemente l'intero allineamento.
DOCX 11/182 incluse didascalie; RTF 19/162 condizionale e 13/162 globale;
EPUB 8/186; TXT 0/89; TeX full WER None, condizionale 1/152, globale
0 TRIVIAL_ZERO. Zero bound non è full PASS; qualità non media dei componenti.

### Sonde proprie e fallimenti preservati

Due sole piccole raccolte reader → audit → score: 16 casi, poi 6 con gold
proprio fissato prima. Nessuna conversione/render/browser/build/reinstallazione/
download/M/V o suite. [R-probes-002](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r008-chatgpt/R-probes-002.json)
e [R-probes-003](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r008-chatgpt/R-probes-003.json)
PASS, isolamento Firejail/rete0/guardie socket e daemon, timeout 15s, stream
1MiB/file32MiB, figli raccolti e binding prima/dopo invariati. Tutti gli argv,
cwd, target, preflight e receipt dentro R sono conservati. R PASS significa
raccolta eseguita: il collector restituisce 0 e registra anche mismatch prodotto.

La prima partenza `R-probes-001` era IMPEDITA per EPERM sulla socket guard prima
del workload, non un rifiuto auto-review. Diagnosi conservata, escalation dello
stesso runner approvata, nessun indebolimento R/no-network né privilegio persistente.

I 12 edge F36 bare/wrapper/raw-HTML × comment/CDATA × chiuso/EOF confermano
prefisso incomplete-start locale e token intero, corpo inerte, zero riferimenti
o aperture di immagini nascoste anche ../, q/page2 missing e audit INCOMPLETE
UNSUPPORTED_RESOURCE_SYNTAX. I miei attesi iniziali omettevano il legittimo
token incomplete-start e nodi unknown vuoti: mismatch di checker preservati,
diagnosticati in `probe-html-postcheck.json`, non rilievi prodotto né PASS
retroattivi delle assertion originali. Nei sei casi con gold, due positivi
conformi e quattro controesempi dello stesso difetto F37 (quote/list/note/Unicode).

Preservati anche due errori dei checker stdlib propri: stage gate-only non ha
command/reap (`static-audit-001`), e final storico nel summary non coincide per
contratto con final corrente permanente (`supplement`). Versioni dei checker
precedenti conservate, correzioni circoscritte e nuove prove -002 PASS. Nessun
input/prodotto altrui modificato. Copie fisiche proprie di reader/helper/casi
prese **dopo** le sonde in `copies-after`; i gate e binding anteriori identificano
i byte consumati, senza fingere copie fisiche precedenti.

## Rilievi

| ID | Severità e blocco | Posizione | Raggiungibilità/input ammessi | Attribuzione e requisito | Problema e impatto | Evidenza | Criterio di risoluzione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-R008-I001 | MEDIUM, sì | observations.py:394, 531, 810–814; scanner 236–239; boundary 39, 47–60 | Label con code soft-line in quote/list/note; `- ` letterale sulla continuazione dopo il prefisso | Residuo F37, C03/C05; consumer default ereditato non riallineato dal fix r008, incide sulla chiusura F38 | Scanner ammette il code nel producer reale, consumer senza limite lo tronca dopo stripping: perde link, inventa testo sintattico e false missing/unexpected/FAIL su input valido pur con COMPLETE | [Traccia statica e runtime](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r008-chatgpt/finding-I001.md), R-probes-003, quattro controesempi e due positivi | Unico dominio/intervallo effettivo nei due percorsi, senza scavalcare producer; testo/link/ordine/offset corretti e automatic PASS/score5 nel gold PROVISIONAL, negativi fuori dominio e opacità preservati |

## Motivazione dell'esito e limiti

**NO_GO sul comune r008 valido.** F35 e F36 hanno correzioni concrete sostenute
dalle prove ricevute e dalle sonde pertinenti. F37 migliora il caso originario,
ma non condivide l'intervallo code tra scanner e consumer nei producer supportati.
Il rilievo è essenziale per fedeltà di testo/riferimenti e validità delle metriche;
non dipende dal conteggio dei test o da una pretesa di Markdown universale.

| Criterio | Conclusione nel perimetro controllato |
| --- | --- |
| C01 F35 | Progresso dei passaggi modificati e raw-fence/boundary verificato staticamente; tre before/after ricevuti confermati. Nessun nuovo rilievo; non prova di terminazione di ogni possibile documento. |
| C02 F36 | Matrice e edge confermati; token intero, contenuto opaco, omissioni reali ancora FAIL. Sonde proprie HTML incomplete-start chiuso/EOF discriminano il prefisso locale. |
| C03 F37 | **Non chiuso**, GPT-R008-I001. Positivi originari/Unicode autore non coprono il marker letterale dopo stripping. |
| C04 HTML/guardie | Guardie di aperture reali prima della proiezione seguite staticamente in HTMLTree/trasporti, vincoli URI/attributi/path nel PassiveHTML invariato; edge runtime sopra. Nessuna attenuazione rilevata nei percorsi controllati; nessuna certificazione di HTML universale. |
| C05 regressioni | Seguiti F32/F27/F30/F28/F26/F16 e domini F31/F33/F34, F29/F09/F13/F14 pertinenti; letterali/raw/metadati/foglie non promossi a risorse. Resta la divergenza label del rilievo. Non rieseguita l'intera regressione. |
| C06 F38 | Contratti e storia distinguono prove finite/gold PROVISIONAL/unknown; chiusura F37 va rettificata dopo fix. TIMEOUT, fixture/test/helper/edge/CLI-image e H12 non sanati retroattivamente. |
| C07 misure/V | Catene e aritmetica 64 coerenti; nessun nuovo motore o V. Checker non certifica allineamento completo. V13/A03 chiuso riguarda riferimenti, non L04 formule output. |
| C08 toolchain | Attribuzione ricevuta e gate propri circoscritti distinti; S007 generale STALE diagnosticato, origine/interprete/installato/ten e R propri correnti PASS; nessun PASS generico trasferito. |
| C09 risorse | Disposizione 24h/entry originale rispettate per le prove proprie, CPU lower/RSS sampled e uso concorrente non certificato distinti; nessun reset o sanatoria del vecchio stop. |
| C10 ciclo | Ruoli separati, nessun GO autore, fix nello stesso mandato; report indipendente senza mutazioni condivise/Git automatico/fase2. Prossimo ruolo supervisore. |

Errori intermedi restano fallimenti: before-loop TIMEOUT; after-matrix001,
targeted002 (regressione + attesi propri poi corretti), edge001, CLI-image002
(image nella gold.references non-image), image003 KeyError harness; final image004
corregge fixture/attesi propri senza cambiare scorer/gold canonici. Restano
Chrome NONPASS/helper12 gap/H15 ed E9 storico con atteso errato. Nessuna nuova
conferma umana: L04 formule output **UNAVAILABLE** distinta, PDF/OCR L01 e
compile L02 **NOT_MEASURED**, L05 CPU lower/RSS sampled/hardware locale. Review
non certifica altri motori, migrazione toolchain o fase1 intera.

### Risorse e consegna

[costs.json](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r008-chatgpt/costs.json) all'istante
2026-10-10T13:39:26.009479Z: ingresso proprio osservato 13:22:26Z; wall run
196351,640411s dall'entry originale 2026-10-08T07:06:54.369068Z (non somma di
wall parallele). CPU propria misurata dei figli **70,185594s lower**, senza
duplicare R annidato; CPU autore 11610,304609507s lower + Claude r007 45s
approssimati separati → saldo visibile lower+approssimazione 11725,490203507s.
Non è un upper bound globale certificato: letture amministrative senza receipt,
wrapper/inferenza e altri reviewer concorrenti non inclusi. Storia autore wall
185399,827848s e ricezione supervisore 53,109h restano istanti storici diversi.

Output propri logici ~10,06MB + snapshot tecnici 142660B al controllo; storia
retained 2384268251B → somma visibile ~2,394GB, documenti finali aggiunti dopo
accounting. RSS payload sampled massimo 27754496B; maxrss esterno dei receipt
315531264B è misura diversa. Nessun processo workload proprio pendente,
reap esplicito dei due R; rete/acquisizioni/document-send/€ **0**.

Stop nuovi workload 2026-10-11T06:37:31Z (08:37:31 Roma), deadline
09:37:31Z (11:37:31), riserva3h; stopCPU19800/max21600s, stopdisco5/max6GiB,
stopRSS sampled6/max8GiB, limiti originali non aumentati. Nessun ulteriore
workload necessario per questa consegna.

Checkpoint [review-implementation-r008-chatgpt](../handovers/review-implementation-r008-chatgpt.md).
Il **supervisore** riceve entrambi i report r008 e arbitra sul comune corretto;
questa review non chiude rilievi o aggiorna stato/arbitrati. Nessun commit,
merge, push, pulizia o fix prodotto eseguito.
