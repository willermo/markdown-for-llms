# Implementazione a002 r002 — fix del benchmark dopo doppio NO_GO

Agisci esclusivamente come **implementatore**, proseguendo preferibilmente nella
stessa chat che ha consegnato r001. Completa tutti i fix accolti, prove dipendenti
e consegna r002 nella stessa chat. Non supervisionare/revisionare te stesso e
non modificare i registri comuni. Non serve una nuova pianificazione generale.

## Mandato e input

Run `run-a002-fase1-corpus`, feature `feature/run-a002-corpus`, base/HEAD/dev
`80c31da46e5caab990d3596c79c44e524ec1dc95`; indice vuoto all’arbitrato.
Consegna r001 revisionata realmente da ChatGPT e Claude: NO_GO/NO_GO.
Arbitrato vincolante `arbitrations/arbitration-implementation-r001.md`: NO_GO
locale con F01–F12; piano r001+errata e GO piano r002 restano il mandato originale,
con prompt05/addendum V/prompt06 e presente correzione di requisiti puntuali.

Percorsi abbreviati sotto `temp/run-a002-fase1-corpus/`. Leggi:

1. AGENTS.md, temp/HANDOVER.md e sezione corrente PROJECT-CONTEXT, STATE/HANDOVER/
   brief e checkpoint handovers/supervisor-implementation-arbitration-r001.md.
2. Skill manage-implementation-run e verify-conversion-fidelity e protocollo
   run-lifecycle; arbitrato implementazione r001 **integrale** e questo prompt.
3. Entrambi i report `reviews/review-implementation-r001-chatgpt.md` e
   `review-implementation-r001-claude.md`, checkpoint/evidenze citate. Sono input
   reali dei fix: puoi leggerli entrambi, non sei più un reviewer indipendente.
4. Piano r001/errata E01–E08, arbitrato piano r002, prompt05 e06/addendumV,
   report autore r001/v2-fix e proprio checkpoint nella parte corrente;
   report benchmark verified-reference/human-comparison/native per attribuzione.
5. Codice e test colpiti, protocollo benchmark/guide, manifest/spec/gold/controlli,
   toolchain uv/S/B/I/R pertinenti. Non rileggere tutta la storia a001.

Verifica feature/HEAD/base/indice/delta reale prima di modificare. Non basta
`git diff base...HEAD`. Il contesto tecnico nuovo è
`snapshots/implementation-fix-context-r001.json`; receipt/hash/impronta nel
checkpoint supervisore ed evidenze supervisor-implementation-arbitration-r001.

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r001
```

Verifica all’ingresso prima dei cambiamenti autorizzati, conserva argv/cwd/exit.
I20PDF oggi ignorati sono copiati/hashati nella ricezione: confronta anche i
file reali con `ignored-pdf-inputs.json` prima del fix Git. Un MATCH tecnico
non approva codice. Lo snapshot r001 è storico, non rigenerarlo o pretendere
MATCH dopo nuove modifiche autorizzate. Identifica autonomamente i target di
esecuzione e ripeti le prove invalidate; non chiedere un altro freeze al supervisore.

## Fix richiesti e prove di chiusura

| Fix | Rilievi e risultato | Prova discriminante richiesta |
| --- | --- | --- |
| F01 | GPT-I001: mantenere contenuti inline tipizzati/coordinate/formule/ref/unknown in celle e caption, incluse tabelleHTML. Audit deve vedere ogni asset/link; scorer deve vedere aggiunte/perdite, senza doppio conteggio. | K01 con cella davvero vuota positivo; `$x_i$` aggiunto alla cella vuota deve dare errore automatico o unavailable motivato senza PASS spurio; immagine assente nella cella→INCOMPLETE/run2/score3. Anche immagine/link/formula/unknown nelle caption e celleHTML, positivi con contenuti leciti e target presenti. Non bastano stringhe text identiche. |
| F02 | GPT-I002: alignment formula coerente con equivalenze AST finite E01, preservando raw e matching one-to-one. | Tutti FP01–FP13 attraverso observe→align→score, incluso `\frac{a}{b}`/`\frac a b`: gate automaticoPASS, L04 resta unavailable. FN/FU restano distinti, duplicati/ambigui non ottengono PASS. Non si chiede algebra universale. |
| F03 | GPT-I003: monitor risorse e R custodiscono tutta l’esecuzione dei figli propri, anche dopo exit del leader/setsid. Caps/timeout/namespace/RSS sampled pertinenti e controllo finale dopo chiusura writer; reap certo e niente segnali a processi estranei. | Figli che scrivono stdout/stderr1MiB+8192B dopo100/200ms, figlio setsid, timeout oltre leader e positivo entrocap: stop/NONPASS corretti senza receiptverde o pendenti. Provare sia resources.run sia vero monitored_command e invocazione esterna R completa, non solo probe sottoR. Preservare H15storico/ChromeNONPASS,20contratti e positivi. |
| F04 | GPT-I004: subset finito passivo per XHTML/HTML/CSS/SVG/ZIP; srcset e attributi risorsa tutti validati/rifiutati, CSS escape gestiti o rifiutati, ogni membroSVG controllato o rifiutato. | Copia EPUB canonico con srcset file/remoto, CSS `u\72l(...)`, SVG script/resource vietata: INVALID_INPUT/exit4 **prima** di estrazione/motore; marker/test assicurano tool e target non aperti. Positivi asset locali/link esterni inerti. Non eseguire browser/Pandoc sui payload per dimostrare che fanno danni. |
| F05 | GPT-I005: inserzioni di parole unmatched entrano nel WER dello scope testuale dichiarato, denominatore atteso invariato. | `Known words.` contro aggiunta `Additional invented content.`: WER>0 con numeratore atteso; positivoWER0. Omissioni/sostituzioni/aggiunte miste e formula raw equivalente non ripenalizzata come prosa. Rettifica misure/report dipendenti. |
| F06 | CLA-I001: eccezione .gitignore solo per PDF sintetici del corpus; i20PDF entrano nel set versionabile e futura impronta comune. Niente `git add -f` come sostituto del contratto. | check-ignore con esito non-ignorato per20file, status-uall/ls-files-others e confronto20hash invariati con manifest/ricezione; verifica set completo includendo asset/derivati. Nessun staging/commit. Nota nel README corpus. La sola regola Git non impone rigenerazioni o conversioni. |
| F07 | CLA-I002: header semantico esplicito nel gold/spec, controllato indipendentemente sul sorgente; header sintattico pipe distinto dal corpo, numeri/celle coordinati correttamente. Sorgenti DOCX/RTF/TeX attuali restano headerless, EPUBth resta semantico. | Headerless→placeholder header vuoto+corpo identico: corpo9/9,6/6,9/9 e numeri di cella recuperati. Promozione indebita del corpo→erroreheader; vera cella/riga vuota vs assente distinta; header reale/header vuoto reale e perdita di cella falliscono solo quando dovuto. Nessun drop generico della prima riga né matching arbitrario contro output ufficiali. Nuovo M markup/fonti applicabili e derivati coerenti. |
| F08 | CLA-I003: missing/page loss restanoFAIL davanti a unknown estraneo; unresolved attribuito solo localmente con regola conservativa versionata e prova, non wildcard globale. | K02 senza pagina2 più `\unknown{x}` estraneo→score3/missing_pagep2 e unavailable separato. Unknown che plausibilmente contiene uno specifico blocco→unresolved attribuito, senza coprire altri mancanti o darePASS. Ambiguità ripetuti, omissioni/inversioni/aggiunte indipendenti preservate. |
| F09 | CLA-I004: target e titolo separati in link/immagini inline e figure; raw_attribute inline tipizzato/rawunknown invece di code. | Titoli doppi/singoli/parentesi, target con parentesi/escape/URI vietata, titolo con colon che non rende URIunsafe; assetpresente positivo e assentevero2. RawLaTeXinline conservato/localizzato come unsupported appropriato, code veroimmutato. Profile finito esplicito, nessun falsoCOMPLETE su sintassi ignorata. |
| F10 | CLA-I005: transcript completo include11caption tabella già presenti in annotations/vista. Aggiornamento derivati distinto da sorgenti. | Tutte9trascrizioni interessate contengono caption corrette nell’ordine, senza duplicati/codicepoint persi; generator goldentest contro atteso autonomo. Corpus/sourcehash stabili, inventario/riproducibilità dei soli derivati aggiornati. |
| F11 | CLA-I006: controllo E08 tokenPandoc risolto secondo PATH effettivo allowlist prima/dopo, realpath/hash/versione registrati e coerenti tra figliolegacy/diretto. | PATH con Pandoc falso anteposto/user-bin/cwd/segmentovuoto/dirrelativa→input4/H10 prima del figlio; positivoPATHchiuso usa binario atteso; mutazione tra controlli rilevata. Nessun ripristino PATH ambientale implicito. |
| F12 | CLA-I007: provenance ispezionabile degli helper consumati nuovi, vecchi12raw marcati storici con limite non sanato. | Nuovi attempt su codice corrente con byte degli helper e manifest prima/dopo conservati; nuoviH06/H08 e contratti colpiti. Recupero byte storici eventuale solo se hash coincide, mai diff inventato. Nuova campagna dipendente descritta sotto evita basare la consegna corrente su helper non ispezionabili. |

Usa le riproduzioni originali dei reviewer come casi di regressione, preservando
i loro file. Crea copie proprie di test; non alterare i confronti protetti o
vincere il test eliminando il fenomeno. Questi test verificano comportamenti
promessi, non sono nuove funzionalità applicative. Sono consentite soluzioni
reversibili diverse da quelle enumerate se soddisfano requisiti/costi/confini.

## Gold, riferimento umano e rettifiche

F07 aggiunge semantica header **indipendente dal motore**, verificabile da M
sul sorgente. Non riscrivere le tabelle originali per far passare Pandoc.
Versiona gold/manifest/metadata/derivati, preserva rev4 e hash/byte di confronto;
non rigenerare PDF/canonici/render invariati per un campo nuovo o un transcript.
Per PDF/raster esplicita fonte/policy e limiti, non attribuire proprietà encoded
che il formato non offre. ReferenceMarkdown/HTML e mapping devono preservare
headerless/header reale; nessun output motore diventa gold.

La V sui13documenti è reale e non è annullata. Se i nuovi metadati cambiano
hash, produci nuove receipt **derivate**, non copiare VERIFIED su hash nuovi:
conserva risposta originale/vecchi pair/packet/gold, delta, nuovoM, checker
indipendente di tutti i contenuti visibili approvati. Invarianti: sorgenti,
testo/numeri/formule/ordine/celle/caption/asset/note/riferimenti. Header tecnico
qualificato daM, non attribuito a un nuovo giudizio dell’utente. Distinguere
inheritance della V sul contenuto invariato da nuova conferma umana inesistente.
Non imporre nuovi TSV,215firme o13review umane per un cambiamento di metadati.
Solo una modifica visibile sostanziale o un’invarianza non dimostrabile torna al
supervisore per perimetro mirato; intanto completa il lavoro indipendente.

Rescore storico: nuove osservazioni/audit/metriche dei32raw selezionati r001,
senza motori, con nuove identità e vecchie osservazioni/metriche/final immutati.
Rettifica celle/numeri/WER/rawinline nei report **nuovi** permanenti con confronto
prima/dopo e collegamenti ai report storici. Non riscrivere le vecchie prove
per nascondere l’errore; non vincolare il risultato al vecchio4PASS/28FAIL.
Le perdite reali Pandoc restano ammissibili/pubblicate; L04formuleoutputUNAVAILABLE
non diventaPASS da Vgold o AST. L01PDF/OCR,L02TeXcompile,L05hardware non misurati.

## Esecuzione completa delle prove dipendenti

Prima chiudi F03/R e verifica guardiepositive/negative stdlib con isolamento;
poi nuovo stage S/B/I, build/reinstall wheel canonica/diecihash e inventario
dei helper fuori wheel, managedCPython3.12.13 e figli nello stesso interprete,
target/preflight/R PASS prima degli import e `--no-sync`. Conserva byte dei
helper/config/test consumati, non soli hash. Crea autonomamente snapshot di
esecuzione con output vivi esclusi; verifica i target prima/dopo.

Esegui test mirati F01–F11,20contratti pertinenti (H06/H08/H10/H15 e dipendenze),
rescore storico e **nuova baseline8percorsi×4attempt** con stessi sorgenti,
profiliwriter/registry/asset. Nuove identitàcampagna/attempt, raw/errori/logs/risorse,
config/input/helperbytes/inventari e manifest atteso congelato prima del workload.
R expectedexit0 tramite harness conserva exit operativi/score reali; niente
expectedexit scelto dopo il risultato. Il rerun32 è motivato da validator,
adapter, risorse eR cambiati che ogni tentativo consuma, e chiude provenance F12.
Non è ripetizione indiscriminata a001 o nuova scelta motori. Finisci con suite
veloce prescritta dopo gate, documentando le esclusioni e collaudo parziale.

Niente Docker/API/campagnea001 globali, ML/OCR/Docling/Marker, rete/fetch/install
da provider, nuovi lock/pin/pesi/macchine o costi esterni. Input sintetici,
risorse già locali, network/document-send0 e nuovi servizi€0. R non è sandbox
universaleFS; payloadpericolosi si fermano prima di tool/estrazione/targetfollow.

## Quota unica, recupero e confini

52hwall187200s dalla entry **2026-10-08T07:06:54.369068Z**, CPU21600s,
disco6GiBnuovi/RAMoperativa8GiB, stop sampled6GiB; stop nuovo workload49h/
19800CPU-s/5GiB, riserva3h/1800CPU-s/1GiB. Deadline **2026-10-10T11:06:54.369068Z**,
stopnuoviworkload **2026-10-10T08:06:54.369068Z**. Nessun reset o nuova quota di
singoli comandi. Aggiorna ledger alla ripresa includendo attesa delle review:
checkpoint008~11h35wall/1465.67CPUnota/465182720Bconservati è un dato storico,
non il residuo attuale. CPUlowerbound/RSSsampled/discoretained/cacheOS restano
limiti, non PASS di cap universali. Prevedi fix/registri/riserva prima dei workload.

Timeout/argv/log/path/launcher propri sono mezzi reversibili entro i limiti:
diagnostica, correzione e riprova nella stessa chat, con fallimenti conservati.
EPERM/EACCES non è rifiuto d’approvazione: usa escalation prevista per azioni
già autorizzate se necessaria mantenendo isolamento; ferma un rifiuto effettivo.
No aumenti quota o allentamenti sicurezza impliciti. Escalazione sostanziale
richiede supervisore, continuando il lavoro indipendente disponibile.

## Output e chiusura della consegna

- Nuovo `reports/implementation-r002.md`: chiusuraF01–F12 con prova/esito/input,
  delta esatto, rettifica storica e nuova baseline separati, qualità perfenomeno,
  metriche manuali unavailable, risorse/provenance/limiti e nessunGOautore.
- Nuovo report permanente in documentation/benchmarks/ con data/giro reali e
  rettifiche ai tre report r001; guide/READMEcorpus/PROTOCOL coerenti sul nuovo
  comportamento. Non aggiornare changelog/STATE/HANDOVER/indici supervisore:
  indica aggiornamenti permanenti necessari nel report, li riceve il supervisore.
- Evidenze sotto `evidence/implementation-r002/`, lavoro sotto
  `work/implementation-r002/`; ledgerappend-only senza reset; niente overwrite
  dei reviewer/r001/packet/receipt/snapshot di confronto.
- Aggiorna `handovers/implementation-r001.md` con sezione corrente r002 e storia
  conservata, o consegna `handovers/implementation-r002.md` e collega il precedente.
  Nessun processo proprio pendente alla consegna, errori ordinari non sono
  una consegna parziale o richiesta di nuovi prompt per ogni sonda.

Snapshot comune r001 e contestofix non sono approvazioni dei nuovi output.
Consegna completa al supervisore per nuovo snapshot e **due nuove review mirate
indipendenti**, poi arbitrato. Non crearle, non attribuire GO futuro, non avviare
Git di scrittura/cleanup/fase2. Il risultato del mandato resta l’intero incremento.
