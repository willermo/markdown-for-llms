# Implementazione a002 r003 — fix locali HTML, audit e metriche testuali

Agisci esclusivamente come **implementatore**, continuando nella stessa chat
che ha consegnato r002. Completa F13–F16, regressioni, prove dipendenti e report
r003 nella stessa chat. Non supervisionare o revisionare te stesso e non
modificare i registri comuni. Non serve un nuovo piano.

## Mandato e identità

Run `run-a002-fase1-corpus`, branch `feature/run-a002-corpus`,
HEAD/dev/origin-dev locale/merge-base `80c31da46e5caab990d3596c79c44e524ec1dc95`.
Indice vuoto alla ricezione. Piano r001+errata/GO piano r002 e prompt05/06,
addendum V e prompt09 restano il mandato, con la correzione locale corrente.

Due nuove review reali dell’**implementazione r002**: NO_GO/NO_GO.
Arbitrato vincolante `arbitrations/arbitration-implementation-r002.md`:
F13 contenuto HTML inline, F14 risorse HTML, F15 CER/WER parziali,
F16 attribuzione della componente strutturale WER. I primi tre bloccano;
F16 è un chiarimento non bloccante accolto da completare nella consegna.
F02/F03/F04/F06/F07/F08/F10/F11/F12 e V derivata restano chiusi nel perimetro
review; F01/F05 e il ramo risorse HTML di F09 richiedono questi completamenti.
Non ripetere l’analisi integrale del piano, V umana o le campagne a001.

Percorsi relativi della run sotto `temp/run-a002-fase1-corpus/`. Leggi:

1. AGENTS.md, temp/HANDOVER e sezione iniziale corrente PROJECT-CONTEXT;
   STATE/HANDOVER/brief della run e
   `handovers/supervisor-implementation-arbitration-r002.md`.
2. Skill manage-implementation-run e verify-conversion-fidelity, protocollo
   run-lifecycle, arbitrato implementazione r002 **integrale**, questo prompt.
3. Entrambi i report `reviews/review-implementation-r002-chatgpt.md` e
   `reviews/review-implementation-r002-claude.md`, checkpoint e prove citate.
   Sei autore dei fix: entrambi sono input ammessi, non review concorrenti.
4. Report/CP autore r002, sezione finale “Consegna completa”; report/dati
   permanenti `documentation/benchmarks/2026-10-09-a002-r002-*`, PROTOCOL e
   guide benchmark. Piano/errata/prompt05/06/09 solo per requisiti pertinenti.
5. Reader, audit, scorer e dipendenze raggiunte, test benchmark/confronto/fix,
   manifest/spec/gold/receipt V e prove di baseline003/rescore-historical002.
   Non importare il prodotto per una mera lettura del codice.

Lo snapshot comune revisionato `snapshots/implementation-r002.json` è storico:
SHA256 `da3ecd3c352afe15bc07a0dba3ed27e52831e779fa1ab19981ad7fe92621291a`,
impronta `9dae2da3d81663f51189fab604effc659ca02fc85bc38bfb2c899e72e52418ca`.
Dopo l’arbitrato solo i registri documentali del supervisore sono aggiornati;
la transizione è registrata nelle evidenze del ruolo. Non rigenerare r002.

Il contesto tecnico **corrente d’ingresso** è
`snapshots/implementation-fix-context-r002.json`. SHA256 `19c15ab6851d14d0e50f878e58b8614d3595d099767bcf6f5fda8cf82c3f0e0c`,
impronta `915fb1f8290596f244282a8778923d9789502043fcf34220c7ec55660cfb8c2c`;
307 file di worktree e26012 artefatti espliciti. Receipt nel checkpoint
del supervisore e nelle sue evidenze.
Prima di modificare verifica branch/HEAD/base/indice/delta e contesto:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r002
```

Registra argv/cwd/exit; MATCH tecnico non è GO. I cambiamenti autorizzati
rendono poi storico questo contesto: crea autonomamente target, inventari e
snapshot di esecuzione per le nuove prove, con output vivi esclusi. Non
chiedere un nuovo freeze come consegna intermedia.

## Risultato e prove di chiusura F13–F16

La ricorrenza della stessa classe F01/F05 richiede controllare il contratto
produttore→trasporto→audit/scoring anche per fallback e sintassi equivalenti.
Non limitare il fix agli esatti payload dei reviewer. Non è richiesto un
parser universale: resta un subset finito, con unsupported visibile e stati
conservativi. Non modificare sorgenti/gold per ottenere PASS, né cancellare
HTML/formule/risorse/unknown durante normalizzazione.

### F13 — contenuto HTML inline completo e localizzato

Rilievi GPT-R002-I001 e CLA-R002-I001. Nel ramo inline span/a di
`observations.py:198–206`, html_nodes estrae formule e unknown ma il ponte
propaga solo testo, riferimenti e anchor. Correggi il trasporto tipizzato,
raw/segmenti/offset/coordinate e scope di prosa/matematica/codice, senza
cancellazioni, doppio conteggio o segmenti spurii. Se una struttura supera il
subset, conserva il raw e il contenuto noto, rendendo esplicita la parte
unsupported. Non promuoverla a vuota, COMPLETE o PASS per il solo text vuoto.

Prove discriminanti, con copie proprie dei casi originali:

- K01 cella vuota positiva resta PASS automatico; aggiunta `$x_i$` resta FAIL.
- `<span><math><semantics><mi>x</mi><annotation encoding="application/x-tex">x_i</annotation></semantics></math></span>`
  nella cella: formula supportata osservata/localizzata, aggiunta FAIL.
- `<span><math><mi>x</mi></math></span>`: unknown visibile/localizzato,
  gate non PASS spurio; non inventare equivalenza/AST non supportati.
- Positivo con formula attesa indipendente e rappresentazione HTML annotata
  equivalente: gate PASS nel subset, L04 output resta UNAVAILABLE.
- Stesso contratto in caption e paragrafo, span/a annidati, contenuto misto,
  duplicati e ordine. I campi devono arrivare fino ad audit e scorer; non basta
  assert su html_nodes isolato. Code literal resta code e non diventa risorsa.

Le originali sonde GPT empty/direct-math/html-math/html-unknown e Claude
P0/A1/A2/A3 sono preservate, non modificate per farle passare.

### F14 — ogni risorsa HTML raggiunge audit

Rilievo CLA-R002-I002, residuo F01/F09 distinto dal validatore input F04.
`<img>` in cella/paragrafo e `<figure>` fuori dal dispatch possono perdere
referenze o la marcatura resource_syntax, mentre `<p><img>` è auditato.

Per ogni frammento del profilo HTML e fallback non pienamente osservato,
riconosci/valida/estrai le risorse oppure rendi esplicita la sintassi risorsa
unsupported e il bundle INCOMPLETE. Un URI proibito noto deve restare
UNSAFE_REFERENCE, senza aprire il target, fetch o esecuzioni da dati. Nessun
COMPLETE da unknown silenzioso. Consenti solo il subset passivo dichiarato;
non occorre supportare semanticamente video/picture o tutto HTML/CSS.

Prove end-to-end osservazione→audit→run/score/evaluate:

- B1 img assente in cella, B2 figure con img assente, B3 img assente inline:
  audit INCOMPLETE, run_exit2 ed evaluate_exit2, non COMPLETE/exit0 o solo5.
- B5 img con `../outside.png`: UNSAFE_REFERENCE/INCOMPLETE e exit2;
  sentinella o spy provano che il target evasivo non viene aperto.
- Controllo B4 `<p><img>` assente resta INCOMPLETE; risorsa locale lecita
  presente, immagine/alt/caption attesi e hash indipendenti danno COMPLETE.
- Celle/caption/paragrafi, span/a/figure e contenitori unsupported che
  contengono risorse: un solo contratto di conservazione/guardia, positivi e
  negativi. Case/attributi/annidamenti del subset gestiti o rifiutati
  esplicitamente. Se unsupported, non COMPLETE per omissione di attributi.
- Immagine extra può produrre qualità FAIL pur con bundle COMPLETE quando
  l’asset esiste; bundle e qualità restano dimensioni distinte.

Non serve eseguire Pandoc o un browser sui payload, né dimostrare ora che
Pandoc emetta figure: i bundle sintetici sono ammessi dal profilo raw_html.
Non allentare i controlli input F04 o i confini path/symlink/hardlink/rete.

### F15 — CER/WER completi indisponibili per testo non misurabile

Rilievi GPT-R002-I002 e CLA-R002-I003. Oggi un unknown osservato non allineato
è escluso dalle inserzioni e dalla lista text_unavailable: CER/WER AVAILABLE0
anche se il reader globale è UNAVAILABLE. La qualità INDETERMINATE non basta.

Considera sia blocchi attesi irrisolti/ambigui sia osservazioni testuali non
misurabili nel dominio effettivo della metrica. CER/WER completi sono
UNAVAILABLE/None con motivo, ID osservazione/blocco e scope; denominatori
attesi invariati ed edit **certi** come lower-bound attribuito. Testo noto
conservato non va cancellato, ma non inventare il numero di edit delle parti
non interpretabili. Eventuale conteggio locale deve avere regola conservativa
versionata/provata; separa lower-bound, completezza e giudizio automatico.

Prove positive/negative attraverso il vero reader/scorer:

- Gold `Known words.`: positivo WER0/CER0 AVAILABLE; inserzione nota
  `Additional invented content.` resta WER3/2=1,5, denominator2 e FAIL.
- Stesso gold con paragrafo `\unknown{Additional invented content.}`:
  CER/WER completi UNAVAILABLE, non AVAILABLE0; lower-bound e provenance
  dichiarati, unknown preservato e mai assorbito in un blocco arbitrario.
- K01+C1 `Parole aggiunte inventate. \foo{x}` contro C0 senza comando:
  C0 WER3/30 invariato; C1 completo UNAVAILABLE con testo noto preservato.
- Unknown estraneo e attribuito, inline/list item/caption pertinenti, parte
  supportata+unsupported e scope math/code; unknown solo fuori dal dominio
  testuale non deve invalidare metriche estranee senza dipendenza motivata.
- Perdita nota di blocco/pagina con unknown estraneo resta FAIL/exit3,
  denominatori/missing/ordine coerenti, disponibilità parziale distinta.
  Non riaprire l’attribuzione F08 come wildcard globale.

Versiona regola/metriche e PROTOCOL. Assert su contenuto, denominatore,
provenance, disponibilità, gate ed exit, non solo presenza del JSON.

### F16 — spiegare la composizione strutturale del WER

Rilievo CLA-R002-I004, non bloccante accolto. Nei dati consegnati DOCX11/182
è da due caption come paragrafi unmatched (2+9 parole); RTF19/162 comprende
8 errori strutturali (titolo declassato3+3, caption2) e11 edit nei blocchi
allineati. Non presentare automaticamente questi edit come parole perse.

Pubblica scomposizione verificabile per tentativo nei **nuovi** dati/report:
edit entro blocchi allineati, inserzioni unmatched, omissioni attese e
attribuzione a tipo/ruolo/caption con denominatori e lower-bound. Mostra la
regola vigente e gli eventuali conteggi di uno stesso fenomeno in più gate;
non sommarli come perdite indipendenti o occultarli in una media. Corrobora
le cifre del reviewer sui file effettivi, non copiarle senza controllo.

Non è necessario cambiare alignment o normalizzare WER per abbassarlo. Se
una correzione inevitabile cambia la misura, motiva/versiona la regola e
pubblica prima/dopo senza sovrascrivere r002. Nessun valore o4PASS/28FAIL è
un atteso da imporre al misuratore.

## Prove, input protetti e riuso motivato

Sorgenti, corpus rev5, PDF/asset/render/include, gold, riferimento Markdown,
packet e 13 receipt V derivate restano input di confronto protetti. F13–F16
non ne richiedono la modifica: nessuna nuova V umana, derivazione generale,
rigenerazione corpus/render o acquisizione. L04 sulle formule output rimane
UNAVAILABLE e distinta dalla V dei riferimenti.

Crea test dedicati dai payload preservati dei due reviewer, più la matrice
minima dei rami produttore/trasporto/fallback/consumatore. Preserva fallimenti
prima e risultati dopo nella tua directory. Chiudi il comportamento, non
una lista di stringhe: controlla le altre dipendenze inline toccate. Limita
le modifiche a reader/audit/scorer/test/protocollo/report e mezzi reversibili
necessari al risultato. Segnala un vero scostamento sostanziale.

Prima delle prove prodotto usa managed locale CPython3.12.13, origine e
startup verificati. Dopo il delta ricostruisci sdist/wheel, reinstalla la
wheel canonica, confronta dieci hash e S/B/I secondo la guida uv; nuovi
helper fuori wheel con byte e manifest/hash prima/dopo. Figli nello stesso
interprete, target/inventario/preflight correnti e R PASS prima degli import
produttivi o --no-sync. Vecchio s013 o verify stdlib non sostituiscono i gate.

I sei documenti post-s013 e i tre Chrome esterni diversi sono storia nota.
Nuovi preflight devono riferirsi agli input realmente consumati; non copiare
un gate sull’intero worktree stale né richiedere un restore Chrome. Nessun
nuovo browser/render: non consumare quei binari senza necessità autorizzata.
I primi fallimenti dei reviewer e receipt NONPASS restano immutati.

Esegui test mirati e contratti dipendenti da observe/audit/score/evaluate:
H01–H08, H12 positivo/negativo e altri realmente invalidati dal delta.
H06 deve mantenere UNSAFE/exit2; expected-exit fissato prima, harness che
conserva exit interni previsti, non adattamento dell’atteso dopo il risultato.
Non ripetere guardie F03/H15 o sicurezza input F04 senza una dipendenza
modificata. Completa la suite veloce prescritta dopo i gate (API/packaging/
Docker esclusi, smoke radice esclusi dalla selezione tests/), documentando
che non è collaudo completo. Niente installazioni/download per rimediare
una collection fallita.

**Nessuna nuova conversione dei motori è dovuta al delta locale previsto.**
Le review constatano che i 64 raw di rescore-historical-002 e baseline-003
non contengono HTML e l’unico unknown TeX è già attribuito. Verificalo sui
file reali e attribuisci la consegna r003 con una delle due prove equivalenti:

- nuovo observe/audit/score senza motori per **entrambi i set32**, con nuove
  identità, input/byte/hash e delta per fenomeno prima/dopo; oppure
- prova di invarianza indipendente e riproducibile dei risultati semantici
  delle osservazioni/audit/metriche di tutti i64 dopo il fix, con confronto
  esplicito dei campi di disponibilità/contatori/exit e differenze di versione
  o schema spiegate. F16 deve comunque produrre una decomposizione attuale.

Non basta una scansione “nessun HTML” o una lista di hash vecchi per provare
l’invarianza del nuovo codice. I raw, vecchie osservazioni/audit/score/final,
report, V, receipt, reviewer e snapshot restano immutati. La baseline003
rimane **conversione r002**, con misure r003 derivate se riescorata: non
chiamarla nuova conversione r003. Il rescore storico mantiene engine_runs0
e la lacuna degli helper dei12 output precedenti.

Se il delta effettivo cambia una dipendenza del motore/adapter/validator/
PATH/risorse/R o gli input di conversione, determina la prova invalidata e
ripetila con motivazione e nuove identità entro quota. Non rieseguire32motori
per il solo cambio del reader, né trasferire una receipt vecchia a nuovo input.

## Quota e recupero nella stessa chat

Quota unica **52h/187200s** dalla entry reale
**2026-10-08T07:06:54.369068Z**, CPU21600s, disco6GiB nuovi,
RAMoperativa8GiB e stop RSS campionato6GiB. Stop nuovi workload49h/
19800CPU-s/5GiB, riserva3h/1800CPU-s/1GiB:

- Stop nuovi workload: **2026-10-10T08:06:54.369068Z**.
- Deadline: **2026-10-10T11:06:54.369068Z**.

Nessun reset o nuova tranche di comandi. Ledger r002 finale55151,43s wall,
6195,769s CPU nota lower-bound e918503388B conservati+1890137B repo prudenziali
è storico alla consegna, non saldo corrente. Alla ricezione dell’arbitrato
sono trascorse circa25,12h dalla entry; aggiorna all’istante reale di ripresa,
comprendendo attese delle review, nuove prove, registri e riserva. Attribuisci
CPU/costi dei probe noti senza inventare misure mancanti o hard-cap aggregati.
Costi esterni€0, acquisizioni/rete/invio documenti0; risorse locali già presenti.

Errori ordinari, diagnostici, argv, log, timeout e launcher reversibili si
correggono nella stessa chat, conservando fallimenti e aggiornando target/
prove invalidate. EPERM/EACCES non è rifiuto d’approvazione: diagnosi ed
escalation prevista dagli strumenti per azione autorizzata, con isolamento
inalterato; un effettivo rifiuto richiede stop, senza aggirarlo. Nessun aumento
quota, privilegio persistente o allentamento dei confini impliciti. Se emerge
un vero limite sostanziale, registra saldo/impedimento e continua il lavoro
indipendente possibile; non consegnare una richiesta intermedia di freeze.

## Consegna completa r003

Scrivi solo output propri nuovi e documentazione tecnica prevista:

- `reports/implementation-r003.md`: F13–F16 con prova prima/dopo, chiusura
  F01/F05/F09 e regressioni; delta completo, S/B/I/origini/input consumati,
  test/contratti/suite, rescore o invarianza, WER decomposto, stati e limiti.
- `evidence/implementation-r003/`: comandi reali argv/cwd/exit, output,
  byte/manifest/snapshot di esecuzione, ricevute e ledger aggiornati.
- `handovers/implementation-r003.md`: checkpoint autosufficiente, sezione
  corrente chiara, processi propri pendenti e prossimo ruolo supervisore.
- Nuovo report permanente sotto `documentation/benchmarks/` e dati con
  identità r003, collegamenti ai r002 storici e alle prove effettive; aggiorna
  PROTOCOL/guide pertinenti senza affermare funzionalità applicative future.

Non modificare STATE/HANDOVER comune, eventi, indici, changelog, arbitrati,
checkpoint del supervisore/reviewer o confronti protetti. Non eseguire Git
scrivente, pulizia, API/Docker/ML/OCR/Docling/Marker/remoto, download, nuove
macchine o fase2. A001 e pulizia già ricevute, niente nuova campagna generale.

Consegna soltanto dopo risultato completo e fix ordinari, senza GO autore.
Prossimo ruolo **supervisore per ricezione completa r003, nuovo snapshot
comune e due NUOVE review indipendenti mirate**, poi arbitrato reale.
