# Implementazione a002 r009 — consolidare i confini del reader

Agisci esclusivamente come **implementatore in una NUOVA chat**. L’utente ha
chiesto un contesto fresco dopo più compattazioni: questo mandato sostituisce
l’indicazione precedente di proseguire nella chat autrice originaria. È la
stessa run `run-a002-fase1-corpus`, con byte/storia/risorse conservati. Completa
F39–F42, regressioni e fix ordinari nella stessa chat; nessuna consegna parziale
per test rossi, argv, permessi o freeze delle prove. Non impersonare reviewer.

Configurazione consigliata per questo recupero: **GPT-6.1 Sol / xhigh**;
se l’utente mantiene Sol/high il mandato resta valido. Astra/high è soltanto
un’alternativa proposta, non selezionata o richiesta dal mandato. Registra
provider/modello/effort effettivi se esposti senza inventarli. Nessuna delega.

## Ingresso autorevole, selettivo

Branch `feature/run-a002-corpus`, HEAD/dev/origin-dev locale/merge-base
`80c31da46e5caab990d3596c79c44e524ec1dc95`, indice vuoto alla ricezione.
Git commit/merge/push manuali dopo GO reale. Nessun reset/checkout di input per
nascondere differenze, nessuna query remota implicita. Leggi nell’ordine:

1. AGENTS, temp/HANDOVER, parte iniziale corrente PROJECT-CONTEXT,
   STATE/HANDOVER/brief della run e questo prompt. Skill
   manage-implementation-run e verify-conversion-fidelity, run-lifecycle.
2. `arbitrations/arbitration-implementation-r008.md` e
   `notes/recovery-method-and-models-r009.md` integrali: tre cause distinte,
   NO_GO reale, consolidamento locale. Addendum-resources-r008-r001 effettivo.
3. Entrambi i report/checkpoint reviewer **r008** integrali. ChatGPT:
   `evidence/review-implementation-r008-chatgpt/finding-I001.md`,
   cases-002/probe-summary-002 e R-probes-003. Claude: out-main-001,
   out-r007-html-001, out-extra-r007/008-001, sonde/attesi/receipt pertinenti.
   Prima leggi report e casi selezionati, non tutti i log o gli ambienti.
4. Report/CP implementazione r008 integrali, sezioni finali autorevoli,
   PROTOCOL e due guide attuali, report/dati permanenti r008. Prompt27 e
   arbitrato r007 solo per le invarianti F35–F38 da preservare. Per requisiti
   originari: piano/errata, mandati05/06 nei contratti pertinenti; non tutta
   la storia di a001 o dei sette fix precedenti.
5. Reader integrale, audit/observe_bundle/metrics/alignment/safety/PassiveHTML,
   CLI/input validator pertinenti; test r008 e regressioni precedenti dipendenti.
   ADR0001/workflow benchmark, guida uv e reference toolchain. Nessun import
   o workload prima dei gate effettivi.

Comune revisionato `snapshots/implementation-r008.json`, SHA256
`2f6d690a8f3db7af8f192f78a9e6f5c0e3704b9757256f9dfdf72e35ee3b4152`,
impronta `448b854a88dc3dc883f7337f6fe037fd27846513e3a19104fc88529841e07eda`:
325 file/76333 artefatti, MATCH in entrambe le review e alla ricezione.
Diventa storico dopo quattro registri generali del supervisore. Non rigenerarlo.

**Ingresso corrente** `snapshots/implementation-fix-context-r008.json`;
identità esatta/conteggi in
`evidence/supervisor-implementation-arbitration-r008/snapshot-receipt.json`
e `handovers/supervisor-implementation-arbitration-r008.md`. Nuovo contesto
tecnico dopo soli registri, non GO. Verifica all’ingresso con argv/cwd/output/exit:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r008
```

Leggi branch/HEAD/delta anche non committato, hash di report e sorgenti
consumati. Reader ricevuto v8 SHA256
`817852a8f8b2645a3bbc6212d982aca4546ab1c4b3f128374137313124aa6061`.
Conserva byte di confronto prima di cambiarli. Nuovi snapshot di esecuzione
r009 autonomi, label nuove, output vivi esclusi; non sono approvazioni e non
richiedono un freeze al supervisore. Vecchi snapshot e receipt immutati.

## Risultato: correggere le cause e consolidare le invarianti

L’obiettivo non è aggiungere una condizione per ciascun ultimo payload.
Dentro lo stesso reader finito, rendi coerenti questi tre rapporti:

- Ogni apertura HTML reale fuori dai corpi inerti viene validata anche se
  la costruzione/proiezione dell’albero incontra un errore di struttura.
- Confine del tag/quote/valore attributo e vero prefisso incompleto hanno
  una semantica coerente con il tokenizer CPython fissato; scanner,
  validazione e proiezione non inventano aperture o corpi attivi diversi.
- Il consumer usa il dominio e gli intervalli decisi dal producer sul
  sorgente, anche dopo stripping/merge dei prefissi; la mappa verso char e
  byte originali resta esplicita. Non dedurre di nuovo quei confini dal
  testo trasformato con regole divergenti.

**Prima delle modifiche**, scrivi nel tuo report una breve nota tecnica:
quali punti decidono oggi i confini, quali duplicazioni causano i tre difetti,
quali invarianti assicura la soluzione e quali input/test sono invalidati.
Poi implementa e verifica nella stessa chat. La nota non è un nuovo piano
né un passaggio di approvazione. Sono autorizzati refactoring/moduli interni
reversibili necessari al contratto; scegli i mezzi, non introdurre una
libreria/download o parser universale per compensare il problema.

### F39 — guardia completa, indipendente dalla proiezione

CLA-R008-I001, D1/D2/D3/D6. Oggi HTMLTree si arresta su duplicati/depth>16
prima di validare script/onerror/path evasivi successivi; audit resta
INCOMPLETE ma declassa UNSAFE_REFERENCE a UNSUPPORTED_RESOURCE_SYNTAX.

Tutte le aperture reali riconosciute dalla semantica lessicale scelta devono
passare PassiveHTML, anche dopo errori della proiezione. Scegli validazione
separata o raccolta degli errori senza interrompere la validazione, mantenendo
limiti di profondità/nodi e token opachi. Non ripristinare semplicemente un
`html(raw)` che interpreta corpi di commenti/CDATA/prefissi F36 come attributi.
Definisci e prova precedenza delle diagnosi: unsafe reale non diventa unknown
per un precedente errore strutturale. Zero fetch/esecuzioni da dati.

Prima/dopo sui D1/D2/D3/D6, controlli D4/D5, varianti ordine/profondità/attributi
ed HTML raw/inline/contenitori pertinenti. Dopo: UNSAFE_REFERENCE/detail,
nessuna apertura di path evasivi; positivi realmente sicuri e corpi inerti
con URI/script letterali restano distinti. Nessuna pretesa di evasione riuscita
nel prima: i reviewer hanno osservato classificazioni e omissioni, non fetch.

### F40 — confine HTML e valori attributi coerenti

CLA-R008-I002, D7 e variante sicura: `<p title=it's><img …>` e testo `Don't`
prima di un vero token. Il tokenizer3.12.13 chiude p al primo `>`;
incomplete_tag_token oggi attribuisce quote a un apice interno non quotato
e assorbe tag/testo nel prefisso inventato.

Usa la semantica reale dei valori quotati/non quotati e della fine del tag;
confronta con tokenizer fissato su input piccoli ammessi. Non creare un secondo
lexer parzialmente diverso tra tag_end/incomplete_tag_token/HTMLTree/safety.
Prefisso incompleto solo se realmente incompleto, senza inventare attributi
attivi da un vero token inerte. Documenta quale decisione è condivisa e come.

D7 dopo UNSAFE_REFERENCE; con p.png conserva ref immagine, testo e asset
verificato (il vero token può comunque rendere il bundle INCOMPLETE).
Copri apici/virgolette in valori quoted/unquoted, `>` e `<` letterali ammessi,
comment/CDATA/PI/declaration dopo tag e dentro attributi, tag selfclosing e
EOF pertinenti. F36 `<alpha` prima del token resta localeunsupported;
negativi con URI unsafe reali e positivi quotati inerti discriminano.

### F41 — domini e mapping nei producer con prefissi

GPT-R008-I001, residuo F37. Caso minimo:

````markdown
> See [Known `first
> - <!--` words.](https://example.invalid)

# Visible heading

After words.
````

Scanner accetta code entro quote fisica; read_parts strippa `> ` e consumer
senza quel dominio scambia `- ` nel code per nuova lista. Link perso, testo
sintattico inventato, falso FAIL pur con COMPLETE.

Non fare una patch per il marker `- `: porta al consumer il dominio/intervallo
valido del producer e la mappa del sorgente, o altra soluzione coerente
motivata. Su quote/list/note/Unicode conserva contenuto senza parentesi/
destinazione spurie, ref corretto, heading e paragrafo esterni, raw/offset
char+UTF8. Su gold PROVISIONAL automatic PASS/evaluate5/qualitàINDETERMINATE;
non firma V. Copri link/image label, uno/due backtick, marker letterali
ammessi e i trasporti pertinenti. Controlli senza marker/originale conformi.
Negativi con chiusura in un producer successivo non devono prendere quella
chiusura o assorbire blocchi estranei. Veri token fuori code restano opachi.

### F42 — prova delle classi, documenti e consegna completa

Costruisci una **matrice finita delle interazioni**, con tabella che associa
invariante, dimensioni ammesse, positivo/negativo, risultato atteso e prova.
I tre casi exact sono necessari ma non sufficienti. Usa trasformazioni
ammesse che conservano contenuto/reference: trasporto del frammento in
quote/list/note, prefisso rimosso con mapping, Unicode, marker dentro
letterali; validazione prima/dopo errore strutturale; valori quoted/unquoted;
token chiuso/EOF e blocchi esterni. Copri le interazioni fra queste classi
nei producer supportati senza prodotto cartesiano arbitrario o test tutti
copiati dal reader. Oracoli sintetici piccoli indipendenti e attesi fissati
prima; il codice del prodotto non calcola i suoi attesi.

Proprietà da osservare: progressione terminante; token opaco intero e nessun
asset/prosa dal corpo; ogni tag attivo validato; nessun link/asset/testo perso
su input valido per effetto del trasporto; mapping char/UTF8 e blocchi esterni
stabili; unknown/FAIL/prevalenze/exit e denominatori corretti. Asserisci
queste proprietà nel percorso audit→observe→score, non solo la funzione
interna o assenza di eccezioni. Test di completezza da soli non chiudono F41.
Se aggiungi generazione deterministica, limiti/seed/dimensioni vanno registrati;
nessun fuzzing illimitato. Minimizza i controesempi e correggili nella chat.

Rettifica PROTOCOL/due guide, nuovo report/dati permanenti r009 datati
all’esecuzione. Mantieni storia r008; nuova readerID e input finali.
Spiega in breve quali interpretazioni duplicate sono state eliminate o
vincolate e quali classi le prove coprono; non dichiarare verifica universale.

## Prove finali dovute ai byte nuovi

- Gate origine managed3.12.13/startup/package prima del primo import; dopo
  codice/input nuovi S/B/I, build sdist/wheel canonica/reinstallazione,
  dieci hash e installazione nello stesso interprete dei figli prima di
  --no-sync. Helpers fuori wheel con manifest/hash/copie prima-dopo.
  S007 integrale storico STALE per soli documenti non diventa PASS: nuovi
  input/gate reali autonomi, niente restauro pyc o receipt trasferite.
- Riproduci i tre difetti sul v8 conservato e prova la soluzione finale.
  Regressioni dei domini r003–r008 toccati; F35/F36/F32/F27/F30/F28locale/F26/F16
  preservati. Non riaprire indiscriminatamente campagne a001/API/Docker/ML.
- Suite veloce con preflight/R attuali: `"$RUN_DEV_PY" -I -B -m pytest tests/
  --ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q`;
  smoke radice esclusi. Suite selezionata, non certificazione universale.
- CLI reali run/evaluate su discriminanti, exit OS/errori/run/final conservati,
  writer sintetico dichiarato/goldPROVISIONAL, expected prima. Contratti
  dipendenti dai cambiamenti; H12 FAIL voluti non sanati. Errori di wrapper/
  fixture propri correggibili nella chat, versioni/failures conservati.
- Rescore o prova di invarianza64raw sui byte finali, checker indipendente
  di aritmetica/catena/domini di distanza, senza motori. Non forzare4PASS/28FAIL:
  registra l’esito reale. Counts condizionati/boundglobal/full distinti;
  TRIVIAL_ZERO non fullPASS. Non spacciare checker distanza per allineamento
  completamente indipendente. Corpus/gold/raw/V13/A03 restano protetti;
  nessuna nuova V generale, L04formula output UNAVAILABLE distinta,
  L01PDF/OCR/L02compile NOT_MEASURED e L05/RSSsampled/costi lower preservati.

Il mandato copre il risultato completo e i fix ordinari: timeout/log/argv/
launcher/percorsi adattabili nei limiti, niente nuovo prompt per ogni sonda.
EPERM/EACCES: diagnosi ed escalation tool della stessa azione autorizzata
se necessaria, mantenendo R/firejail/no-network/registry/shell=False/confini;
fermarsi a un rifiuto effettivo, senza aggirarlo. Nessun privilegio persistente.

## Risorse già disposte: nuove chat senza reset

Addendum-resources-r008-r001:24h dalla conferma osservata2026-10-10T09:37:31Z,
entry originale2026-10-08T07:06:54.369068Z. **Stop nuovi workload
2026-10-11T06:37:31Z =08:37:31 Europe/Rome; deadline09:37:31Z =11:37:31**,
riserva3h. Wall cumulativo268236,630932s/stop257436,630932s.
CPU21600s/stop19800,disco6GiB/stop5GiB,RAM8GiB/stop sampled6GiB;
riserva1800CPU-s/1GiB;R≤900s/1MiB per stream/32MiB file. Rete/acquisizioni/
invii corpus/servizi benchmark0/€0, nessun nuovo modello motore/download.
Stop al primo limite reale, comprende preparazione/attese/prove/registri/
consegna e future review; programma lavoro e riserva prima di iniziare.

Ledger autore r008 storico51,500h/11610,304610CPU-s lower; reviewerGPT r008
70,185594CPU-s lower,Claude r008104,81CPU-s lower,Claude r00745s approssimati
separati. Accounting visibile11830,300204CPU-s lower+approssimazione,
non upper totale; amministrazione non integralmente misurata. Retained
storico2384268251B + output nuovi reviewer e supervisore: receipt di ricezione
aggiorna wall/costi visibili, fai misura attuale circoscritta senza doppio
conteggio. RSS storico740855808B sampled non picco certificato. Vecchie quote
52h/stop/FAIL conservati, nessuna sanatoria. Leggi clock reale all’ingresso.

## Output e prossimo ruolo

Output tuoi sotto la run:

- `reports/implementation-r009.md`, con nota tecnica iniziale e consegna completa.
- `evidence/implementation-r009/`, input/prove/fallimenti/ledger reali.
- `handovers/implementation-r009.md`, input/output/costi/processi pendenti.
- Nuovi snapshot esecuzione con label implementation-r009-*; vecchi immutati.
- Nuovi report/dati permanenti in documentation/benchmarks, guide/protocollo.

Non modificare STATE/HANDOVER/eventi/artifacts/changelog/indici/arbitrati/
checkpoint supervisore o review. Completa l’intero risultato; limiti
sostanziali reali tornano al supervisore, continuando il lavoro indipendente
consentito. Non dichiarare GO autore. Alla consegna completa prossimo ruolo
**supervisore per nuovo comune e due NUOVE chat review indipendenti**,
poi arbitrato. Nessuna fase2/Git automatico/cleanup/nuova V o motore scelto.
