# Review implementazione a002 r013 — ChatGPT/OpenAI

**NO_GO.** Gli originali F54 e F55 sono corretti nei domini dimostrati; F57 è
sostenuto. Restano due violazioni essenziali della stessa area: un errore di
parsing URI interrompe il drenaggio e sostituisce la prima unsafe; un opener
tilde con info non valida presta proprietà di codice a HTML reale, con
`COMPLETE` e asset perso. **GPT-R013-I001 e GPT-R013-I002, entrambi media/bloccante**,
sono riprodotti sul comune r013 e sul V12 fisico. Non sono nuove regressioni
attribuite a r013, né difetti dei motori di conversione.

## Ruolo, oggetto e indipendenza

11 ottobre 2026, Europe/Rome. Reviewer esclusivo ChatGPT/OpenAI in nuova chat,
estranea all'autore e al supervisore. Codex basato su GPT-6 secondo le istruzioni
esposte; deployment preciso, effort e ID piattaforma non esposti. Nessuna delega,
impersonificazione o lettura di review/evidenze/checkpoint concorrenti r013 o
arbitrato futuro. Skill applicate:
[manage-implementation-run](../../../../../.agents/skills/manage-implementation-run/SKILL.md)
e [verify-conversion-fidelity](../../../../../.agents/skills/verify-conversion-fidelity/SKILL.md).
Autorità: prompt44/mandato43, arbitrato r012 e addendum risorse corrente.

| Identità | Oggetto |
| --- | --- |
| Comune | `snapshots/implementation-r013.json` |
| SHA256 | `ac928bb37a34e637b352013241652132e03daa86412c32e2705655e92549bccc` |
| Impronta | `062e6fd310234f0e3953f0c48b3c9478f94b5ac2223a8d2f430534ec4d141685` |
| Perimetro | 344 file generali/102949 artefatti/20 PDF generali/28527756 B snapshot |
| Branch | `feature/run-a002-corpus`, indice vuoto all'ingresso |
| HEAD/dev/origin-dev locale/merge-base ricevuti | `80c31da46e5caab990d3596c79c44e524ec1dc95` |
| Reader V13 | 79328 B, `27d1b5809bc2dca51b75e50222aaab1eef86b2c907732c33537fbb5df20c43ad` |
| V12 fisico | 74476 B, `5cabe6778ed53149ad02e1f50aaea6f334fc1c7ff3f9ffd38dd7f87c010abc22` |
| Piano/errata | `3ac23bce82817c3a2781a1dded78b8ac6352d78675aea355cd794950dedd4ab8` / `a09cbb8bcedd0fd9c7cca5dc45f45de82d66776539e573ec1c716011aee0dfd9` |

Verify host stdlib all'ingresso, cwd repository, realmente eseguito:

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-r013
```

**MATCH/exit0**,47.95 CPU-s,48.04s wall; output e costo in
[verify-entry.out](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r013-chatgpt/verify-entry.out)
e `verify-entry.time`. Il verify finale dopo questi output è registrato in
`verify-exit.*`; nota finale sotto. Il comune non è stato rigenerato.

## Letture e prove

Letti contesto AGENTS/HANDOVER/STATE/brief e checkpoint supervisore; skill,
ciclo delle run, ADR0001, addendum/fonte no-deadline. Arbitrato r012,
mandato43, recovery-method-r013, entrambi report/checkpoint r012 e
report/checkpoint autore r013 letti integralmente. Reader V13 letto e analizzato
nei rami scanner/producer/consumer/ricorsioni, eventi/drain, attrs/peel_attrs,
InlineDecision/source-map, HTMLGuard/HTMLTree, observe/finally/node/located e
fence; diff dal V12 fisico conservato. Consultati safety/PassiveHTML/local_uri,
audit/observe_bundle/CLI, checker/generatore/copier/launcher/gate, test r013 e
delta r006/r012, le tre fixture r013, permanente/dati e inventario anteriore.

Letture documentali selettive: PROTOCOL/due guide con sezioni r013 e contratti
precedenti pertinenti, guida uv/toolchain, piano/errata/mandati05–06, arbitrati
F31/F36/F40/F41/F43/F44/F46/F50–F53. Non dichiaro lettura integrale di tutti i
log storici o di ogni documento del piano originario.

[Static-check001](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r013-chatgpt/static-check-001.json)
registra372 letture strutturate e586 descriptor verificati, senza import prodotto:
285 raw/attesi/risultati,199 AST ricevuti/139 ownership del corpo ordinario,
60 raw separati, mapping16, overlay20, mutazioni002, regressioni e receipt finali,
S008/S009/S010 e37 helper,102 CLI e64 catene/19 campi. I risultati autore sono
**ricevuti**, non nuove esecuzioni proprie. Originali GPT26/27, prove/gate/R003
e Claude A1–A7/B1–B9/D1, AST/correzioni/R005 sono recuperati nel record; V11
GPT resta attribuzione storica statica, V11 Claude runtime storico distinto.

Sonde proprie: **18+6 raw**, fissati prima del rispettivo runtime in
[cases001](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r013-chatgpt/cases-001.json) e
[cases002](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r013-chatgpt/cases-002.json).
Attesi manuali senza helper prodotto, massimo24/2048B, nessun generatore casuale.
Visite su una sola observe per versione: snippet/char/UTF8, sequenza e molteplicità;
audit separato senza spia. Dodici AST propri complessivi, pinned Pandoc3.1.3,
`/usr/bin/pandoc -f markdown -t json`, stdin, exit0, hash
`3dd273647f0265cb439f22976d5366a54b071a3783f6fec50838b47fb53d701b`.

Tre previsioni iniziali sui backtick con info invalida erano errate: l'AST
mostra **Code inline contenente il corpo discriminante**, quindi quel corpo è
inerte. Correzione separata in `oracle-corrections-001.json`, senza cambiare
raw/attesi originari o ricavare il gold dal reader. Non sono finding. Il checker
proprio [own-check001](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r013-chatgpt/own-check-001.json)
riscontra **19/24** attesi corretti e cinque fallimenti riconducibili alle due
cause sotto. Sono proprietà mirate di guardia/ownership, non un collaudo completo
del contenuto di ogni costrutto code. La cella sicura conserva link/location,
formula caption `x_1` e heading esterna; coverage privata assente. Reset verificato
per le chiamate raccolte e per una successiva observe sicura.

## GPT-R013-I001 — eccezione URI scavalca prima unsafe e coda

**OpenAI, media/bloccante.** C01/C02 e mandato43 F54: prima diagnosi reader,
prosecuzione delle guardie indipendente da errori e adapter dei validatori.

Input `first-unsafe-invalid-uri`:

```markdown
È <script>x</script> <img src="http://["> <img src="../last.png">
```

Atteso congelato: detail `script`; aperture ai char **2,21,42**, byte **3,22,43**,
una volta in quell'ordine. La URI successiva va comunque rifiutata, conservando
la prima causa e la visita finale. V13 osservato: `ValueError('Invalid IPv6 URL')`,
detail Unsafe assente, visite solo script e prima img; ultima img non visitata.
Audit propagato come `ValueError`, senza risultato `INCOMPLETE/UNSAFE_REFERENCE`.
V12 sugli **stessi byte** dà lo stesso errore e le stesse due visite.
La prima sonda `uri-validator-error`, con URI malformata prima dello script,
conferma il blocco alla prima img e zero visite alle due aperture seguenti.

Percorso: [drain_guards](../../../../../scripts/benchmarks/observations.py#L202)
chiama PassiveHTML per ogni evento ma cattura soltanto `Unsafe`.
[local_uri](../../../../../scripts/benchmarks/safety.py#L92) usa `urllib.parse.urlsplit`,
che per questo input solleva `ValueError` prima del rifiuto URI previsto.
L'errore attraversa drain e [observe](../../../../../scripts/benchmarks/observations.py#L1049)
prima del rilancio di `session['first']`; finally resetta correttamente la sessione.
Nella raccolta V13 gli eventi della coda esistono: manca la loro **validazione**,
non la sola proiezione. La promessa di completamento non è sostenuta dal ledger.

Impatto dimostrato: prima causa persa, coda non guardata e audit senza esito
contrattuale. `observe_bundle` cattura Unsafe/UnicodeError, non questo ValueError;
la CLI generale cattura Exception come errore interno exit4: **traccia statica**,
non nuova CLI OS eseguita su questo input. Nessun fetch/esecuzione/falso COMPLETE
dimostrato in questa causa.

Attribuzione: adapter URI ereditato, non modifica della policy PassiveHTML r013.
Il difetto è però raggiungibile dentro il nuovo protocollo F54 e viola la
prosecuzione richiesta; la preesistenza non lo rende chiuso. Chiusura richiesta:
gestire questa classe di errore dell'adapter entro il protocollo di rifiuto,
senza accettare URI invalide, preservando prima causa, tutte le visite e reset.
Nessuna campagna generale o modifica della whitelist è necessaria per provarlo.

Prova: [probe-results002](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r013-chatgpt/probe-results-002.json),
caso identificato con raw SHA e traceback V13/V12; own-check001 e R-probe003 PASS.

## GPT-R013-I002 — info invalida del tilde nasconde HTML reale

**OpenAI, media/bloccante.** C03/C04/C05/C06 e mandato43 F55/F56:
ownership del corpo dimostrata rispetto a opener/profilo, mai COMPLETE vuoto
per una risorsa non rappresentata. Input `tilde-info-root`:

```markdown
~~~foo bar
<script>x</script>
~~~

# End
```

AST proprio Pandoc3.1.3: Plain con `~~~foo bar`, **RawBlock html del corpo script**,
Para con `~~~`, Header End. Nessun Code/CodeBlock proprietario del corpo.
Atteso congelato: Unsafe `script` e una visita char/byte11. V13 e V12:
nessuna visita, nessun detail/errore, audit **COMPLETE**, corpo in nodo/segmento
code. La variante quote (`tilde-info-quote`) ha lo stesso AST dentro BlockQuote
e lo stesso COMPLETE; script fisico char/byte15, senza visita.

Variante asset `tilde-info-asset`, medesimo opener con `<img src="p.png">`:
AST **RawInline html reale**, nel contenuto Strikeout/Para, nessun owner code.
Il PNG sintetico redistribuibile è presente nel bundle. V13 e V12 producono
**COMPLETE, assets=[], references=[], unknown assente**, zero visita all'img
char/byte11: perdita effettiva senza residuo esplicito. Non basta che il file
PNG esista. I controlli `~~~python` e `~~~{.python}` hanno invece CodeBlock del
corpo nell'AST e restano correttamente inerti in entrambe le versioni.

Percorso: [fence_decision](../../../../../scripts/benchmarks/observations.py#L49)
accetta qualsiasi `(.*)` dopo il marker, verifica separazione e chiusura ma
non l'info dell'opener ordinario. `foo bar` diventa metadata=None/kind=code_block.
Scanner radice salta il corpo; producer radice crea code con language `foo bar`.
Nel container l'atomo viene consumato come code, senza evento HTML/unknown;
coverage del dominio esenta poi il fallback. La decisione condivisa conserva
un'assunzione errata condivisa da tutti i consumer: consolidamento non equivale
a ownership valida.

Attribuzione: ereditato V12 sul medesimo raw, compresi i container. Nessuna
colpa del parser Pandoc, che qui fornisce la controprova sintattica. L'input
fuori dalla forma di opener supportata deve conservare HTML reale/residuo,
non ottenere code ownership e COMPLETE. Questo è il confine finito F55
`opener/marker/profilo`, non una richiesta di supportare ogni sintassi Pandoc.
Non rientra in F36, html5/html-space, raw esatto o backtick-run r010.

Chiusura richiesta: decisione dell'opener coerente con il profilo ordinario,
prove pertinenti della classe info e guardia/residuo per il corpo senza owner;
preservare i controlli CodeBlock/Code inline reali e i contratti raw ricevuti.
Non rendere tutto HTML attivo con un secondo feed indiscriminato.

Prova: probe-results002, cinque AST mirati del followup, V13/V12 e audit;
own-check001, R-probe003 PASS. Nessuna esecuzione HTML o fetch osservato.

## Criteri e chiusure

| ID | Giudizio sul comune r013 |
| --- | --- |
| C01 F54 fonti | **Originali chiusi, invariante parziale.** GPT26/27/ClaudeA1–A7 sostenuti dai risultati285; quattro producer e metadata-first verificati propriamente, eventi esclusi dai valori. I001 scavalca prima causa. A8 reader/audit resta limite separato. |
| C02 visite | **FAIL bloccante I001.** Duplicati fisici distinti e depth18/duplicate attrs passano nelle sonde; tree-error/declaration gestita non tronca la coda. URI parser può interrompere drain. Reset passa; zero accesso URI proprio. |
| C03 F55 fence | **Originali sostenuti, FAIL di classe I002.** Testo→tilde/EOF e vero tilde iniziale confermati propriamente; B4 ricevuto preserva risorsa/residuo. Info invalida presta falso codeowner. |
| C04 ownership | **Parziale I002.**199 AST autore/139 assert ordinari sul corpo verificati nei record;12 AST propri sul corpo, inclusi3 errori delle mie premesse corretti.60 raw e4 list/backtick/10 tilde-after-text separati, niente equivalenza universale. |
| C05 F56 prove | **Metodo migliorato, copertura incompleta.** Censimento reale e16 righe, congelamenti,199+16 AST e due mutazioni002 pertinenti. Generatore ordinario varia contesto/chiusura/host ma info sempre vuota;13 adapter case non includono l'eccezione urlsplit. I001/I002 mostrano due buchi concreti, non soltanto un numero di test insufficiente. |
| C06 regressioni | **350/153/303/308 ricevute e sostenute nei propri attesi, non universali.** Overlay20 e F29 giustificati dal requisito EOF/AST, originali preservati. F50/F51raw/F52label e F31/F36/F40/F41/F44/F46/F48 senza nuovo controesempio in quei rami; nessuna riapertura generale. I002 resta difetto essenziale ereditato raggiungibile. |
| C07 F57 coverage | **Sostenuto.** public pulisce solo dopo node/located e drain; own cell/link/formula/location/heading preservati, nessun privato nei JSON propri.640 rimozioni/56 obs e altri delta0 ricevuti; unknown non esenta una guardia mai effettuata: ownership falsa di I002 distinta dalla pulizia. |
| C08 attribuzione | **Sostenuta nei limiti.** S008 runtime/S009CLI/S010documentale distinti,10 origini/37 helper/byte/receipt verificati; own gate corrente separato. Baseline132/133 chiarita sotto. CLI102, writer/gold/H12/64metriche/19campi attribuiti, non rieseguiti. |
| C09 documenti/limiti | **Limiti conservati, promessa globale ancora parziale.** Ordine reader/audit, raw e coverage descritti correttamente per i casi chiusi; I001/I002 smentiscono prosecuzione completa e solo vero codeowner. Nessuna deadline/reset/sanatoria o GO autore. |

## Baseline132/133, regressioni e prove ricevute

Discrepanza **chiarita senza riscritture**: `check-baseline-004.json` ha133/285;
`check-baseline-005.json` ha132/285 ed è identico al `before_checks` permanente.
Entrambi si riferiscono ai raw di baseline004, identici tra cases004/005.
Cambiano tre attesi dopo AST (`freeze-005.json`): context-list-code e due
context-note-heading. Per list-code il vecchio risultato V12 passava e il nuovo
fallisce detail/visite: **−1**. Le due note fallivano prima AST_literal_owner e
falliscono poi detail/visite: saldo0. La sintesi permanente132 è quindi coerente
con gli **attesi finali**, la receipt133 con i precedenti. Nessun nuovo runtime
inventato, dato originale riscritto o falsa conferma di check004. Dopo finale285.

20 attesi EOF r012 aggiornati in overlay, non raw/gold originali. Ordinario EOF
non possiede code secondo il requisito/AST; F29 aggiornato conserva il commento
opaco del tokenizer Python, non trasferisce la grammatica generale Pandoc dei
commenti. Non affermo «tutti gli attesi storici invariati». Mapping16/16 LF/CRLF
ricevuto sui byte finali; campagne antecedenti/pre-mapping/errori restano storia.

Fast003 **2997+310** ricevuto, API/packaging/Docker/due smoke esclusi.
CLI43+8/102 exit ricevuti, writer sintetico/goldPROVISIONAL/zeroengine/V.
25 receipt finali **23PASS+2H12FAIL deliberati**, binding003PASS separato;
60 receipt storiche non sono60 finaliPASS. H01–H08PASS; H12harness1/wrapper2/
expectedR0 non sanati.64 rescore engine0:19 campi/gate delta0,4PASS/28FAIL per
set32,56EXACT_GLOBAL_DISTANCE/8TRIVIAL_ZERO. Catene conversion_finalr001,
previousderivedr012 e finalr013 distinte/verificate. Checker distanza autore
letto/PASS ricevuto; propria verifica hash/catene/confronto19 campi, **nessuna
nuova distanza, suite o alignment indipendente eseguiti**. TRIVIAL_ZERO non fullPASS.

## Gate proprio, recuperi e accounting

Canonica S010 esistente usata senza build/reinstall. Prima degli import:
origine managed3.12.13/binary hash/startup pth e assenza customize,
input immutabili, dieci moduli source/wheel/install e37 helper confrontati;
nuovi S/B/I **validano gli stessi archivi e installazione**, con snapshot
tecnico corrente. S010 generale resta STALE per amministrazione supervisore,
non sanato né attribuito PASS. `gate-binding.json`, `current-*.json`, input-store
e manifest prima/dopo legano i Python/helper/fixture effettivamente usati.

R-probe002 e003 **PASS**, preflight corrente prima import, Firejail netnone/
namespace distinta, binding prima/dopoMATCH, inputs_unchanged/socketcleanup true,
workload exit0 e figli raccolti. Exit0 raccoglie anche i cinque fallimenti:
non24PASS. Argv/cwd/ambiente/stdout/stderr/clock/costi nei record `probe-00*.json`
e cartelle R. Nessuna suite/build/reinstall/rescore/CLI generale/conversione/
motore/modello/download/nuovaV propri.

Errori propri conservati: prepare-gate iniziale non trovava admin sotto `-I`,
corretto prima degli import prodotto; tentativo successivo interrotto exit130
quando si è riconosciuto il costo del comune molto ampio, sostituito da snapshot
tecnico piccolo; primo snapshot tecnico finale richiedeva un artifact fuori
temp/run e fallisce, recuperato con descriptor/copia nel target e artifact ammessi.
R-probe001 IMPEDITA per EPERM socket, poi escalation strumenti approvata mantenendo
R/Firejail/netnone, probe002 reale. Nessun rifiuto auto-review aggirato.
Static-check001 conclude il controllo ma collide col nome della receipt admin:
risultato/stdout/errore preservati, non attribuito exit0 al wrapper. Non sono
difetti del prodotto. Tre premesse AST proprie errate conservate e corrette sopra.

Scritture fuori evidence propria: soltanto questi output review/checkpoint e
tre snapshot tecnici **review-r013-chatgpt-inputs-001, probe-002, probe-003**;
nessuna approvazione o nuovo comune. Nessun prodotto, stato condiviso, arbitrato,
prompt o output altrui modificato. Nessun Git mutativo/cleanup/fase2/ML/motore/V.

Base supervisore corrente **19168.993788 CPU-s lower**, allocated4576026624B;
aggiunti soli costi propri, una volta, senza CPU R annidata/basi/45sapprox già
assorbiti. [Accounting finale](../../../../../temp/run-a002-fase1-corpus/evidence/review-implementation-r013-chatgpt/accounting-final.json)
riporta lower e scope5 deduplicato. Tentativo amministrativo interrotto con file
GNU time vuoto: **costo non misurato**, riserva prudenziale di pianificazione120s,
non spacciata per misura/upper. Letture/amministrazione e collisione checker in
parte non misurate; altro reviewer r013 non letto, costi concorrenti non ricevuti,
riserva di pianificazione200s senza aggiungere una quota. Budget proprio pianificato
≤200s misurati; nessun upper CPU/RSS certificato.

CPU21600/stop19800, disco6/stop5GiB, RAM8/stop sampled6GiB, riserva1800CPU-s/1GiB,
R≤900s/stream1MiB/file32MiB invariati. Entry08ott07:06:54.369068Z e contatori
invariati, **nessuna deadline**; wall solo accounting. Rete/acquisizioni/invii
benchmark0/€0; soli socket diagnostici prescritti daR. Nessun workload proprio
pendente alla consegna.

Limiti ricevuti preservati: A8, F36B1–B4, quattro osservazioni r010;
L01/L02NOT_MEASURED,L04outputUNAVAILABLE distinta da V13/A03 del riferimento,
L05lower/sampled, ChromeNONPASS/helper12gap/H15/E9/FAIL/STALE ed errori intermedi.
Il prossimo ruolo è **supervisore**, dopo entrambi i report reali indipendenti,
per arbitrato dei due finding e relativo passaggio. Questo NO_GO è una review,
non l'arbitrato finale. Git resta manuale dopo eventuale GO finale.

## Chiusura della review

Verify finale realmente eseguito dopo la scrittura del report/checkpoint, stesso
argv e cwd dell'ingresso: **MATCH, exit0**,30.34 CPU-s,30.37s wall, stderr vuoto;
record `verify-exit.out/.err/.time`. Le successive annotazioni di chiusura sono
nei soli output propri, esclusi dal comune.

Accounting finale: **110.188566 CPU-s propri misurati lower**, cumulativo visibile
**19279.182354 CPU-s lower**; scope5 deduplicato **4586946560 B allocated**, sotto
stop5GiB. Riserva prudenziale120s per l'interruzione non misurata e200s per l'altro
reviewer restano ipotesi di pianificazione, non consumi certificati. L'ultimo
controllo amministrativo degli output è successivo al campionamento: la sua
receipt separata non è inclusa in questo lower. RSS resta sampled, nessun upper
certificato; nessun limite non temporale osservato e nessun workload proprio
pendente. `git diff --check` exit0; controllo locale finale di collegamenti,
spazi finali e receipt R in `output-check.json`/`local-output-check.json`.
