# Implementazione a002 r011 — ownership delle foglie inline

Ruolo esclusivo implementatore nella chat autrice r009/r010. OpenAI/Codex
basato su GPT-6 secondo identità esposta; deployment, effort e chat ID non
esposti. Nessuna delega, review impersonata o GO autore.
Ingresso reale 2026-10-10T20:40:13Z; feature feature/run-a002-corpus,
HEAD/dev/origin-dev/merge-base80c31da46e5caab990d3596c79c44e524ec1dc95,
indice vuoto. Fix-context-r010 MATCH/exit0 con argv/cwd/output/costo in
evidence/implementation-r011/entry-verify.json. Reader v10 conservato
fisicamente in work/implementation-r011/r010-observations.py, SHA256
382d6be9ad2c7d376bcfdb44ca2c6ba7acaa12132c2545960692ed2a7c8394b5.

## Nota tecnica anteriore alle modifiche del prodotto

Il producer possiede il blocco e rimuove i prefissi costruendo il dominio
inline e la mappa esplicita al sorgente. Soltanto il blocco heading possiede
i suoi attributi finali: una riga logica `# … {…}` dentro una foglia non
installa metadata di heading. La label accettata possiede l'intero valore,
anche quando un code span chiuso attraversa LF. Il consumer ricorsivo e
la guardia devono usare quel medesimo limite, senza un secondo taglio LF.
Una LF libera rimane fuori dal profilo delle label accettate.

Lo stack HTML possiede il contenuto dal proprio tag aperto alla chiusura
riconosciuta. I ledger degli atomi Markdown della label non possono
saltarne aperture: code/math sono inerti solo nel dominio Markdown. Dopo
la chiusura dell'owner gli atomi chiusi tornano inerti. Un delimitatore
non chiuso non possiede il resto: il walk conserva ogni apertura reale;
il fallback segnala risorse sicure non rappresentabili come unsupported.
La guardia valida tutti gli eventi prima della proiezione, mantiene la
prima unsafe in ordine sorgente e continua dopo unsafe/errori d'albero.
F36 resta estensione finita deliberata, senza secondo feed puro.

Scelta locale: vincolare riconoscimento dei metadata al blocco, eliminare
il bound LF aggiuntivo nelle label già accettate, subordinare il ledger
code all'ownership HTML. Le regole di code_end, tokenizer F36 e mapping
r009 restano contratti ricevuti. Le prove sui vecchi byte non si trasferiscono:
nuovi S/B/I, sdist/wheel canonica/reinstallazione e dieci hash; confronto
fisico v10 con stessi raw/oracoli manuali, matrice per classi/CLI/regressioni,
rescore64 e checker indipendente. Il gold math includerà la formula raw,
senza usare reader/scorer per costruire gli attesi o certificare il subset
matematico unsupported. Nessuna nuova V o conversione motore.

## Risorse iniziali

Entry originale2026-10-08T07:06:54.369068Z, disposizione24h09:37:31Z del10,
stop nuovi workload11ottobre06:37:31Z, deadline09:37:31Z/riserva3h.
CPU21600/stop19800s,disco6/stop5GiB,RAM8/stop sampled6GiB,
R≤900s/stream1MiB/file32MiB; acquisizioni/rete/invii/servizi0/€0.
Base ricevuta14448.500381CPU-s lower; entry14489.155945s,
retained3372822528B allocati scoped/dedup, non somma storico stesso scope.
45s Claude r007 approx già storici, non risommati; admin parzialmente non
misurata, RSS storico749760512B sampled, nessun upper certificato.
Gate prima del primo import managed PASS_SCOPED_BYTES: dieci moduli
source=wheel=installato, package/startup/origine managed3.12.13 verificati.
S009 generale storico STALE per quattro registri, nessun PASS trasferito.

Nota registrata prima del codice; lavoro e prove da completare nella stessa
chat. Questa nota non è un nuovo piano, una consegna o un'approvazione.

## Recuperi ordinari e vincolo aggiuntivo sui token

La prima raccolta v10/v9 e after001 conserva gli oracoli iniziali; after001
144/153, con errori propri di check delle visite safe e posizione della
link note (il primo `[` era la definizione). Corrette le asserzioni, stessi
raw/gold/expected, nuovo stage004 e confronto finale153/153 contro27/153v10
e37/153v9. Due closed-code quote espongono un feed HTML ridondante nel node
unknown: la proiezione inline completa era già validata, ora il fallback
raw si applica ai soli blocchi privi di tale proiezione.

Targeted001 R FAIL reale:2098pass/4fail F31r007. Eliminare ogni bound LF
permetteva di prendere una chiusura math nel corpo di token opachi delle
label. Raffinamento locale: soltanto LF dentro intervalli code accettati
dal walk della label permette lookahead oltre LF. LF del token conserva
il limite, il token sorgente resta barriera opaca nella ricorsione. Non
si riattivano commenti/CDATA per chiudere il literal precedente. Nuovo
S/B/I e ripetizione delle prove dipendenti; raccolte/fallimento preservati.

## Consegna completa r011

F46–F49 implementati e verificati nel perimetro finito del mandato37.
Nessun GO autore: prossimo ruolo supervisore per ricezione, nuovo comune,
due NUOVE review indipendenti ChatGPT/Claude e arbitrato reale. Nessuna
promessa di ultimo giro. Registri condivisi, indici/changelog/arbitrati,
review/checkpoint supervisore e storia ricevuta non modificati; Git manuale.
Skill manage-implementation-run, verify-conversion-fidelity e
write-diataxis-docs; nessuna delega o decisione della nuova architettura.

Reader finale observations-v11-inline-ownership. Quattro file ricevuti
modificati: reader, PROTOCOL e due guide. Nuovi: test_conversion_fix_r011.py
e report/dati permanenti2026-10-10-a002-r011-ownership-inline. Scorer/alignment/
F26, normalizzazioni/denominatori, corpus/gold/raw/V13/A03, F40/F41 e gli
altri delta preesistenti sono preservati.

### Criteri verificati

F46: metadata heading solo in modalità blocco. H1/H2/H4–H11 e HC1 sono
INCOMPLETE/UNSAFE_REFERENCE con prima diagnosi; HC2 vero heading conserva
metadata. H7 distingue testo interno `# …` e attributi `.c` del blocco.
H3 sceglie alternativa autorizzata INCOMPLETE/UNSUPPORTED per la risorsa
sicura non rappresentabile, senza apertura p.png o COMPLETE con perdita.

F47: LF posseduta dal code chiuso accettato della label non taglia math.
LF di un token opaco conserva il limite, poi il token sorgente è barriera
nella ricorsione: nessuna chiusura dal corpo comment/CDATA. LF libera non
diventa label accettata. Gli otto originali e16varianti conservano formula
raw, link esterno, testo, tre blocchi e char/UTF8; controlli fuori label,
code non chiuso e F31r007 delimitano la regola. Gold manuale con formula,
subset math unsupported/L04 espliciti: score5/INDETERMINATE, senza falso
UNSAFE; non un PASS semantico della formula. Gold negativi e errori reviewer
storici conservati, non inclusi fra i nuovi attesi positivi.

F48: ledger code della label subordinato a html_stack. Owner aperto possiede
HTML, backtick sono testo e tutte le aperture reali vengono visitate;
prima unsafe globale stabile, visite successive anche dopo errore d'albero.
Owner chiuso restituisce il dominio Markdown e code esterno rimane inerte.
Il node unknown di una quote complessa già guardata/proiettata inline non
rilegge il raw come HTML; fallback raw soltanto senza proiezione inline.
Mappa sorgente r009 e tokenizer/attributi F40 invariati.

F49:153oracoli manuali finiti,34originali accolti (13ClaudeH/HC+21GPTcases003),
39metadata/chiusure/continuazioni,16math-label,48ownership e16ordine/errori/
controlli. Trasformazioni dichiarate, nessun fuzzing illimitato o oracolo
riusato dal reader/scorer. Guard/diagnosi/visite, audit→observe→score,
raw/testo/formule/ref/asset/blocchi/coordinate e CLI effettivi verificati.
F36 B1–B4 restano deliberatamente UNSUPPORTED, divergenza dal tokenizer puro,
nessun feed extra obbligatorio. F43/F44 originali e F40/F41 preservati nei
loro contratti, interazioni ora coperte da questo giro, senza claim universale.

### Prove finali

| Prova | Esito effettivo |
| --- | --- |
| Gate prima del primo import managed | PASS_SCOPED_BYTES origine/startup/package3.12.13 e10source=wheel=installato; s009r010 generale storico STALE non trasferito |
| Confronto finale | before-v10-final27/153; before-v9-final37/153; after-final-002153/153, stessi raw/gold/expected |
| Originali reviewer |34raw identici,21GPTcases003 e13ClaudeH/HC con terminatori originali; binding finale |
| Regresso r010 | matrix-r010-002303/303, inclusi F36B1–B4, sotto R |
| Regresso r009 | matrix-r009-002308/308 sotto R |
| Mirati | R-r011-targeted-002PASS,2102pass; r003–r011/benchmark/comparison/writer; -k notF04 effettivo senza deselection perché r002 non incluso |
| Suite veloce | R-r011-fast-finalPASS/s005,2327pass/310subtest,37.95s workload |
| CLI reale |37fixture/74comandi OS, all_expected=true, attesi/manifest prima dei figli, writer sintetico dichiarato |
| Contratti | H01–H08R PASS; H12±R FAIL previsti, expectedR0/harness1/wrapper2 e contract-exit match=true |
| Rescore64 | R PASS, motori0,0delta numerico/gate nei19campi;4PASS/28FAIL osservati/set32 senza imporli |
| Checker indipendente | derived-check-001PASS: catena/raw/domìni/denominatori/distanze char/token e aritmetica;56EXACT_GLOBAL_DISTANCE/8TRIVIAL_ZERO |
| Binding finale | delivery-check-001PASS_SCOPED_BYTES sotto R PASS/s006;65963protetti/129corpus/109581artefatti ricevuti/327filecomuni immutati |
| Installazione finale | s006 S/B/I,sdist/wheelcanonico/reinstallazione;10moduli installati identici a fasts005,37input Python esterni identici, runtime/test/diagnostica byte uguali nelle copie |
| Documenti/processi | document-check-001PASS38link/anchor,whitespace e gitdiffcheck;28receipt,figli reaped/socket puliti,pending0 |

Suite effettiva: stage-s005/dev/bin/python -I -B -m pytest tests/
--ignore=tests/api --ignore=tests/packaging --ignore=tests/docker -q.
API/packaging/Docker e due smoke radice esclusi; non collaudo completo della
migrazione. F35/F36/F40/F41/F32/F27/F30/F28locale/F26/F16 e quattro F31token
label nei controlli correnti, non PASS trasferiti. S006 segue soli documenti/
dati, Python identico provato: non nuova suite inventata su s006. Nessun
--no-sync prima dei gate, download o installazione per collection fallita.
Stesso managed locale CPython3.12.13 nei figli.

CLI H1/H6unsafe2/2, H3unsupported2/2/score3; F47run0/evaluate5,
F48run2/evaluate2/detail prima apertura. GoldPROVISIONAL, nuovaV0, writer
fixture-cli.py copia byte e mantiene registry: non conversioni native reali.
Raw/output/run/final/stdout/stderr/argv/exitOS conservati. Spy delle aperture
nella matrice, non telemetria aggiunta alla CLI.

64misure conservano conversion_final originale r001, distinto dal
previous_derived_final r010 e dalla provenienza rescore r002r009; nuova
final identificata nei dati permanenti. Condizionato/globalbound/full
separati, TRIVIAL_ZERO non fullPASS. Checker senza import prodotto,
allineamento strutturale non interamente indipendente. Corpus/gold/raw/
V13/A03 chiusi protetti, L04outputUNAVAILABLE distinta; L01/L02NOT_MEASURED,
L05lower/sampled/hardware/cache/costi e gap12helper/H15/E9/ChromeNONPASS.

Quattro osservazioni Claude non bloccanti conservate: run backtick ereditato
diverso da CommonMark; rawhtml5/html-space fuori profilo esatto e attivazione
Pandoc non confermata; tempi ricevuti4000commenti~7.3sv10/~1.4sv9, debito
finito senza soglia violata, nessuna ottimizzazione; fallback unsafe
observations-v1 in benchmark.py fuori delta, distinto dal reader eseguito.
Attribuzione reale mediante byte/gate/R/snapshot, mai quell'ID fallback.

### Identità, errori e limiti

[Manifest finale](../../../../../temp/run-a002-fase1-corpus/evidence/implementation-r011/final-delivery-inputs-001.json)
identifica7file, stage/gate, helper, prove e copie before/after/input-store.
[Report permanente](../../../../benchmarks/2026-10-10-a002-r011-ownership-inline.md)
e [dati](../../../../benchmarks/2026-10-10-a002-r011-ownership-inline-data.json)
contengono153attesi/risultati prima/dopo,64catene e27receipt disponibili prima
del binding; receipt successiva nel manifest/report. Snapshot tecnico
identificato sotto, report/checkpoint/outputlive esclusi, non comune/GO.
Il supervisore includerà i finali nel nuovo common per due NUOVE review.

Errori preliminari preservati: due check miei safevisit/linknote corretti
con nuovi stage/confronti; quoteclosedcode feed ridondante corretto;
Targeted0014FAILF31r007/2098pass corretto, nuovoS/B/I e prove ripetute.
S001precede cambio del checker, nessuna prova prodotto attribuita. Letture
amministrative con path inesistenti senza workload. Primo writer del report
finale KeyError per chiave snapshot working_tree_sha256 inesistente, prima
della scrittura: usata worktree_sha256, senza cambiare prodotto/prove;
outputtool preservato, CPU amministrativa non misurata. Nessun rifiuto
approvazione reale: escalation socket autorizzata con R/Firejail/netnone,
shellFalse,registry/limiti invariati; nessun privilegio persistente.

CPU annidata contata una volta dal wrapper, admin parzialmente non misurata,
45sClaude r007approx non risommati, riserve non azzerate. RSSsampled proprio
659263488B/storico749760512B, non upper certificato, workload propri conclusi.
Budget e verify finali sotto e nel checkpoint. Stop11ottobre06:37:31Z=08:37:31
Europe/Rome,deadline09:37:31Z=11:37:31,riserva3h; CPUstop19800/discostop5GiB/
RAMsampledstop6GiB invariati, rete/acquisizioni/invii/servizi0/€0.

## Verify tecnico e contabilità di consegna

Reader 67260byte, SHA256`af5299bb0c52bd4a0634a1aebb8a28743e2793c315eec5bbaf9b6424a98d8482`. Snapshot
implementation-r011-delivery-001 SHA256`9cfbeac35e1dee02e711339d686d102597652f0e264a4d50feb6f55d45fe4adb`,
impronta`e1766aecedc651554e7bc623b9fff7641839ca342275218328576a81eb764675`,334file/3artefatti.
author-verify-001MATCH/exit0 con argv/cwd/output/costo; verifiche Git finali
branch/HEAD/dev/origin-dev/indicevuoto/diffcheck immutati e PASS. Snapshot
tecnico autore, non common o GO; report/CP/outputlive esclusi.

Budget-delivery-001 2026-10-10T21:06:48.423597+00:00, wall223194.054529s dall'entryoriginale,
CPUvisibilelower15194.393237999999s, propria figli misurata
745.892857s, base14448.500380999998s.
Retained dedup logical3135697675B/allocated
3591094272B, include tmpwork, non somma storico
stesso scope.45sClaude r007approx storici non risommati, admin parzialmente
non misurata; nessun upper CPU/RSS certificato. Stop/riserva invariati e
non raggiunti nei valori osservati; nessun workload proprio pendente.

PROSSIMO RUOLO: supervisore per ricezione completa e nuovo common, due
NUOVE chat indipendenti ChatGPT/Claude, poi arbitrato reale. Nessun GO
autore/Git automatico/cleanup/fase2/motore/nuovaV.

## Controllo conclusivo dopo gli output di ruolo

Role-output-check-001PASS:8link locali e whitespace.
author-verify-after-role-output-001MATCH/exit0; argv/cwd/stdout/stderr/costo
nella receipt. Budget-delivery-002 2026-10-10T21:08:15.727253+00:00,wall223281.358185s,
CPUlower15194.479391999997s/propria745.979011s,
retainedscoped dedup logical3135712056B/allocated
3591122944B. Tutti i limiti/riserve precedenti
invariati,admin parzialmente non misurata,nessun upper inventato.
Consegna completa al supervisore; nessun workload proprio pendente.
