# a002 r002 — rettifica del benchmark e baseline finale

Registrato il 9 ottobre 2026, Europe/Rome; prove concluse l'8 ottobre UTC.
Run `run-a002-fase1-corpus`, implementazione r002, feature `feature/run-a002-corpus`,
base/HEAD `80c31da46e5caab990d3596c79c44e524ec1dc95`. Risultati realizzati,
non integrati: nessun commit/merge/push o GO autore. Nuove review indipendenti
ed arbitrato spettano al supervisore.

I fix F01–F12 correggono la misura e la custodia del benchmark: contenuti
inline in celle/caption, alignment formula AST finito, WER delle inserzioni,
header semantico indipendente dal motore, unknown attribuito localmente,
target/titoli/raw inline, sicurezza passiva, PATH Pandoc effettivo, custody
figli fino a chiusura writer e byte helper consumati. I percorsi e il writer
Pandoc rimangono quelli approvati; nessun output motore è diventato gold.

Questa rettifica si riferisce ai report storici immutati:
[native r001](2026-10-08-a002-native.md),
[confronto umano r001](2026-10-08-a002-human-comparison.md),
[riferimento verificato r001](2026-10-08-a002-verified-reference.md).
Le loro osservazioni/metriche rimangono attribuite agli input realmente usati;
i nuovi conteggi seguenti sostituiscono l'interpretazione della fedeltà,
senza riscrivere quelle prove.

## Riferimento e conferma umana

Gold/spec/manifest rev5 aggiungono header_rows/policy. DOCX/RTF/TeX sono
headerless, EPUB th è semantico. Per PDF/raster policy manuale esplicita,
senza pretendere proprietà encoded. Una riga pipe tecnicamente vuota è
placeholder solo per gold headerless; corpo e header sono controllati separati.
Undici caption tabella già annotate sono ora nei nove transcript, una volta
nell'ordine. Un checker indipendente dimostra che sono le sole inserzioni nei
byte rev4. Canonici, PDF, render, asset e include sono invariati: nessuna
rigenerazione per metadati o transcript.

Nuovo M: 415 controlli corpus e93 controlli PASS, inclusi15 controlli header
sul sorgente/policy. Confronto completo e checker indipendente di entrambe le
viste, vecchie e nuove, PASS. Tredici receipt schema3 derivate conservano
risposta V reale, vecchi packet/pair/gold, delta, nuovo M e prova di invarianza
su testo/numeri/formule/ordine/celle/caption/asset/note/riferimenti.
**Zero nuove conferme umane**: header tecnico qualificato da M; V ereditata
soltanto sul contenuto invariato. Non è stato copiato VERIFIED su nuovi hash.

Risposta originale SHA256 `2b34e76549d084988bb09ba56d121e66b0b20d81443e2922f01ef1a3cda87f9e`;
packet approvato `ac0921defac79ca063ef89fe270ed09e8d991e7dc17d9a0aeae3c2c4c06a5c6f`.
I260 minuti concordati non sono durata umana misurata. EPUB renderer storico
NONPASS/Chrome resta tale; capitoli ereditati per hash/invarianza, senza nuovo render.

## Rescore storico: zero motori

Nuove observe/audit/score dei32 raw r001 selezionati (12campaign002,
20campaign003), su copie byte identiche e nuove identità;606 file protetti
verificati invariati prima/dopo. Vecchi raw/observazioni/metriche/final/receipt
rimangono immutati. Esiti effettivi4PASS/28FAIL, senza imporre quel conteggio.

| Fenomeno/percorso | Prima r001 | Rettifica r002 |
| --- | --- | --- |
| Celle corpo DOCX |0/9|9/9|
| Celle corpo RTF |0/6|6/6|
| Celle corpo TeX |0/9|9/9|
| Numeri DOCX |8/13|12/13|
| Numeri RTF |3/8|5/8|
| Numeri TeX |4/9|8/9|
| WER DOCX |0|11/182|
| WER RTF |14/162|19/162|
| WER TeX |3/152 dichiarato completo|UNAVAILABLE; errori certi lowerbound1/152|

DOCX/RTF righe iniziali erano corpo: il confronto precedente le trattava
indebitamente come header. Numeri delle celle sono recuperati. Inserzioni
unmatched entrano nel WER con denominatore atteso invariato. RawLaTeX inline
TeX è unsupported localizzato, non code: CER/WER completi non vengono affermati
quando l'attribuzione è incerta. Unknown estranei non sanano missing/page loss.
TXT/EPUB conservano i conteggi. I12 raw storici TXT/EPUB con helper consumati
non recuperabili restano storici e limitati: nessun diff o sanatoria inventati.

## Baseline finale: nuovi32 tentativi

Experiment `a002-r002-baseline-003`, otto percorsi×4, stesso sourcehash,
registry, writer e opzioni r001. Nuove identità/manifest/attesi congelati prima
del workload, byte di helper/config/test conservati. Stage s013 S/B/I e
wheel canonica reinstallata, dieci hash, managed CPython3.12.13, preflight e R
PASS prima dei workload. Figli stesso interprete/installazione. PATH `/usr/bin`,
Pandoc3.1.3: which/realpath/hash/versione prima/dopo; proof parent/child legacy
coerenti. Nessun ripristino implicito del PATH ambientale.

Tutti32 tentativi SUCCESS/COMPLETE, harness0 e R PASS per ogni gruppo;
qualità4PASS/28FAIL. A10 soddisfatto su DOCX/EPUB diretto e legacy con quattro
figure attese referenziate e contenuto verificato. Le perdite Pandoc rimangono
pubblicate; bundle completo e qualità sono dimensioni separate.

| Percorso | Qualità ×4 | Celle | Numeri | Formule AST | Figure+caption | Riferimenti | WER |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct-txt | PASS:4 | 0/0 | 9/9 | 0/0 | 0/0 | 0/0 | 0.000000 |
| direct-docx | FAIL:4 | 9/9 | 12/13 | 2/2 | 0/1 | 2/3 | 0.060440 |
| direct-epub | FAIL:4 | 9/9 | 11/13 | 2/2 | 1/1 | 1/5 | 0.043011 |
| direct-rtf | FAIL:4 | 6/6 | 5/8 | 0/0 | 0/0 | 1/2 | 0.117284 |
| direct-tex | FAIL:4 | 9/9 | 8/9 | 5/5 | 1/1 | 0/3 | UNAVAILABLE |
| legacy-docx | FAIL:4 | 9/9 | 12/13 | 2/2 | 0/1 | 2/3 | 0.060440 |
| legacy-epub | FAIL:4 | 9/9 | 11/13 | 2/2 | 1/1 | 1/5 | 0.043011 |
| legacy-rtf | FAIL:4 | 6/6 | 5/8 | 0/0 | 0/0 | 1/2 | 0.117284 |

| Percorso | Conversione primo s | Mediana3 s (min–max) | Audit mediana s | Score mediana s | Evaluate mediana s | RSS motore sampled MiB |
| --- | --- | --- | --- | --- | --- | --- |
| direct-txt | 0.0947 | 0.1156 (0.0944–0.1489) | 0.0023 | 0.5629 | 2.9724 | 17.7 |
| direct-docx | 0.0641 | 0.0649 (0.0640–0.0659) | 0.0083 | 0.6005 | 4.8363 | 120.1 |
| direct-epub | 0.0648 | 0.0648 (0.0645–0.0651) | 0.0087 | 0.5341 | 4.7673 | 119.6 |
| direct-rtf | 0.0543 | 0.0543 (0.0536–0.0658) | 0.0063 | 0.6716 | 5.0154 | 118.4 |
| direct-tex | 0.0642 | 0.0644 (0.0644–0.0647) | 0.0096 | 0.5126 | 4.5304 | 119.0 |
| legacy-docx | 1.9235 | 2.0167 (1.8868–2.0842) | 0.0084 | 0.5888 | 6.7491 | 225.2 |
| legacy-epub | 1.8542 | 1.8394 (1.7225–1.9838) | 0.0088 | 0.5572 | 6.8285 | 225.3 |
| legacy-rtf | 1.8010 | 2.0333 (1.9541–2.0793) | 0.0062 | 0.5449 | 6.6701 | 225.3 |

0/0 significa fenomeno assente. Figure+caption distinto dai byte asset: A10
passa anche se la caption non è fedele. AST esatto non completa L04.
Latenza conversione è processo Pandoc diretto oppure child/API legacy; audit,
score con startup/check ed evaluate completo sono separati. Primo e tre
ripetizioni hanno processo/workspace nuovi, cache OS non controllata; nessuna
pretesa cold OS. RSS sampled10ms non è hard cap RAM aggregato.

## Verifiche, provenienza e limiti

247 test mirati PASS;18 guardie resources.run/monitored_command e otto R
esterne corrette, inclusi writer1MiB+8192 B tardivi, setsid, timeout dopo leader,
positivi200/400ms e processo estraneo preservato.20 contratti correnti con
esiti attesi; H15 R FAIL per STREAM_LIMIT indipendentemente dall'exit del leader,
H15/Chrome storici conservati. CSS escape/srcset/SVG/XML resource pericolosi
sono INVALID_INPUT4 prima di estrazione/tool, marker target/tool assenti.
I payload pericolosi non sono stati eseguiti in browser/Pandoc per provarli.

Suite veloce prescritta: **388 test e310 subtest PASS** sotto R s013; API,
packaging,Docker esclusi, due smoke radice esclusi scegliendo tests/. Collaudo
parziale: migrazione/applicazione progettata non convalidate. Nessun nuovo
motore, rete/fetch/provider, modello/peso/lock/pin o costo esterno.

208 file helper/config/test/corpus preservati,230 input campagna fisicamente
copiati prima e verificati invariati dopo.19 receipt R positive finali con
binding prima/dopo; branch/HEAD/indice verificati, nessun processo proprio
pendente. Errori ordinari e giri precedenti sono conservati con identità reali;
rerun motivati da input/helper cambiati, nessun reset quota o errore nascosto.

L04 tipografia/semantica formule output UNAVAILABLE, anche con AST e Vgold
validi. L01 PDF/OCR,L02 TeX compile,L05 hardware non misurati. CPU lowerbound,
RSS sampled/processi brevi/shared pages/cache OS restano limiti; R non è sandbox
universale filesystem. Rete/acquisizioni/document-send0, nuovi servizi€0.

[Dati effettivi e rettifica prima/dopo](2026-10-09-a002-r002-data.json) e
[report implementatore con prove F01–F12](../../temp/run-a002-fase1-corpus/reports/implementation-r002.md).
Le evidenze complete e i byte sono in temp per le due review; prima della futura
pulizia il supervisore dovrà archiviarli. Nessuna pulizia o aggiornamento dei
registri comuni è stato eseguito. Prossimo: snapshot comune, due review mirate
indipendenti, arbitrato; Git di scrittura resta all'utente dopo GO.

Bilancio cumulativo finale dall'entry unica:15,32h wall;6195,769CPU-s noti
come lowerbound;918503388B retained noti più 1890137B conservativi per
il delta repository; RSS peak sampled735604736B. Nessun reset, limiti di
misurazione dichiarati. Cap52h/21600CPU-s/6GiB nuovi, stop49h/19800CPU-s/5GiB
e riserva3h/1800CPU-s/1GiB invariati; tutti i workload implementatore conclusi.
