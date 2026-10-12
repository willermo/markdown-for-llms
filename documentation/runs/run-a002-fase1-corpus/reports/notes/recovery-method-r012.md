# Diagnosi dei NO_GO e recupero del reader — a002 r012

2026-10-11. Supervisore Codex/OpenAI; valutazione su codice/arbitrati/prove
locali, non esperimento comparativo dei modelli né nuova review. Nessun
import/workload prodotto proprio. Fatti runtime ricevuti distinti da tracce
statiche r007/r011. [Disposizione tempo](../arbitrations/addendum-resources-r012-r001.md).

## Origine e storia

Undici consegne revisionate NO_GO non significano undici ripetizioni dello
stesso elenco. [r001](../arbitrations/arbitration-implementation-r001.md)
comprende reader/audit/scorer, formula/allineamento, risorse/processi,
validator, corpus/gold, Git e provenienza.
[r002](../arbitrations/arbitration-implementation-r002.md)–
[r004](../arbitrations/arbitration-implementation-r004.md) includono ancora
omissioni di foglie/caption, metriche/bound e diagnostica/mapping HTML.
Da [r005](../arbitrations/arbitration-implementation-r005.md)–
[r008](../arbitrations/arbitration-implementation-r008.md), poi
[r009](../arbitrations/arbitration-implementation-r009.md)–
[r011](../arbitrations/arbitration-implementation-r011.md), i difetti
si concentrano sui confini del reader: token comment/CDATA, blocchi, code,
math, autolink/label, HTML, source-map e fallback. Non tutto resta aperto:
numerosi fix originali e misure sono conservati, ma le classi restano parziali.

Il reader osserva Markdown prodotto, prima di audit/scoring: non è il motore
che converte DOCX/PDF. Nel codice v11:
- markdown_lexemes decide intervalli, LF, metadata, stack HTML ed eventi.
- markdown_label accetta code, token e escape+LF con proprie regole.
- inline ricostruisce code/math/link/HTML e ricorre sulle label.
- producer quote/list/note/caption/cella cambiano il dominio mediante stripping.
- node decide fallback anche tramite presenza della chiave segments.
- audit consuma reference/unknown emessi: una risorsa non emessa non può
  essere correttamente auditata dopo.

Non basta una funzione condivisa se viene richiamata su domini/limiti diversi.
Le decisioni duplicate divergono e i guardrail a valle vedono ciò che il reader
ha emesso, non ciò che ha omesso. Mix Markdown/Pandoc/math/HTML e sintassi
incompleta rende il compito difficile, ma il difetto è nel nostro codice.
L'assenza di parser universale non giustifica COMPLETE silenzioso nel subset.

## Impatto reale, senza equiparare tutti i finding

| Causa r011 | Impatto deducibile | Evidenza/limite |
| --- | --- | --- |
| F50 label escape+LF/math | Falso UNSAFE su formula chiusa; fallback perde formula/link/blocchi tipizzati, falsando audit/misure | Traccia statica, regressione r010 non chiusa. Raw originale conservato, non file sorgente distrutto. |
| F51 raw fence nella quote | Corpo raw scambiato per code; risorsa/tag reale omesso, audit COMPLETE silenzioso. Asset sicuro perso senza diagnosi | Nuova regressione r011 statica. Caso autore conferma ramo per code letterale soltanto; T2/Pandoc non eseguiti. |
| F52 raw inline nella label | Prima apertura saltata, diagnosi della seconda unsafe | Ereditato r010, statico. Documento già bloccato dalla seconda violazione, impatto pratico minore; requisito completo/ordine ancora violato. |

Non è dimostrata esecuzione script, fuga di file o fetch indebito nei finding
recenti. In r006 erano state osservate letture/hash di asset commentati:
problema di interpretazione, non prova di path evasivo riuscito.
Audit COMPLETE non equivale a PASS globale di qualità. Un finding blocking
può violare un contratto essenziale senza avere severità alta o un exploit.
Le forme sintetiche sono ammesse anche se rare; frequenza in documenti reali
non misurata. Le64misure storiche invariate nei fix recenti mostrano assenza
di variazioni sui campioni, non assenza di difetti nelle classi non coperte.
4PASS/28FAIL di conversione per set è altra misura, non11NO_GO del codice.
NO_GO dello strumento di misura non dimostra undici fallimenti di Pandoc.

## Motori, modelli e responsabilità

Finding attuali hanno causa/percorso nel codice Python locale observations.py.
Non sono fenomeni di OCR/ML o allucinazione del motore di conversione.
Il confronto ML/scelta motori resta successivo. Pandoc è rilevante per il
dialetto/sintassi, ma l'incoerenza guardia/consumer è locale e riproducibile.
Le chat AI autrici possono produrre correzioni incomplete; non abbiamo prove
che questi difetti siano limiti intrinseci dei modelli o che cambiarne uno
li elimini. Non confondere modello che scrive codice con motore di conversione.

Supervisione: la strategia ha chiuso riproduzioni, ma non sta convergendo
abbastanza sulle classi. Dopo r008 avevo già indicato invarianti e refactoring
nel mandato30. R011 mostra che quella richiesta non si è tradotta in una
rimozione sufficiente delle decisioni duplicate. La sola formula "matrice
finita/oracolo indipendente" non garantisce copertura: nei153raw r011 nessun
{=html}, i24math-label coprono LFincode e non escape+LF. Molti test generati
moltiplicano un'assunzione comune. Le review trovano interazioni prima della
consegna, ma non devono essere l'unico strumento di esplorazione.

Non è corretto attribuire tutto a nuovi requisiti dei reviewer, né dichiarare
gli11giri inutili. Vecchi difetti chiusi convivono con residui ereditati e
regressioni nuove (r010 due regressioni principali; r011 una nuova più due
classi non chiuse). I gate hanno impedito di dichiarare affidabile un reader
incoerente; non abbassarne la severità per far terminare il ciclo.
La crescita di test/documenti e gli stop che hanno imposto review statiche
aumentano attrito, ma non sono causa dimostrata delle divergenze del parser.

## Cambiamento concreto per r012

Il prompt40 non resta tre patch ai payload. Prima dei cambiamenti l'autore
censisce tutti i punti che decidono queste classi, i domini e i dati passati:
scanner/producer/guardia/consumer/fallback. Crea una sola decisione riusabile
sul tipo e intervallo dell'atomo nel dominio corretto, compresi metadata raw,
LF ammesse e mapping al sorgente. Mezzi interni liberi, ad esempio descrittori
o stream/eventi tipizzati; non serve imporre nuovo parser o schema pubblico.
Guardia e proiezione consumano quella decisione, senza riscoprire confini da
stringhe trasformate con euristiche divergenti. Errore di proiezione non
interrompe visite; segments non prova che l'intero dominio sia stato guardato.

Prima delle prove finali:
1. Attesi indipendenti di sintassi/contenuto/coordinate fissati prima;
   microcasi Pandoc locale solo se la sintassi è davvero dubbia, non gold
   automatico da Pandoc o dal reader.
2. Generatore finito separato dal prodotto: contesto/container × tipo atomo
   × via di LF ammessa × stato delimitatore × tipo raw × ordine safe/unsafe.
   Trasformazioni motivate, non ogni combinazione grammaticalmente invalida.
   Nuovi casi non tutti derivati dagli ultimi tre input.
3. Proprietà esplicite: dominio concorde fra scanner/consumer, nessuna ref
   da code/math chiusi, ogni reale raw/tag visitato, prima unsafe in ordine,
   risorsa non rappresentata diagnosticata, raw/char/byte/blocchi esterni
   conservati, cursore avanza/termina. Confronto copertura per decisione/
   classe, non obiettivo di raggiungere un numero di test.
4. Ricerca di controesempi guidata dalle proprietà prima della consegna;
   seed e limiti riproducibili, casi piccoli/R/no rete. Riduzione degli errori
   trovati a fixture indipendenti. Correggere nella stessa chat, non inviare
   test rossi all'utente come altra consegna.
5. Nella nota finale indica quali duplicazioni sono state eliminate,
   come si verifica guardia completa, classe residua unsupported e perché
   resta dichiarata. Finale canonical/SBI/R/suite/CLI/64 previsto, ripetere
   solo prove invalidate, non freeze o nuova approvazione per ogni comando.

Criteri di uscita stabili: fedeltà, guardia delle risorse, diagnosi/ordine
nel profilo ricevuto, regressioni pertinenti e prove attribuite. F36 e quattro
osservazioni r010 restano limiti non bloccanti; O001 stima di costo non misura,
O002 trasformazioni della classe F51. Non inseguire intera grammatica
CommonMark/HTML/Pandoc, aggiungere motori o cambiare metriche/gold per GO.
Doppia review reale sul risultato completo mantenuta. Questa strategia ha
una giustificazione causale migliore delle patch locali ripetute; la sua
efficacia sarà verificata dai risultati, non garantita dall'assenza di deadline.

