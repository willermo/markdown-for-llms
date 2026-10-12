# Diagnosi dopo r012: controllo indipendente delle assunzioni prima della consegna

2026-10-11. Supervisore;due review r012 reali NO_GO ricevute/arbitrate.
Fonte:[arbitrato](../arbitrations/arbitration-implementation-r012.md),
[Claude](../reviews/review-implementation-r012-claude.md),
[OpenAI](../reviews/review-implementation-r012-chatgpt.md).

Non sono undici ricomparse dello stesso bug. I primi giri comprendevano
corpus/metriche/risorse/prove;gli ultimi sono concentrati sul reader locale
che osserva e valida Markdown/HTML,mischiando sintassi di blocco e inline,
domini dopo stripping e fonte fisica,guardia e proiezione. Diverse chiusure
sono reali,ma i fix dei confini hanno prodotto anche nuove regressioni.
Questa run non misura la capacità di un motore OCR/ML di correggere questi
bug:sono decisioni Python del nostro misuratore. In r012 Pandoc offre una
controprova sintattica al nostro riconoscimento,non causa del difetto.

R012 ha fatto refactoring reale,non solo una regex:atomi condivisi e sessione
di guardia. Due lacune ne limitano il risultato:

- Sessione HTML nuova,ma attrs continua a sollevare direttamente:la politica
  degli errori non è stata integrata per tutti i produttori. La scansione
  raise/callsite lo avrebbe mostrato prima dei test finali.
- Generatore fence indipendente dagli helper,ma tutti i payload iniziano
  con la fence. Autore e oracolo manuale condividono l'assunzione che ogni
  inizio riga dia un blocco code;la variante testo→tilde è assente. La
  competenza grammaticale non diventa indipendente solo cambiando file.

2688PASS indica assenza di fallimenti nei test eseguiti;non prova copertura
di un ramo mancante o validità degli attesi. Byte/MATCH e gate R provano
identità/isolamento,non correttezza. Il metodo supervisore precedente era
insufficiente anche dopo la richiesta r009 di consolidare e quella r012 di
generare casi:non aveva reso verificabili queste due assunzioni.

Autore e reviewer OpenAI dichiarano famiglia GPT-6;deployment/effort non
esposti,identità esatta della configurazione non dimostrata. Claude è altro
provider. Stesso modello può ricevere contesti/obiettivi diversi:produrre una
soluzione e verificare un artefatto finito cercando controesempi sono compiti
diversi. Un risultato in una chat non implica stesso percorso/giudizio in
un'altra. Le review concordano sull'esito,non sempre sul medesimo finding.
In r012 si sovrappongono sugli attributi,il finding fence è di Claude.
Ipotesi di ancoraggio dell'autore ai fix precedenti è plausibile ma non
misurata;compattazione e inferiorità del modello non sono cause provate.

La documentazione ufficiale OpenAI descrive variabilità e necessità di
valutazioni specifiche,ed evidenzia la differenza fra confronto/classificazione
e generazione aperta: [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
Questo spiega perché l'identità del modello non certifica autocontrollo;
non diagnostica da sola le nostre undici iterazioni. Nessun servizio di eval
remoto,abbonamento o cambio modello automaticamente autorizzato.

Il mandato43 rende concreto il cambiamento di metodo:

1. Un solo protocollo della guardia per tutte le fonti Unsafe del reader,
   eventi/posizioni e prosecuzione indipendente dall'albero. Censire ogni
   eccezione e callsite,non solo report_unsafe/HTML.
2. Condizione fence condivisa e corretta rispetto al dialetto pinned:
   contesto prima del marker,chiusura/EOF,marker e raw sono dimensioni delle
   prove. AST Pandoc locale obbligatorio come oracolo di queste decisioni,
   non nuovo parser universale,non gold semantico o validatore di sicurezza.
3. Prima della consegna,due controprove su copie private fanno riapparire
   le assunzioni attrs fail-fast/fence-ogni-riga:i checker devono respingerle.
   La mappa rami→proprietà→casi deve includere ogni fonte/callsite censito.
   Un generatore indipendente ma basato sulla stessa premessa non basta.
4. Correggere tutti i controesempi della classe nella chat autrice prima
   del report finale. Doppia review resta verifica finale indipendente.

È una soluzione circoscritta ai meccanismi che hanno fallito,con un gate
oggettivo aggiunto. Non garantisce zero bug o GO al giro successivo. Se
l'adapter locale non riesce a mantenere queste proprietà nel profilo finito,
il supervisore deve riconoscere il limite strutturale e valutare una scelta
architetturale esplicita di parser/AST maturo,senza continuarne la simulazione
per patch. Sostituire oggi tutto con un AST Pandoc non è una correzione già
dimostrata:offset raw,unsupported e sicurezza pre-proiezione vanno conservati.

Nessuna deadline della run;altri limiti e contatori invariati. Nessuna
severità ridotta per terminare o nuovo requisito di grammatica universale.
