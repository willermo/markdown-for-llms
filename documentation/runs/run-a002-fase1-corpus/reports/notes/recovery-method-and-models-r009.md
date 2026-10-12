# Recupero a002 dopo r008 — metodo, contesto e modelli

2026-10-10 Europe/Rome. Valutazione del supervisore richiesta dall’utente,
non prova comparativa dei modelli, non GO del prodotto o aumento del budget.

## Diagnosi e responsabilità

Le ultime review documentano tre cause precise: la guardia HTML dipende dal
successo della proiezione dell’albero; l’euristica dei tag incompleti interpreta
le virgolette diversamente dal tokenizer fissato; dopo stripping dei prefissi,
il consumer ricostruisce un limite diverso da quello del producer. Sono
interazioni tra rappresentazioni dello stesso input. F35/F36 migliorati non
implicano F37 completo: Claude ha provato positivi conformi, ChatGPT ha quattro
controesempi ammessi. Nessuna contraddizione da risolvere con un voto medio.

La serie r003–r008 mostra correzioni di sintomi e regressioni dei confini
lexicali. La crescita dei test fino a1373mirati/1530veloci non costituisce una
prova di copertura delle interazioni. Anche la supervisione ha mantenuto
mandati troppo orientati alle ultime riproduzioni: serve ora rendere esplicite
le invarianti comuni e verificarle su trasformazioni dei contesti supportati.
Le doppie review hanno svolto una funzione utile; abbassare i gate per ottenere
GO lascerebbe difettoso il misuratore destinato al confronto dei motori.

Non abbiamo un esperimento che dimostri causalità tra compattazione e difetti,
né che un altro modello li eviti. Chat fresca significa però poter rivalutare
il codice attuale senza portarsi tutte le correzioni conversazionali come
premesse. È giustificato cambiare entrambe le chat di supervisione e
implementazione, conservando run/byte/storia/risorse. La preferenza corrente
dell’utente prevale sull’indicazione precedente della stessa chat autrice.

## Intervento concreto, locale

Mandato r009 F39–F42: consolidamento circoscritto dei confini del reader.
Tre invarianti: (1) la validazione dei tag reali continua indipendentemente
dagli errori della proiezione; (2) token/tag incompleto e valori attributi
seguono una sola semantica lessicale coerente con CPython fissato, rispettando
i corpi inerti; (3) il consumer riceve il dominio e la mappa sorgente stabiliti
dal producer, anche dopo stripping/merge. Mezzi e refactoring reversibili
lasciati all’implementatore; non imporre nuova libreria o parser universale.

Prima di cambiare codice, una breve nota tecnica nel proprio report indica
i punti decisionali duplicati e come vengono eliminati o resi coerenti. Poi
riproduzioni prima/dopo e matrice finita per interazioni, con oracoli sintetici
indipendenti e trasformazioni che conservano contenuto/link/asset. Test sul
percorso audit→observe→score, positivi e negativi, nuovi gate/installazione/
prove dipendenti, rescore/invarianza64. La nota non è una nuova pianificazione
né un gate al supervisore: la nuova chat completa tutto il risultato.

La consegna deve spiegare perché la classe di difetti viene chiusa, non solo
mostrare il payload dell’ultimo report passato. Se emerge un errore ordinario
si corregge nella stessa chat. Nuovo comune e due nuove review dopo consegna.
Nessuna promessa che r009 otterrà GO; nessun numero di test imposto come uscita.

## Combinazioni consigliate, da selezionare nell’interfaccia

| Ruolo | Raccomandazione per questo recupero | Motivo |
| --- | --- | --- |
| Nuova implementazione | GPT-6.1 Sol, xhigh | Concentrare ragionamento aggiuntivo nel consolidamento e nell’analisi degli effetti del cambiamento; stessa famiglia di costo |
| Nuova supervisione | GPT-6.1 Sol, high | Contesto fresco e mandato concreto; nessun motivo osservato per aumentare il costo di tutti gli atti amministrativi |
| Nuovo reviewer ChatGPT | GPT-6.1 Sol, high | Ha già trovato controesempi utili; mantenere indipendenza e aggiungere controllo delle invarianti |
| Nuovo reviewer Claude | Modello e livello già scelti dall’utente; report r008: Claude Opus 5.5, effort non esposto | Conservare il secondo provider e la capacità osservata di trovare regressioni; non inventare mapping degli effort OpenAI |

Alternativa per la sola implementazione, se l’utente preferisce spendere di
più in un singolo tentativo: GPT-6 Astra, high. Supervisore/reviewer invariati.
Non raccomando passaggio indiscriminato ad Astra/max: non abbiamo prova locale
che compensi il costo; prima viene il cambio di metodo. Restare Sol/high
rimane compatibile col mandato rinforzato; xhigh è una raccomandazione,
non una condizione di validità e non è stato selezionato automaticamente.
Non proporre due reviewer OpenAI chiamando uno di essi Claude.

Fonti primarie OpenAI consultate e aperte2026-10-10:

- [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol):
  effort low/medium/high/xhigh/max; API Standard testo2USD input/10USD output
  per milione di token.
- [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra):
  modello per lavoro più esigente,effort low/medium/high/xhigh/max;10USD input/
  50USD output per milione API Standard. A pari token non cached nelle fasce
  base, tariffe5volte quelle Sol; non è una stima della fattura Codex/abbonamento
  o del costo per risultato (token,cache,contesto e tier possono differire).
- [Reasoning](https://developers.openai.com/api/docs/guides/reasoning):
  effort superiore concede più ragionamento e può usare più token/tempo;
  nessun moltiplicatore di costo universale high→xhigh.
- [Model selection](https://developers.openai.com/api/docs/guides/model-selection):
  confrontare sul proprio compito e mantenere l’impostazione più leggera che
  soddisfa la qualità. Qui la superiorità sul caso è un’ipotesi del supervisore,
  non un benchmark eseguito.

Disponibilità/tariffe del piano e consumo inference attuale non verificati.
La rete usata per documentazione pubblica sui modelli è distinta dal vincolo
rete0 sulle prove e sui documenti del corpus. €0 dei workload indica servizi
esterni del benchmark, non costo totale delle chat. Le raccomandazioni non
modificano file di configurazione, provider dei motori o risorse della run.
