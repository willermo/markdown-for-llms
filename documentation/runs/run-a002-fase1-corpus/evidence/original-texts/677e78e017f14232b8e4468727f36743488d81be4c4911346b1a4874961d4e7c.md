# Disposizione supervisore — rendere V eseguibile per l'utente — r001

2026-10-08, Europe/Rome. Codex/OpenAI, ruolo esclusivo supervisione.
Autorità: risposta esplicita dell'utente in questa chat. V a blocchi/TSV è
respinta come troppo complicata e non eseguibile; l'utente indica come modalità
possibile il confronto di un originale con un output Markdown.

Questa disposizione sostituisce la **presentazione e raccolta del giudizio
umano** di E02/prompt05 e del packet006. Corregge un requisito del supervisore;
non attribuisce il problema alla capacità dell'utente. Le precedenti spiegazioni
non hanno reso il compito praticabile. Non occorre un'altra guida al TSV.

## Istruzione vigente

La chat implementatrice prosegue con
[prompt06](../prompts/06-implementation-r001-human-usable-comparison.md) nella
stessa run e quota. **Non chiedere all'utente di completare packet005/006,
215 righe TSV, ID di blocco, AST, code point, hash o timestamp manuali.**
Packet e istruzioni precedenti rimangono evidenze storiche, non vengono riscritti.
Nessun giudizio umano completo è stato ricevuto/validato da questa supervisione;
tentativi e feedback già espressi non sono firma o ACCEPTED impliciti.

Preparare un confronto diretto offline: originale effettivo a sinistra e
**Markdown di riferimento leggibile** a destra, un documento per volta.
Il riferimento è la rappresentazione completa del gold indipendente, non
un output Pandoc promosso a gold. Indicarlo con parole semplici («Originale» /
«Markdown di riferimento»); spiegare in una frase che si sta controllando il
riferimento da usare per misurare le conversioni. Output effettivi Pandoc e
relativi FAIL restano distinti nei report; non confondere questo controllo
con la valutazione dei motori o con giudizi matematici semantici L04.

Testo completo, formule leggibili in notazione matematica, tabelle vere con
celle vuote, figure/caption, note/bibliografia e riferimenti devono comparire
nel Markdown reso visibile. Nessun campo atteso vuoto che richieda trovare il
valore in JSON. Originale/render collegato alla sua identità reale, zoom/ordine
pagine e capitoli chiari. Per TXT/TeX e controlli, source leggibile come lato
originale; TeX non compilato esplicitamente, senza certificazione tipografica
simulata. Limiti reali di presentazione restano visibili e non scaricati sull'utente.

**Un giudizio umano per documento**, massimo10corpus+3controlli; mai215giudizi
atomici. L'utente può scrivere «C02: corrisponde», «C02: manca ...» oppure
«C02: non riesco a valutarlo». Nessuna necessità di tradurre la risposta in
schema tecnico o fornire la data per ogni riga. Risposte incomplete restano
pendenti; dubbi/differenze reali non diventano conformità. L'interfaccia può
offrire tre scelte semplici e nota opzionale, senza rete/JS/registri tecnici
esposti; rispondere in chat o con testo ordinario deve essere sufficiente.

Registrare il giudizio reale **per documento**, con copertura del confronto
mostrato e identità/hash della coppia. Non espanderlo in215false firme umane:
le proprietà tecniche hanno i propri controlli M e la receipt dichiarerà
granularità documentale. Import fedele di risposta originale, autore/data
attribuiti alla risposta reale con limiti, ambito visibile. M required PASS e
V documentale completo ACCEPTED sono necessari per VERIFIED. Il rifiuto di un
documento non annulla i giudizi reali sugli altri; A03 complessivo richiede tutti
i documenti dovuti verificati. Nessuna blanket approvazione sulla sola UI.

## Carico tecnico e difetto C02

Gli strumenti e l'implementatore verificano codifica, code point, mapping,
completezza delle annotazioni e della vista, hash, licenze, denominatori,
stati/parser/metriche e negativi. Questo lavoro non è chiesto all'utente.
La vista del gold non è una prova autonoma che il gold sia corretto: M deve
ancora confrontarlo sul canonico con logica distinta da reader/scorer.

Osservazione **agent_static/agent_visual** del supervisore su packet006:
C02/b04 ha `caption="Celle misurate"` nel riferimento HTML; la scritta è
assente dai due render osservati e da word/document.xml del DOCX copiato nel
packet. Questa è una discrepanza concreta da diagnosticare/correggere prima
del nuovo confronto. Non è un rifiuto umano già ricevuto e non è una review
completa del codice. Aggiungere un controllo M/test discriminante del caso:
una caption attesa ma assente non può essere nascosta da PASS delle sole celle.
Correggere sorgente/generatore o riferimento contro specifica indipendente,
non contro perdite del converter; conservare copertura e vecchie revisioni.

## Limiti e ciclo

Il risultato del brief, i gold indipendenti, le metriche, i negativi, la baseline
e le due future review del codice restano dovuti. Questa è una correzione locale
del workflow umano per istruzione utente, non una nuova pianificazione generale,
rinuncia alla fedeltà o GO sull'implementazione. Gli altri contratti approvati
restano vigenti salvo le sostituzioni sopra; codice/import/verification devono
essere adeguati e verificati, non aggirati con record accettati fittizi.

260min era la durata concordata sul vecchio lavoro, non tempo umano già svolto
né durata da trasferire meccanicamente alla nuova modalità. Presentare conteggio
e stima semplici sulla coppia C02 pronta; l'utente potrà valutare il carico sulla
vista concreta. Nessuna nuova finestra o durata accettata inventata. Quota52h/
21600CPU-s/6GiB e riserva invariati, entry07:06:54UTC conservata, niente reset.
Continua il lavoro tecnico autorizzato senza attendere nuove approvazioni di
singoli comandi; l'utente ora non ha lavoro da fare sui packet precedenti.

Snapshot/stage vecchi non attestano nuovi helper o gold; ripetere solo prove
dipendenti. Se cambia solo viewer/import, non ripetere tutte32conversioni;
se cambia gold, riscore dipendente; se cambia input canonico C02, rerun solo
percorsi che consumano C02 con nuove identità e generazioni/M/render pertinenti.
Le modifiche amministrative di questa supervisione sono dichiarate nel nuovo
contesto d'ingresso, non attribuite a modifiche del codice consumato prima.

Lavoro tecnico dichiarato nel report implementatore letto, A03 ancora PENDING;
107test/suite/20contratti/32attempt sono esiti **dichiarati dall'autore**, non
ricertificati da questa supervisione e non GO. Stato IMPLEMENTATION /
V_WORKFLOW_FIX; niente review codice finché manca consegna completa.
