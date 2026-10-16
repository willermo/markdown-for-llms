# Questioni aperte

Data: 2026-10-02. Questi punti affinano la roadmap; non impediscono di predisporre
documentazione e governance.

| Tema | Informazione o decisione necessaria | Quando serve |
| --- | --- | --- |
| Hardware | Sistemi operativi, architetture CPU, RAM, GPU/VRAM delle macchine reali | Fase 1: benchmark e requisiti minimi |
| Carico | Paper/libri, pagine tipiche e massime, file per batch, frequenza | Fasi 1–2: corpus, limiti e worker |
| Utenti | Uso personale sullo stesso PC, LAN o servizio per più utenti | Fase 2: accesso, concorrenza e distribuzione |
| Corpus | Campioni redistribuibili di scansioni IT/EN, formule, colonne e formati nativi | Fase 1: confronto e soglie di accettazione |
| Cloud | Provider preferiti, budget per conversione e dati ammessi al remoto | Prima di benchmark remoti e integrazione provider |
| Motori e licenze | Versioni, pesi, licenze e capacità misurate | Chiusura fase 1 |
| Documento interno | DoclingDocument o schema indipendente e adattatori | Fasi 1–2 |
| Markdown | Estensioni ammesse, tabelle complesse, note e riferimenti | Fase 2 con fixture rappresentative |
| Frontend | React/TypeScript o alternativa, visualizzatore e componenti | Prima della fase 3 |
| Export | Priorità PDF, HTML e DOCX; eventuale esigenza di RAG immediata | Prima della fase 7 |

Ipotesi operative iniziali: applicazione su singolo host, uso locale, italiano e
inglese, pieno documento come risultato principale, proposta di correzioni con
revisione manuale. Sono ipotesi di progetto, non limiti definitivi del prodotto.

## Priorità16ott2026: prodotto prima della certificazione

[ADR0008 aggiornato](decisions/0008-maintained-parser-and-provisional-benchmark.md):
percorso utilizzabile con output/asset conservati e qualità non certificata;
parser e debito differiti, senza bloccare le funzioni operative. Da decidere nel
piano prodotto: prima configurazione PDF, frontend minimo, confini dell'anteprima,
sequenza degli incrementi e dati da conservare per futura validazione.
€0/remoti0 invariati, niente adozione del trace o certificazioni retroattive.
Le note seguenti conservano le decisioni precedenti, superate solo nei trigger.

Il debito è esplicitamente riconosciuto, da consolidare quando avremo modo/tempo,
con responsabilità e ripresa nel [registro](development/technical-debt.md).
Ambienti/build del trace rimossi selettivamente per liberare8,18GiB; nessun
risultato di conversione o sorgente cancellato, ricostruzione necessaria alla ripresa.

## Decisione2026-10-12: parser e debito a002

[ADR0008](decisions/0008-maintained-parser-and-provisional-benchmark.md) accetta
il parser mantenuto; [registro](benchmarks/a002-residual-debt.md) vincola
ripresa/chiusura. Da verificare idoneità dell’AST pinned a intervalli/source-map,
profilo e adattatore; la decisione non è un’implementazione. La scelta dei
motori e il modello documentale restano aperti per il prossimo incremento fase1.

Aggiornamento 2026-10-15: a003 r007 arbitrata NO_GO per qualificazione sintattica
non convergente. Si verifica ora la fattibilità circoscritta del metodo mantenuto
con coordinate sul raw e copertura del confronto; candidato e adattatore ancora
da convalidare. Successiva disposizione utente dello stesso giorno: anticipata la ripresa
del debito a002 nel medesimo piano, con fattibilità condivisa e prove di chiusura
per ID. Debito ancora aperto; fasi successive non avviate.

Esito verificato del prototipo, 2026-10-16: wrapper AST e testimone di soli
valori insufficienti per il contratto pieno. Prossima prova autorizzata:
provenienza nei rami originali di Pandoc con AST invariata e contabilità raw.
Manutenzione del delta locale e sua eventuale adozione restano aperte;
nessun fork permanente o nuovo motore prodotto già scelti.
Il prototipo trace r001 è stato poi arbitrato NO_GO dopo due review, per
trim non equivalente e lacune del verificatore. Nucleo r002 poi approvato con
due GO e arbitrato. Ruoli r003 poi NO_GO per provenienza HTML; fix locale
r004 ora NO_GO: cursore non prova termine del token e origine della parte non
prova origine del nodo. Fix r005 richiede osservazione nativa e qualificazione
esclusiva, preservando enfasi/note chiuse. T00/C02/T05 rinviata al contratto dei
canali; T07 inline non adottata ora, nessuna operazione T07 nelle23mappe.
Sufficienza a002 non confermata; adozione permanente e adattatori ancora aperti.
