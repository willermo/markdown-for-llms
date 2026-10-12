# Valutazione richiesta dall’utente — debito residuo r014

2026-10-12. Consiglio del supervisore, NON decisione di accettazione o GO.
L’utente contesta il ciclo di fix e chiede conseguenze e alternative; nessun
nuovo fix o workload prodotto avviato durante questa valutazione.

Una review r014 Claude NO_GO con traccia statica: la premessa AST della
combinazione lista→blocco non è stata eseguita. Altre ipotesi sono condizionate.
Review ChatGPT incompleta per errore di costo del proprio checker: 35,510746s
nel controllo, 41,042049s misurati complessivi, nessun nuovo finding dimostrato.
Non esiste una coppia completa da cui inferire un arbitrato r014 finale.

Il reader in scripts/benchmarks misura/audita gli output: il suo difetto non
prova un difetto del motore o una perdita nei byte grezzi. Può perdere asset
nel modello osservato o assegnare COMPLETE/UNSAFE erroneamente; run_exit nel
benchmark dipende dall’audit. Non è dimostrato fetch/esecuzione HTML dei casi.

Consiglio: interrompere il ciclo di correzioni del reader attuale e conservarlo
come misuratore sperimentale, mantenendo raw/corpus/gold/ricevute e test utili.
Uso esplorativo consentito solo con risultati provvisori e limiti per fenomeno;
nessuna attestazione di completezza/sicurezza o ranking automatico fondato
sulle classi non affidabili. Accettare formalmente il debito richiede un mandato
esplicito di consegna ridotto e criteri verificabili; niente GO retroattivo.
Un’ipotesi statica resta ipotesi finché manca la premessa decisiva.

Il verificatore non ripara il motore. Serve a scegliere configurazioni/motori,
scoprire regressioni, verificare numeri/formule/asset/ordine e instradare la
revisione. Sintassi e fedeltà vanno separate: valutare in un intervento distinto
l’AST di un parser mantenuto, con versione/profilo fissati, anziché replicare
la grammatica Markdown; l’AST dello stesso motore non è un gold indipendente
per la correttezza semantica della conversione. Pandoc documenta reader→AST→writer:
https://pandoc.org/filters.html . Proposta, nessun cambio architetturale attuato.

Il tetto10CPU-s per reviewer era una scelta operativa inadeguata del supervisore.
Non è un criterio di qualità. Limiti di risorse servono a controllare workload
inattesi, ma il budget va dimensionato al lavoro e separato dall’accettazione;
non deve produrre sonde mancanti e review bloccate per amministrazione ripetuta.
Il consumo misurato resta distinto dagli accantonamenti prudenziali; nessuna
quota modificata o reset assunto dalla richiesta di consiglio.
