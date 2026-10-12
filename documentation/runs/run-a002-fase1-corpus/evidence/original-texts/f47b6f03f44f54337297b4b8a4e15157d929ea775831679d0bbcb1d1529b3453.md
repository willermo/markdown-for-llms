# R014 — requisiti di accettazione e stato del chiamante

R013 ha chiuso gli originali attrs/ordine e coverage,ha pubblicato il censimento,
ha usato AST indipendenti eha dimostrato che due mutazioni sono intercettate.
Non è un giro senza progressi. I quattro nuovi finding accolti mostrano però
premesse ancora duplicate o troppo ampie prima della guardia:

1. La validazione URI può uscire con un tipo di errore non previsto dal protocollo.
2. Una riga somigliante a un fence è accettata senza validare la sua info.
3. Una decisione comune riceve stato diverso o riparte da zero nei producer.
4. Un gruppo delimitato da graffe viene accettato senza validare la grammatica
   dell'attributo;il contenuto diventa opaco prima di raccogliere aperture HTML.

Il generatore è separato dal prodotto,ma seleziona lo stesso contesto interno
e le stesse info vuote. La separazione delle funzioni non basta per indipendenza
delle premesse. Una mutazione locale che fallisce non prova il chiamante:
gli input devono discriminare anche stato esterno,validità sintattica eerrori
possibili dell'adapter. Responsabilità condivisa del mandato e dell'autore:
non attribuire l'insuccesso alla compattazione o al modello senza evidenza.

Per r014 la nota prima del codice fissa **tre contratti finiti**,con callsite:
errori ammessi/rifiutati dell'adapter URI;accettazione blocco con opener e
stato fisico precedente;accettazione metadata con intervallo e grammatica.
Decisioni condivise consumate dai rami,tipo/intervallo/owner/posizioni immutabili
o equivalenti controlli espliciti:nessun nuovo parser universale o schema pubblico.
Un intervallo non provato come code/metadata non ottiene proprietà di contenuto
inerte. Nei rami fuori proiezione resta una diagnosi esplicita eHTML reale
riceve la guardia;un secondo feed indiscriminato inventerebbe falsi unsafe.

Prove per partizioni positive e negative,attesi congelati prima eASTdel corpo:
contesto **esterno×interno**,inizio/separazione/lazy/figura,info valida/invalida,
metadata validi/invalidi equoted-values conHTMLletterale,safeasset/script/URI,
ordine/charUTF8/reset. I casi corretti V13 devono essere inclusi come controlli.
Si selezionano combinazioni sintatticamente pertinenti,non un prodotto cartesiano
universale o un minimo numerico di test. Censimento fa riferimento a ogni
callsite effettivo eai tipi di uscita dell'adapter,non solo alle sue raiseUnsafe.

Le prove devono respingere privatamente le quattro assunzioni errate:
URIValueError sfuggito,infoopener qualsiasi,contesto chiamante azzerato,metadata
qualsiasigraffe. Una copia privata per assunzione euno/due discriminanti bastano;
zero mutazioni del prodotto protetto,non una campagna mutation. I checker
devono guardare il corpo discriminante AST euna observe,non unCodetrovataltrove
o due chiamate sommate. Correzioni di premesse rese esplicite,non copiate dal reader.

Budget CPU molto ridotto,tempo della run senza limite. Evitare il costo di
verifiche ripetute su inventari di100kfile per ogni sonda:manager invariato,
snapshot tecnico puntuale dei byte effettivi,copie/archivi/10moduli/37helper
legati prima e dopo,R/preflightcorrente mantenuti. Lo storico è collegato da
snapshot originale immutato;nessuna cancellazione o sanatoria. Accorpare prove
compatibili sui medesimi byte finali,non fare freeze per argv o per ogni caso.
Non ripetere campagne invalidate soltanto da commenti nei registri;ripetere
sempre quelle realmente invalidate da codice o input modificati.

Le due review reali restano necessarie. Non promettiamo che questo giro sarà
l'ultimo:questa impostazione affronta le assunzioni identificate e rende le
prove capaci di smentirle prima della consegna,senza garanzia di assenza di
tutti i difetti. Il budget corrente non è ampliato dalla rimozione delle deadline.
