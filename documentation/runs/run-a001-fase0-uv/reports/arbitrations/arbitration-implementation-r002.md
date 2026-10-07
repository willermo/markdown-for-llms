# Arbitrato — implementazione — run-a001-fase0-uv — r002

- Data: 2026-10-07. Supervisore Codex/OpenAI, chat corrente; variante del modello
  e ID chat non esposti. Nessuna delega o sostituzione dei reviewer.
- Oggetto: fix del primo arbitrato, report-r019, delta e regressioni pertinenti.
- Snapshot comune: `impl-r001-stage-final-s002`, SHA256
  `e0cb33d16a5766f653923c2d25cf365f910d7e738d542893db2a1ac589272c1d`.
- [ChatGPT](../reviews/review-implementation-r002-chatgpt.md): **NO_GO**,
  GPT-I002 bloccante, GPT-I003 non bloccante; Codex/OpenAI GPT-6 dichiarato.
- [Claude](../reviews/review-implementation-r002-claude.md): **GO**,
  CLA-I006–I008 non bloccanti; Anthropic/Claude Opus 5.5 dichiarato.
- **Esito supervisore: NO_GO per GPT-I002, residuo CLA-I001.**

## Validità e ricezione

Entrambi i report dichiarano chat nuove indipendenti, mancata lettura del report
concorrente e MATCH prima/dopo sullo stesso oggetto. Il supervisore ha verificato
nuovamente **MATCH prima degli aggiornamenti propri**, con il managed 3.12.13.
Branch feature/run-a001-uv, HEAD/dev/base
`66ba82200e5def5a4db76f9bafccb0731b506091`, indice vuoto.
La [ricezione — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
identifica 41 file pertinenti, inclusi report, checkpoint, evidenze e sorgenti.
Letture del codice e ricostruzione del tempo; nessuna suite, R, build, installazione,
Docker, rete o operazione Git di scrittura del supervisore.

Il manifest final-s002 rimane immutato. Dopo questi aggiornamenti documentali e
dei registri è storico: non richiedere all'autore un MATCH sul contesto modificato
né riscrivere lo snapshot o le vecchie prove. I registri precedenti sono conservati
in `evidence/supervisor-arbitration-implementation-r002/before/`.

## Decisione sui rilievi

| ID | Decisione | Requisito, raggiungibilità ed evidenza | Azione |
| --- | --- | --- | --- |
| CLA-I001 | **Aperto, bloccante**, limitato al residuo GPT-I002 | Standalone e gate S/I generalizzati sono dimostrati; il launcher R permanente rifiuta ancora altre run. Arbitrato r001 e mandato55 richiedono il percorso ufficiale riutilizzabile. | Generalizzare anche l'ingresso R e provarlo realmente con un secondo run_id valido. Non rifare la suite standalone già dimostrata per il solo cambio del launcher. |
| CLA-I002 | **Risolto** | Compose r007 reale network:none, produttore versionato identificato, contesto/supply verificati; secondo Compose equivalente. Concordanza delle review e hash indipendenti. | Conservare le prove ai loro input; nessuna nuova build per questo fix. |
| CLA-I003 | **Risolto** | C-docker versionato realmente eseguito, nove sottocasi, driver completo e isolamento verificati. CLA-I008 è una diversa incoerenza nell'interfaccia di selezione. | Conservare il test e le prove Docker; rendere esplicito il modo supportato nelle guide. |
| CLA-I004 | Parte principale risolta; **residuo documentale aperto** | Riconciliazione 74 fence +9 inline verificata, configurazione/backend recuperati; resta il flag inesistente GPT-I003/CLA-I007. | Correggere guida e reference; controlli documentali mirati. |
| CLA-I005 | **Risolto** | README durevole identico a B56, AGENTS aggiornato, stato temporaneo nei registri di sviluppo. | Non modificare README/input build per questi fix. |
| GPT-I001 | **Risolto** | Cleanup del path reale, errore primario e cleanup distinti, receipt nei FAIL; tre negativi mock e R reale ricevuti. | Verificare regressione mirata del cleanup dopo l'edit del launcher. |
| GPT-I002 | **Accolto, bloccante**; stesso residuo di CLA-I001 | `run_offline.py:97` confronta letteralmente il run_id con a001; `outside()` chiama questo ramo prima del runner. La prova ricevuta b002 passa da S/I/preflight ma non dal launcher. Sonda mock del reviewer e lettura diretta concordano. | Validazione generalizzata di schema/ID/label/path e confinamento, verify reale e binding conservati; R reale con a001 e altra run valida, negativi e regressione cleanup. |
| GPT-I003 | **Accolto, non bloccante**, unificato con la prima parte CLA-I007 | `verify_distribution.py` non espone `--standalone`; il comando documentato fallisce prima di produrre I. La CLI effettiva deriva lo scope da S. | Correggere il comando; non aggiungere un flag superfluo per adattare il codice all'errore della guida. |
| CLA-I006 | **Accolto; correzione del supervisore registrata** | Riscritture finali posteriori alla misura e non conservate. Formula e mtime riproducono almeno 8321,418589 s, non 8285,287443 s. Il limite superiore è ignoto. | Ledger corretto, vedi sotto. Originali immutati; nessuna riesecuzione per cancellare lo sforamento. |
| CLA-I007 | **Accolto, non bloccante**; chiusura nel fix | Flag inesistente come GPT-I003 e `RUN_IMAGE_CONTEXT_RECEIPT` omessa dalla reference, benché richiesta dal test. | Correggere entrambi i documenti, controllare opzioni/variabili contro il codice, link e whitespace. |
| CLA-I008 | **Accolto, non bloccante**; scelta dell'alternativa documentale | C-docker legge la modalità solo dall'ambiente; il solo flag pytest standalone porta a un rifiuto, senza falso PASS. Il percorso documentato nella guida Docker con RUN_TEST_MODE funziona. | Limitare esplicitamente C-docker a RUN_TEST_MODE nelle guide/reference. Conservare il codice del test/driver e la prova precedente: nessun nuovo Docker richiesto per questa nota. |

I suggerimenti su Compose up e sull'inventario `.venv-python/.lock` restano limiti
operativi/backlog, senza nuove attività di runtime o architettura in questa run.

## Perché prevale il NO_GO

Le review concordano sulla sostanza dei fix packaging, Docker, documentazione e
cleanup. Claude dimostra il gate S/I con b002; Sol controlla anche l'ingresso R e
trova una seconda validazione non aggiornata. Nel report Claude è dichiarata la
lettura di `outside()`, ma non una prova della funzione `snapshot_check()` con b002.
La chiusura completa di CLA-I001 non è quindi sostenuta per quel ramo.

Il difetto impedisce le future run ufficiali, è attribuito a un diagnostico
introdotto qui e viola un requisito approvato. È un fix locale: nessun nuovo piano,
nessuna ripetizione indiscriminata delle prove applicative. I PASS di a001 restano
validi per il codice e gli input effettivamente eseguiti; non diventano prove del
nuovo launcher senza una verifica pertinente.

## Correzione del tempo e risorse prospettiche

Il [ledger corretto — riferimento locale non archiviato](../../notes/materiale-locale-conservato.md)
ricostruisce la formula dell'autore sull'ultima scrittura del manifest:
7194,886280 +45 +1791367898,304797 −1791366789,2481403 −27,524348
= **8321,418589 s**. Sforamento rispetto a 8280: **almeno 41,418589 s**,
arrotondato **41,42 s**. Il costo dopo l'ultima scrittura non è ricostruibile;
non dichiarare questo limite inferiore come totale esatto o limite superiore.

I campi finali e il paragrafo del report-r019 provengono da una riscrittura
successiva non conservata. R020 e ricezione r001 conservano la precedente misura
incompleta come storia; questo arbitrato/R021 e il ledger correggono il dato
corrente. **Quota storica NON PASS**, nessuna sanatoria. L'ultimo workload noto
è concluso a 8251,006318 s; non risulta un workload prodotto oltre il cap.

[R021](addendum-operational-protocol-r021.md) autorizza ora **+600 s prospettici
core, cap cumulativo 8880 s**, con ingresso prudenziale almeno 8380 s e al massimo
500 s nuovi a quell'ingresso, contando preparazione, tentativi e consegna.
È un addebito prudenziale, non una nuova misura della chiusura storica.
Storage 352 MiB e quote/confini precedenti invariati; nessuna rete/build/Docker
nuova. Scope e derivazione sono in
[authorized-scope](../../evidence/supervisor-arbitration-implementation-r002/authorized-scope.json).

## Passaggio successivo

**IMPLEMENTATION / FIX_R002_AUTHORIZED**.
[Prompt59](../prompts/59-implementation-r001-fix-launcher-and-close-review-r002.md)
alla stessa chat implementatrice: risultato completo, report-r020, prove mirate
e checkpoint proprio. Poi snapshot comune del supervisore, due chat nuove per
review mirate di questi fix/regressioni e arbitrato r003. Nessun GO finale,
commit/merge/push/deploy, cleanup globale o fase successiva.
