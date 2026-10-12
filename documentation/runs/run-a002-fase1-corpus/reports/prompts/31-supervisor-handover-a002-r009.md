# Handover supervisione a002 — contesto fresco dopo r008, mandato r009

Agisci esclusivamente come **supervisore**, in una NUOVA chat. L’utente ha
chiesto questo handover perché anche la supervisione ha compattato più volte
il contesto. È la stessa run, senza reset della storia o delle risorse.
Usa manage-implementation-run e verify-conversion-fidelity. Non implementare,
non impersonare/delegare reviewer e non riaprire tutta la pianificazione.

## Decisione corrente e prossimo passo concreto

**NO_GO reale r008** registrato in arbitration-implementation-r008.md.
Entrambi i report sono ricevuti e arbitrati. F35/F36 chiusi nei domini verificati;
F37 parziale, F38 parziale. Tre cause accolte:

- F39/CLA-R008-I001: errori di proiezione interrompono la guardia di tag reali
  successivi; UNSAFE diventa UNSUPPORTED. Nessun fetch/apertura indebita.
- F40/CLA-R008-I002: apice dentro valore unquoted crea incomplete-start spurio,
  nasconde tag/ref/asset/testo reali e guardia.
- F41/GPT-R008-I001: dopo stripping quote/list/note il consumer ricostruisce
  un diverso dominio e perde link su code soft-line con marker letterale.
- F42: consolidamento delle decisioni di confine, prove delle interazioni e
  documenti coerenti. Non nuovo piano o nuovo parser universale.

[Prompt30](30-implementation-r009-consolidate-reader-boundaries.md) è completo
per una **NUOVA chat implementatrice**, cambio richiesto dall’utente e già
registrato. Non rinviarlo alla vecchia chat né introdurre un’ulteriore chat
pianificatrice. Riesamina il mandato all’ingresso per coerenza col contesto
attuale, conserva l’arbitrato; poi il passaggio operativo è l’utente che invia
prompt30 alla nuova chat autrice. Se la consegna r009 è già presente, ricevila
anziché far ripetere lavoro concluso. Nessun GO futuro.

## Letture mirate nell’ordine

1. AGENTS,temp/HANDOVER, solo prima parte corrente PROJECT-CONTEXT,
   STATE/HANDOVER/brief della run; skill e protocollo run-lifecycle.
2. `handovers/supervisor-implementation-arbitration-r008.md`, arbitrato r008,
   `notes/recovery-method-and-models-r009.md`, prompt30 integrali,
   addendum-resources-r008-r001 e nota originale utente sulla finestra24h.
3. Entrambi i report/checkpoint reviewer r008 integrali, finding GPT e casi
   Claude D1/D2/D3/D6/D7/variante safe, ricezione/premesse/transizione r008.
4. Report/CP implementazione r008 finali e protocollo attuale; sorgenti
   pertinenti solo per questioni decisive. Mandati/piano/errata/ADR0001
   per contratti ammessi, non tutti i log o tutti i giria001–r007.
5. Se già presente, report/CP/evidenze finali r009 e relativo delta. Non
   inventare quella consegna se non esiste.

## Identità e contesto corrente

Repo `/home/davide/workarea/markdown-for-llms`. Feature
`feature/run-a002-corpus`; HEAD/dev/origin-dev locale/merge-base
`80c31da46e5caab990d3596c79c44e524ec1dc95`, indice vuoto all’ultima ricezione;
ricontrolla delta anche non committato, niente reset/checkout o query remota
implicita. Git commit/merge/push restano manuali dopo GO finale.

Comune revisionato r008 SHA256
`2f6d690a8f3db7af8f192f78a9e6f5c0e3704b9757256f9dfdf72e35ee3b4152`,
impronta`448b854a88dc3dc883f7337f6fe037fd27846513e3a19104fc88529841e07eda`:
325file/76333artefatti,20PDF generali. MATCH reviewer e supervisore prima
dei registri; storico dopo quattro documenti amministrativi, non rigenerare.
Nuovo ingresso `snapshots/implementation-fix-context-r008.json`, identità
nella receipt di `evidence/supervisor-implementation-arbitration-r008/`.

```bash
python3 -I -B scripts/run_context.py verify run-a002-fase1-corpus --label implementation-fix-context-r008
```

Registra argv/cwd/output/exit. Se autore ha già modificato input r009, STALE
può essere atteso per delta autorizzato: attribuiscilo, non restaurare v8
per forzare MATCH. Il nuovo comune di review si crea sulla consegna completa.
Snapshot tecnici attribuiscono prove, non approvazioni; niente freeze ad ogni
sonda o argv. Preflight S007 storico STALE per soli quattro registri va
conservato, non trasferito comePASS; nuove prove con gate/input attuali.

## Metodo di recupero e modello

Diagnosi: scanner/producer/consumer/validazione interpretano byte/domìni
diversamente. I fix hanno chiuso payload ma introdotto regressioni dei
confini. Anche i mandati supervisore dovevano coprire meglio queste
interazioni; contesto fresco utile, causalità compattazione non provata.
Prompt30 consente refactoring locale e impone invarianti, nota tecnica breve
prima dei cambiamenti, oracoli indipendenti/trasformazioni finite e prova
end-to-end. È esecuzione nella chat autrice, non ritorno di ruolo o nuovo piano.

Raccomandazioni, non configurazioni automaticamente cambiate:
nuovo autoreSol/xhigh, supervisoreSol/high, reviewerOpenAISol/high,
Claude configurazione corrente (reportr008 Opus5.5/effort non esposto).
Alternativa più costosa soloautore Astra/high. Mandato valido ancheSol/high;
nessuna garanzia di GO da modello superiore. Fonti OpenAI aggiornate/aperte
nella nota, tariffeAPI distinte dall’abbonamento, benchmark€0 distinto dal
consumochat. Se utente sceglie un’altra combinazione registra il provider
reale; non leggere la selezione proposta come autorizzazione a nuovi servizi.

## Prove, limiti, risorse da non perdere

r008:1373mirati/1530veloci+310subtest,114exitCLI e64raw verificati dall’autore;
25receipt finali23PASS+2FAILH12, prima3TIMEOUT3s e dopo3PASS conservati.
Reviewer hanno prove proprie mirate, non suite/conversioni.64osservazioni
invarianti salvoIDreader e metriche identiche:4PASS/28FAIL per set32,
non forzare questi numeri in r009. F32/F27/F30/F28locale/F26/F16 preservati.
Corpus/gold/raw/V13/A03 chiuso invariati,L04formuleoutputUNAVAILABLE distinta;
L01PDF/OCR,L02compile,L05CPUlower/RSSsampled/hardware restano limiti.
ChromeNONPASS/helper12gap/H15/E9 storico e erroriintermedi non sanati.

**24h effettive** già disposte dalla conferma osservata2026-10-10T09:37:31Z;
entry originale2026-10-08T07:06:54.369068Z,nessun reset. Stop nuovi workload
**2026-10-11T06:37:31Z =08:37:31 Europe/Rome**,deadline09:37:31Z =11:37:31,
riserva3h. CPU21600/stop19800s,disco6/stop5GiB,RAM8/stop sampled6GiB,
riserva1800CPU-s/1GiB,R≤900s/1MiBstream/32MiBfile,rete0/€0benchmark;
stop al primo limite. Preparazione/attese/review/registri/consegna inclusi,
nessuna finestra nuova per il cambio di chat. Leggi clock reale all’ingresso.

Accounting visibile11830,300204CPU-s include autore e due reviewer r008,
con45sClaude r007 approssimati separati, non upper certificato. Ledger
r00851,500h storico distinto dalla wall alla ricezione aggiornata nella
receipt. Retained2,384GB autore più reviewer/supervisore, misura attuale
circoscritta senza doppio conteggio; amministrazione non tutta misurata.
Non domandare nuova approvazione per ogni comando; erroriordinari e recuperi
argv/log/permessi nella chat, escalation tool se necessaria mantiene isolamento
e rispetta un rifiuto effettivo. Al limite sostanziale reale conservare
checkpoint e richiedere disposizione necessaria; continuare lavoro indipendente
consentito, nessuna sanatoria o estensione implicita.

## Ciclo da proseguire

Ricevi la consegna **completa** r009, controlla byte/prove dipendenti e delta,
aggiorna checkpoint/stato/eventi/artefatti/changelog. Nuovo snapshot comune
e due NUOVE chat indipendenti ChatGPT/Claude con stessi criteri, chiusure
F39–F42 e regressioni delle classi modificate. Controlla anche le invarianti
condivise e l’indipendenza degli oracoli, non solo i payload accolti; non
espandere a parser universale. Piccole sonde reviewer con gate/R nei limiti,
nessuna review simulata o delegata. Arbitra report reali con raggiungibilità,
requisito/impatto/attribuzione, senza cambiare severità per ottenere GO.

GO/GO più GO supervisore: prepara archivio/chiusura e comandiGit manuali;
NO_GO: correzioni minime accolte nello stesso incremento. Non promettere
che il prossimo giro sarà l’ultimo. Nessuna nuova V generale, motore scelto,
fase2/ML/download/cleanup. Pulizia a001 già ricevuta, non ripeterla.
