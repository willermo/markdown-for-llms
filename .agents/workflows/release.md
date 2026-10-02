# Workflow — Integrazione e rilascio

**Ingresso:** versione pronta alla revisione o richiesta di integrare/rilasciare.
Per una run è necessario il GO finale valido sullo snapshot e stato
`READY_FOR_MANUAL_INTEGRATION` nel [ciclo supervisionato](implementation-run.md).
L'utente esegue commit, merge, eventuale push e promozione: l'agente prepara i comandi
e ne verifica l'esito comunicato/osservato. Questa procedura non li esegue automaticamente.

1. Verificare criteri della [roadmap](../../documentation/roadmap.md), diff, migrazioni,
   limiti noti e prove sul corpus. Distinguere piattaforme collaudate e non verificate.
2. Verificare package/immagini, configurazione, download dei modelli, persistenza,
   backup e ripristino per i profili dichiarati. Registrare versioni e risultati.
3. Completare README e documentazione Diátaxis, note di migrazione e cambiamenti di
   compatibilità. Preparare un riepilogo revisionabile con comportamento e verifiche.
4. Fornire all'utente comandi per commit e integrazione in `dev`, basati su branch e
   stato Git reali. Se emergono conflitti o variazioni rispetto allo snapshot approvato,
   valutarne l'effetto e riaprire le verifiche/review necessarie prima della promozione.
5. Fornire eventuali comandi di push e promozione da `dev` a `main` nel perimetro del
   rilascio richiesto. Registrare commit/tag reali e distinguere stato locale e remoto.
6. Valutare redeploy, ambiente destinatario e verifica di funzionamento; eseguirlo
   soltanto nel perimetro autorizzato. Completare archivio e pulizia selettiva della
   run come descritto nel protocollo, preservando evidenze e contesto utili.

**Uscita:** stato del rilascio e delle verifiche esplicito, con documentazione coerente.
