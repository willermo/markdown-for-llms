# Workflow — Integrazione e rilascio

**Ingresso:** versione pronta alla revisione o richiesta di integrare/rilasciare.
Questa procedura descrive il percorso futuro; non autorizza da sola push, merge o release.

1. Verificare criteri della [roadmap](../../documentation/roadmap.md), diff, migrazioni,
   limiti noti e prove sul corpus. Distinguere piattaforme collaudate e non verificate.
2. Verificare package/immagini, configurazione, download dei modelli, persistenza,
   backup e ripristino per i profili dichiarati. Registrare versioni e risultati.
3. Completare README e documentazione Diátaxis, note di migrazione e cambiamenti di
   compatibilità. Preparare un riepilogo revisionabile con comportamento e verifiche.
4. Integrare il feature branch in `dev` quando compreso nella richiesta di integrazione.
   Risolvere conflitti preservando il comportamento concordato e rieseguire le sole
   verifiche giustificate dai cambiamenti risultanti.
5. Promuovere la versione verificata da `dev` a `main` nell'ambito della richiesta di
   rilascio; registrare versione/tag e istruzioni di ripristino secondo le convenzioni
   che saranno definite per il prodotto. Non presentare una branch locale come pubblicata.

**Uscita:** stato del rilascio e delle verifiche esplicito, con documentazione coerente.
