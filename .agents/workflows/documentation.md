# Workflow — Documentazione

**Ingresso:** funzionalità disponibile o documentazione esistente da correggere.

1. Stabilire bisogno del lettore e categoria Diátaxis. Per scelte ancora in discussione
   usare `documentation/`; per istruzioni operative usare `docs/`.
2. Leggere codice, schema e test pertinenti, distinguendo il sistema legacy da quello
   nuovo. Indicare esplicitamente gli eventuali contenuti pianificati.
3. Scrivere passi, riferimenti o spiegazioni adatti al bisogno. Mantenere credenziali
   fittizie negli esempi e non usare documenti privati come materiale dei tutorial.
4. Provare i comandi quando applicabile, oppure indicare la verifica mancante.
   Controllare i collegamenti relativi e l'indice della sezione. Usare `git diff --check`
   e ispezionare anche i file nuovi non ancora tracciati.
5. Aggiornare il README solo quanto serve per rendere raggiungibile il contenuto.
   Per modifiche puramente editoriali non avviare benchmark OCR o test con modelli.

**Uscita:** documentazione utilizzabile, aderente alla versione descritta.
