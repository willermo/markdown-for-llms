# Archivio permanente delle run

Qui si conservano piani, review, arbitrati, evidenze e sintesi utili dopo la pulizia
del contesto locale. Non sono ancora presenti run completate con il nuovo ciclo.

Ogni run archiviata ha una directory `<run-id>/`, un `manifest.md` derivato dal
[template](../development/templates/archive-manifest.md) e sottocartelle per report,
evidenze e note effettivamente conservati. Aggiungere qui il collegamento alla run
solo quando esiste un archivio reale, con stato e risultato di integrazione.

Probe e test riutilizzabili vanno nelle posizioni eseguibili indicate dal
[protocollo](../development/run-lifecycle.md), collegandoli dal manifest. Questa
cartella contiene documentazione ed evidenze, non dipendenze eseguibili del prodotto.

I file di questa cartella sono esclusi dall'impronta del codice del helper di run
per consentire l'archiviazione dei report dopo le review. L'integrità dell'archivio
va verificata separatamente e registrata nel manifest.
