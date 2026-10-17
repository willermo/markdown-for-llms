# Chiusura a002 — consegna sperimentale accettata

Disponibilità17ott2026: alcune prove e copie storiche sono state rimosse
dalla pulizia autorizzata; i riferimenti successivi descrivono la consegna
storica. Corpus/gold e casi decisivi restano; nessuna nuova firma umana
o chiusura del debito. [Dettagli](../../development/cleanup-2026-10-17.md).

A002 chiusa il2026-10-12 per accettazione esplicita del perimetro ridotto, con debito aperto. Reader r014 sperimentale; ClaudeNO_GO statico e reviewGPTincompleta conservati, nessun GOtecnico retroattivo. Corpus/raw/prove archiviati, parser mantenuto adottato come direzione (ADR0008), implementazione futura obbligatoria. Primo incremento fase1 concluso nel perimetro accettato; confronto/scelta motori e fase1 complessiva ancora aperti. Nessuna integrazione o UI dichiarata.

La disposizione utente è conservata nell’archivio, insieme alla decisione di
chiusura; essa supera per questa consegna il requisito di arbitrato tecnico
finale positivo, senza cambiare le review o il protocollo delle run future.
Non è un report di GO prodotto. I problemi restano da risolvere secondo
[debito](../../benchmarks/a002-residual-debt.md) e
[ADR0008](../../decisions/0008-maintained-parser-and-provisional-benchmark.md).
Misure delle classi non affidabili provvisorie, nessuna garanzia universale
diCOMPLETE/UNSAFE. L’ispezione umana usa sorgenti/gold/output originali.

Prove ricevute:157/157,159input,3156+310subtest,12CLI,64ricalcoli,zero delta
numerici/gate,16diagnosiEPUB eotto metadataTeX attribuiti. H12FAIL previsti,
56EXACT/8TRIVIAL_ZERO e4PASS/28FAIL per set restano tali. Questi sono risultati
autori/reviewer ricevuti, non test rieseguiti nella chiusura. Nessun nuovo motore.

Il parser mantenuto risolve la responsabilità sintattica da convalidare; non
certifica da solo una conversione corretta. Fase1 deve ancora confrontare i
motori, quindi non si salta automaticamente a fase2. La prima applicazione
valutabile è prevista in fase3, non già disponibile. Nessun commit/merge/deploy.
