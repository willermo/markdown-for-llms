# Convertire il primo PDF locale

Per una persona che usa il prodotto sperimentale I1 su Linux x86_64 e CPU.
Sono necessari package installato con extra `app,marker-cpu`, CPython managed
3.12.13, Bubblewrap e i checkpoint/font preparati. La [guida di preparazione](../how-to/prodotto-locale.md)
spiega questi prerequisiti. Non sono previsti invii remoti, revisione AI o GPU.

Copia [.env.example](../../.env.example) in un file di lavoro e imposta percorsi
assoluti `MDF_DATA_ROOT` e `MDF_MODEL_ROOT`. Scegli una directory dati nuova,
privata e su filesystem locale. Non usare una directory di output della pipeline
legacy. Se la porta 8765 è occupata, scegli un’altra `MDF_PORT` prima di creare
lo storage. API, worker e CLI devono leggere lo stesso file e ambiente.

In due terminali, usando il Python della venv installata:

```bash
"$RUN_APP_PY" -I -B -m markdown_for_llms.cli --env-file /percorso/prodotto.env serve
"$RUN_APP_PY" -I -B -m markdown_for_llms.cli --env-file /percorso/prodotto.env worker
```

Apri `http://127.0.0.1:8765` (o la porta configurata). Attendi «Worker locale
pronto», scegli un PDF e premi **Converti**. La coda mostra fase e identificativo;
non mostra percentuali stimate. Ricaricare la pagina conserva la selezione.
L’archivio rimane disponibile anche quando il worker non è avviato.

Quando la conversione termina, leggi il **Markdown originale**. Il pannello è
testuale: HTML, collegamenti e LaTeX non vengono eseguiti. Per testi lunghi usa
«Leggi porzione successiva»; il contatore indica quando hai consultato tutto.
La galleria mostra solo PNG/JPEG che hanno superato la verifica raster separata.
Scarica **sorgente** e **Markdown e asset (ZIP)**. Lo ZIP contiene `source/document.pdf`,
`result/document.md`, le immagini nelle posizioni relative originali e `manifest.json`.

Controlla il risultato rispetto al PDF, soprattutto formule, numeri, ordine,
tabelle e figure. «Terminata» e «bundle disponibile» non significano fedeltà
verificata. Un risultato `unchecked` resta scaricabile quando i file sono integri,
con motivi e ambito visibili; un asset noto mancante o alterato blocca lo ZIP.

**Annulla** ferma il lavoro in coda o il processo attivo. **Ripeti** crea una nuova
esecuzione con altro identificativo, mantenendo sorgente e tentativi precedenti.
Dopo un arresto del worker, riavvialo: riconcilia le pubblicazioni e marca le
conversioni rimaste senza proprietario come interrotte. La ripetizione è esplicita.

Per usare la CLI sullo stesso archivio:

```bash
"$RUN_APP_PY" -I -B -m markdown_for_llms.cli --env-file /percorso/prodotto.env convert esempio.pdf --wait
"$RUN_APP_PY" -I -B -m markdown_for_llms.cli --env-file /percorso/prodotto.env jobs
"$RUN_APP_PY" -I -B -m markdown_for_llms.cli --env-file /percorso/prodotto.env download ID --output risultato.zip
```

`--wait` richiede il worker separato già attivo. Un batch con errori restituisce
exit nonzero; leggi gli esiti per documento. Non rimuovere manualmente membri
del DB o dello storage per liberare spazio: I1 non include cancellazione
selettiva dell’archivio. [Reference](../reference/prodotto-locale.md),
[limiti](../explanation/prodotto-sperimentale.md).
