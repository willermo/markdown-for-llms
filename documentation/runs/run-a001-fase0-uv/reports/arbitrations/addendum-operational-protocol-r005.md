# Addendum operativo r005 — ammissione costi managed/lock e ricezione core

2026-10-06. Supervisore; richiesta concreta del mandato36, GO tecnico piano r003
nei limiti. Nessun nuovo pin, motore o backend, nessun GO finale sul codice.

## Decisione

Ricevuti codice core/statiche/test/docs e due richieste. Baseline-s006 ha input
presenti: freeze prima del comando ufficiale R→S→V0. I PASS R preliminari restano
legati al wrapper sondato, che è stato modificato; R ufficiale va ripetuta.
53 test stdlib PASS d'autore e Compose parser non attestano convalida applicativa.

La richiesta acquisizioni r002 stima304MiB incrementali e72MiB di rete, più gli
80MiB di riserve. Il budget500/stop384MiB del recupero non la copre. Ammetto per
questa tranche un ledger **cumulativo run + .venv-python**, max1GiB/stop896MiB,
con storico/cache/tmp/evidenze compresi e riserve80MiB ancora conteggiate. Non
si sposta lo storage per nasconderlo né si cambia alcuna receipt precedente.
Misura propria comprende blocchi delle directory; la misura d'autore dei soli
file non è chiamata equivalente. La stima cumulativa è circa671MiB, su oltre
151GiB liberi all'ingresso; libero minimo1GiB. Quote incrementali304MiB e
log1MiB/stream,JSON8MiB,file32MiB restano; monitor periodico non quota atomica.
Il pool più ampio è un'ammissione prospettica circoscritta, non un costo sostenuto.

L'ammissione riguarda soltanto CPython3.12.13 GNU/build20260310/catalogue checksum,
metadata/lock universale no-build, lock-check offline e setuptools84 wheel-only
in ambiente backend nuovo. Niente runtime ML/native completo, product sync/build,
Docker, modelli/font/inferenza o benchmark remoti. V7/V8 restano essenziali con
mandato pesante distinto; V10/V11 escluse. S/B/I/E e V0–V9 non cambiano.

## Correzione concreta della richiesta

La richiesta d'autore r002 rimane immutata/not-ready alla sua consegna. La scope
supervisore e request package-s008 sono nuovi input identificati. Ambiente
UV_PYTHON_DOWNLOADS=manual **solo** per python install esplicito; never nei passi
successivi, come --no-python-downloads. Il never unico della proposta avrebbe
vietato anche l'acquisizione richiesta. Fonti: [setting uv](https://docs.astral.sh/uv/reference/settings/#python-downloads)
e [catalogo pinned](https://raw.githubusercontent.com/astral-sh/uv/0.10.10/crates/uv-python/download-metadata.json).
Nessuna rimozione della guardia persistente manual nel progetto.

Ambiente chiuso, retries HTTP0/download concorrenti1, configurazioni esterne
verificate assenti/immutate; sorgenti copiati con hash. Checksum/origine/versione
managed prima del resolver; checksum/inventario wheel backend e startup prima
del suo uso. Metadata PEP658/range o piccole wheel entro quote; fallback a payload
ML/native pesante o metadata dinamici non ammessi ⇒ sospendere l'operazione,
conservare errore/parziale. Non fabbricare hash/backend/grafo o abbassare i gate.
La stima rete non è una quota atomica attestata: riportare osservabilità e misure,
non dichiarare byte di traffico che il tool non ha misurato.

Entrambi i freeze vengono prodotti in questa ricezione, prima di attività reali,
sullo stesso codice stabile. Package-s008 include soltanto input già esistenti
più manifest baseline già creato; managed, lock, receipt e backend futuri sono
output. Non è tests-s001 o il prodotto convalidato; lock resta nella copia finché
ricevuto. Una modifica tracciata dei sorgenti invalida il freeze pertinente.

La baseline conserva i suoi limiti500/384 nel comando iniziale; dopo quel comando,
per la tranche acquisizione vale il nuovo ledger cumulativo. Il main storico di
check_preserved contiene il vecchio budget: non modificarlo o scambiarne l'uscita
per la nuova ammissione. Riusa la sua funzione verify per sola integrità s007,
poi calcola separatamente il ledger della scope package-s008. Nessuna reinstallazione
baseline, pulizia del target o trasferimento di PASS.

Prossimo implementatore: un mandato37 per baseline e acquisizioni ammesse, con fix
ordinari dei propri launcher nella stessa chat. Scostamenti sostanziali sospendono
solo attività dipendenti. Nessuna nuova pianificazione o consegna di sola sintassi.
Git manuale e doppie review indipendenti/arbitrato finale ancora necessari.
