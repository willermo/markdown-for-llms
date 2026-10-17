# Contratti del prodotto sperimentale

Fixture `reference_contract.json`: venti esempi sintetici E01 del supervisore a004,
creati per questo progetto e redistribuiti con la licenza del repository. Sono
attese del contratto operativo, non output del motore né certificazioni di fedeltà.
Gli asset opachi nei test verificano conservazione/inventario; il decoder raster
è collaudato separatamente. C06/C07 reali e le ricevute sono nella run sotto temp.

I test importano il package installato: eseguire con l'interprete managed della
venv, `-I -B`, dopo la preparazione S/B/I corrente. Nessun import ML/download.
Il TestClient ASGI necessita delle primitive locali di socket/event loop del
runner; il divieto di rete esterna si applica separatamente.
