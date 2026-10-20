# Contracts of the experimental product

Fixture `reference_contract.json`: twenty synthetic E01 examples of the a004 supervisor, created for this project and redistributed under the licence of the repository. They are expectations of the operational contract, not engine outputs or fidelity certifications. The opaque assets in the tests verify retention and inventory; the raster decoder is tested separately. The real C06/C07 and the receipts are in the run under temp.

The tests import the installed package: run them with the managed interpreter of the venv, `-I -B`, after the current S/B/I preparation. No ML/download import. The ASGI TestClient needs the local socket/event loop primitives of the runner; the ban on the external network applies separately.
