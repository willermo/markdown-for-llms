"""The conversion engine of markdown-for-llms (MFL032): one library for the CLI, the worker and
the bench.

It recognizes the format of a file by extension and signature (`formats`), chooses the route of
the conversion matrix (`routes`), runs the converter of that route in its isolation and writes
the output contract (`contract`: `document.md`, `assets/`, `engine.json`). `convert` and
`availability` are in `core`. Importing the package imports nothing else: the API process uses
only `formats` and `routes`, which need the standard library alone.
"""
