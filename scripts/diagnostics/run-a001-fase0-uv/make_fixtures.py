#!/usr/bin/env python3
"""Genera una sola volta F1–F6 sintetiche; non converte e non misura token.

L'eventuale tuning di F6 richiede una nuova versione e un nuovo freeze baseline.
Il generatore resta nel repository per rendere verificabile la provenienza.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib


def png_chunk(name, data):
    return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)


def fixtures():
    stable = """# F1 Documento sintetico italiano

## Primo esperimento

Il campione contiene Unicode: città, perché, più, quantità. I numeri esatti
sono 123,45, -7 e 2026. La relazione inline è $E=mc^2$. Il riferimento [1]
rimane in questa posizione dopo la descrizione del primo esperimento.

La misura riguarda una sequenza di parole semplici. Ogni frase offre contenuto
sufficiente per la validazione legacy senza richiedere servizi remoti.
Gli stessi caratteri devono essere confrontati dopo ogni fase. Le soglie del
validatore non sostituiscono il confronto del testo effettivo.

## Secondo esperimento

Questo paragrafo viene dopo il primo esperimento e conserva tale ordine.
La temperatura osservata è 18,25 e la differenza è -7. La tabella seguente
contiene dati sintetici che non rappresentano misure di un motore OCR.

| Caso | Anno | Valore |
| --- | --- | --- |
| A | 2026 | 123,45 |
| B | 2025 | -7 |

## Riferimenti

[1] Autore sintetico. Titolo del riferimento. Edizione 2026.
"""
    sensitive = r"""# F2 Formule, codice e riferimenti

## Matematica

La formula esatta è $x_{i}^{2}+\frac{a}{b}=7$ e il valore firmato è -7.
La sequenza di righe seguente è parte del contenuto da inventariare:

$$
\begin{aligned}
a_{1} &= 123,45 \\
b_{2} &= \sum_{i=1}^{n} x_i
\end{aligned}
$$

## Codice

```python
record = {"anno": 2026, "numero": -7}
comparazione = "a < b e c > d"
nome = "D[AN] S. K[ENNEDY]"
```

Il collegamento [riferimento locale](#bibliografia) non usa HTTP.
La figura è ![Figura sintetica](assets/f5.png), con didascalia e numero 2.

## Bibliografia

[1] Autore sintetico, Titolo sintetico, 2026, pagina 123.

Il campione presenta volutamente graffe, indici e delimitatori che il cleaning
legacy può rimuovere. Queste perdite devono restare visibili nell'inventario.
Una nuova versione non può normalizzare il testo per nascondere una differenza.
Il documento è interamente artificiale e non contiene dati o documenti privati.
"""
    html = """<!doctype html>
<html lang="it"><head><meta charset="utf-8"><title>F4 sintetico</title></head>
<body><h1>Primo esperimento</h1><p>Città, perché e quantità: 123,45, -7 e 2026.
La relazione testuale è E = mc². Il riferimento [1] segue il primo risultato.</p>
<h2>Secondo esperimento</h2><p>Questa sezione deve restare dopo la prima.</p>
<table><thead><tr><th>Caso</th><th>Valore</th></tr></thead><tbody>
<tr><td>A</td><td>123,45</td></tr><tr><td>B</td><td>-7</td></tr></tbody></table>
<p id="riferimento">[1] Autore sintetico, Titolo, 2026.</p></body></html>
"""
    asset = """# F5 Asset sintetico

![Figura 5: quadrato rosso e blu](assets/f5.png)

La didascalia associa il numero 5 all'immagine sintetica. Il file PNG deve
essere identificato separatamente dal riferimento Markdown. Il cleaning legacy
non copia gli asset e può rimuovere il riferimento. Questo limite viene
inventariato senza dichiarare un bundle completo. I dati sono 2026 e -7.

## Riferimenti

[1] Disegno sintetico generato con libreria standard, nessuna immagine esterna.
"""
    paragraphs = ["# F6 Sequenza multi-chunk sintetica\n"]
    for index in range(1, 161):
        if (index - 1) % 10 == 0:
            paragraphs.append(f"## Sezione {(index - 1) // 10 + 1:02d}\n")
        paragraphs.append(
            f"Paragrafo P{index:04d}. Il campione numero {index:04d} conserva testo italiano e ASCII. "
            f"Ogni parola resta nella stessa sequenza per verificare contenuto ordine e confini. "
            f"Anno 2026 valore -7 misura 123,45 riferimento [{index:04d}]. "
            "Questa frase conclude il paragrafo sintetico con dati distinti e riproducibili.\n"
        )
    long_text = "\n".join(paragraphs)
    # PNG RGBA 2x2, scanline senza filtro; nessun asset recuperato da terzi.
    scanlines = b"\x00\xff\x00\x00\xff\x00\x00\xff\xff" * 2
    png = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(scanlines, 9)) + png_chunk(b"IEND", b"")
    config = {
        "directories": {"source_pdfs": "source_pdfs", "source_documents": "source_documents",
                        "converted": "converted_markdown", "cleaned": "cleaned_markdown",
                        "validated": "validated_markdown", "chunked": "chunked_markdown", "logs": "pipeline_logs"},
        "validation_thresholds": {"min_content_length": 100, "max_content_length": 10000000, "min_words": 50,
                                  "max_artifact_ratio": 0.05, "min_readability_score": 20, "min_structure_score": 50},
        "cleaning_settings": {"aggressive_cleaning": True, "preserve_tables": True, "min_line_length": 3,
                              "remove_html_tags": True, "normalize_whitespace": True},
        "chunking_settings": {"target_llm": "custom", "chunk_size": 1000, "overlap": 100,
                              "min_chunk_size": 1000, "max_chunk_size": 8000, "strategy": "semantic"},
        "conversion_settings": {"document_converter": "pandoc", "supported_document_formats": ["html"],
                                "marker_document_formats": [], "use_llm": False},
        "pipeline_settings": {"skip_existing": False, "max_workers": 1, "log_level": "INFO"},
    }
    return {"f1-stable.md": stable.encode(), "f2-sensitive.md": sensitive.encode(),
            "f3-cleaned-stable.md": stable.encode(), "f3-cleaned-sensitive.md": sensitive.encode(),
            "f4-pandoc.html": html.encode(), "f5-asset.md": asset.encode(), "assets/f5.png": png,
            "f6-multichunk.md": long_text.encode(), "fixture-config.json": (json.dumps(config, indent=2) + "\n").encode(),
            "dotenv-1200-80.txt": b"CHUNK_SIZE=1200\nOVERLAP_SIZE=80\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(path.is_symlink() for path in (args.output, *args.output.parents)):
        parser.error("Output symlink non ammesso")
    args.output.mkdir(parents=True, exist_ok=False)
    entries = []
    for name, data in fixtures().items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        entries.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    provenance = {"schema": 1, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "author": "Codex/OpenAI, famiglia GPT-6", "date_context": "2026-10-03", "origin": "sintetica, generata nel repository",
                  "rights": "fixture originali per il repository con licenza MIT; nessun contenuto di terzi",
                  "files": entries, "F6": {"paragraphs": 160, "section_count": 16, "target_llm": "custom", "chunk_size": 1000,
                                          "overlap": 100, "token_count": None, "multi_chunk_status": "NON_ESEGUITA, attende R e stage"},
                  "F3": "Copie dirette F1/F2 per validation; nessuna normalizzazione",
                  "limits": ["Soglie configurate e costanti effettive legacy saranno caratterizzate in V0.",
                             "F5 verifica riferimento e hash; copia asset non garantita dal legacy."]}
    with (args.output / "provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print(f"Create {len(entries)} fixture/input sintetici in {args.output}")


if __name__ == "__main__":
    main()
