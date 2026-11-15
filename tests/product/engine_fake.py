"""A fake of `engine.convert` for the product tests: it writes what the engine writes into
`out` (document.md, assets, diagnostics, engine.json) without converting anything, after
handing `events` to the observer. The engine itself is tested in tests/unit/engine."""
from pathlib import Path

from markdown_for_llms.files import identify, write, write_json


def fake_convert(document=b'converted', assets=None, problems=None, code=None, events=(), calls=None,
                 before=None, counts=None):
    def convert(source, out, request=None, limits=None, work=None, observer=None, name=None, model_root=None,
                datalab_key=None):
        if calls is not None:
            calls.append(dict(source=Path(source), request=request, limits=limits, work=work, name=name,
                              model_root=model_root, datalab_key=datalab_key))
        stops = [observer(event) for event in events] if observer else []
        if before is not None:
            before(observer)
        out = Path(out)
        out.mkdir(parents=True)
        result = {"schema": "mdf-engine-result/1", "attempt": (request or {}).get("attempt"),
                  "route": {"converter": "marker", "family": "pages", "chosen_by": "default", "reason": "matrix-default",
                            "steps": []},
                  "components": {"marker": {"marker-pdf": "2.2.0"}}, "timings": {"total_seconds": 1.0},
                  "limits": [], "stops": stops}
        failed = code or next((s for s in stops if s), None)
        if failed:
            result.update(status="failed", error={"code": failed, "fatal": False}, inventory_complete=False)
            write_json(out, 'engine.json', result)
            return result
        write(out, 'document.md', document)
        for name_, data in (assets or {}).items():
            (out / name_).parent.mkdir(parents=True, exist_ok=True)
            write(out, name_, data)
        (out / 'diagnostics').mkdir()
        write(out, 'diagnostics/marker-1.log', b'log of the child')
        listed = list(problems or [])
        totals = counts or {s: sum(1 for p in listed if p['severity'] == s) for s in ('incomplete', 'warning')}
        inventory = [identify(out, n) for n in ['document.md', *(assets or {}), 'diagnostics/marker-1.log']]
        result.update(status="complete", problems=listed, problem_counts=totals,
                      problem_codes={}, inventory=inventory, inventory_complete=True)
        write_json(out, 'engine.json', result)
        return result
    return convert
