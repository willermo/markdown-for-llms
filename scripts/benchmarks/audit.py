"""Audit bundle inerte gold-blind, nessun fetch o lettura di target evasivi."""
from pathlib import Path
from data import digest_bytes
from observations import observe
from safety import regular, tree_inventory, local_uri, png, svg, Unsafe


def audit(root):
    root=Path(root);result={'schema_version':1,'status':'UNAVAILABLE','reason':'RAW_MISSING','assets':[],'references':[],'inventory':[],'raw_sha256':None}
    if not root.exists():return result
    inventory=tree_inventory(root);result['inventory']=inventory
    if any(r['kind'] in ('symlink','special','unsafe_regular') for r in inventory):
        result.update(status='INCOMPLETE',reason='UNSAFE_REFERENCE');return result
    try:raw=regular(root,'document.md')
    except FileNotFoundError:return result
    result['raw_sha256']=digest_bytes(raw)
    if not raw.strip():result.update(status='INCOMPLETE',reason='EMPTY');return result
    try:observed=observe(raw)
    except UnicodeError as exc:result.update(status='INCOMPLETE',reason='INVALID_ENCODING',detail=str(exc));return result
    except Unsafe as exc:result.update(status='INCOMPLETE',reason='UNSAFE_REFERENCE',detail=str(exc));return result
    result['observations_sha256']=__import__('data').digest(observed)
    refs=[r for n in observed['nodes'] for r in n.get('references',[])]
    missing=[]
    for r in refs:
        uri=r['target'];record={'kind':r['kind'],'uri':uri}
        try:
            path=None if r['kind'] in ('ref','note_call') else local_uri(uri,allow_external=r['kind']=='link')
            # TeX/note IDs are semantic references, not resource paths.
            if r['kind'] in ('ref','note_call'):path=None
            if path is not None:
                content=regular(root,path);record.update(path=path,sha256=digest_bytes(content),bytes=len(content))
                if r['kind']=='image':
                    if path.endswith('.png'):record['dimensions']=list(png(content))
                    elif path.endswith('.svg'):svg(content);record['vector_content']='passive SVG; pixel equivalence unavailable'
                    else:record['content_verification']='UNAVAILABLE: image decoder not in finite profile'
            if r['kind']=='image':result['assets'].append(record)
            else:result['references'].append(record)
        except FileNotFoundError:missing.append(uri)
        except Unsafe as exc:result.update(status='INCOMPLETE',reason='UNSAFE_REFERENCE',detail=str(exc));return result
    resource_unknown=[u for n in observed['nodes'] for u in n.get('unknown',[]) if u.get('resource_syntax')]
    if resource_unknown:result.update(status='INCOMPLETE',reason='UNSUPPORTED_RESOURCE_SYNTAX',unknown=resource_unknown)
    elif missing:result.update(status='INCOMPLETE',reason='MISSING_RESOURCE',missing=missing)
    else:result.update(status='COMPLETE',reason=None)
    return result
