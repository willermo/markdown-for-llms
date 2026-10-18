"""Promozione amministrativa di due generazioni, senza eseguire motori."""
import argparse
from copy import deepcopy
from pathlib import Path
import re
import shutil

from data import descriptor, digest_bytes, load, provenance, write_new, counts


def promote(first, second, destination, external):
    if any((destination/n).exists() for n in ('inputs','assets','gold')):
        raise ValueError('Promozione già presente')
    differences = []
    comparison = []
    for name in ('inputs','assets','gold','sources','renders'):
        ap = {p.relative_to(first).as_posix():p for p in (first/name).rglob('*') if p.is_file()}
        bp = {p.relative_to(second).as_posix():p for p in (second/name).rglob('*') if p.is_file()}
        if set(ap) != set(bp):
            raise ValueError('File inventory diverge')
        for key, p in sorted(ap.items()):
            q = bp[key]
            same = p.read_bytes() == q.read_bytes()
            comparison.append({'path':key,'first':provenance(p),'second':provenance(q),'byte_identical':same})
            if not same:
                differences.append(key)
    if any(not (n.startswith('renders/') and n.endswith('.pdf')) for n in differences):
        raise ValueError('Byte determinism FAIL outside LibreOffice PDF metadata')
    # All fixed page PNGs are byte-identical. Keep both PDF hashes, fonts and
    # actual page count receipts; do not rewrite PDF IDs/timestamps.
    for name in ('inputs','assets','gold'):
        shutil.copytree(first/name,destination/name)
    for path in (first/'sources').rglob('*'):
        if path.is_file():
            target=destination/path.relative_to(first)
            if target.exists():
                if target.read_bytes() == path.read_bytes():
                    continue
                raise ValueError('Source collision with different bytes')
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    shutil.copytree(first/'derivations',destination/'sources/derivations')
    shutil.copytree(first/'renders',destination/'assets/renders')
    documents = deepcopy(load(first/'generation.json')['documents'])
    for d in documents:
        g=load(destination/d['gold'])
        for b in g['blocks']:
            loc=b['locator']
            for key in ('raster_parent','progenitor_pdf'):
                if loc.get(key,'').startswith('derivations/'):
                    loc[key]='sources/'+loc[key]
        # Only provenance paths change; annotations/canonical bytes untouched.
        (destination/d['gold']).write_bytes(__import__('data').canonical(g)+b'\n')
        d['gold_sha256']=descriptor(destination/d['gold'])['sha256']
        d['blocks']=g['blocks']
        d['denominators']=counts(g)
        literals=[b['text'] for b in g['blocks']]
        literals += [b['caption'] for b in g['blocks'] if b.get('caption') and b['caption'] != b['text']]
        literals += [c for b in g['blocks'] for row in b.get('cells',[]) for c in row]
        literals += [ref['text'] for b in g['blocks'] for ref in b.get('references',[])]
        d['word_count']=sum(len(re.findall(r'\w+',s)) for s in literals)
        d['renders']=[p.relative_to(destination).as_posix() for p in sorted((destination/'assets/renders'/d['document_id']).glob('*.png'))]
        d.pop('render',None)
        if d.get('render_pdf','').startswith('renders/'):
            d['render_pdf']='assets/'+d['render_pdf']
    fonts=[r for r in load(external)['files'] if '/fonts/' in r['path'] or '/fonts-' in r['path']]
    notices=destination/'assets/font-notices';notices.mkdir()
    for name,origin in [('dejavu.txt','/usr/share/doc/fonts-dejavu-core/copyright'),('liberation.txt','/usr/share/doc/fonts-liberation/copyright'),('opensymbol.txt','/usr/share/doc/fonts-opensymbol/copyright'),('MPL-2.0.txt','/usr/share/common-licenses/MPL-2.0'),('Apache-2.0.txt','/usr/share/common-licenses/Apache-2.0')]:
        shutil.copyfile(origin,notices/name)
    manifest={'schema_version':1,'corpus_id':'a002-synthetic-v1','revision':1,'documents':documents,'word_count':sum(d['word_count'] for d in documents),'word_count_method':'Unicode word runs of authored block text, captions once, cells and reference labels; formulas excluded','pdf_pages':sum(d.get('pages',0) for d in documents if d['format']=='pdf'),'fixed_renders':sum(len(d['renders']) for d in documents),'provenance':{'spec':provenance(destination/'sources/documents.json'),'generator':provenance(Path(__file__).with_name('generate_corpus.py')),'generations':[provenance(first/'generation.json'),provenance(second/'generation.json')],'comparison':comparison,'content_reproducibility':'Canonical files byte-identical; all page PNGs byte-identical. LibreOffice PDF IDs/timestamps not rewritten; pages/fonts/render receipts retained.'},'fonts':fonts,'verification':'PROVISIONAL: M on promoted locators and real human V required'}
    manifest['revision']=load(destination/'sources/documents.json')['corpus_revision']
    write_new(destination/'manifest.json',manifest)
    (destination/'NOTICE').write_text('Testi, annotazioni, sorgenti e figure originali sintetici: MIT, copyright 2026 autori del progetto Markdown for LLMs. Vedere LICENSE del repository. Nessun dato privato.\nFont incorporati: DejaVu Sans/Mono e Bold (licenza DejaVu/Bitstream, assets/font-notices/dejavu.txt); Liberation Sans/Serif (SIL OFL 1.1, liberation.txt); OpenSymbol (MPL 2.0 / Apache 2.0, opensymbol.txt). Questi font non sono MIT. Hash dei file locali effettivi nel manifest. LibreOffice può scegliere i fallback identificati nei pdffonts conservati; i PDF e i PNG non attestano un font runtime futuro.\n',encoding='utf-8')
    (destination/'README.md').write_text('# Corpus sintetico di conversione\n\nDieci canonici originali con gold manuali per documento. `manifest.json` identifica input, gold, asset, nove pagine PDF e dodici render fissi. EPUB è reflowable; TeX ha un main e due include e non è compilato. Le derivazioni raster sono sotto `sources/derivations`.\n\nIl gold rimane PROVISIONAL finché M e la verifica umana V non accettano esattamente gli hash correnti. I render e la presenza del file non certificano una conversione. `controls/` contiene controlli autonomi del misuratore, non output di motori. Licenze e font in `NOTICE`.\n',encoding='utf-8')
    print('CORPUS_PROMOTED',manifest['word_count'],manifest['fixed_renders'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first',required=True,type=Path);p.add_argument('--second',required=True,type=Path)
    p.add_argument('--destination',required=True,type=Path);p.add_argument('--external',required=True,type=Path)
    a=p.parse_args();promote(a.first,a.second,a.destination,a.external)


if __name__=='__main__':main()
