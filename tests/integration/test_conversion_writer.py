"""Pandoc writer autonomo e contratti di processo. Nessun gold del corpus."""
from pathlib import Path
import sys
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
from data import descriptor,digest_bytes
from observations import observe
from profile import config
from resources import run
from png_codec import encode
from audit import audit


@pytest.fixture
def environment():
    import os
    return dict(os.environ)


@pytest.mark.parametrize('name,source,assertion',[
 ('P01-P06','"quoted" - −7 NFC Caffè NFD Cafè.\n\n# Explicit {#explicit}\n\n# Automatic\n', 'unicode-and-anchors'),
 ('P02-P04-P05-P10','|A|B|C|\n|---|---|---|\n|0012|−7||\n\n: Caption\n\nInline $x_i$ and escaped \\$7.\n\n$$\nE=mc^2\n$$\n\n````python\n$x_i$ a_b −7\n````\n','typed-content'),
 ('P07-P08-P11-P13','::: {#outer}\n\n[Visible]{#span}\n\n::: \n\nCall[^n]\n\n[^n]: Definition with −7.\n\n```{=html}\n<p>HTML visible <span id="anchor">span</span></p>\n```\n','containers-notes-raw'),
])
def test_real_writer_profile_on_independent_sources(tmp_path,environment,name,source,assertion):
    cfg=config('docx','pandoc-direct-v1');(tmp_path/'data').mkdir();(tmp_path/'source').mkdir();input=tmp_path/'source/independent.md';input.write_text(source)
    version=run(['/usr/bin/pandoc','--version'],tmp_path,tmp_path/'version',environment,10)
    assert version['adapter_exit']==0 and (tmp_path/'version/stdout.bin').read_text().startswith('pandoc 3.1.3\n')
    # This control input is Markdown with smart disabled explicitly. Real
    # campaign readers remain docx/epub/rtf/latex, without reader extension tune.
    argv=['/usr/bin/pandoc',str(input),'--from=markdown-smart-auto_identifiers',*cfg['ordered_options'][1:],'--output=document.md']
    result=run(argv,tmp_path,tmp_path/'process',environment,20)
    assert result['adapter_exit']==0 and result['stop_reason'] is None
    assert descriptor('/usr/bin/pandoc')['sha256']==cfg['pandoc']['sha256']
    raw=(tmp_path/'document.md').read_bytes();observed=observe(raw)
    assert observed['unknown']==0,observed
    if assertion=='unicode-and-anchors':
        assert observed['nodes'][0]['text']=='"quoted" - −7 NFC Caffè NFD Cafè.'
        headings=[n for n in observed['nodes'] if n['type']=='heading'];assert headings[0]['anchors']==['explicit'];assert headings[1]['anchors']==[]
    elif assertion=='typed-content':
        table=next(n for n in observed['nodes'] if n['type']=='table');assert table['cells']==[['A','B','C'],['0012','−7','']] and table['caption']=='Caption'
        assert next(n for n in observed['nodes'] if n['type']=='code')['text']=='$x_i$ a_b −7\n'
        assert sum(len(n['formulas']) for n in observed['nodes'])==2
        assert all(f['math']['fully_supported'] for n in observed['nodes'] for f in n['formulas'])
    else:
        assert any(n['text']=='Visible' and 'span' in n['anchors'] for n in observed['nodes'])
        assert any(n['type']=='note' and n['text']=='Definition with −7.' for n in observed['nodes'])
        assert any(n['text']=='HTML visible span' and 'anchor' in n['anchors'] for n in observed['nodes'])


def test_png_bundle_positive_and_altered_content_discriminating(tmp_path):
    # Original synthetic 24px plate, independent of corpus generator.
    (tmp_path/'media').mkdir();asset=encode(24,24,3,[b'\x20\x40\x80'*24 for _ in range(24)]);(tmp_path/'media/figure.png').write_bytes(asset);(tmp_path/'document.md').write_text('![Original plate](media/figure.png)\n')
    result=audit(tmp_path);assert result['status']=='COMPLETE';assert result['assets'][0]['sha256']==digest_bytes(asset)
    (tmp_path/'media/figure.png').write_bytes(encode(24,24,3,[b'\x80\x40\x20'*24 for _ in range(24)]));after=audit(tmp_path);assert after['status']=='COMPLETE' and after['assets'][0]['sha256']!=digest_bytes(asset)


def test_empty_anchor_binding_and_display_multiline_are_literal_inputs():
    # Standalone fixture, not fitted to a corpus engine output.
    observed=observe(b'[]{#landmark}\n\n# Visible heading\n\n$$\nE=mc^2\n$$\n')
    assert observed['nodes'][0]['type']=='anchor';assert observed['nodes'][1]['anchors']==['landmark'];assert observed['nodes'][1]['raw']=='# Visible heading\n'
    assert observed['nodes'][2]['type']=='formula';assert observed['nodes'][2]['formulas'][0]['raw']=='\nE=mc^2\n'
