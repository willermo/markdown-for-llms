"""F46–F49: manual finite classes, no oracle from the reader/scorer.
Exact reviewer raw values, followed by documented deterministic transports.
Gold formulas are manual lexical references; unsupported AST is unavailable.
"""
from copy import deepcopy
from pathlib import Path
import shutil,sys,json
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmarks'))
import observations as reader
import audit as auditor
from benchmark import observe_bundle
from metrics import score,evaluate_exit
ASSET=ROOT/'tests/fixtures/conversion-corpus/assets/figure-it.png'
TAIL='\n\n# Visible heading\n\nAfter words.'
BAD='<img src="../outside.png">'
SCRIPT='<script>x</script>'
HOSTS=['paragraph','quote','list','note-tab','cell','heading','caption']
def transport(value,host):
    if host=='cell':return '| h |\n| --- |\n| '+value
    if host=='heading':return '## '+value+' {.c}'
    if host=='caption':return '| h |\n| --- |\n| a |\n\nTable: '+value
    return {'paragraph':value,'quote':'> '+value.replace('\n','\n> '),'list':'- '+value.replace('\n','\n    '),'note-tab':'[^n]: '+value.replace('\n','\n\t')}[host]
def gold(text,host):
    first={'block_id':'p','type':'heading' if host=='heading' else 'note' if host=='note-tab' else 'paragraph','text':text,'page':1}
    if host=='note-tab':first.update(note_id='n',anchor='n')
    if host in ('cell','caption'):
        first={'block_id':'t','type':'table','text':'','header_rows':1,'cells':[['h'],[text if host=='cell' else 'a']],'page':1}
        if host=='caption':first['caption']=text
    return {'document_id':'r011-synthetic','blocks':[first,{'block_id':'h','type':'heading','text':'Visible heading','page':1},{'block_id':'a','type':'paragraph','text':'After words.','page':1}]}
CASES=[{'id': 'H1-list',
  'group': 'H',
  'raw': '- # x {`<img src="../outside.png">}\n\n# Visible heading\n\nAfter words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': '# x {`<img src="../outside.png">}',
                       'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSAFE_REFERENCE',
  'kind': 'unsafe',
  'detail': '../outside.png',
  'logical': '# x {`<img src="../outside.png">}',
  'visits': ['img']},
 {'id': 'H2-list-nodelim',
  'group': 'H',
  'raw': '- # x {<img src="../outside.png">}\n\n# Visible heading\n\nAfter words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 'p', 'type': 'paragraph', 'text': '# x {}', 'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSAFE_REFERENCE',
  'kind': 'unsafe',
  'detail': '../outside.png',
  'logical': '# x {<img src="../outside.png">}',
  'visits': ['img']},
 {'id': 'H3-list-safe',
  'group': 'H',
  'raw': '- # x {`<img src="p.png">}\n\n# Visible heading\n\nAfter words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': '# x {`<img src="p.png">}',
                       'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSUPPORTED_RESOURCE_SYNTAX',
  'kind': 'safe-fallback',
  'detail': None,
  'logical': '# x {`<img src="p.png">}',
  'visits': ['img']},
 {'id': 'H4-quote',
  'group': 'H',
  'raw': '> # x {`<img src="../outside.png">}\n\n# Visible heading\n\nAfter words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': '# x {`<img src="../outside.png">}',
                       'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSAFE_REFERENCE',
  'kind': 'unsafe',
  'detail': '../outside.png',
  'logical': '# x {`<img src="../outside.png">}',
  'visits': ['img']},
 {'id': 'H5-note',
  'group': 'H',
  'raw': '[^n]: # x {`<img src="../outside.png">}\n\n# Visible heading\n\nAfter words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 'p',
                       'type': 'note',
                       'text': '# x {`<img src="../outside.png">}',
                       'page': 1,
                       'note_id': 'n',
                       'anchor': 'n'},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSAFE_REFERENCE',
  'kind': 'unsafe',
  'detail': '../outside.png',
  'logical': '# x {`<img src="../outside.png">}',
  'visits': ['img']},
 {'id': 'H6-cell',
  'group': 'H',
  'raw': '| h | i |\n'
         '| --- | --- |\n'
         '| a | # x {`<img src="../outside.png">}\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 't',
                       'type': 'table',
                       'text': '',
                       'header_rows': 1,
                       'cells': [['h', 'i'], ['a', '# x {`<img src="../outside.png">}']],
                       'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSAFE_REFERENCE',
  'kind': 'unsafe',
  'detail': '../outside.png',
  'logical': '# x {`<img src="../outside.png">}',
  'visits': ['img']},
 {'id': 'H7-heading',
  'group': 'H',
  'raw': '## # x {`<img src="../outside.png">} {.c}\n\n# Visible heading\n\nAfter words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 'p',
                       'type': 'heading',
                       'text': '# x {`<img src="../outside.png">}',
                       'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSAFE_REFERENCE',
  'kind': 'unsafe',
  'detail': '../outside.png',
  'logical': '# x {`<img src="../outside.png">}',
  'visits': ['img']},
 {'id': 'H8-list-math',
  'group': 'H',
  'raw': '- # x {$y <img src="../outside.png">}\n\n# Visible heading\n\nAfter words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 'p', 'type': 'paragraph', 'text': '# x {$y }', 'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSAFE_REFERENCE',
  'kind': 'unsafe',
  'detail': '../outside.png',
  'logical': '# x {$y <img src="../outside.png">}',
  'visits': ['img']},
 {'id': 'H9-list-continuation',
  'group': 'H',
  'raw': '- item\n    # x {`<img src="../outside.png">}\n\n# Visible heading\n\nAfter words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': 'item\n# x {`<img src="../outside.png">}',
                       'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSAFE_REFERENCE',
  'kind': 'unsafe',
  'detail': '../outside.png',
  'logical': 'item\n# x {`<img src="../outside.png">}',
  'visits': ['img']},
 {'id': 'H10-list-script',
  'group': 'H',
  'raw': '- # x {`<script>alert(1)</script>}\n\n# Visible heading\n\nAfter words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': '# x {`<script>alert(1)</script>}',
                       'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSAFE_REFERENCE',
  'kind': 'unsafe',
  'detail': 'script',
  'logical': '# x {`<script>alert(1)</script>}',
  'visits': ['script']},
 {'id': 'H11-caption',
  'group': 'H',
  'raw': '| h |\n'
         '| --- |\n'
         '| a |\n'
         '\n'
         'Table: # x {`<img src="../outside.png">}\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 't',
                       'type': 'table',
                       'text': '',
                       'header_rows': 1,
                       'cells': [['h'], ['a']],
                       'page': 1,
                       'caption': '# x {`<img src="../outside.png">}'},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSAFE_REFERENCE',
  'kind': 'unsafe',
  'detail': '../outside.png',
  'logical': '# x {`<img src="../outside.png">}',
  'visits': ['img']},
 {'id': 'HC1-list-nohash',
  'group': 'H',
  'raw': '- x {`<img src="../outside.png">}\n\n# Visible heading\n\nAfter words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': 'x {`<img src="../outside.png">}',
                       'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': 'UNSAFE_REFERENCE',
  'kind': 'unsafe',
  'detail': '../outside.png',
  'logical': 'x {`<img src="../outside.png">}',
  'visits': ['img']},
 {'id': 'HC2-heading-metadata',
  'group': 'H',
  'raw': '# x {`<img src="../outside.png">}\n\n# Visible heading\n\nAfter words.\n',
  'gold': {'document_id': 'r010-claude',
           'blocks': [{'block_id': 'p', 'type': 'heading', 'text': 'x', 'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'reason': None,
  'kind': 'metadata-control',
  'detail': None},
 {'id': 'math-code-label-paragraph-img',
  'raw': 'È See [Known $x `first\n'
         'second` <img src="../outside.png"> y$ words](https://example.invalid) end.\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': None,
  'formula': 'x `first\nsecond` <img src="../outside.png"> y',
  'refs': ['https://example.invalid'],
  'no_assets': True,
  'logical': 'È See [Known $x `first\n'
             'second` <img src="../outside.png"> y$ words](https://example.invalid) end.',
  'link_accepted': True,
  'direct_reason': None,
  'host': 'paragraph',
  'kind': 'formula',
  'detail': None,
  'visits': [],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': 'È See Known  words end.',
                       'page': 1,
                       'formulas': [{'formula_id': 'f',
                                     'reference_latex': 'x `first\nsecond` <img src="../outside.png"> y',
                                     'symbols_expected': [],
                                     'structure_expected': None}],
                       'references': [{'kind': 'link',
                                       'text': 'Known  words',
                                       'target': 'https://example.invalid'}]},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'È See Known  words end.'},
 {'id': 'math-code-label-paragraph-script',
  'raw': 'È See [Known $x `first\n'
         'second` <script>x</script> y$ words](https://example.invalid) end.\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': None,
  'formula': 'x `first\nsecond` <script>x</script> y',
  'refs': ['https://example.invalid'],
  'no_assets': True,
  'logical': 'È See [Known $x `first\n'
             'second` <script>x</script> y$ words](https://example.invalid) end.',
  'link_accepted': True,
  'direct_reason': None,
  'host': 'paragraph',
  'kind': 'formula',
  'detail': None,
  'visits': [],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': 'È See Known  words end.',
                       'page': 1,
                       'formulas': [{'formula_id': 'f',
                                     'reference_latex': 'x `first\nsecond` <script>x</script> y',
                                     'symbols_expected': [],
                                     'structure_expected': None}],
                       'references': [{'kind': 'link',
                                       'text': 'Known  words',
                                       'target': 'https://example.invalid'}]},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'È See Known  words end.'},
 {'id': 'math-code-label-quote-img',
  'raw': '> È See [Known $x `first\n'
         '> second` <img src="../outside.png"> y$ words](https://example.invalid) end.\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': None,
  'formula': 'x `first\nsecond` <img src="../outside.png"> y',
  'refs': ['https://example.invalid'],
  'no_assets': True,
  'logical': 'È See [Known $x `first\n'
             'second` <img src="../outside.png"> y$ words](https://example.invalid) end.',
  'link_accepted': True,
  'direct_reason': None,
  'host': 'quote',
  'kind': 'formula',
  'detail': None,
  'visits': [],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': 'È See Known  words end.',
                       'page': 1,
                       'formulas': [{'formula_id': 'f',
                                     'reference_latex': 'x `first\nsecond` <img src="../outside.png"> y',
                                     'symbols_expected': [],
                                     'structure_expected': None}],
                       'references': [{'kind': 'link',
                                       'text': 'Known  words',
                                       'target': 'https://example.invalid'}]},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'È See Known  words end.'},
 {'id': 'math-code-label-quote-script',
  'raw': '> È See [Known $x `first\n'
         '> second` <script>x</script> y$ words](https://example.invalid) end.\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': None,
  'formula': 'x `first\nsecond` <script>x</script> y',
  'refs': ['https://example.invalid'],
  'no_assets': True,
  'logical': 'È See [Known $x `first\n'
             'second` <script>x</script> y$ words](https://example.invalid) end.',
  'link_accepted': True,
  'direct_reason': None,
  'host': 'quote',
  'kind': 'formula',
  'detail': None,
  'visits': [],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': 'È See Known  words end.',
                       'page': 1,
                       'formulas': [{'formula_id': 'f',
                                     'reference_latex': 'x `first\nsecond` <script>x</script> y',
                                     'symbols_expected': [],
                                     'structure_expected': None}],
                       'references': [{'kind': 'link',
                                       'text': 'Known  words',
                                       'target': 'https://example.invalid'}]},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'È See Known  words end.'},
 {'id': 'math-code-label-list-img',
  'raw': '- È See [Known $x `first\n'
         '    second` <img src="../outside.png"> y$ words](https://example.invalid) end.\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': None,
  'formula': 'x `first\nsecond` <img src="../outside.png"> y',
  'refs': ['https://example.invalid'],
  'no_assets': True,
  'logical': 'È See [Known $x `first\n'
             'second` <img src="../outside.png"> y$ words](https://example.invalid) end.',
  'link_accepted': True,
  'direct_reason': None,
  'host': 'list',
  'kind': 'formula',
  'detail': None,
  'visits': [],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': 'È See Known  words end.',
                       'page': 1,
                       'formulas': [{'formula_id': 'f',
                                     'reference_latex': 'x `first\nsecond` <img src="../outside.png"> y',
                                     'symbols_expected': [],
                                     'structure_expected': None}],
                       'references': [{'kind': 'link',
                                       'text': 'Known  words',
                                       'target': 'https://example.invalid'}]},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'È See Known  words end.'},
 {'id': 'math-code-label-list-script',
  'raw': '- È See [Known $x `first\n'
         '    second` <script>x</script> y$ words](https://example.invalid) end.\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': None,
  'formula': 'x `first\nsecond` <script>x</script> y',
  'refs': ['https://example.invalid'],
  'no_assets': True,
  'logical': 'È See [Known $x `first\n'
             'second` <script>x</script> y$ words](https://example.invalid) end.',
  'link_accepted': True,
  'direct_reason': None,
  'host': 'list',
  'kind': 'formula',
  'detail': None,
  'visits': [],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': 'È See Known  words end.',
                       'page': 1,
                       'formulas': [{'formula_id': 'f',
                                     'reference_latex': 'x `first\nsecond` <script>x</script> y',
                                     'symbols_expected': [],
                                     'structure_expected': None}],
                       'references': [{'kind': 'link',
                                       'text': 'Known  words',
                                       'target': 'https://example.invalid'}]},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'È See Known  words end.'},
 {'id': 'math-code-label-note-tab-img',
  'raw': '[^n]: È See [Known $x `first\n'
         '\tsecond` <img src="../outside.png"> y$ words](https://example.invalid) end.\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': None,
  'formula': 'x `first\nsecond` <img src="../outside.png"> y',
  'refs': ['https://example.invalid'],
  'no_assets': True,
  'logical': 'È See [Known $x `first\n'
             'second` <img src="../outside.png"> y$ words](https://example.invalid) end.',
  'link_accepted': True,
  'direct_reason': None,
  'host': 'note-tab',
  'kind': 'formula',
  'detail': None,
  'visits': [],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'note',
                       'text': 'È See Known  words end.',
                       'page': 1,
                       'note_id': 'n',
                       'anchor': 'n',
                       'formulas': [{'formula_id': 'f',
                                     'reference_latex': 'x `first\nsecond` <img src="../outside.png"> y',
                                     'symbols_expected': [],
                                     'structure_expected': None}],
                       'references': [{'kind': 'link',
                                       'text': 'Known  words',
                                       'target': 'https://example.invalid'}]},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'È See Known  words end.'},
 {'id': 'math-code-label-note-tab-script',
  'raw': '[^n]: È See [Known $x `first\n'
         '\tsecond` <script>x</script> y$ words](https://example.invalid) end.\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': None,
  'formula': 'x `first\nsecond` <script>x</script> y',
  'refs': ['https://example.invalid'],
  'no_assets': True,
  'logical': 'È See [Known $x `first\n'
             'second` <script>x</script> y$ words](https://example.invalid) end.',
  'link_accepted': True,
  'direct_reason': None,
  'host': 'note-tab',
  'kind': 'formula',
  'detail': None,
  'visits': [],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'note',
                       'text': 'È See Known  words end.',
                       'page': 1,
                       'note_id': 'n',
                       'anchor': 'n',
                       'formulas': [{'formula_id': 'f',
                                     'reference_latex': 'x `first\nsecond` <script>x</script> y',
                                     'symbols_expected': [],
                                     'structure_expected': None}],
                       'references': [{'kind': 'link',
                                       'text': 'Known  words',
                                       'target': 'https://example.invalid'}]},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'È See Known  words end.'},
 {'id': 'HTML-owner-label-first-paragraph-0',
  'raw': 'È See [<span>`<script>x</script>`</span>](https://example.invalid) <img src="../second.png">\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': 'UNSAFE_REFERENCE',
  'detail': 'script',
  'logical': 'È See [<span>`<script>x</script>`</span>](https://example.invalid) <img '
             'src="../second.png">',
  'link_accepted': True,
  'direct_reason': 'UNSAFE_REFERENCE',
  'host': 'paragraph',
  'kind': 'unsafe',
  'visits': ['span', 'script', 'img'],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p', 'type': 'paragraph', 'text': 'Known words.', 'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'Known words.'},
 {'id': 'HTML-owner-label-first-paragraph-1',
  'raw': 'È See [<span>`<img src="../first.png">`</span>](https://example.invalid) '
         '<script>second</script>\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': 'UNSAFE_REFERENCE',
  'detail': '../first.png',
  'logical': 'È See [<span>`<img src="../first.png">`</span>](https://example.invalid) '
             '<script>second</script>',
  'link_accepted': True,
  'direct_reason': 'UNSAFE_REFERENCE',
  'host': 'paragraph',
  'kind': 'unsafe',
  'visits': ['span', 'img', 'script'],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p', 'type': 'paragraph', 'text': 'Known words.', 'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'Known words.'},
 {'id': 'HTML-owner-label-first-quote-0',
  'raw': '> È See [<span>`<script>x</script>`</span>](https://example.invalid) <img '
         'src="../second.png">\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': 'UNSAFE_REFERENCE',
  'detail': 'script',
  'logical': 'È See [<span>`<script>x</script>`</span>](https://example.invalid) <img '
             'src="../second.png">',
  'link_accepted': True,
  'direct_reason': 'UNSAFE_REFERENCE',
  'host': 'quote',
  'kind': 'unsafe',
  'visits': ['span', 'script', 'img'],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p', 'type': 'paragraph', 'text': 'Known words.', 'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'Known words.'},
 {'id': 'HTML-owner-label-first-quote-1',
  'raw': '> È See [<span>`<img src="../first.png">`</span>](https://example.invalid) '
         '<script>second</script>\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': 'UNSAFE_REFERENCE',
  'detail': '../first.png',
  'logical': 'È See [<span>`<img src="../first.png">`</span>](https://example.invalid) '
             '<script>second</script>',
  'link_accepted': True,
  'direct_reason': 'UNSAFE_REFERENCE',
  'host': 'quote',
  'kind': 'unsafe',
  'visits': ['span', 'img', 'script'],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p', 'type': 'paragraph', 'text': 'Known words.', 'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'Known words.'},
 {'id': 'HTML-owner-label-first-list-0',
  'raw': '- È See [<span>`<script>x</script>`</span>](https://example.invalid) <img '
         'src="../second.png">\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': 'UNSAFE_REFERENCE',
  'detail': 'script',
  'logical': 'È See [<span>`<script>x</script>`</span>](https://example.invalid) <img '
             'src="../second.png">',
  'link_accepted': True,
  'direct_reason': 'UNSAFE_REFERENCE',
  'host': 'list',
  'kind': 'unsafe',
  'visits': ['span', 'script', 'img'],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p', 'type': 'paragraph', 'text': 'Known words.', 'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'Known words.'},
 {'id': 'HTML-owner-label-first-list-1',
  'raw': '- È See [<span>`<img src="../first.png">`</span>](https://example.invalid) '
         '<script>second</script>\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': 'UNSAFE_REFERENCE',
  'detail': '../first.png',
  'logical': 'È See [<span>`<img src="../first.png">`</span>](https://example.invalid) '
             '<script>second</script>',
  'link_accepted': True,
  'direct_reason': 'UNSAFE_REFERENCE',
  'host': 'list',
  'kind': 'unsafe',
  'visits': ['span', 'img', 'script'],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p', 'type': 'paragraph', 'text': 'Known words.', 'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'Known words.'},
 {'id': 'HTML-owner-label-first-note-tab-0',
  'raw': '[^n]: È See [<span>`<script>x</script>`</span>](https://example.invalid) <img '
         'src="../second.png">\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': 'UNSAFE_REFERENCE',
  'detail': 'script',
  'logical': 'È See [<span>`<script>x</script>`</span>](https://example.invalid) <img '
             'src="../second.png">',
  'link_accepted': True,
  'direct_reason': 'UNSAFE_REFERENCE',
  'host': 'note-tab',
  'kind': 'unsafe',
  'visits': ['span', 'script', 'img'],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'note',
                       'text': 'Known words.',
                       'page': 1,
                       'note_id': 'n',
                       'anchor': 'n'},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'Known words.'},
 {'id': 'HTML-owner-label-first-note-tab-1',
  'raw': '[^n]: È See [<span>`<img src="../first.png">`</span>](https://example.invalid) '
         '<script>second</script>\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': 'UNSAFE_REFERENCE',
  'detail': '../first.png',
  'logical': 'È See [<span>`<img src="../first.png">`</span>](https://example.invalid) '
             '<script>second</script>',
  'link_accepted': True,
  'direct_reason': 'UNSAFE_REFERENCE',
  'host': 'note-tab',
  'kind': 'unsafe',
  'visits': ['span', 'img', 'script'],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'note',
                       'text': 'Known words.',
                       'page': 1,
                       'note_id': 'n',
                       'anchor': 'n'},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'Known words.'},
 {'id': 'control-math-code-body-0',
  'raw': 'È See Known $x `first\n'
         'second` <img src="../outside.png"> y$ words end.\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'logical': 'È See Known $x `first\nsecond` <img src="../outside.png"> y$ words end.',
  'reason': None,
  'direct_reason': None,
  'formula': 'x `first\nsecond` <img src="../outside.png"> y',
  'refs': [],
  'no_assets': True,
  'host': 'paragraph',
  'kind': 'formula',
  'detail': None,
  'visits': [],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': 'È See Known  words end.',
                       'page': 1,
                       'formulas': [{'formula_id': 'f',
                                     'reference_latex': 'x `first\nsecond` <img src="../outside.png"> y',
                                     'symbols_expected': [],
                                     'structure_expected': None}],
                       'references': []},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'È See Known  words end.'},
 {'id': 'control-HTML-owner-label-only-0',
  'raw': 'È See [<span>`<script>x</script>`</span>](https://example.invalid)\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'logical': 'È See [<span>`<script>x</script>`</span>](https://example.invalid)',
  'reason': 'UNSAFE_REFERENCE',
  'direct_reason': 'UNSAFE_REFERENCE',
  'detail': 'script',
  'link_accepted': True,
  'host': 'paragraph',
  'kind': 'unsafe',
  'visits': ['span', 'script'],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p', 'type': 'paragraph', 'text': 'Known words.', 'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'Known words.'},
 {'id': 'control-math-code-body-1',
  'raw': 'È See Known $x `first\n'
         'second` <script>x</script> y$ words end.\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'logical': 'È See Known $x `first\nsecond` <script>x</script> y$ words end.',
  'reason': None,
  'direct_reason': None,
  'formula': 'x `first\nsecond` <script>x</script> y',
  'refs': [],
  'no_assets': True,
  'host': 'paragraph',
  'kind': 'formula',
  'detail': None,
  'visits': [],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': 'È See Known  words end.',
                       'page': 1,
                       'formulas': [{'formula_id': 'f',
                                     'reference_latex': 'x `first\nsecond` <script>x</script> y',
                                     'symbols_expected': [],
                                     'structure_expected': None}],
                       'references': []},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'È See Known  words end.'},
 {'id': 'control-HTML-owner-label-only-1',
  'raw': 'È See [<span>`<img src="../first.png">`</span>](https://example.invalid)\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'logical': 'È See [<span>`<img src="../first.png">`</span>](https://example.invalid)',
  'reason': 'UNSAFE_REFERENCE',
  'direct_reason': 'UNSAFE_REFERENCE',
  'detail': '../first.png',
  'link_accepted': True,
  'host': 'paragraph',
  'kind': 'unsafe',
  'visits': ['span', 'img'],
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p', 'type': 'paragraph', 'text': 'Known words.', 'page': 1},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]},
  'text': 'Known words.'},
 {'id': 'label-code',
  'raw': '[Known <span>words.</span> `<script>x</script>`](https://example.invalid)\n'
         '\n'
         '# Visible heading\n'
         '\n'
         'After words.',
  'reason': None,
  'text': 'Known words. <script>x</script>',
  'refs': ['https://example.invalid'],
  'no_assets': True,
  'positive': True,
  'host': 'paragraph',
  'kind': 'literal',
  'detail': None,
  'visits': None,
  'gold': {'document_id': 'r011-synthetic',
           'blocks': [{'block_id': 'p',
                       'type': 'paragraph',
                       'text': 'Known words. <script>x</script>',
                       'page': 1,
                       'references': [{'kind': 'link',
                                       'text': 'Known words. <script>x</script>',
                                       'target': 'https://example.invalid'}]},
                      {'block_id': 'h', 'type': 'heading', 'text': 'Visible heading', 'page': 1},
                      {'block_id': 'a', 'type': 'paragraph', 'text': 'After words.', 'page': 1}]}}]
def add(ident,value,host,reason,kind,**extra):
    CASES.append(dict(id=ident,raw=transport(value,host)+TAIL,logical=value,host=host,reason=reason,kind=kind,gold=gold(value,host),**extra))
# 42 metadata-leaf cases: six owning producers, one/two ticks or math,
# first logical line/continuation where supported. Closed literal controls
# carry the same marker and tag, rather than a different payload.
for host in HOSTS[1:]:
    for marker in ('`','``','$y '):
        for cont in (False,True):
            if cont and host in ('cell','heading','caption'):continue
            value=('È item\n' if cont else '')+'# x {'+marker+BAD+'}'
            add(f'metadata-{host}-{len(marker)}-{cont}',value,host,'UNSAFE_REFERENCE','unsafe',detail='../outside.png',visits=['img'])
    for ticks in ('`','``'):
        value='# È x {'+ticks+SCRIPT+ticks+'}'
        # Safe closed literal in a finite leaf, no actual script opening.
        # Quote is structurally complex by the received profile, but audit complete.
        add(f'metadata-closed-{host}-{len(ticks)}',value,host,None,'closed',visits=[],literal=SCRIPT)
# 16 additional accepted-label math cases: two tick lengths across LF in
# four producer domains; outside-label controls already in the originals.
for host in HOSTS[:4]:
    for ticks in ('`','``'):
        for tag in (BAD,SCRIPT):
            payload='x '+ticks+'first\nsecond'+ticks+' '+tag+' y'
            value='È See [Known $'+payload+'$ words](https://example.invalid) end.'
            add(f'label-math-{host}-{len(ticks)}-{tag==SCRIPT}',value,host,None,'formula',formula=payload,refs=['https://example.invalid'],visits=[],link_accepted=True,text='È See Known  words end.')
            c=CASES[-1];c['gold']=gold(c['text'],host)
            c['gold']['blocks'][0].update(formulas=[{'formula_id':'f','reference_latex':payload,'symbols_expected':[],'structure_expected':None}],references=[{'kind':'link','text':'Known  words','target':'https://example.invalid'}])
# 48 ownership cases, not an unconstrained Cartesian campaign: four hosts,
# three owner endings, two unsafe source orders, one/two backticks.
for host in HOSTS[:4]:
    for ending in ('</span>','</em>',''):
        for first_script in (False,True):
            for ticks in ('`','``'):
                first,second=(SCRIPT,BAD) if first_script else (BAD,SCRIPT)
                value='È See [<span>'+ticks+first+ticks+ending+'](https://example.invalid) '+second
                add(f'owner-{host}-{len(ending)}-{first_script}-{len(ticks)}',value,host,'UNSAFE_REFERENCE','unsafe',detail='script' if first_script else '../outside.png',visits=['span','script' if first_script else 'img','img' if first_script else 'script'])
# Safe HTML before unsafe, and a tree error before both violations: all
# openings must still be visited, in source order.
for host in HOSTS[:4]:
    value='È See [<span class="a" class="b">`<img src="p.png">` '+SCRIPT+'</span>](https://example.invalid) '+BAD
    add('owner-safe-tree-error-'+host,value,host,'UNSAFE_REFERENCE','unsafe',detail='script',visits=['span','img','script','img'])
    for ticks in ('`','``'):
        value='È See [<span>words.</span> '+ticks+SCRIPT+ticks+'](https://example.invalid) '+BAD
        add('owner-closed-external-code-'+host+'-'+str(len(ticks)),value,host,'UNSAFE_REFERENCE','unsafe',detail='../outside.png',visits=['span','img'])
    # LF outside closed code is deliberately an unaccepted label; the real
    # HTML after an unclosed literal remains an actual opening.
    value='È See [Known `first\nsecond '+BAD+' words](https://example.invalid) end.'
    add('free-LF-unclosed-'+host,value,host,'UNSAFE_REFERENCE','unsafe',detail='../outside.png',visits=['img'],link_accepted=False)

def pipeline(path,raw,g):
    path.mkdir(parents=True,exist_ok=True);(path/'document.md').write_bytes(raw.encode());shutil.copyfile(ASSET,path/'p.png')
    a=auditor.audit(path);o=observe_bundle(path,path/'observations.json');m=score(deepcopy(g),o,a,verification=None)
    return o,a,m

def diagnostics(c):
    if 'logical' not in c:return {},{}
    logical=c['logical'];events=[]
    if 'v9' not in reader.VERSION:reader.markdown_lexemes(logical,False,_html_events=events)
    visits=[];original=reader.PassiveHTML.handle_starttag
    def spy(self,tag,attrs):visits.append(tag);return original(self,tag,attrs)
    reader.PassiveHTML.handle_starttag=spy
    try:
        try:p=reader.inline(logical,producer_limit=len(logical));reason=None;detail=None
        except reader.Unsafe as exc:p=None;reason='UNSAFE_REFERENCE';detail=str(exc)
    finally:reader.PassiveHTML.handle_starttag=original
    link=reader.markdown_link(logical,logical.find('['),len(logical)) if '[' in logical else None
    checks={'direct-reason':reason==('UNSAFE_REFERENCE' if c['reason']=='UNSAFE_REFERENCE' else None)}
    if c.get('detail'):checks['first-direct']=c['detail'] in (detail or '')
    if c.get('visits') is not None:
        checks['all-openings']=visits==c['visits']
        checks['event-order']=events==sorted(events,key=lambda e:e['start'])
    if 'link_accepted' in c:checks['label-domain']=bool(link)==c['link_accepted']
    return checks,dict(events=events,visits=visits,reason=reason,detail=detail,link=link,parsed=p)

def checks(c,o,a,m,opened):
    raw=c['raw'];reason=c['reason'];refs=[r for n in o['nodes'] for r in n['references']];fs=[f for n in o['nodes'] for f in n['formulas']]
    result=dict(audit=a['reason']==reason,status=a['status']==('COMPLETE' if reason is None else 'INCOMPLETE'),raw=all(n['raw']==raw[n['offset']['char_start']:n['offset']['char_end']] for n in o['nodes']),no_evasive_open=not any(v.startswith('../') for v in opened),denominator=m['totals']['text_ref_codepoints']==sum(len(b['text']) for b in c['gold']['blocks']),arithmetic=m['totals']['word_errors']==sum(v['known_word_edits'] for v in m['WER_decomposition']['entries']),no_assets=not a['assets'] and 'p.png' not in opened)
    result['utf8']=all(v['source_position']['byte']==len(raw[:v['source_position']['char']].encode()) for n in o['nodes'] for field in ('references','formulas','unknown','segments') for v in n.get(field,[]) if 'source_position' in v)
    if c.get('detail'):result['first-unsafe']=c['detail'] in a.get('detail','')
    if reason:
        result.update(score_FAIL=m['score_exit']==3,evaluate=evaluate_exit(2,m['score_exit'])==2)
    else:
        result['external']=len(o['nodes'])==3 and o['nodes'][-2]['text']=='Visible heading' and o['nodes'][-1]['text']=='After words.'
    if c['kind']=='formula':
        result.update(formula=len(fs)==1 and fs[0]['raw']==c['formula'],references=[r['target'] for r in refs]==c.get('refs',[]),text=o['nodes'][0]['text']==c['text'],types=[n['type'] for n in o['nodes']]==['note' if c.get('host')=='note-tab' else 'paragraph','heading','paragraph'],formula_denominator=m['denominators']['formulas']==1,no_false_formula_failure=not any(v['metric']=='formula' for v in m['failures']),indeterminate=m['score_exit']==5 and m['quality']=='INDETERMINATE')
        if fs:
            at=raw.index('$');result['formula-map']=fs[0]['source_position']['char']==at and fs[0]['source_position']['byte']==len(raw[:at].encode())
        if refs:
            at=raw.index('[Known');result['link-map']=refs[-1]['source_position']['char']==at
    if c['kind']=='metadata-control':result['heading-metadata']=o['nodes'][0]['type']=='heading' and o['nodes'][0]['text']=='x'
    if c['kind']=='literal':result['literal']=o['nodes'][0]['text']==c['text'] and m['automatic_gates']=='PASS' and m['score_exit']==5
    if c['kind']=='closed':result['closed-inert']=not any(r['kind']=='image' for r in refs) and not fs
    return result

# R014 overlay: HC2's braces are not metadata in pinned Pandoc 3.1.3.
# Keep the original CASES/raw/gold above intact; AST/provenance in the fixture.
R014=json.loads((ROOT/'tests/fixtures/conversion-reader-r014-legacy-expectations.json').read_text())['cases']
CASES=[{**c,**R014.get(c['id'],{})} for c in CASES]

@pytest.mark.parametrize('case',CASES,ids=[c['id'] for c in CASES])
def test_inline_ownership_pipeline(tmp_path,monkeypatch,case):
    opened=[];original=auditor.regular
    def spy(root,name,*args,**kw):opened.append(name);return original(root,name,*args,**kw)
    monkeypatch.setattr(auditor,'regular',spy);o,a,m=pipeline(tmp_path,case['raw'],case['gold']);result=checks(case,o,a,m,opened);direct,_=diagnostics(case);result.update(direct)
    assert all(result.values()),result
