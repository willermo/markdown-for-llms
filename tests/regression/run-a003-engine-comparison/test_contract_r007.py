"""R007: prove raw finite, maschere additive e confini; nessun parser/motore/gold."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('r006', Path(__file__).with_name('test_contract_r006.py'))
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
p = t.p
K = t.K
E = t.E
M = t.M
L = t.L
T6 = lambda value: (value, 'T06', dict(evidence='finite raw tag'))
D = lambda value, role: (value, 'T04', dict(relation='link', role=role, evidence='finite raw link'))


def finite_cases():
    rows = []
    def add(name, parts, accepted=False, uri=None, link=False):
        row = t.t.case(name, parts, accepted)
        a = row['annotation']
        if uri:
            a['contexts'].append(dict(kind='uri', span=uri))
        if link:
            segments = [s for s in a['segments'] if s['operation'] == 'T04']
            spans = [s['span'] for s in segments]
            label = [spans[0][1], spans[1][0]]
            a['relations'] = dict(link=dict(channel='references',kind='link',segments=spans,declarations=[],label_span=label,label=row['raw'][slice(*label)].decode(),target=next(s['raw'] for s in segments if s.get('role')=='destination')))
        rows.append(row)
    # G: entrambe le famiglie, maschere raw e dichiarate, closer backslash.
    for delimiter in ('`', '``', '$', '$$'):
        for cause in ('see https://a.b/', 'a\\(b', 'see www.a.b/', 'see mailto:a@b'):
            suffix = ' c \\) ' if cause == 'a\\(b' else ' then '
            add('G/raw/'+repr((delimiter,cause)), [K(delimiter+cause+delimiter+suffix+delimiter+'x'),E(),K('y'+delimiter+' z')])
        for field in ('domain', 'channel'):
            for domain in ('literal','code','math','cell','uri'):
                add('G/declared/'+repr((delimiter,field,domain)),[K(delimiter+'a'),(delimiter,'KEEP',{field:domain}),K(' b '+delimiter+'x'),E(),K('y'+delimiter+' z')])
        value = delimiter+'see '
        url = 'https://a.b/'+delimiter
        add('G/uri-context/'+delimiter,[K(value),K(url),K(' then '+delimiter+'x'),E(),K('y'+delimiter+' z')],uri=[len(value.encode()),len((value+url).encode())])
        add('G/control/'+delimiter,[K(delimiter+'see a.b/'+delimiter+' then '+delimiter+'x'),E(),K('y'+delimiter+' z')])
        add('P1/'+delimiter,[K(delimiter+'a'+delimiter+' https://a.b/ prosa x'),E(),K('y')],True)
    for count in (1,2,3,4):
        add('G/backslash-closer/'+str(count),[K('`'+'\\'*count+'` `x'),E(),K('y`')])
    for name,raw in [('G7','Il totale `a\n2. ```\nb` e\n   ```\n`c` e `x'),('G7c','Il totale `a\nb` e\n   ```\n`c` e `x'),('G8','    ```\n    x\n\ntesto `a\n    ```\nb` e `x'),('G8c','testo `a\n    ```\nb` e `x')]:
        add(name,[K(raw),E(),K('y` z\n')])
    add('P2',[K('\\(x\\) prosa '),M('*','open'),K('non'),M('*','close')],True)
    # H4: nessun dichiarativo può escludere un opener reale.
    for marker in ('*','_','**','__'):
        for field in ('domain','channel'):
            for domain in ('literal','code','math','cell','uri'):
                add('H4/'+repr((marker,field,domain)),[M(marker,'open'),K('foo '),(marker,'KEEP',{field:domain}),K('bar baz'),M(marker,'close')])
        add('H4c/'+marker,[M(marker,'open'),K('foo '+marker+'bar baz'),M(marker,'close')])
        add('H/later/'+marker,[K(marker+'foo '),M(marker,'open'),K('bar baz'),M(marker,'close')],True)
        add('H/simple/'+marker,[M(marker,'open'),K('non 0012'),M(marker,'close')],True)
        for nl in ('\n','\r\n'):
            add('H/LF/'+repr((marker,nl)),[M(marker,'open'),K('non'+nl+'0012'),M(marker,'close')],True)
            for space in ('',' ','\t','   '):
                sep=nl+space+nl
                add('H3/KEEP/'+repr((marker,sep)),[M(marker,'open'),K('non'+sep+'0012'),M(marker,'close')])
                if not space:
                    add('H3/T08/'+repr((marker,sep)),[M(marker,'open'),K('non'),(sep,'T08',dict(kind='blank-layout',evidence='raw blank')),K('0012'),M(marker,'close')])
            add('H3/heading/'+repr((marker,nl)),[M(marker,'open'),K('non'+nl+'## 0012'),M(marker,'close')])
            add('H3/local-pairs/'+repr((marker,nl)),[M(marker,'open','a'),K('non'),M(marker,'close','a'),K(nl+nl),M(marker,'open','b'),K('0012'),M(marker,'close','b')],True)
        for form in ('cross-left','cross-right','enclosing','internal'):
            left=[M(marker,'open')] if form in ('cross-left','enclosing') else []
            right=[M(marker,'close')] if form in ('cross-right','enclosing') else []
            body=[*([] if form in ('cross-left','enclosing') else [M(marker,'open')]),K('foo'),*([] if form in ('cross-right','enclosing') else [M(marker,'close')])]
            add('H1/'+repr((marker,form)),[*left,D('[','delimiter'),*body,D('](','delimiter'),D('/uri','destination'),D(')','delimiter'),*right],form in ('enclosing','internal'),link=True)
    for marker in ('*','_','**','__'):
        for mode in ('competing','later','enclosing'):
            head = [K(marker+'a '),M(marker,'open')] if mode=='later' else [M(marker,'open')]
            text = 'b ' if mode=='later' else 'a '+marker+'b ' if mode=='competing' else 'a '
            add('H/target-competition/'+repr((marker,mode)),[*head,K(text),D('[','delimiter'),K('foo'),D('](','delimiter'),D('/x'+marker,'destination'),D(')','delimiter'),K(' c'),M(marker,'close')],mode!='competing',link=True)
    for tag in ('span','a','em','p'):
        for attr in ('id','name'):
            for tail in ('_','__','xy'):
                add('H2/'+repr((tag,attr,tail)),[M('_','open'),K('a _b '),T6('<'+tag+' '+attr+'="x'+tail+'">'),K('c'),M('_','close')])
    for tag in ('<span data-x="x_">','<custom id="x_">','<span id="x_"'):
        add('H2/unproved/'+tag,[M('_','open'),K('a _b '+tag+'c'),M('_','close')])
    add('H/nested',[M('**','open','a'),K('non '),M('_','open','b'),K('−7'),M('_','close','b'),M('**','close','a')],True)
    # F03: item vuoto in interruzione, incluso EOF; positivi a confine lecito.
    for nl in ('\n','\r\n'):
        for marker in ('1. ','1) ','01. ','01) '):
            for after in ('',nl+'Poi', '  '+nl+'Poi',nl+nl+'Poi'):
                for prefix in ('','> '):
                    q=[(prefix,'T01',dict(wrapper='quote',evidence='raw quote'))] if prefix else []
                    add('F03/interrupt/'+repr((nl,marker,after,prefix)),[K(prefix+'Il totale è'+nl),*q,L(marker),K(after)])
            for before in ('','Il totale è'+nl+nl,'# Titolo'+nl,'1. prima'+nl,'```'+nl+'code'+nl+'```'+nl):
                add('F03/empty-legal/'+repr((nl,marker,before)),[K(before),L(marker),K(nl+'Poi')],True)
            add('F03/nonempty/'+repr((nl,marker)),[K('Il totale è'+nl),L(marker),K('Poi')],True)
    return [dict(row,name=row['name']+'/'+variant,annotation=t.t.partition(row['annotation'],variant)) for row in rows for variant in t.t.VARIANTS]


class RawProofContract(unittest.TestCase):
    def test_cases_and_keep_four_partitions(self):
        for row in finite_cases():
            with self.subTest(name=row['name']):
                got=t.t.result(row['raw'],row['annotation'])
                self.assertEqual(got['accepted'],row['accepted'],got)
                raw,a=t.t.annotation([K(row['raw'].decode())])
                self.assertEqual(t.t.result(raw,a)['emission'],raw.decode())

    def test_annotation_masks_only_add_protection(self):
        # Compara insiemi, non solo il rifiuto della singola stringa G.
        for row in finite_cases():
            a=copy.deepcopy(row['annotation']);raw=row['raw']
            base=copy.deepcopy(a)
            base['contexts']=[c for c in base['contexts'] if c['kind']!='uri']
            for s in base['segments']:
                s.pop('channel',None)
                if s['operation']=='KEEP':s.pop('domain',None)
            def covered(annotation):
                ctx=p.finite_contexts(raw,annotation,p.raw_boundaries(raw))
                return {i for r in p.qualified_domains(raw,annotation,ctx) for i in range(*r['span'])}
            before=covered(base)
            self.assertLessEqual(before,covered(a),row['name'])
            for segment in base['segments']:
                if segment['operation']=='KEEP':
                    segment['domain']='literal'
                    self.assertLessEqual(before,covered(base),row['name'])

    def test_raw_reads_survive_estimates(self):
        for row in finite_cases():
            raw=row['raw'];a=row['annotation']
            regions=p.qualified_domains(raw,a,p.finite_contexts(raw,a,p.raw_boundaries(raw)))
            # Le sole maschere semanticamente comprovate sono ammesse alla base.
            proved=[r for r in regions if r.get('inline_opaque')]
            covered={i for r in regions for i in range(*r['span'])}
            for ch,kind in ((b'`','code'),(b'$','math')):
                for region in p.inline_regions(raw,ch,kind,proved):
                    self.assertLessEqual(set(range(*region['span'])),covered,row['name'])

    def test_previous_r007_setup_is_readonly(self):
        import json
        c=t.t.c
        source=c.RUN/'evidence/implementation-r007/preflight-r001/step-04-manifest.json'
        manifest=json.loads(source.read_text());parent=Path(manifest['scratch'])
        for name in ('setup-r006-002','dist-fix-r006-002','setup-r007-001','dist-fix-r007-001'):
            value=copy.deepcopy(manifest);value['setup_writable']=[str(parent/name)]
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Setup writable destination not disposed'):
                c.validate(value)
