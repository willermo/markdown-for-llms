"""Chiusura finita r006: sintetici byte-esatti, nessun motore/gold/R autentico."""
import copy
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).with_name('test_contract_r005.py')
spec=importlib.util.spec_from_file_location('contract_previous',path)
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
p=t.p
K=lambda value:(value,'KEEP',{})
E=lambda:('\\_','T02',dict(domain='prose',emit='_'))
M=lambda marker,side,ident='p':(marker,'T03',dict(domain='prose',side=side,pair_id=ident))
L=lambda value:(value,'T01',dict(wrapper='list',evidence='raw block position'))

def finite_cases():
    rows=[]
    def add(name,parts,accepted=False):
        row=t.case(name,parts,accepted)
        for seg in row['annotation']['segments']:
            if seg['operation']=='T07':
                a,b=seg['span'];row['annotation']['contexts'].append(dict(kind='html-text',span=[a,b],opening=[a-3,a],closing=[b,b+4]))
        rows.append(row)
    bodies=[('T02',[K('x'),E(),K('i')]),
            ('T03',[K(' '),M('*','open'),K('non −7 0012'),M('*','close'),K(' ')]),
            ('T07',[K('<p>'),('&amp;','T07',dict(domain='html-text',emit='&')),K('</p>')]),
            ('T06',[('<b>','T06',dict(evidence='finite tag')),K('non −7'),('</b>','T06',dict(evidence='finite tag'))]),
            ('T00-padding',[K('x '),(' ','T00',dict(domain='prose',mode='padding')),K('i')]),
            ('T00-wrap',[K('x'),('\n','T00',dict(domain='prose',mode='soft-wrap',emit=' ')),K('i')]),
            ('T01',[K('\n'),('# ','T01',dict(wrapper='heading',evidence='line start')),K('non 0012')]),
            ('T08',[K('x'),('\n\n','T08',dict(kind='blank-layout',evidence='blank')),K('i')])]
    # Due versi, delimiter interni a un dominio/URI, non solo $HOME.
    for code in ('`','``'):
        for math in ('$','$$'):
            for inside in ('a','echo non −7','*x*','<b>x</b>','x y'):
                for op,body in bodies:
                    add('F01/code-first/'+repr((code,math,inside,op)),[K('usa '+code+inside+' '+math+'q'+code+' e poi '+math),*body,K(math+' fine')])
                    add('F01/math-first/'+repr((code,math,inside,op)),[K(math+inside+code+'q'+math+' e '+code),*body,K(code+' fine')])
            for outer in ('<code>','<pre>','    ','https://e.invalid/?q=','mailto:'):
                close={'<code>':'</code>','<pre>':'</pre>','    ':'\n\n'}.get(outer,' ')
                for delimiter in (code,math):
                    for op,body in bodies:
                        add('F01/opaque/'+repr((outer,delimiter,op)),[K(outer+'echo'+delimiter+'non'+close+' prosa '+delimiter),*body,K(delimiter+' fine')])
    # Operazioni oltre una regione valida: non si rende indiscriminatamente raw indisponibile.
    for raw in ('`non 0012`','$x_i$','$$x_i$$','<code>non $x</code>', '<pre>`x</pre>',
                '    non $x\n\n', 'https://e.invalid/?q=$x ', '\\(x`i\\)'):
        add('F01/external/'+repr(raw),[K(raw+' prosa '),E(),K('i')],True)
    # 32 varianti richieste, più concorrenti, chiusure, run lunghi e marker protetti.
    for marker in ('*','_','**','__'):
        for prefix,middle in (('', 'foo '),('prosa ','non −7 0012 ')):
            add('F02/competing/'+repr((marker,prefix)),[K(prefix),M(marker,'open'),K(middle+marker+'bar baz'),M(marker,'close')])
        for middle in ('foo '+marker+'bar ', 'non '+marker+'0012 '+marker+' altro ', 'x '+marker*3+'y '):
            add('F02/class/'+repr((marker,middle)),[M(marker,'open'),K(middle),M(marker,'close')])
        add('F02/later/'+marker,[K(marker+'foo '),M(marker,'open'),K('bar baz'),M(marker,'close')],True)
        add('F02/simple/'+marker,[M(marker,'open'),K('non −7 0012'),M(marker,'close')],True)
        for literal in ('\\'+marker,'`'+marker+'x`','$'+marker+'x$', 'https://e.invalid/'+marker+'x '):
            add('F02/literal/'+repr((marker,literal)),[K(literal+' prosa '),M(marker,'open'),K('non'),M(marker,'close')],True)
    add('F02/nested',[M('**','open','a'),K('non '),M('_','open','b'),K('−7'),M('_','close','b'),K(' si'),M('**','close','a')],True)
    # Fence in contenitori: riga esterna qualificata, mai run midline/corto.
    for prefix,continuation in (('> ','> '),('> > ','> > '),('- ','  '),('1. ','   '),('    ','    '),('> - ','>   '),('\t','\t')):
        for marker in ('```','````','~~~','~~~~'):
            for info in ('','text'):
                for closing in (marker,marker[:-1],'a '+marker,''):
                    for op,body in bodies:
                        add('F03/payload/'+repr((prefix,marker,info,closing,op)),[K(prefix+marker+info+'\n'+continuation+'a '+marker+' b '),*body,K('\n'+continuation+closing+'\n')])
                add('F03/incomplete/'+repr((prefix,marker,info)),[K(prefix+marker+info),E(),K('i')])
                add('F03/external/'+repr((prefix,marker,info)),[K(prefix+marker+info+'\n'+continuation+'a '+marker+' b \\_i\n'+continuation+marker+'\nprosa '),E(),K('i')],True)
    # Numero/anno, dieci cifre e contenitori; ricostruzione del blocco dal raw.
    for prefix in ('','> '):
        before=([(prefix,'T01',dict(wrapper='quote',evidence='raw'))] if prefix else [])
        for value in ('12. ','2023. ','1234567890. ','12) '):
            add('F04/paragraph/'+repr((prefix,value)),[K(prefix+'Il totale è\n'),*before,L(value),K('Poi')])
            add('F04/blank/'+repr((prefix,value)),[K(prefix+'Il totale è\n'+prefix.rstrip()+'\n'),*before,L(value),K('Poi')],len(value.split('.')[0].split(')')[0])<=9)
        for preceding,marker,ok in (('', '12. ',True),('Il totale è\n','1. ',True),('12. prima\n','13. ',True),('testo\n12. prima\n','13. ',False),('testo\n\n12. prima\n','13. ',True),('testo\n    continua\n','12. ',False),('testo\n','123456789. ',False)):
            add('F04/context/'+repr((prefix,preceding,marker)),[K(''.join(prefix+line for line in preceding.splitlines(keepends=True))),*before,L(marker),K('Poi')],ok)
    return [dict(row,name=row['name']+'/'+v,annotation=t.partition(row['annotation'],v)) for row in rows for v in t.VARIANTS]

class FiniteClosure(unittest.TestCase):
    def test_finite_classes_and_four_partitions(self):
        for row in finite_cases():
            with self.subTest(name=row['name']):
                got=t.result(row['raw'],row['annotation'])
                self.assertEqual(got['accepted'],row['accepted'],got)
                if not row['accepted']:
                    raw,a=t.annotation([K(row['raw'].decode())])
                    self.assertEqual(t.result(raw,a)['emission'],row['raw'].decode())
    def test_independent_regions_retained_through_closure(self):
        for raw in (b'$a`b$ e `x\\_y`',b'`a$b` e $x\\_1$',b'`a$b` $c`d$ `e$f` $g`h$'):
            _,a=t.annotation([K(raw.decode())]);all_regions=p.qualified_domains(raw,a,[])
            for ch,kind in ((b'`','code'),(b'$','math')):
                for region in p.inline_regions(raw,ch,kind,[]):
                    self.assertIn(region,all_regions)
    def test_declared_literal_and_uri_with_delimiters(self):
        for domain in ('code','literal','math','cell','uri'):
            raw,a=t.annotation([('echo $x`y','KEEP',dict(domain=domain)),K(' prosa $'),K('x'),E(),K('i$')])
            a['contexts']=[dict(kind='prose',span=[0,len(raw)])]
            self.assertFalse(t.result(raw,a)['accepted'])
    def test_literal_marker_cannot_be_promoted_from_a_fragment(self):
        for marker in ('*','_','**','__'):
            raw,a=t.annotation([M(marker,'open'),K(marker+'non'),M(marker,'close')])
            self.assertFalse(t.result(raw,a)['accepted'])
    def test_setup_history_is_readonly_in_current_delivery(self):
        import json
        c=t.c
        manifest=json.loads((c.RUN/'evidence/implementation-r005/preflight-r005/step-03-manifest.json').read_text())
        parent=Path(manifest['scratch'])
        for name in ('setup-r005','dist-fix-r005-002','setup-r006','dist-fix-r006-001'):
            with self.subTest(name=name):
                value=copy.deepcopy(manifest);value['setup_writable']=[str(parent/name)]
                with self.assertRaisesRegex(ValueError,'Setup writable destination not disposed'):
                    c.validate(value)
    def test_layout_observation_preserved_with_limit(self):
        raw,a=t.annotation([K('frammen'),('\n\n','T08',dict(kind='blank-layout',evidence='blank')),K('to')])
        self.assertEqual(t.result(raw,a)['emission'],'frammento')
        a['segments'][0]['unit_id']='left';a['segments'][-1]['unit_id']='right'
        got=p.project(raw,a);self.assertEqual(got['units']['left'],'frammen');self.assertEqual(got['units']['right'],'to')
        raw,a=t.annotation([K('# Titolo\n    x'),E(),K('i\n')]);a['contexts']=[dict(kind='prose',span=[0,len(raw)])]
        self.assertTrue(t.result(raw,a)['accepted']) # limite storico fuori dal profilo indent.

if __name__=='__main__':unittest.main()
