"""math-structure-v1: parser finito gold-blind, nessun CAS o macro."""
import re

VERSION='math-structure-v1'


class Unsupported(ValueError):
    def __init__(self,reason,offset):
        super().__init__(reason);self.reason=reason;self.offset=offset


def seq(items):return {'type':'Seq','items':items}
def arg(items):return {'type':'Arg','body':seq(items)}


class Parser:
    def __init__(self,raw):
        self.raw=raw;self.i=0;self.depth=0;self.spacing=[];self.cells=0
        if len(raw)>4096:raise Unsupported('length>4096',0)
    def fail(self,reason):raise Unsupported(reason,self.i)
    def skip(self):
        while self.i<len(self.raw):
            match=re.match(r'[ \t\r\n]+|\\(?:[,;:!]|quad(?![A-Za-z])|qquad(?![A-Za-z]))',self.raw[self.i:])
            if not match:break
            self.spacing.append({'offset':self.i,'raw':match[0]});self.i+=len(match[0])
    def eat(self,value):
        self.skip()
        if not self.raw.startswith(value,self.i):self.fail('expected '+value)
        self.i+=len(value)
    def sequence(self,stops=()):
        self.depth+=1
        if self.depth>16:self.fail('depth>16')
        items=[]
        while True:
            self.skip()
            if self.i==len(self.raw) or any(self.raw.startswith(s,self.i) for s in stops):break
            items.append(self.atom(scripts=True))
        self.depth-=1
        return seq(items)
    def argument(self):
        self.skip()
        if self.i==len(self.raw):self.fail('missing argument')
        if self.raw[self.i]=='{':
            self.i+=1;result=self.sequence(('}',));self.eat('}')
            return {'type':'Arg','body':result}
        return arg([self.atom(scripts=False,bare=True)])
    def text(self):
        self.eat('{');result=[]
        while self.i<len(self.raw) and self.raw[self.i]!='}':
            c=self.raw[self.i];self.i+=1
            if c=='{':self.fail('nested text group')
            if c=='\\':
                if self.i==len(self.raw) or self.raw[self.i] not in '{}%$_&#':self.fail('text macro unsupported')
                c=self.raw[self.i];self.i+=1
            result.append(c)
        self.eat('}');return {'type':'Text','value':''.join(result),'command':'text'}
    def environment(self):
        match=re.match(r'\\begin\{(aligned|bmatrix|matrix)\}',self.raw[self.i:])
        if not match:self.fail('environment unsupported')
        env=match[1];self.i+=len(match[0]);end='\\end{'+env+'}';rows=[];row=[]
        while True:
            row.append(self.sequence(('&','\\\\',end)));self.cells+=1
            if self.cells>256:self.fail('cells>256')
            self.skip()
            if self.raw.startswith('&',self.i):self.i+=1;continue
            rows.append(row);row=[]
            if self.raw.startswith(end,self.i):self.i+=len(end);break
            self.eat('\\\\')
        if env=='aligned':return {'type':'Aligned','rows':rows}
        return {'type':'Matrix','delimiter':'square' if env=='bmatrix' else 'none','rows':rows}
    def atom(self,scripts,bare=False):
        self.skip()
        if self.i>=len(self.raw):self.fail('truncated atom')
        c=self.raw[self.i]
        if self.raw.startswith('\\begin{',self.i):node=self.environment()
        elif re.match(r'\\frac(?![A-Za-z])',self.raw[self.i:]):
            self.i+=5;node={'type':'Fraction','numerator':self.argument(),'denominator':self.argument()}
        elif re.match(r'\\sum(?![A-Za-z])',self.raw[self.i:]):
            self.i+=4;node={'type':'Sum','lower':None,'upper':None}
        elif re.match(r'\\text(?![A-Za-z])',self.raw[self.i:]):self.i+=5;node=self.text()
        elif self.raw.startswith('\\left[',self.i):
            self.i+=6;body=self.sequence(('\\right]',));self.eat('\\right]')
            node=self.group('square',body)
        elif c in '{([':
            self.i+=1;body=self.sequence(({ '{':'}','(':')','[':']'}[c],));self.eat({'{':'}','(':')','[':']'}[c])
            node=self.group({'{':'braces','(':'round','[':'square'}[c],body)
        elif c in '=+-−<>':
            self.i+=1;node={'type':'Operator','kind':{'=':'equals','+':'plus','-':'minus','−':'minus','<':'lt','>':'gt'}[c]}
        elif c.isascii() and c.isalpha():self.i+=1;node={'type':'Symbol','value':c}
        elif c.isascii() and c.isdigit():
            value=c if bare else re.match(r'[0-9]+(?:[.,][0-9]+)?',self.raw[self.i:])[0]
            self.i+=len(value);node={'type':'Number','lexeme':value}
        else:self.fail('unsupported token')
        if scripts or node['type']=='Sum':
            fields={}
            while True:
                self.skip()
                if self.i>=len(self.raw) or self.raw[self.i] not in '_^':break
                key=self.raw[self.i];self.i+=1
                if key in fields:self.fail('duplicate script')
                fields[key]=self.argument()
            if node['type']=='Sum':node.update(lower=fields.get('_'),upper=fields.get('^'))
            elif fields:node={'type':'Scripts','base':node,'sub':fields.get('_'),'sup':fields.get('^')}
        return node
    def group(self,kind,body):
        # Equivalence only of a full one-matrix wrapper, not a cell group.
        if kind=='square' and len(body['items'])==1 and body['items'][0]['type']=='Matrix' and body['items'][0]['delimiter']=='none':
            node=body['items'][0];node['delimiter']='square';return node
        return {'type':'Group','kind':kind,'body':body}
    def parse(self):
        result=self.sequence()
        if self.i!=len(self.raw):self.fail('trailing token')
        if len(result['items'])==1 and result['items'][0]['type'] in ('Aligned','Matrix'):return result['items'][0]
        return result


def observe(raw):
    parser=None
    try:
        parser=Parser(raw);ast=parser.parse()
        return {'version':VERSION,'raw':raw,'fully_supported':True,'structure':ast,'spacing_tokens':parser.spacing,'recognized_defects':[],'reason':None,'offset':None}
    except Unsupported as exc:
        return {'version':VERSION,'raw':raw,'fully_supported':False,'structure':None,'spacing_tokens':parser.spacing if parser else [],'recognized_defects':[],'reason':exc.reason,'offset':exc.offset}


def differences(expected,actual,path='$'):
    if type(expected)!=type(actual):return [{'path':path,'expected':expected,'actual':actual}]
    if isinstance(expected,dict):
        result=[]
        for key in sorted(set(expected)|set(actual)):
            if key not in expected or key not in actual:result.append({'path':path+'.'+key,'expected':expected.get(key),'actual':actual.get(key)})
            else:result.extend(differences(expected[key],actual[key],path+'.'+key))
        return result
    if isinstance(expected,list):
        result=[]
        if len(expected)!=len(actual):result.append({'path':path+'.length','expected':len(expected),'actual':len(actual)})
        for i,(e,a) in enumerate(zip(expected,actual)):result.extend(differences(e,a,path+f'[{i}]'))
        return result
    return [] if expected==actual else [{'path':path,'expected':expected,'actual':actual}]
