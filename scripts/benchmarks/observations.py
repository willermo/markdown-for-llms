"""Reader pandoc-md-bench-v1 deterministico sui soli byte. Nessun gold."""
from bisect import bisect_right
from contextvars import ContextVar
from html.parser import HTMLParser, tagfind_tolerant, attrfind_tolerant, locatetagend
import re
from data import digest_bytes
from math_structure import observe as math_observe
from safety import html, PassiveHTML, Unsafe

VERSION='observations-v14-syntax-contracts'
LIMIT=32*1024*1024
COMMENT_CLOSE=re.compile(r'--!?>')
COMMENT_ABRUPT=re.compile(r'-?>')


def html_lexeme(raw,start):
    """Finite inert-token intervals, matching the pinned 3.12.13 tokenizer.

    Normal comment closure is searched before the abrupt opener closure,
    exactly as parse_comment does. No interpretation of the enclosed bytes.
    EOF is part of the token when no closure exists. This same policy is used
    by the callback transport and by Markdown's physical line boundaries.
    """
    if raw.startswith('<!--',start):
        closing=COMMENT_CLOSE.search(raw,start+4)
        if closing:return 'comment',closing.end()
        closing=COMMENT_ABRUPT.match(raw,start+4)
        return 'comment',closing.end() if closing else len(raw)
    if raw.startswith('<![CDATA[',start):
        at=raw.find(']]>',start+9)
        return 'cdata',at+3 if at>=0 else len(raw)
    if raw.startswith('</>',start):return 'empty-close',start+3
    if raw.startswith(('<!','<?'),start):
        at=raw.find('>',start+2)
        return ('processing-instruction' if raw.startswith('<?',start) else 'declaration'),at+1 if at>=0 else len(raw)
    return None


# This predicate is shared with the actual paragraph producer. A finite
# inline leaf never searches for its delimiter in the next producer's block.
AUTOLINK=re.compile(r'<(https?://[^ <>\n]+|mailto:[^ <>\n]+)>')
ATTRIBUTE_TOKEN=re.compile(r'''(?P<id>\#[\w:.+-]+)|(?P<class>\.[A-Za-z_][\w:.+-]*)|(?P<special>-(?![\w-]))|(?P<key>[A-Za-z][\w:.-]*)=(?P<value>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s{}"']*)''')
SEMANTIC_COMMAND=re.compile(r'\\(label|ref|eqref)\{([^{}\n]+)\}')
RAW_COMMAND=re.compile(r'\\[A-Za-z]+(?:\{[^{}]*\})?')
ESCAPES=r'\`*_{}[]()#+-.!$<>|:;"\'=&'


def fence_decision(raw,start,nested=False,paragraph_open=False,host=None):
    """Finite Pandoc-3.1.3 block ownership, shared by all block consumers.

    Tilde requires separation from a paragraph; a closed backtick fence may
    interrupt it. Ordinary EOF markers have no block ownership. Exact raw
    formats retain the profile's explicit EOF diagnostics. Nested quotes
    preserve their prefix while looking for a closing line.
    """
    end=raw.find('\n',start);end=len(raw) if end<0 else end+1
    line=raw[start:end].rstrip('\r\n')
    match=re.match(r'^(?: {0,3}> ?)* {0,3}(`{3,}|~{3,})(.*)$' if nested else r'^ {0,3}(`{3,}|~{3,})(.*)$',line)
    if not match:return None
    marker=match[1];info=match[2].strip();prefix=line[:match.start(1)]
    metadata=metadata_at(info,0,raw_format=True) if info.startswith('{') else None
    if info and not (metadata and metadata.end()==len(info)) and not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_+.#-]*',info):return None
    if marker[0]=='~' and paragraph_open:return None
    closing=re.compile((re.escape(prefix) if prefix.strip() else r' {0,3}')+re.escape(marker[0])+r'{'+str(len(marker))+r',}[ \t]*[\r\n]*')
    body_end=end
    while body_end<len(raw):
        next_end=raw.find('\n',body_end);next_end=len(raw) if next_end<0 else next_end+1
        if closing.fullmatch(raw[body_end:next_end]):break
        body_end=next_end
    closed=body_end<len(raw)
    format=info[2:-1] if info.startswith('{=') and info.endswith('}') else None
    if not closed and format not in ('html','latex'):return None
    closing_end=raw.find('\n',body_end);closing_end=len(raw) if closing_end<0 else closing_end+1
    inline_owner=host=='list' and marker[0]=='`' and not prefix.strip() and (start==0 or paragraph_open)
    return {'kind':'raw_block' if format else 'code_block','start':start,'end':closing_end,
            'body_start':end,'body_end':body_end,'mark':marker,'info':info,
            'metadata':format,'closed':closed,'owner':'inline-code' if inline_owner and not format else 'block',
            'paragraph_after':inline_owner}


def paragraph_boundary(raw,start):
    # Pinned blank_before_header/blockquote and lists_without_preceding_blankline
    # disabled: a marker alone cannot terminate an already open paragraph.
    return bool(fence_decision(raw,start,paragraph_open=True))


def inline_scope_end(raw,start):
    """End of the finite producer domain, before blank/next block markers.

    Inline callers already receive a cell/caption/heading/label's own string.
    This additional bound prevents an inert-token merge from enlarging the
    lookahead for a preceding unmatched literal. Soft lines remain supported.
    """
    at=raw.find('\n',start)
    while at>=0:
        following=at+1;end=raw.find('\n',following)
        line=raw[following:len(raw) if end<0 else end]
        if not line.strip() or paragraph_boundary(raw,following):return following
        at=end
    return len(raw)


class MetadataDecision:
    """Accepted interval and tokens; match-like access keeps source coordinates."""
    def __init__(self,raw,start,end,tokens=()):
        self.raw=raw;self.lo=start;self.hi=end;self.tokens=tokens
    def __getitem__(self,key):return self.raw[self.lo:self.hi] if key==0 else self.raw[self.lo+1:self.hi-1]
    def start(self,group=0):return self.lo+int(group==1)
    def end(self,group=0):return self.hi-int(group==1)


def metadata_at(raw,start,limit=None,raw_format=False):
    limit=len(raw) if limit is None else limit
    if start>=limit or raw[start]!='{':return None
    if raw_format:
        match=re.compile(r'\{=[A-Za-z0-9_+-]+\}').match(raw,start,limit)
        if match:return MetadataDecision(raw,start,match.end())
    at=start+1;tokens=[]
    while at<limit:
        whitespace=re.compile(r'[ \t\r\n]*').match(raw,at,limit);at=whitespace.end()
        if re.search(r'\n[ \t\r]*\n',raw[start:at]):return None
        if at<limit and raw[at]=='}':return MetadataDecision(raw,start,at+1,tokens)
        token=ATTRIBUTE_TOKEN.match(raw,at,limit)
        if not token:return None
        tokens.append(token);at=token.end()
    return None


def trailing_metadata(raw):
    for match in re.finditer(r'\{',raw):
        decision=metadata_at(raw,match.start())
        if decision and decision.end()==len(raw):return decision
    return None


def span_at(raw,start,limit):
    label=re.compile(r'\[([^\]\n]*)\]').match(raw,start,limit)
    metadata=metadata_at(raw,label.end(),limit) if label else None
    return (label,metadata) if metadata else None


def div_at(line):
    match=re.match(r'^ {0,3}(:{3,})(.*)$',line)
    if not match:return None
    tail=match[2].strip()
    if not tail:return {'close':True,'metadata':None}
    metadata=metadata_at(tail,0)
    if metadata and metadata.end()==len(tail):return {'close':False,'metadata':metadata,'start':line.index(tail)}
    if re.fullmatch(r'[A-Za-z_][\w.-]*',tail):return {'close':False,'class':tail,'start':line.index(tail)}
    return None


def command_at(raw,start,limit=None):
    limit=len(raw) if limit is None else limit
    semantic=SEMANTIC_COMMAND.match(raw,start,limit)
    if semantic:return semantic
    if start+1<limit and raw[start+1] in ESCAPES:return None
    return RAW_COMMAND.match(raw,start,limit)


def code_end(raw,start,limit=None):
    tick=re.match(r'`+',raw[start:])[0]
    limit=len(raw) if limit is None else limit
    end=raw.find(tick,start+len(tick),limit)
    return tick,end


def math_end(raw,start,limit=None):
    mark='$$' if raw.startswith('$$',start) else '$'
    limit=len(raw) if limit is None else limit
    end=start+len(mark)
    while True:
        end=raw.find(mark,end,limit)
        if end<0:return mark,-1
        if raw[end-1]!='\\' and (mark=='$$' or not raw[end-1].isspace()) and (mark=='$$' or end+1==limit or not raw[end+1].isdigit()):return mark,end
        end+=len(mark)


def literal_atom(raw,start,limit):
    """One finite decision, including code metadata and incomplete bounds.

    Callers must supply the producer/label barrier bound, not a stripped
    string's inferred block scope. Guard and projection share this record.
    """
    if raw[start]=='`':
        mark,close=code_end(raw,start,limit)
        end=close+len(mark) if close>=0 else limit
        metadata=metadata_at(raw,end,limit,raw_format=True) if close>=0 else None
        return {'kind':'raw_inline' if metadata and metadata[1].startswith('=') else 'code',
                'start':start,'end':metadata.end() if metadata else end,
                'body_start':start+len(mark),'body_end':close if close>=0 else limit,
                'mark':mark,'closed':close>=0,'metadata':metadata[1] if metadata else None,
                'owner':'markdown'}
    if raw[start]=='$' and start+1<limit and not raw[start+1].isspace():
        mark,close=math_end(raw,start,limit)
        if close>=0 or mark=='$$' or raw[start+1].isalpha() or raw[start+1]=='\\':
            return {'kind':'math','start':start,'end':close+len(mark) if close>=0 else limit,
                    'body_start':start+len(mark),'body_end':close if close>=0 else limit,
                    'mark':mark,'closed':close>=0,'metadata':None,'owner':'markdown'}
    return None


class InlineDecision:
    """Private ledger in a single producer domain; slices preserve decisions."""
    def __init__(self,raw,inert=(),blocks=False,source_map=None):
        self.raw=raw;self.atoms={};self.events=[];self.metadata=[]
        self.source_map=list(range(len(raw)+1)) if source_map is None else source_map
        assert len(self.source_map)==len(raw)+1
        self.spans=markdown_lexemes(raw,False,_html_events=self.events,_inert=inert,
                                    _atoms=self.atoms,_container_blocks=blocks,_metadata_events=self.metadata)
        self.coverage=(0,len(raw));self.blocks=blocks
        for event in self.events:event['source_start']=self.source_map[event['start']]
    def slice(self,start,end):
        value=object.__new__(InlineDecision);value.raw=self.raw[start:end]
        value.atoms={}
        for at,atom in self.atoms.items():
            if start<=at and atom['end']<=end:
                copy=dict(atom)
                for key in ('start','end','body_start','body_end'):
                    if key in copy:copy[key]-=start
                if 'link' in copy:
                    copy['link']=dict(copy['link'])
                    for key in ('label_start','label_end'):copy['link'][key]-=start
                value.atoms[at-start]=copy
        value.events=[{**e,**{k:e[k]-start for k in ('start','tag_end','end','body_start') if e.get(k) is not None}}
                      for e in self.events if start<=e['start'] and e['tag_end']<=end]
        value.spans=[(lo-start,hi-start) for lo,hi in self.spans if start<=lo and hi<=end]
        value.metadata=[{**e,'start':e['start']-start} for e in self.metadata if start<=e['start']<end]
        value.coverage=(0,end-start);value.blocks=False
        value.source_map=self.source_map[start:end+1]
        return value

    def attributes(self,raw,start):
        return attrs(raw,self.source_map[start],self.source_map[start:start+len(raw)+1])


_GUARD=ContextVar('reader_guard',default=None)

def guard_event(event):
    """Queue source-owned validation, independent of projection order/failure."""
    session=_GUARD.get()
    if session is None:
        if event['kind']=='html':PassiveHTML().handle_starttag(event['tag'],event['attributes'])
        else:raise Unsafe('Active markdown attribute')
        return
    key=(event['kind'],event['source_start'],event['snippet'])
    session['events'].setdefault(key,event)


def drain_guards(session):
    for event in sorted(session['events'].values(),key=lambda e:e['source_start']):
        session['current']=event
        try:
            if event['kind']=='html':PassiveHTML().handle_starttag(event['tag'],event['attributes'])
            else:raise Unsafe('Active markdown attribute')
        except Unsafe as exc:report_unsafe(exc)
    session.pop('current',None)


def report_unsafe(exc):
    session=_GUARD.get()
    if session is None:raise exc
    if session['first'] is None:session['first']=exc


def html_tag_boundary(raw,start,limit=None):
    """One tag/attribute decision for scanner, guard and tree.

    Use the pinned CPython matchers, including bare values containing quotes
    or '<'. The sole finite extension is F36: an inert opener in tag/attribute
    *syntax*, before a closing '>', separates the unfinished prefix. Openers
    in attribute values are literal, quoted or bare; no second quote lexer.
    Return (complete end, prefix end), in original character coordinates.
    """
    limit=len(raw) if limit is None else limit
    name_start=start+(2 if raw.startswith('</',start) else 1)
    name=tagfind_tolerant.match(raw,name_start,limit)
    if not name:return None,None
    def syntax_token(lo,hi):
        at=raw.find('<',lo,hi)
        while at>=0:
            if html_lexeme(raw,at):return at
            at=raw.find('<',at+1,hi)
        return None
    token=syntax_token(name.start(1),name.end(1))
    if token is not None:return None,token
    at=name.end()
    while at<limit:
        if html_lexeme(raw,at):return None,at
        attribute=attrfind_tolerant.match(raw,at,limit)
        if not attribute:break
        token=syntax_token(attribute.start(1),attribute.end(1))
        if token is not None:return None,token
        at=attribute.end()
    match=locatetagend.match(raw,name_start,limit)
    end=match.end() if match and raw[match.end()-1]=='>' else None
    return end,None


def tag_end(raw,start,limit):return html_tag_boundary(raw,start,limit)[0]


def incomplete_tag_token(raw,start):return html_tag_boundary(raw,start)[1]


def source_scope_end(raw,start):
    """Same finite prefix/continuation domains as observe, before merging."""
    first_end=raw.find('\n',start);first_end=len(raw) if first_end<0 else first_end+1
    first=raw[start:first_end];kind=None
    if re.match(r'^ {0,3}#{1,6}[ \t]+',first):return first_end
    if re.match(r'^ {0,3}(?:[-+*]|[0-9]{1,9}[.)])[ \t]+',first):kind='list'
    elif re.match(r'^ {0,3}>',first):kind='quote'
    elif re.match(r'^\[\^[^\]]+\]:',first):kind='note'
    if kind:
        at=first_end
        while at<len(raw):
            end=raw.find('\n',at);end=len(raw) if end<0 else end+1;line=raw[at:end]
            if kind=='list' and not line.startswith('    '):break
            if kind=='quote' and not re.match(r'^ {0,3}>',line):break
            if kind=='note' and not line.startswith(('    ','\t')):break
            at=end
        return at
    if re.match(r'</?[A-Za-z][\w:-]*\b',first) and not AUTOLINK.match(first):
        at=first_end
        while at<len(raw):
            end=raw.find('\n',at);end=len(raw) if end<0 else end+1
            if not raw[at:end].strip():break
            at=end
        return at
    return inline_scope_end(raw,start)


def raw_fence_end(raw,start,marker):
    """The raw inline producer consumes the whole fence body, including blanks.

    A fence-looking line inside an inert token can be crossed by that token;
    the next bound is then searched from its end, never from an expired cursor.
    """
    closing=re.compile(r' {0,3}'+re.escape(marker[0])+r'{'+str(len(marker))+r',}[ \t]*[\r\n]*')
    at=start
    while at<len(raw):
        end=raw.find('\n',at);end=len(raw) if end<0 else end+1
        if (at==0 or raw[at-1]=='\n') and closing.fullmatch(raw[at:end]):return at
        at=end
    return len(raw)


def markdown_lexemes(raw,block_context=True,_metadata_start=None,_html_events=None,_inert=(),_atoms=None,_container_blocks=False,_metadata_events=None,_block_decisions=None):
    """Boundary walk uses the same finite atoms as the inline consumers.

    Literal lookahead is bounded by the actual producer. Only an inert token
    recognized outside these atoms may extend through Markdown boundaries.
    Quoted HTML attrs, Markdown metadata, destinations, commands and autolinks
    consume exactly their own intervals; none installs a false HTML stack.
    """
    spans=[];i=0;html_stack=[];raw_fence=None;raw_mode=None;destinations={};metadata={};labels=[];label_atoms={};pipe_table=False;caption_pending=False;paragraph_open=False;physical_host=None;div_depth=0;limit=0 if block_context else len(raw)
    while i<len(raw):
        opaque=next((hi for lo,hi in _inert if lo<=i<hi),None)
        if opaque is not None:
            if _atoms is not None:_atoms[i]={'kind':'opaque_token','start':i,'end':opaque,'owner':'source-token'}
            i=opaque;continue
        line_start=i==0 or raw[i-1]=='\n'
        if line_start:
            end=raw.find('\n',i);end=len(raw) if end<0 else end+1
            line=raw[i:end].rstrip('\r\n')
            before=paragraph_open
            block={'paragraph_before':before,'fence':None,'heading':False,'div':None,'container':None}
            if _block_decisions is not None:_block_decisions[i]=block
            if not line.strip() and not raw_fence:
                caption_pending=caption_pending or pipe_table;pipe_table=False;paragraph_open=False;physical_host=None;html_stack.clear();i=end;limit=i if block_context else len(raw);continue
            if i>=limit:
                html_stack.clear()
                if block_context and not raw_fence:
                    next_end=raw.find('\n',end);next_end=len(raw) if next_end<0 else next_end+1
                    begins_table='|' in line and end<len(raw) and all(re.fullmatch(r':?-{3,}:?',c.strip()) for c in pipe_row(raw[end:next_end]))
                    pipe_table=begins_table or pipe_table and '|' in line
                    limit=end if pipe_table or caption_pending and re.match(r'^(?:Table:|:) ',line) else source_scope_end(raw,i)
                    caption_pending=False
            if raw_fence and re.fullmatch(r' {0,3}'+re.escape(raw_fence[0])+r'{'+str(len(raw_fence))+r',}[ \t]*',line):
                raw_fence=None;raw_mode=None;paragraph_open=False;html_stack.clear();i=end;limit=i;continue
            fence=fence_decision(raw,i,nested=_container_blocks,paragraph_open=paragraph_open,host=_container_blocks) if (block_context or _container_blocks) and not raw_fence and not html_stack else None
            if fence:
                block['fence']=fence
                marker=fence['mark']
                if _container_blocks:
                    atom=dict(fence)
                    if _atoms is not None:_atoms[i]=atom
                    if _metadata_events is not None and atom['metadata'] is None and atom['info'].startswith('{'):
                        _metadata_events.append({'start':i+line.index(atom['info'])+1,'raw':atom['info'].strip('{}')})
                    if atom['metadata']=='html' and _html_events is not None:
                        _html_events.append({'start':i,'tag_end':atom['end'],'end':atom['end'],
                                             'body_start':atom['body_start'],'raw_html':raw[atom['body_start']:atom['body_end']]})
                    i=atom['end'];paragraph_open=atom['paragraph_after'];html_stack.clear();continue
                if fence['metadata'] in ('html','latex'):
                    raw_fence=marker;raw_mode=fence['metadata'];i=fence['body_start'];limit=fence['body_end'];continue
                i=fence['end'];paragraph_open=False;html_stack.clear();limit=i;continue
            logical_line=re.sub(r'^(?: {0,3}> ?)+','',line) if _container_blocks else line
            first_note_text=_container_blocks=='note' and i==0 and logical_line==line
            block['heading']=not before and not first_note_text and bool(re.match(r'^ {0,3}#{1,6}[ \t]+',logical_line))
            div=div_at(logical_line)
            block['div']=div if not before or div and div['close'] and div_depth else None
            if block['div']:div_depth=max(0,div_depth+(-1 if div['close'] else 1))
            if block_context and not before:
                if re.match(r'^ {0,3}(?:[-+*]|[0-9]{1,9}[.)])[ \t]+',line):block['container']='list'
                elif re.match(r'^ {0,3}>',line):block['container']='quote'
                elif re.match(r'^\[\^[^\]]+\]:',line):block['container']='note'
            if block['container']:physical_host=block['container']
            elif physical_host=='list' and not line.startswith(('    ','\t')):physical_host=None
            paragraph_open=bool(logical_line.strip()) and not (block['heading'] or block['div'])
            # A root fence after a list item starts outside that item (K2c).
            # Quotes/notes and a figure remain a paragraph for lazy continuation.
            if block_context and physical_host=='list':paragraph_open=False
            # Heading metadata belongs to the source block, never to an
            # already-produced inline value with a literal logical '# '.
            if block_context and not raw_fence and block['heading']:
                attr=trailing_metadata(line)
                if attr:metadata[i+attr.start()]=i+attr.end()
            if block_context and block['div'] and block['div'].get('metadata'):
                at=block['div']['start'];metadata[i+at]=i+at+block['div']['metadata'].end()
            if not line.strip() and not raw_fence:html_stack.clear()
        if i==_metadata_start:return spans
        if i in destinations:i=destinations.pop(i);continue
        if i in metadata:i=metadata.pop(i);continue
        if (not raw_fence or raw_mode=='latex') and not html_stack:
            labels=[v for v in labels if i<v['end']]
            bound=min([limit,*(v['end'] for v in labels),*(lo for lo,hi in _inert if lo>i)])
            if pipe_table:
                row_start=raw.rfind('\n',0,i)+1;cuts=pipe_cuts(raw[row_start:limit]);bound=min([bound,*(row_start+v for v in cuts if row_start+v>i)])
            # Label acceptance records LF barriers once. Escaped LF and LF
            # in closed code are owned; opaque-token LF lends no closure.
            for label in labels:
                index=bisect_right(label['lf_barriers'],i)
                if index<len(label['lf_barriers']):bound=min(bound,label['lf_barriers'][index]+1)
            if raw[i]=='\\':
                command=command_at(raw,i,bound)
                i=command.end() if command else min(i+2,bound);continue
            atom=label_atoms.get(i) or literal_atom(raw,i,bound)
            if atom:
                if _atoms is not None:_atoms[i]=atom
                if atom['closed']:
                    if _metadata_events is not None and atom['metadata'] is not None and not atom['metadata'].startswith('='):
                        _metadata_events.append({'start':atom['body_end']+len(atom['mark'])+1,'raw':atom['metadata']})
                    if _html_events is not None and atom['kind']=='raw_inline' and atom['metadata']=='=html':
                        _html_events.append({'start':i,'tag_end':atom['end'],'end':atom['end'],
                                             'body_start':atom['body_start'],'raw_html':raw[atom['body_start']:atom['body_end']].replace('\n',' ')})
                    i=atom['end'];continue
                i+=len(atom['mark']);continue
            if raw[i] in '![':
                local=raw[:bound];link=markdown_link(raw,i,bound)
                if link and i+link['length']>bound and not any(lo<bound<hi for lo,hi in link['label_tokens']):link=None
                if link:
                    label_atoms.update(link['label_atoms'])
                    labels.append({'end':link['label_end'],'lf_barriers':link['lf_barriers']})
                    if _atoms is not None:_atoms[i]={'kind':'link','start':i,'end':i+link['length'],'link':link}
                    end=i+link['length'];attr=metadata_at(raw,end,max(bound,end))
                    if attr and _metadata_events is not None:_metadata_events.append({'start':attr.start(1),'raw':attr[1]})
                    destinations[i+(2 if link['image'] else 1)+len(link['label'])]=attr.end() if attr else end
                else:
                    walk=markdown_label(raw,i,bound)
                    if walk:label_atoms.update(walk['atoms'])
                    if walk and any(lo<bound<hi for lo,hi in walk['tokens']):
                        # A label's literal atoms cannot borrow closure from
                        # the body of its opaque EOF token. The same walk is
                        # used by the constructor, even when destination lost.
                        labels.append({'end':walk['end'],'lf_barriers':walk['lf_barriers']})
                    span=span_at(raw,i,bound)
                    if span:
                        label,meta=span
                        destinations[label.end()-1]=meta.end()
                        if _metadata_events is not None:_metadata_events.append({'start':meta.start(1),'raw':meta[1]})
        if raw[i]=='<':
            if (not raw_fence or raw_mode=='latex') and not html_stack:
                auto=AUTOLINK.match(raw,i,min(limit,inline_scope_end(raw,i)))
                if auto:i=auto.end();continue
            token=html_lexeme(raw,i)
            if token:
                if _atoms is not None:_atoms[i]={'kind':'opaque_token','start':i,'end':token[1],'owner':'html-token','token_type':token[0]}
                spans.append((i,token[1]));i=token[1]
                if i>limit:
                    limit=raw_fence_end(raw,i,raw_fence) if raw_fence else source_scope_end(raw,i) if not pipe_table else (raw.find('\n',i)+1 or len(raw))
                continue
            tag=re.match(r'</?([A-Za-z][\w:-]*)\b',raw[i:])
            if tag:
                end=tag_end(raw,i,limit)
                if end is None:i+=1;continue
                name=tag[1].lower()
                if raw.startswith('</',i):
                    names=[n for n,event in html_stack]
                    if name in names:
                        index=len(names)-1-names[::-1].index(name)
                        event=html_stack[index][1]
                        if event is not None:event['end']=end
                        html_stack=html_stack[:index]
                else:
                    closed=name in ('br','img','hr','meta','link','input','wbr') or raw[end-2]=='/'
                    event={'start':i,'tag_end':end,'end':end if closed else None} if _html_events is not None else None
                    if event is not None:_html_events.append(event)
                    if not closed:html_stack.append((name,event))
                i=end;continue
        i+=1
    return spans

def markdown_units(raw,_spans=None):
    """Suppress only physical boundaries strictly inside an inert token."""
    spans=markdown_lexemes(raw) if _spans is None else _spans
    physical=raw.splitlines(keepends=True);units=[];numbers=[];offset=0;part='';first=0;cursor=0
    for number,line in enumerate(physical):
        if not part:first=number
        part+=line;offset+=len(line)
        while cursor<len(spans) and spans[cursor][1]<=offset:cursor+=1
        inside=cursor<len(spans) and spans[cursor][0]<offset<spans[cursor][1]
        if not inside:units.append(part);numbers.append(first);part=''
    if part:units.append(part);numbers.append(first)
    return units,numbers,len(physical)


def strip_inline_end(raw,characters='\r\n',inert=(),_source_proven=False):
    """Serialization padding may be removed only outside an inert token."""
    trimmed=raw.rstrip(characters)
    spans=inert if _source_proven else [*inert,*markdown_lexemes(raw,False)]
    if len(trimmed)<len(raw) and any(a<=len(trimmed)<b for a,b in spans):return raw
    return trimmed


def inline_delimiter(raw,marker,start,inert=(),producer_limit=None,html_events=None,decision=None):
    """Find a formatting delimiter without cutting a token or literal leaf."""
    i=start
    while i<len(raw):
        if raw[i]=='\\':
            command=re.match(r'\\[A-Za-z]+(?:\{[^{}]*\})?',raw[i:])
            i+=len(command[0]) if command else 2;continue
        atom=decision.atoms.get(i) if decision else literal_atom(raw,i,min([len(raw),*(lo for lo,hi in inert if lo>i)]))
        if atom and atom['kind'] in ('code','raw_inline','math','raw_block','code_block'):
            if not atom['closed']:return -1
            i=atom['end'];continue
        link=atom.get('link') if atom and atom['kind']=='link' else markdown_link(raw,i,producer_limit) if decision is None and raw[i] in '![' else None
        if link:i+=link['length'];continue
        if raw[i]=='<':
            token=html_lexeme(raw,i)
            if token:i=token[1];continue
            if re.match(r'<[A-Za-z][\w:-]*\b',raw[i:]):
                try:
                    event=(html_events or {}).get(i)
                    end=(event['end'] or event['tag_end']) if event else i
                    snippet=raw[i:end]
                    tree=HTMLTree(snippet,guarded=True);tree.feed(snippet);tree.close();first=tree.root['children'][0]
                    if first['tag']!='#text' and first.get('closed'):i+=first['end'];continue
                except (ValueError,IndexError):pass
        if raw.startswith(marker,i):return i
        i+=1
    return -1


def attrs(raw,source_start=0,source_map=None):
    values={}
    if source_map is not None:assert len(source_map)==len(raw)+1
    decision=metadata_at('{'+raw+'}',0)
    if decision is None or decision.end()!=len(raw)+2:return {'unknown':[raw]}
    for match in decision.tokens:
        token=match[0];offset=match.start()-1
        if match['id']:values['id']=token[1:]
        elif match['class']:values.setdefault('classes',[]).append(token[1:])
        elif match['special']:values.setdefault('classes',[]).append('unnumbered')
        elif match['key']:
            key,value=match['key'],match['value']
            if key.lower().startswith('on') or key in ('style','srcdoc'):
                at=source_map[offset] if source_map is not None else source_start+offset
                guard_event({'kind':'markdown-attribute','source_start':at,'snippet':token})
                continue
            if value.startswith(('"',"'")):
                value=re.sub(r'\\(.)',r'\1',value[1:-1])
                value=re.sub(r'\r?\n',' ',value)
            values[key]=value
    return values


def peel_attrs(value,inert=(),source_start=0):
    match=trailing_metadata(value)
    if match and any(a<=match.start()<b for a,b in [*inert,*markdown_lexemes(value,False,match.start())]):return value,{}
    return (value[:match.start()].rstrip(' '),attrs(match[1],source_start+match.start(1))) if match else (value,{})


def pipe_cuts(line):
    """Delimiters outside the finite leaves consumed by inline, no gold."""
    cuts=[];i=0
    while i<len(line):
        if line[i]=='\\':
            atom=command_at(line,i)
            i=atom.end() if atom else min(i+2,len(line));continue
        if line[i]=='`':
            atom=literal_atom(line,i,len(line))
            if not atom['closed']:break
            i=atom['end'];continue
        if line[i] in '![':
            link=markdown_link(line,i)
            span=re.match(r'\[([^\]\n]*)\]\{([^{}]*)\}',line[i:]) if not link else None
            if link:
                i+=link['length'];meta=metadata_at(line,i)
                if meta:i=meta.end()
                continue
            if span:i+=len(span[0]);continue
        token=html_lexeme(line,i) if line[i]=='<' else None
        if token:i=token[1];continue
        if line[i]=='|':cuts.append(i)
        i+=1
    return cuts


def pipe_row(line,inert=()):
    leading=len(line)-len(line.lstrip(' \t'));line=line[leading:]
    known=[(a-leading,b-leading) for a,b in inert]
    line=strip_inline_end(line,' \t\r\n',known)
    start=1 if line.startswith('|') else 0;cuts=[at for at in pipe_cuts(line) if at>=start]
    cells=[]
    for end in [*cuts,len(line)]:
        if end==len(line) and start==end and cells:break
        value=line[start:end];padding=len(value)-len(value.lstrip(' '));value=value[padding:]
        barriers=[(a-start-padding,b-start-padding) for a,b in known if start+padding<=a and b<=end]
        cells.append(strip_inline_end(value,' ',barriers));start=end+1
    return cells

def markdown_label(raw,start,limit=None):
    """Finite label walk, including opaque EOF even without a destination."""
    image=raw.startswith('![',start);opening=start+(2 if image else 1)
    if not raw.startswith('[',start+int(image)):return None
    i=opening;level=1;label_tokens=[];codes={};atoms={};lf_barriers=[]
    while i<len(raw) and raw[i]!='\n':
        if raw[i]=='`':
            atom=literal_atom(raw,i,inline_scope_end(raw,i) if limit is None else limit)
            if atom['closed']:
                codes[i]=atom['body_end']+len(atom['mark']);atoms[i]=atom;i=codes[i];continue
        token=html_lexeme(raw,i) if raw[i]=='<' else None
        if token:
            label_tokens.append((i,token[1]));lf_barriers.extend(i+m.start() for m in re.finditer('\n',raw[i:token[1]]));i=token[1];continue
        if raw[i]=='\\' and i+1<len(raw):i+=2;continue
        if raw[i]=='[':level+=1
        if raw[i]==']':
            level-=1
            if level==0:break
        if level>16:return None
        i+=1
    # Metadata does not change the historical label acceptance grammar.
    # Bound it to the accepted label, retaining the already-decided code body.
    for at,atom in atoms.items():
        if atom['end']>i:
            atom['end']=codes[at];atom['metadata']=None;atom['kind']='code'
    return {'opening':opening,'end':i,'tokens':label_tokens,'codes':codes,'atoms':atoms,'lf_barriers':lf_barriers}


def markdown_link(raw,start,limit=None):
    """Finite balanced destination and separate optional CommonMark title."""
    image=raw.startswith('![',start);walk=markdown_label(raw,start,limit)
    if not walk:return None
    opening=walk['opening'];i=walk['end'];label_tokens=walk['tokens']
    if i>=len(raw) or not raw.startswith('](',i):return None
    label=raw[opening:i];i+=2
    while i<len(raw) and raw[i] in ' \t':i+=1
    begin=i
    if i<len(raw) and raw[i]=='<':
        i+=1;begin=i
        while i<len(raw) and raw[i] not in '>\n':
            if raw[i]=='\\':i+=1
            i+=1
        if i>=len(raw) or raw[i]!='>':return None
        target=raw[begin:i];i+=1
    else:
        level=0
        while i<len(raw) and raw[i]!='\n':
            if raw[i]=='\\' and i+1<len(raw):i+=2;continue
            if raw[i]=='(':
                level+=1
                if level>16:return None
            elif raw[i]==')':
                if level==0:break
                level-=1
            elif raw[i].isspace() and level==0:break
            i+=1
        if level:return None
        target=raw[begin:i]
    separated=False
    while i<len(raw) and raw[i] in ' \t':separated=True;i+=1
    title=None
    if separated and i<len(raw) and raw[i] in '\"\'(':
        opening=raw[i];closing=')' if opening=='(' else opening;i+=1;begin=i
        while i<len(raw) and raw[i]!='\n':
            if raw[i]=='\\' and i+1<len(raw):i+=2;continue
            if raw[i]==closing:break
            # Parenthesized titles do not have nested parentheses in this profile.
            if opening=='(' and raw[i]=='(':return None
            i+=1
        if i>=len(raw) or raw[i]!=closing:return None
        title=raw[begin:i];i+=1
        while i<len(raw) and raw[i] in ' \t':i+=1
    if i>=len(raw) or raw[i]!=')':return None
    decoded=re.sub(r'\\([!"#$%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~])',r'\1',target)
    return {'image':image,'label':label,'target':decoded,'target_raw':target,'title':title,'length':i+1-start,'label_start':start+(2 if image else 1),'label_end':start+(2 if image else 1)+len(label),'label_tokens':label_tokens,'label_codes':walk['codes'],'label_atoms':walk['atoms'],'lf_barriers':walk['lf_barriers']}


def shift_inline(value,offset):
    if '_source_map' in value:
        mapping=value['_source_map']
        value['_source_map']=[mapping[0]-offset+i for i in range(offset)]+mapping
    for field in ('references','formulas','unknown','segments'):
        for occurrence in value[field]:
            for key in ('start','end','offset','source_offset'):
                if key in occurrence:occurrence[key]+=offset
    return value


def located(parsed,location,raw,offset=0):
    """Keep one typed occurrence per leaf, with container coordinates."""
    result={**parsed,'raw':raw,'location':location,'source_offset':offset}
    result.pop('_source_map',None)
    for field in ('references','formulas','unknown','segments'):
        result[field]=[{**v,'location':location,'source_offset':offset+v.get('start',v.get('offset',0))} for v in parsed[field]]
    return result


def table_contents(rows,caption=None,offsets=None,caption_offset=0,inert=(),source_offset=0):
    def parse(value,at):
        known=[(lo-source_offset-at,hi-source_offset-at) for lo,hi in inert if source_offset+at<=lo and hi<=source_offset+at+len(value)]
        decision=InlineDecision(value,known,source_map=list(range(source_offset+at,source_offset+at+len(value)+1)))
        return inline(value,inert=known,_decision=decision)
    grid=[[located(parse(c,(offsets or {}).get((ri,ci),0)),{'kind':'cell','row':ri,'col':ci},c,(offsets or {}).get((ri,ci),0)) for ci,c in enumerate(row)] for ri,row in enumerate(rows)]
    cap=located(parse(caption,caption_offset),{'kind':'caption'},caption,caption_offset) if caption is not None else None
    leaves=[c for row in grid for c in row]+([cap] if cap else [])
    return {'cells':[[c['text'] for c in row] for row in grid],'cell_contents':grid,'caption':cap['text'] if cap else None,'caption_content':cap,'references':[v for c in leaves for v in c['references']],'formulas':[v for c in leaves for v in c['formulas']],'unknown':[v for c in leaves for v in c['unknown']],'anchors':[v for c in leaves for v in c['anchors']]}


def inline_html_events(raw,inert=(),decision=None):
    """Validate the lexer-owned openings before projection can lose them.

    Closed Markdown atoms and source-proven opaque tokens stay inert. An
    unmatched delimiter is not a literal owner. Feed each actual start tag
    separately so a tree failure or parser state cannot hide the next one.
    The first passive violation retains source order while visits continue.
    """
    decision=decision or InlineDecision(raw,inert)
    events=decision.events
    for metadata in decision.metadata:decision.attributes(metadata['raw'],metadata['start'])
    unsafe=None
    for event in events:
        try:
            snippet=event.get('raw_html',raw[event['start']:event['tag_end']])
            start=event.get('body_start',event['start'])
            validate_html(snippet,decision.source_map[start:start+len(snippet)+1])
        except Unsafe as exc:
            if unsafe is None:unsafe=exc
    if unsafe is not None:report_unsafe(unsafe)
    return {event['start']:event for event in events}


def inline(raw,depth=0,inert=(),producer_limit=None,_decision=None):
    decision=_decision or InlineDecision(raw,inert)
    assert decision.raw==raw and decision.coverage==(0,len(raw))
    html_events=inline_html_events(raw,inert,decision)
    if depth>16:return {'text':raw,'references':[],'formulas':[],'anchors':[],'segments':[],'unknown':[{'raw':raw,'offset':0,'reason':'inline depth>16','scope':'text','resource_syntax':True}]}
    text=[];refs=[];formulas=[];anchors=[];segments=[];unknown=[];i=0
    def add(value,kind,start,end):
        text.append(value);segments.append({'type':kind,'text':value,'start':start,'end':end})
    def transport(parsed,offset):
        parsed=shift_inline(parsed,offset)
        text.append(parsed['text'])
        refs.extend(parsed['references']);formulas.extend(parsed['formulas'])
        anchors.extend(parsed['anchors']);unknown.extend(parsed['unknown']);segments.extend(parsed['segments'])
    def leaf_end(start):return min([len(raw) if producer_limit is None else producer_limit,*(lo for lo,hi in inert if lo>start)])
    def sub_inert(start,end):return [(lo-start,hi-start) for lo,hi in inert if start<=lo and hi<=end]
    def lost_html(start,end):return any(start<=at<end for at in html_events)
    while i<len(raw):
        start=i;c=raw[i];atom=decision.atoms.get(i)
        if c=='\\':
            command=SEMANTIC_COMMAND.match(raw,i,leaf_end(i))
            if command:
                if command[1]=='label':anchors.append(command[2])
                else:refs.append({'kind':'ref','text':'','target':command[2]})
                i=command.end();continue
            if i+1<len(raw) and raw[i+1] in ESCAPES:add(raw[i+1],'text',i,i+2);i+=2;continue
            command=command_at(raw,i,leaf_end(i))
            if command:
                unknown.append({'raw':command[0],'offset':i,'reason':'raw TeX unsupported'});add(command[0],'unknown',i,i+len(command[0]));i=command.end();continue
            add(c,'text',i,i+1);i+=1;continue
        if atom and atom['kind'] in ('raw_block','code_block'):
            i=atom['end'];payload=raw[atom['body_start']:atom['body_end']]
            if atom['kind']=='raw_block':
                unknown.append({'raw':raw[start:i],'offset':start,'end':i,'reason':'container raw block not projected','format':atom['metadata'],'resource_syntax':True,'scope':'structure'})
            if not atom['closed']:
                unknown.append({'raw':raw[start:i],'offset':start,'reason':'unclosed container fence','resource_syntax':True,'scope':'structure'})
            add(payload,'raw_block' if atom['kind']=='raw_block' else 'code',start,i);continue
        if c=='`' and atom and atom['kind'] in ('code','raw_inline'):
            tick=atom['mark'];end=atom['body_end'] if atom['closed'] else -1
            if end<0:end=atom['end'];unknown.append({'raw':raw[i:end],'offset':i,'reason':'unclosed inline code','resource_syntax':lost_html(i,end)});add(raw[i:end],'unknown',i,end);i=end;continue
            payload=raw[i+len(tick):end].replace('\n',' ')
            # CommonMark's single padding-space rule belongs to serialization.
            if payload.startswith(' ') and payload.endswith(' ') and payload.strip(' '):payload=payload[1:-1]
            i=end+len(tick);match=metadata_at(raw,i,atom['end'],raw_format=True) if atom['metadata'] is not None else None
            metadata=decision.attributes(match[1],match.start(1)) if match and not match[1].startswith('=') else {}
            if match and match[1].startswith('='):
                format=match[1][1:]
                # Raw HTML was guarded from the original ledger before projection.
                i+=len(match[0]);unknown.append({'raw':raw[start:i],'offset':start,'reason':'raw inline format unsupported','format':format,'resource_syntax':True} if format=='html' else {'raw':raw[start:i],'offset':start,'reason':'raw inline format unsupported','format':format});add(payload,'raw_inline',start,i);continue
            if match:i+=len(match[0])
            add(payload,'code',start,i);segments[-1]['attrs']=metadata
            if metadata.get('id'):anchors.append(metadata['id'])
            continue
        if c=='$' and atom and atom['kind']=='math':
            mark=atom['mark'];end=atom['body_end'] if atom['closed'] else -1
            if end>=0:
                payload=raw[i+len(mark):end];formulas.append({'raw':payload,'display':mark=='$$','start':i,'end':end+len(mark),'math':math_observe(payload)})
                segments.append({'type':'math','start':i,'end':end+len(mark),'raw':raw[i:end+len(mark)]});i=end+len(mark);continue
            if mark=='$$' or raw[i+1].isalpha() or raw[i+1]=='\\':
                end=atom['end'];unknown.append({'raw':raw[i:end],'offset':i,'reason':'unclosed math delimiter','resource_syntax':lost_html(i,end)});add(raw[i:end],'unknown',i,end);i=end;continue
        note=re.match(r'\[\^([^\]\n]+)\]',raw[i:]) if c=='[' else None
        if note:refs.append({'kind':'note_call','text':'','target':note[1]});i+=len(note[0]);continue
        # A transported producer already decided its domain on the source.
        # Never reclassify a stripped continuation's literal marker as a block.
        link=atom['link'] if atom and atom['kind']=='link' else None
        if link:
            target=link['target'];nested=shift_inline(inline(link['label'],depth+1,sub_inert(link['label_start'],link['label_end']),len(link['label']),decision.slice(link['label_start'],link['label_end'])),i+2 if link['image'] else i+1);label=nested['text']
            refs.extend(nested['references']);formulas.extend(nested['formulas']);unknown.extend(nested['unknown']);anchors.extend(nested['anchors'])
            refs.append({'kind':'image' if link['image'] else 'link','text':label,'target':target,'title':link['title'],'target_raw':link['target_raw'],'start':i,'end':i+link['length']})
            end=i+link['length'];attribute=metadata_at(raw,end,leaf_end(i));metadata=decision.attributes(attribute[1],attribute.start(1)) if attribute else {}
            if attribute:end=attribute.end()
            if metadata.get('id'):anchors.append(metadata['id'])
            refs[-1]['attrs']=metadata;text.append(label)
            for s in nested['segments']:s['in_link']=not link['image'];s['attrs']=metadata
            segments.extend(nested['segments']);i=end;continue
        if c in '![' and re.match(r'!?\[[^\n]*?\]\(',raw[i:leaf_end(i)]):
            end=leaf_end(i);unknown.append({'raw':raw[i:end],'offset':i,'reason':'link/image outside finite profile','resource_syntax':True});add(raw[i:end],'unknown',i,end);i=end;continue
        span=span_at(raw,i,leaf_end(i)) if c=='[' else None
        if span:
            label,attribute=span
            nested=shift_inline(inline(label[1],depth+1,sub_inert(i+1,label.end()-1),len(label[1]),decision.slice(i+1,label.end()-1)),i+1);meta=decision.attributes(attribute[1],attribute.start(1));text.append(nested['text']);segments.extend({**s,'attrs':meta} for s in nested['segments'])
            refs.extend(nested['references']);formulas.extend(nested['formulas']);unknown.extend(nested['unknown']);anchors.extend(nested['anchors'])
            if meta.get('id'):anchors.append(meta['id'])
            i=attribute.end();continue
        if c in '*_':
            mark=c*2 if raw.startswith(c*2,i) else c
            intraword=c=='_' and i>0 and raw[i-1].isalnum() and i+1<len(raw) and raw[i+1].isalnum()
            end=-1 if intraword else inline_delimiter(raw,mark,i+len(mark),inert,producer_limit,html_events,decision)
            if end>i+len(mark):
                nested=shift_inline(inline(raw[i+len(mark):end],depth+1,sub_inert(i+len(mark),end),end-i-len(mark),decision.slice(i+len(mark),end)),i+len(mark));text.append(nested['text']);segments.extend(nested['segments']);refs.extend(nested['references']);formulas.extend(nested['formulas']);unknown.extend(nested['unknown']);anchors.extend(nested['anchors']);i=end+len(mark);continue
        if c=='<':
            auto=AUTOLINK.match(raw,i)
            if auto:refs.append({'kind':'link','text':auto[1],'target':auto[1]});add(auto[1],'link',i,auto.end());i=auto.end();continue
            # One balanced tree fragment; nested equal tags and self-closing
            # leaves share exactly the same producer/transport contract.
            if raw.startswith('</>',i) or re.match(r'<(?:[A-Za-z][\w:-]*\b|!|\?)',raw[i:]):
                try:
                    # A complete opening tag can own a balanced subtree and
                    # its inert tokens. An incomplete tag cannot absorb the
                    # next source-proven token as an attribute or fallback.
                    event=html_events.get(i)
                    token=(atom.get('token_type','source-token'),atom['end']) if atom and atom['kind']=='opaque_token' else html_lexeme(raw,i)
                    end=(event['end'] or event['tag_end']) if event else token[1] if token else leaf_end(i)
                    snippet=raw[i:end]
                    parser=HTMLTree(snippet,guarded=True);parser.feed(snippet);parser.close()
                    first=next((n for n in parser.root['children'] if isinstance(n,dict)),None)
                except ValueError: first=None
                if first and first['tag']!='#text' and first.get('closed'):
                    end=first['end'];snippet=raw[i:i+end];values=html_nodes(snippet,inline_context=True,guarded=True)
                    if len(values)==1:
                        transport(values[0],i);i+=end;continue
                # Openings were validated before any fallback. Only this HTML
                # leaf's interval reaches the tree; the Markdown tail is not HTML.
            if not re.match(r'<(?:/?[A-Za-z]|!|\?)',raw[i:]):
                add(c,'text',i,i+1);i+=1;continue
            end=leaf_end(i);unknown.append({'raw':raw[i:end],'offset':i,'reason':'inline HTML unsupported','resource_syntax':True,'scope':'text'});add(raw[i:end],'unknown',i,end);i=end;continue
        add(c,'text',i,i+1);i+=1
    return {'text':''.join(text),'body_text':''.join(s.get('text','') for s in segments if s['type'] not in ('link','image','math') and not s.get('in_link')),'references':refs,'formulas':formulas,'anchors':anchors,'segments':segments,'unknown':unknown,'_guard_coverage':{'domain':'producer','start':0,'end':len(raw),'raw_sha256':digest_bytes(raw.encode())}}


class FiniteHTML(HTMLParser):
    """Shared tokenizer extension; projection and validation use identical bytes."""
    def parse_starttag(self,i):
        token_at=incomplete_tag_token(self.rawdata,i)
        if token_at is not None:
            self.handle_incomplete_start(token_at-i)
            return token_at
        return super().parse_starttag(i)
    def handle_incomplete_start(self,length):pass


class HTMLGuard(FiniteHTML):
    """Collect every opening independently of the projection tree."""
    def __init__(self,raw,source_map):
        super().__init__(convert_charrefs=True);self.unsafe=None
        self.source_map=source_map;self.starts=[0,*(m.end() for m in re.finditer('\n',raw))]
    def handle_starttag(self,tag,attributes):
        line,col=self.getpos();at=self.starts[line-1]+col
        try:guard_event({'kind':'html','source_start':self.source_map[at],
                         'snippet':self.get_starttag_text(),'tag':tag,'attributes':attributes})
        except Unsafe as exc:
            if self.unsafe is None:self.unsafe=exc
    handle_startendtag=handle_starttag


def validate_html(raw,source_map=None):
    mapping=list(range(len(raw)+1)) if source_map is None else source_map
    assert len(mapping)==len(raw)+1
    guard=HTMLGuard(raw,mapping);guard.feed(raw);guard.close()
    if guard.unsafe is not None:report_unsafe(guard.unsafe)


class HTMLTree(FiniteHTML):
    def __init__(self,raw='',guarded=False):super().__init__(convert_charrefs=True);self.raw=raw;self.guarded=guarded;self.starts=[0];self.starts.extend(m.end() for m in re.finditer(r'\n',raw));self.root={'tag':'root','attrs':{},'children':[]};self.stack=[self.root];self.errors=[];self.node_count=0
    def append_node(self,node):
        self.node_count+=1
        if self.node_count>20000:raise ValueError('HTML nodes>20000')
        self.stack[-1]['children'].append(node)
    def handle_starttag(self,tag,attributes):
        # Validate every actual opening tag before any semantic projection.
        # The same passive validator is used by input validation; inert-token
        # bodies and incomplete prefixes are never passed as resource attrs.
        if not self.guarded:
            try:PassiveHTML().handle_starttag(tag,attributes)
            except Unsafe as exc:report_unsafe(exc)
        if len(attributes)!=len({k for k,v in attributes}):raise ValueError('Duplicate HTML attributes outside finite profile')
        line,col=self.getpos();offset=self.starts[line-1]+col
        node={'tag':tag,'attrs':dict(attributes),'children':[],'start':offset,'start_tag':self.get_starttag_text(),'end':offset+len(self.get_starttag_text()),'closed':tag in ('br','img','hr','meta','link')};self.append_node(node)
        if not node['closed']:self.stack.append(node)
        if len(self.stack)>17:raise ValueError('HTML depth>16')
    def handle_startendtag(self,tag,attributes):
        self.handle_starttag(tag,attributes)
        if self.stack[-1].get('tag')==tag and not self.stack[-1].get('closed'):
            self.stack[-1]['closed']=True;self.stack.pop()
    def handle_incomplete_start(self,length):
        line,col=self.getpos();start=self.starts[line-1]+col
        self.append_node({'tag':'#token','token_type':'incomplete-start','children':[],'attrs':{},'start':start,'end':start+length,'closed':True})
    def parse_endtag(self,i):
        # The pinned tokenizer consumes this finite error without a callback.
        # Emit a leaf at the current parser position, just like other tokens.
        if self.rawdata.startswith('</>',i):self.token('empty-close')
        return super().parse_endtag(i)
    def handle_endtag(self,tag):
        if len(self.stack)>1 and self.stack[-1]['tag']==tag:
            line,col=self.getpos();at=self.starts[line-1]+col;end=self.raw.find('>',at)
            self.stack[-1]['end']=end+1 if end>=0 else len(self.raw);self.stack[-1]['closed']=True;self.stack.pop()
        else:
            line,col=self.getpos();at=self.starts[line-1]+col
            end=self.raw.find('>',at)+1
            self.errors.append({'raw':self.raw[at:end],'offset':at,'end':end,'reason':'unmatched HTML close','scope':'text','resource_syntax':True})
    def handle_data(self,value):
        line,col=self.getpos();start=self.starts[line-1]+col;end=self.raw.find('<',start+1 if value.startswith('<') else start)
        self.append_node({'tag':'#text','value':value,'children':[],'attrs':{},'start':start,'end':end if end>=0 else len(self.raw),'closed':True})
    def token(self,kind,body_length=None):
        line,col=self.getpos();start=self.starts[line-1]+col
        lexical=html_lexeme(self.raw,start)
        if lexical:kind,end=lexical
        else:
            at=self.raw.find('>',start);end=at+1 if at>=0 else len(self.raw)
        self.append_node({'tag':'#token','token_type':kind,'children':[],'attrs':{},'start':start,'end':end,'closed':True})
    def handle_comment(self,value):
        line,col=self.getpos();start=self.starts[line-1]+col
        if self.raw.startswith('<!--',start):self.token('comment',len(value))
        else:self.token('declaration')
    def handle_decl(self,value):self.token('declaration')
    def unknown_decl(self,value):self.token('declaration')
    def handle_pi(self,value):self.token('processing-instruction')
    def close(self):
        super().close()
        # Parser diagnostics survive every semantic projection, including
        # wrappers which only visit their children. They carry no text leaves.
        for n in self.stack[1:]:
            self.errors.append({'raw':self.raw[n['start']:],'offset':n['start'],'end':len(self.raw),'reason':'unclosed HTML container','scope':'structure','resource_syntax':True,'location':{'kind':'html-diagnostic','tag':n['tag']}})


def html_nodes(raw,inline_context=False,source_map=None,guarded=False):
    # A structural error cannot suppress validation of any subsequent opening.
    # The guard stores no tree and uses the same finite tokenizer extension.
    if not guarded:validate_html(raw,source_map)
    parser=HTMLTree(raw,guarded=True)
    try:parser.feed(raw);parser.close()
    except ValueError as exc:
        if isinstance(exc,Unsafe):raise
        return [{'type':'unknown','text':'','body_text':'','references':[],'formulas':[],'anchors':[],'segments':[],'unknown':[{'raw':raw,'offset':0,'reason':str(exc),'scope':'text','resource_syntax':True}],'reason':str(exc)}]
    def covered_end(n):
        return max([n.get('end',0),*(covered_end(c) for c in n['children'])])
    covered=max([covered_end(parser.root),*(u['end'] for u in parser.errors if u['reason']=='unmatched HTML close')])
    if raw[covered:].strip():
        # HTMLParser may discard an incomplete start tag even from rawdata.
        # Source coverage, rather than its buffer, detects that missing tail.
        parser.errors.append({'raw':raw[covered:],'offset':covered,'reason':'unconsumed HTML tokenizer tail','scope':'text','resource_syntax':True})
    nodes=[]
    def text(n):return n if isinstance(n,str) else n['value'] if n['tag']=='#text' else ''.join(text(c) for c in n['children'])
    inline_tags={'span','a','img','math','code','br','em','strong','sup','sub'}
    math_tags={'mrow','mi','mn','mo','msub','msup','mfrac','msubsup','mtable','mtr','mtd','munderover','mtext','semantics','annotation'}
    def parsed(n,in_inline=False,in_math=False):
        out={'text':'','body_text':'','references':[],'formulas':[],'anchors':[],'segments':[],'unknown':[]}
        if isinstance(n,str):
            out.update(text=n,body_text=n,segments=[{'type':'text','text':n}]);return out
        tag=n['tag'];meta=n['attrs'];start=n.get('start',0);end=n.get('end',start)
        if tag=='#token':
            value={'raw':raw[start:end],'offset':start,'end':end,'token_type':n['token_type'],'reason':'HTML lexical token outside finite profile','scope':'math' if in_math else 'text','resource_syntax':True,'profile':'html-tokens-v1'}
            out['unknown']=[value];out['segments']=[{'type':'unknown','raw':value['raw'],'start':start,'end':end}];return out
        if tag=='#text':
            out.update(text=n['value'],body_text=n['value'],segments=[{'type':'text','text':n['value'],'start':start,'end':end}])
            if n['value'].startswith('<') and re.match(r'<(?:/?[A-Za-z]|!|\?)',raw[start:]):
                out['unknown'].append({'raw':raw[start:],'offset':start,'reason':'malformed HTML treated as text by tokenizer','scope':'text','resource_syntax':True})
            return out
        child_inline=in_inline or tag in ('p','span','a','td','th','caption','figcaption','em','strong','sup','sub','h1','h2','h3','h4','h5','h6','code')
        children=[parsed(c,child_inline,in_math or tag=='math') for c in n['children']]
        for c in children:
            out['text']+=c['text'];out['body_text']+=c['body_text']
            for field in ('references','formulas','anchors','segments','unknown'):out[field].extend(c[field])
        if meta.get('id'):out['anchors'].append(meta['id'])
        if tag=='img':
            out['text']=meta.get('alt','');out['body_text']=''
            out['references'].append({'kind':'image','text':out['text'],'target':meta.get('src',''),'title':meta.get('title'),'start':start,'end':end})
            out['segments']=[{'type':'image','text':out['text'],'start':start,'end':end}]
        elif tag=='a' and 'href' in meta:
            out['references'].append({'kind':'link','text':out['text'],'target':meta['href'],'title':meta.get('title'),'start':start,'end':end});out['body_text']=''
            for s in out['segments']:s['in_link']=True
        elif tag=='link' and 'href' in meta:
            out['references'].append({'kind':'resource','text':'','target':meta['href'],'start':start,'end':end})
        elif tag=='math':
            annotations=[]
            def collect(x):
                if isinstance(x,str):return
                if x['tag']=='annotation' and x['attrs'].get('encoding') in ('application/x-tex','application/x-latex'):annotations.append(text(x))
                for c in x['children']:collect(c)
            collect(n);out['text']='';out['body_text']='';out['formulas']=[]
            if len(annotations)==1:out['formulas']=[{'raw':annotations[0],'display':meta.get('display')=='block','start':start,'end':end,'math':math_observe(annotations[0])}]
            else:out['unknown'].append({'raw':raw[start:end],'offset':start,'reason':'MathML without one finite TeX annotation','scope':'math'})
            out['segments']=[{'type':'math','raw':raw[start:end],'start':start,'end':end}]
        elif tag=='code':out['segments']=[{'type':'code','text':out['text'],'start':start,'end':end}]
        elif tag=='br':out.update(text='\n',body_text='\n',segments=[{'type':'text','text':'\n','start':start,'end':end}])
        if tag not in ('root','div','section','body','article','table','caption','thead','tbody','tr','td','th','figure','figcaption','p','span','a','h1','h2','h3','h4','h5','h6','img','link','math','mrow','mi','mn','mo','msub','msup','mfrac','msubsup','mtable','mtr','mtd','munderover','mtext','semantics','annotation','code','br','em','strong','sup','sub'):
            out['unknown'].append({'raw':raw[start:end],'offset':start,'reason':'raw HTML subset unsupported','scope':'text'})
        if in_inline and tag!='root' and tag not in inline_tags and not (in_math and tag in math_tags):
            out['unknown'].append({'raw':raw[start:end],'offset':start,'reason':'block/unsupported structure in HTML inline profile','scope':'text','profile':'html-inline-v1'})
        return out
    def visit(n):
        if isinstance(n,str):
            if n.strip():nodes.append({'type':'paragraph',**parsed(n)})
            return
        tag=n['tag'];base={**parsed(n,inline_context),'html_tag':tag}
        if tag=='#text':
            if base['text'].strip():nodes.append({'type':'paragraph',**base})
            return
        if inline_context and tag!='root':
            # Transport the complete tree, never a block's lossy projection.
            nodes.append({'type':'paragraph',**base});return
        if tag in ('div','section','body','article','root'):
            for c in n['children']:visit(c)
        elif tag=='table':
            grid=[];header_rows=0;complex_table=False;captions=[];residual=[]
            def residue(x,reason):
                start=x['start'];end=x['end']
                value=located(parsed(x),{'kind':'table-residual','index':len(residual)},raw[start:end])
                value['source_offset']=start
                value['reason']=reason;residual.append(value)
            def collect(x,parent):
                nonlocal complex_table,header_rows
                if x['tag']=='#text':
                    if x['value'].strip():residue(x,'non-whitespace outside table cell/caption')
                    return
                if x['tag']=='caption' and parent=='table':
                    value=located(parsed(x),{'kind':'caption','index':len(captions)},raw[x['start']:x['end']])
                    value['source_offset']=x['start'];captions.append(value);return
                if x['tag']=='tr' and parent in ('table','thead','tbody'):
                    cells=[c for c in x['children'] if c['tag'] in ('td','th')]
                    if cells and all(c['tag']=='th' for c in cells) and header_rows==len(grid):header_rows+=1
                    row=[]
                    for ci,c in enumerate(cells):
                        if any(k in c['attrs'] for k in ('rowspan','colspan')):complex_table=True
                        row.append(located(parsed(c),{'kind':'cell','row':len(grid),'col':ci},raw[c['start']:c['end']]))
                    grid.append(row)
                    for c in x['children']:
                        if c['tag'] not in ('td','th'):collect(c,'tr')
                elif x['tag'] in ('thead','tbody') and parent=='table':
                    for c in x['children']:collect(c,x['tag'])
                else:
                    # Consume the entire unsupported branch once; descendants
                    # cannot disappear through selective recursion.
                    residue(x,'branch outside finite table structure')
            for c in n['children']:collect(c,'table')
            cap=captions[0] if len(captions)==1 else None
            leaves=[c for row in grid for c in row]+captions+residual
            if len(captions)>1:
                residual.extend({**c,'reason':'duplicate table caption'} for c in captions)
            base.update(type='unknown' if complex_table else 'table',text='',body_text='',cells=[[c['text'] for c in row] for row in grid],cell_contents=grid,caption=cap['text'] if cap else None,caption_content=cap,header_rows=header_rows,header_origin='html-th',reason='HTML spanning table unsupported' if complex_table else None)
            base['residual_contents']=residual
            for field in ('references','formulas','unknown'):
                base[field]=sorted([v for c in leaves for v in c[field]],key=lambda v:v['source_offset'])
            for value in residual:
                base['unknown'].append({'raw':value['raw'],'offset':value['source_offset'],'location':value['location'],'reason':value['reason'],'scope':'table','profile':'html-table-v1'})
            # Parser/container errors are independent of projected leaves.
            for u in parsed(n)['unknown']:
                if u.get('resource_syntax') and not any((v.get('raw'),v.get('offset'),v.get('reason'))==(u.get('raw'),u.get('offset'),u.get('reason')) for v in base['unknown']):base['unknown'].append(u)
            # Anchors already come from the independent source-order tree walk.
            nodes.append(base)
        elif tag=='figure':
            images=[r for r in base['references'] if r['kind']=='image' and any(c['tag']=='img' and c['start']==r['start'] for c in n['children'])];captions=[c for c in n['children'] if c['tag']=='figcaption']
            residual=[c for c in n['children'] if c['tag'] not in ('img','figcaption') and not (c['tag']=='#text' and not c['value'].strip())]
            image_nodes=[c for c in n['children'] if c['tag']=='img']
            if len(images)==len(captions)==len(image_nodes)==1:
                cap=located(parsed(captions[0]),{'kind':'caption'},raw[captions[0]['start']:captions[0]['end']])
                base.update(type='figure',text=cap['text'],caption=cap['text'],caption_content=cap,asset_uri=images[0]['target'])
                for field in ('references','formulas','unknown'):
                    for v in base[field]:
                        v['location']={'kind':'caption'} if captions[0]['start']<=v.get('start',v.get('offset',0))<captions[0]['end'] else {'kind':'figure','source_offset':v.get('start',v.get('offset',0))}
                base['residual_contents']=[{**located(parsed(c),{'kind':'figure-residual','index':i},raw[c['start']:c['end']]),'source_offset':c['start'],'reason':'branch outside finite figure structure'} for i,c in enumerate(residual)]
                for value in base['residual_contents']:
                    base['unknown'].append({'raw':value['raw'],'offset':value['source_offset'],'location':value['location'],'scope':'figure','reason':value['reason'],'profile':'html-figure-v1'})
            else:
                base.update(type='unknown',reason='HTML figure outside finite single-image/caption profile')
                base['unknown'].append({'raw':raw[n['start']:n['end']],'offset':n['start'],'scope':'text','reason':base['reason']})
            nodes.append(base)
        elif tag in ('p','span','a','img','math','br','code','em','strong','sup','sub','h1','h2','h3','h4','h5','h6'):
            base['type']='heading' if tag.startswith('h') and len(tag)==2 else 'paragraph';nodes.append(base)
        else:base.update(type='unknown',reason='raw HTML subset unsupported');nodes.append(base)
    visit(parser.root)
    if parser.errors:
        parser.errors.sort(key=lambda u:u['offset'])
        if nodes:
            nodes[0].setdefault('unknown',[]).extend(parser.errors)
            nodes[0]['unknown'].sort(key=lambda u:u.get('source_offset',u.get('offset',0)))
        else:nodes.append({'type':'unknown','text':'','references':[],'formulas':[],'anchors':[],'segments':[],'unknown':parser.errors,'reason':'HTML parser diagnostics without projected leaves'})
    return nodes


def observe(content):
    session={'first':None,'events':{}};token=_GUARD.set(session)
    try:
        result=_observe(content)
        drain_guards(session)
        if session['first'] is not None:raise session['first']
        def public(value):
            if isinstance(value,dict):return {k:public(v) for k,v in value.items() if not (k=='_guard_coverage' and isinstance(v,dict) and v.get('domain')=='producer')}
            if isinstance(value,list):return [public(v) for v in value]
            return value
        return public(result)
    finally:_GUARD.reset(token)


def _observe(content):
    if len(content)>LIMIT:raise ValueError('Reader bytes limit')
    raw=content.decode('utf-8',errors='strict');blocks={};lexical_spans=markdown_lexemes(raw,_block_decisions=blocks);lines,line_numbers,physical_count=markdown_units(raw,lexical_spans);starts=[];pos=0
    for line in lines:starts.append(pos);pos+=len(line)
    def barriers(value,at):
        result=[(lo-at,hi-at) for lo,hi in lexical_spans if at<=lo and hi<=at+len(value)]
        for lo,hi in result:assert value[lo:hi]==raw[at+lo:at+hi]
        return result
    # A physical line can start inside a literal accepted on the preceding
    # soft line. Only the complete-source ledger decides whether trim is inert;
    # reparsing that fragment would invent a new HTML token and retain padding.
    def strip_source(value,at,characters='\r\n'):return strip_inline_end(value,characters,barriers(value,at),_source_proven=True)
    def read_inline(value,at):
        mapping=list(range(at,at+len(value)+1))
        decision=InlineDecision(value,barriers(value,at),source_map=mapping)
        parsed=inline(value,inert=barriers(value,at),producer_limit=len(value),_decision=decision)
        parsed['_source_map']=mapping
        return parsed
    def read_parts(parts,host):
        # Producer domain and mapping are built while removing source prefixes,
        # rather than inferred later from marker-looking transformed text.
        value='\n'.join(v for v,at in parts);offset=0;known=[];mapping=[]
        for pi,(v,at) in enumerate(parts):
            known.extend((lo+offset,hi+offset) for lo,hi in barriers(v,at))
            mapping.extend(range(at,at+len(v)))
            if pi+1<len(parts):mapping.append(at+len(v))
            offset+=len(v)+1
        mapping.append(parts[-1][1]+len(parts[-1][0]))
        assert len(mapping)==len(value)+1
        decision=InlineDecision(value,known,blocks=host,source_map=mapping)
        parsed=inline(value,inert=known,producer_limit=len(value),_decision=decision)
        parsed['_source_map']=mapping
        return parsed
    nodes=[];i=0;containers=[];landmark=None;pending_anchors=[]
    def node(kind,start,end,**values):
        nonlocal landmark
        cstart=starts[start] if start<len(starts) else len(raw);cend=starts[end] if end<len(starts) else len(raw)
        inline_offset=values.pop('source_inline_offset',0)
        source_map=values.pop('_source_map',None)
        guard_coverage=values.pop('_guard_coverage',None)
        # Markdown container prefixes are outside the producer's inline string.
        # Map each logical line back to its actual raw line; no text matching.
        syntax=values.get('block_syntax');prefixes=[];logical=0;source=0
        if source_map is None and (syntax in ('list_item','blockquote','atx_heading') or kind=='note'):
            for li,line_raw in enumerate(lines[start:end]):
                line_value=strip_source(line_raw,cstart+source);prefix=0
                if syntax=='blockquote':prefix=len(re.match(r'^ {0,3}> ?',line_value)[0])
                elif syntax=='list_item':prefix=len(re.match(r'^ {0,3}(?:[-+*]|[0-9]{1,9}[.)])[ \t]+',line_value)[0]) if li==0 else 4
                elif syntax=='atx_heading':prefix=len(re.match(r'^ {0,3}#{1,6}[ \t]+',line_value)[0])
                elif kind=='note':prefix=len(re.match(r'^\[\^[^\]]+\]:[ \t]*',line_value)[0]) if li==0 else 1 if line_value.startswith('\t') else 4
                prefixes.append((logical,source+prefix));logical+=len(line_value)-prefix+1;source+=len(line_raw)
        result={'observation_id':f'o{len(nodes)+1:04d}','type':kind,'raw':raw[cstart:cend],'offset':{'char_start':cstart,'char_end':cend,'byte_start':len(raw[:cstart].encode()),'byte_end':len(raw[:cend].encode()),'line_start':line_numbers[start]+1 if start<len(lines) else physical_count+1,'line_end':line_numbers[end] if end<len(lines) else physical_count},'containers':list(containers),'landmark':landmark,'text':'','references':[],'formulas':[],'anchors':[],**values}
        # A complete inline projection (including a complex quote) was
        # already guarded in its producer domain. Refeeding its block raw
        # here would reinterpret its closed Markdown code as active HTML.
        # Keep this fallback only for blocks without an inline projection.
        if kind=='unknown' and guard_coverage is None and not result.get('unknown') and re.search(r'</?[A-Za-z][\w:-]*\b[^>]*>',result['raw']):
            validate_html(result['raw'],list(range(cstart,cend+1)))
            result['unknown']=[{'raw':result['raw'],'offset':0,'reason':result.get('reason','unsupported HTML fallback'),'scope':'text','resource_syntax':True}]
        # Exact table/figure source positions accompany logical coordinates;
        # other inline containers retain their logical offsets and raw bounds.
        if kind!='code':
            for field in ('references','formulas','unknown','segments'):
                for value in result.get(field,[]):
                    relative=value.get('source_offset',value.get('start',value.get('offset',0)))
                    if prefixes:
                        logical_start,source_start=next((p for p in reversed(prefixes) if p[0]<=relative),prefixes[0]);relative=source_start+relative-logical_start
                    at=source_map[relative] if source_map is not None else cstart+inline_offset+relative
                    value['source_position']={'char':at,'byte':len(raw[:at].encode()),'basis':'raw-block-relative-inline-v1'}
        if result.get('attrs',{}).get('id'):result['anchors'].append(result['attrs']['id'])
        if kind=='anchor':pending_anchors.extend(result['anchors'])
        elif pending_anchors:
            result['anchors']=[*pending_anchors,*result['anchors']];pending_anchors.clear();result['anchor_binding']='empty anchor attached to next leaf; original anchor node/raw retained'
        if result['anchors'] and kind=='heading':landmark=result['anchors'][0];result['landmark']=landmark
        nodes.append(result);return result
    while i<len(lines):
        line=strip_source(lines[i],starts[i]);start=i
        block=blocks.get(starts[i],{'paragraph_before':True,'fence':None,'heading':False,'div':None,'container':None})
        if not line.strip():i+=1;continue
        div=block['div']
        if div:
            if div['close']:
                if containers:containers.pop()
                else:node('unknown',i,i+1,text=line,reason='unmatched div close')
            else:
                metadata=attrs(div['metadata'][1],starts[i]+div['start']+1) if div.get('metadata') else {'classes':[div['class']]}
                if len(containers)>=16:node('unknown',i,i+1,text=line,reason='div depth>16')
                else:containers.append(metadata)
            i+=1;continue
        fence=block['fence']
        if fence:
            marker=fence['mark'];info=fence['info'];i+=1;body_start=i
            while i<len(lines) and not re.fullmatch(r' {0,3}'+re.escape(marker[0])+r'{'+str(len(marker))+r',}[ \t]*[\r\n]*',lines[i]):i+=1
            if i==len(lines):
                payload=''.join(lines[body_start:i])
                if info=='{=html}':
                    for h in html_nodes(payload,source_map=list(range(fence['body_start'],fence['body_start']+len(payload)+1))):node(h.pop('type'),start,i,source_inline_offset=starts[body_start]-starts[start] if body_start<len(starts) else len(raw)-starts[start],**h)
                if info=='{=latex}':
                    at=starts[body_start] if body_start<len(starts) else len(raw)
                    parsed=shift_inline(read_inline(payload,at),at-starts[start])
                    parsed['unknown'].append({'raw':''.join(lines[start:i]),'offset':0,'scope':'structure','resource_syntax':True,'reason':'unclosed raw LaTeX fence'})
                    node('unknown',start,i,**parsed,reason='unclosed fence');continue
                node('unknown',start,i,text='' if info=='{=html}' else payload,reason='unclosed fence',unknown=[{'raw':''.join(lines[start:i]),'offset':0,'scope':'structure','resource_syntax':True,'reason':'unclosed raw HTML fence' if info=='{=html}' else 'unclosed literal fence'}]);continue
            payload=''.join(lines[body_start:i]);i+=1
            if info.startswith('{='):
                format=info.strip('{}')[1:]
                if format=='html':
                    for h in html_nodes(payload,source_map=list(range(fence['body_start'],fence['body_start']+len(payload)+1))):node(h.pop('type'),start,i,source_inline_offset=starts[body_start]-starts[start],**h)
                elif format=='latex':node('paragraph',start,i,source_inline_offset=starts[body_start]-starts[start],**read_inline(strip_source(payload,starts[body_start]),starts[body_start]))
                else:node('unknown',start,i,text=payload,reason='raw format unsupported')
            else:node('code',start,i,text=payload,attrs=attrs(info[1:-1],starts[start]+line.index(info)+1) if info.startswith('{') else {'language':info})
            continue
        item=re.match(r'^ {0,3}([-+*]|[0-9]{1,9}[.)])[ \t]+(.+)$',line,re.S) if block['container']=='list' else None
        if item:
            i+=1;values=[item[2]];parts=[(item[2],starts[start]+item.start(2))]
            while i<len(lines) and lines[i].startswith('    '):v=strip_source(lines[i][4:],starts[i]+4);values.append(v);parts.append((v,starts[i]+4));i+=1
            parsed=read_parts(parts,'list');node('unknown' if parsed['unknown'] else 'paragraph',start,i,**parsed,block_syntax='list_item',list_marker=item[1],ordered=item[1][0].isdigit());continue
        if block['container']=='quote':
            i+=1;values=[re.sub(r'^ {0,3}> ?', '',line)];parts=[(values[0],starts[start]+len(line)-len(values[0]))]
            while i<len(lines) and re.match(r'^ {0,3}>',lines[i]):v=strip_source(lines[i],starts[i]);stripped=re.sub(r'^ {0,3}> ?', '',v);values.append(stripped);parts.append((stripped,starts[i]+len(v)-len(stripped)));i+=1
            parsed=read_parts(parts,'quote');logical=0;complex_quote=False
            for v in values:
                # The typed consumer already identified closed literals in the
                # producer domain. A marker inside such a leaf is content.
                literal=any(s['type']=='code' and s['start']<logical<s['end'] for s in parsed['segments'])
                complex_quote=complex_quote or (v.startswith(('>','#','|','```')) and not literal)
                logical+=len(v)+1
            node('unknown' if parsed['unknown'] or complex_quote else 'paragraph',start,i,**parsed,block_syntax='blockquote',reason='nested/complex quote outside finite subset' if complex_quote else None);continue
        heading=re.match(r'^ {0,3}(#{1,6})[ \t]+(.+)$',line,re.S) if block['heading'] else None
        if heading:
            at=starts[i]+heading.start(2);text,metadata=peel_attrs(heading[2],barriers(heading[2],at),at);node('heading',i,i+1,**read_inline(text,at),attrs=metadata,level=len(heading[1]),block_syntax='atx_heading');i+=1;continue
        note=re.match(r'^\[\^([^\]]+)\]:[ \t]*(.*)$',line,re.S) if block['container']=='note' else None
        if note:
            i+=1;values=[note[2]];parts=[(note[2],starts[start]+note.start(2))]
            while i<len(lines) and (lines[i].startswith('    ') or lines[i].startswith('\t')):prefix=4 if lines[i].startswith('    ') else 1;v=strip_source(lines[i][prefix:],starts[i]+prefix);values.append(v);parts.append((v,starts[i]+prefix));i+=1
            result=read_parts(parts,'note');result['anchors'].append(note[1]);node('note',start,i,**result,note_id=note[1]);continue
        if i+1<len(lines) and '|' in line and all(re.fullmatch(r':?-{3,}:?',c.strip()) for c in pipe_row(lines[i+1],barriers(lines[i+1],starts[i+1]))):
            rows=[pipe_row(line,barriers(line,starts[i]))];i+=2
            while i<len(lines) and '|' in lines[i] and lines[i].strip():rows.append(pipe_row(lines[i],barriers(lines[i],starts[i])));i+=1
            caption=None;caption_offset=0;j=i
            while j<len(lines) and not lines[j].strip():j+=1
            if j<len(lines) and re.match(r'^(?:Table:|:) ',lines[j]):
                prefix=re.match(r'^(?:Table:|:) ',lines[j])[0];caption=strip_source(lines[j][len(prefix):],starts[j]+len(prefix));caption_offset=starts[j]-starts[start]+len(prefix);i=j+1
            offsets={};search=0;table_raw=''.join(lines[start:i])
            for ri,row in enumerate(rows):
                for ci,c in enumerate(row):
                    at=table_raw.find(c,search) if c else search
                    offsets[ri,ci]=max(at,0);search=max(at,0)+len(c)
            node('table',start,i,**table_contents(rows,caption,offsets,caption_offset,inert=lexical_spans,source_offset=starts[start]),header_rows=1,header_origin='pipe-syntax');continue
        figure=markdown_link(line,0)
        if figure and figure['image']:
            tail=line[figure['length']:];attribute=metadata_at(tail,0) if tail else None
            if attribute and attribute.end()!=len(tail):attribute=None
            if not tail or attribute:
                cap=located(read_inline(figure['label'],starts[i]+2),{'kind':'caption'},figure['label'],2)
                reference={'kind':'image','text':cap['text'],'target':figure['target'],'target_raw':figure['target_raw'],'title':figure['title'],'location':{'kind':'figure'},'start':0,'end':figure['length']}
                node('figure',i,i+1,text=cap['text'],caption=cap['text'],caption_content=cap,asset_uri=figure['target'],attrs=attrs(attribute[1],starts[i]+figure['length']+attribute.start(1)) if attribute else {},references=[reference,*cap['references']],formulas=cap['formulas'],unknown=cap['unknown']);i+=1;continue
        if line.startswith('<') and re.match(r'</?[A-Za-z][\w:-]*\b',line) and not AUTOLINK.match(line):
            i+=1
            while i<len(lines) and lines[i].strip():i+=1
            try:
                for h in html_nodes(''.join(lines[start:i]),source_map=list(range(starts[start],starts[start]+len(''.join(lines[start:i]))+1))):node(h.pop('type'),start,i,**h)
            except ValueError as exc:
                if isinstance(exc,Unsafe):raise
                node('unknown',start,i,text=''.join(lines[start:i]),unknown=[{'raw':''.join(lines[start:i]),'offset':0,'scope':'text','resource_syntax':True,'reason':str(exc)}],reason=str(exc))
            continue
        i+=1
        while i<len(lines) and lines[i].strip():
            next_block=blocks.get(starts[i],{})
            if next_block.get('fence') or next_block.get('div') or next_block.get('heading') or next_block.get('container'):break
            i+=1
        value=strip_source(''.join(lines[start:i]),starts[start]);observed=read_inline(value,starts[start])
        if not observed['text'].strip() and observed['formulas']:
            node('formula',start,i,**observed)
        elif not observed['text'] and observed['anchors']:
            node('anchor',start,i,**observed)
        else:node('unknown' if observed['unknown'] else 'paragraph',start,i,**observed,reason='inline unsupported' if observed['unknown'] else None)
    if containers:node('unknown',len(lines),len(lines),reason='unclosed div')
    return {'schema_version':1,'reader':VERSION,'raw_sha256':digest_bytes(content),'raw_bytes':len(content),'nodes':nodes,'unknown':sum(n['type']=='unknown' or bool(n.get('unknown')) for n in nodes)}
