"""Scoring separato dalle osservazioni: denominatori completi, zero loss."""
from collections import Counter
from difflib import SequenceMatcher
import re
from data import digest, counts, canonical
from math_structure import differences

VERSION='metrics-v4-conditional-and-bounds'
NUMBER=r'[−+-]?[0-9]+(?:[.,][0-9]+)?(?:[eE][−+-]?[0-9]+)?'


def soft(value):return value.replace('\r\n','\n').replace('\r','\n').replace('\n',' ').strip(' ')


def expected_text(b,format):
    text=b['text']
    if format in ('docx','rtf','tex'):
        text+=''.join(' '+r['text'] for r in b.get('references',[]) if r['kind']=='link')
    elif format=='epub':text+=''.join(' '+r['text'] for r in b.get('references',[]))
    return text


def kind(b):return {'literal_text':'code','bibliography':'paragraph'}.get(b['type'],b['type'])
def key(b):
    if b['type']=='table':return '\n'.join('|'.join(row) for row in b.get('alignment_cells',b.get('cells',[])))
    if b['type']=='formula':return '\n'.join(f.get('reference_latex',f.get('raw','')) for f in b.get('formulas',[]))
    return b.get('caption',b.get('text',''))


def formula_key(b):
    values=[]
    for f in b.get('formulas',[]):
        if 'structure_expected' in f:values.append(f['structure_expected'])
        elif f.get('math',{}).get('fully_supported'):values.append(f['math']['structure'])
        else:return None
    return canonical(values).decode() if values else None


def table_projection(g,o):
    """Separate declared header/body regions; one finite pipe-placeholder rule."""
    expected=g.get('header_rows',1);actual=o.get('header_rows',1)
    result={**o};drop=False
    grid=o.get('cell_contents',[]);cells=o.get('cells',[])
    empty=bool(cells) and all(v=='' for v in cells[0])
    if grid:empty=empty and all(not c.get(k) for c in grid[0] for k in ('references','formulas','unknown','anchors'))
    if expected==0 and actual==1 and o.get('header_origin')=='pipe-syntax' and empty:
        drop=True;actual=0;body=cells[1:];body_grid=grid[1:]
    else:body=cells[actual:];body_grid=grid[actual:]
    observed_header=cells[:actual];observed_header_grid=grid[:actual]
    # Missing header is represented by absent cells, never genuine empty ones.
    aligned_header=[observed_header[i] if i<len(observed_header) else [] for i in range(expected)]
    aligned_grid=[observed_header_grid[i] if i<len(observed_header_grid) else [] for i in range(expected)]
    result['cells']=aligned_header+body;result['cell_contents']=aligned_grid+body_grid;result['header_rows']=actual
    result['source_header_cells']=observed_header;result['source_body_cells']=body
    result['header_projection']={'version':'header-regions-v2','expected_rows':expected,'source_rows':o.get('header_rows',1),'actual_rows':actual,'empty_pipe_placeholder':drop,'source_row_offset':int(drop),'header_cells':observed_header}
    # Region mismatch remains observable during alignment, then fails locally.
    # No exact output becomes expected gold; only source-declared regions match.
    if actual!=expected:result['alignment_cells']=cells
    return result


def unknown_attribution(gs,os,mapping,ambiguous):
    """Versioned conservative local proof: exact visible payload, unique each side."""
    candidates=[]
    for i,g in enumerate(gs):
        if i in mapping or g['block_id'] in ambiguous:continue
        value=soft(g.get('text',''))
        if len(value)<8 or g['type'] in ('formula','table','figure'):continue
        for j,o in enumerate(os):
            if o['type']!='unknown':continue
            payload=soft(o.get('text',''))
            # Boundaries prevent a short matching word inside unrelated prose.
            if re.search(r'(?<!\w)'+re.escape(value)+r'(?!\w)',payload):candidates.append((i,j))
    return [(i,j) for i,j in candidates if sum(a==i for a,b in candidates)==1 and sum(b==j for a,b in candidates)==1]


def levenshtein(a,b):
    if len(a)*len(b)>4_000_000:return None
    previous=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        current=[i]
        for j,d in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(c!=d)))
        previous=current
    return previous[-1]


def textual_unknown(o):
    """text-domain-unknown-v1: math/cell/caption outside whole-block CER/WER."""
    if o['type'] in ('formula','table','figure','code'):return []
    values=[u for u in o.get('unknown',[]) if u.get('scope','text')!='math']
    if o['type']=='unknown' and not o.get('unknown'):
        values=[{'reason':o.get('reason','unsupported textual block'),'raw':o.get('raw',''),'scope':'text'}]
    return values


def known_text(o):
    """Only typed, measurable leaves; never interpret raw unknown as prose."""
    return ''.join(s.get('text','') for s in o.get('segments',[]) if s['type'] not in ('unknown','raw_inline','math'))


def role_evidence(value,gs):
    """Diagnostic exact-text candidates only; does not alter alignment/counts."""
    evidence=[]
    for g in gs:
        if g.get('caption') and soft(value)==soft(g['caption']):
            evidence.append({'block_id':g['block_id'],'role':'caption','source_type':g['type'],'text':g['caption']})
        if g['type']=='heading' and soft(value)==soft(g.get('text','')):
            evidence.append({'block_id':g['block_id'],'role':'heading','source_type':g['type'],'text':g['text']})
    return evidence


def text_lower_bound(gs,os,format,unavailable,totals):
    """A guarantee for the unrestricted text stream, NOT structural matching.

    Unknown completions may emit any text, including nothing, and may alter
    token boundaries. We make no positive claim in that case: edit distances
    are nonnegative, so zero is valid for every completion/correspondence.
    With measurable leaves the stream is fixed and its exact distance is a
    lower bound (indeed the minimum) over every edit correspondence. Resource,
    block and page decisions are independent and never use this bound.
    """
    kinds=('formula','table','figure','anchor')
    expected=[(g['block_id'],expected_text(g,format),kind(g)) for g in gs if g['type'] not in kinds]
    observed=[(o['observation_id'],o.get('text',''),kind(o)) for o in os if o['type'] not in kinds]
    def stream(values):
        return [(ident,value if typ=='code' else soft(value)) for ident,value,typ in values if (value if typ=='code' else soft(value))]
    es,actual=stream(expected),stream(observed)
    reason='unmeasurable text: unrestricted completions, including zero emission' if unavailable else None
    chars=words=0
    if not unavailable:
        chars=levenshtein('\n'.join(v for i,v in es),'\n'.join(v for i,v in actual))
        words=levenshtein([w for i,v in es for w in v.split()],[w for i,v in actual for w in v.split()])
        if chars is None or words is None:
            chars=words=0;reason='global comparison cell limit: nonnegative bound only'
    return {'version':'global-text-completion-bound-v1','guaranteed':True,
            'domain':'non-table/non-figure/non-formula whole-block text, code literal; omit empty text projections; chars join blocks with LF, tokens concatenate split blocks',
            'correspondence':'unrestricted edit correspondence of global streams; independent of structural alignment',
            'status':'TRIVIAL_ZERO' if reason else 'EXACT_GLOBAL_DISTANCE','reason':reason,
            'char_errors':chars,'word_errors':words,'reference_codepoints':totals['text_ref_codepoints'],'reference_words':totals['word_ref'],
            'CER':chars/totals['text_ref_codepoints'] if totals['text_ref_codepoints'] else None,
            'WER':words/totals['word_ref'] if totals['word_ref'] else None,
            'expected_ids':[i for i,v in es],'observed_ids':[i for i,v in actual],
            'denominator_policy':'same full expected block denominators; LF separators do not add reference codepoints'}


def align(gold,observations,format):
    gs=list(gold['blocks']);os=[o for o in observations['nodes'] if o['type']!='anchor']
    if gold.get('note_order_policy')=='definitions-collected-at-end':gs=[g for g in gs if g['type']!='note']+[g for g in gs if g['type']=='note']
    contexts={};last=None
    for g in gs:
        if g['type']=='heading' and g.get('anchor'):last=g['anchor']
        contexts[g['block_id']]=last
    # An exact visible heading maps the source's landmark for alignment only.
    # Its actual anchor is still compared separately and never rewritten.
    context_alias={}
    for o in os:
        if o['type']=='heading' and o.get('anchors'):
            exact=[g for g in gs if g['type']=='heading' and g.get('anchor') and soft(expected_text(g,format))==soft(o.get('text',''))]
            if len(exact)==1:
                for anchor in o['anchors']:context_alias[anchor]=exact[0]['anchor']
    candidates=[];ambiguous=[];ambiguous_observations=set()
    for gi,g in enumerate(gs):
        row=[]
        for oi,o in enumerate(os):
            if kind(g)!=kind(o):continue
            observed_context=context_alias.get(o.get('landmark'),o.get('landmark'))
            if g['type']!='heading' and contexts[g['block_id']] and observed_context and contexts[g['block_id']]!=observed_context:
                continue
            a=soft(expected_text(g,format)) if g['type'] not in ('table','formula','figure') else key(g)
            b=soft(o.get('text','')) if g['type'] not in ('table','formula','figure') else key(table_projection(g,o) if g['type']=='table' else o)
            if g['type']=='formula':
                ga,oa=formula_key(g),formula_key(o)
                if ga is not None and oa is not None:a,b=ga,oa
                # Unsupported math still aligns by its raw occurrence for a
                # localized unavailable result; it can never establish equality.
                elif len(g.get('formulas',[]))==len(o.get('formulas',[]))==1:a=b='localized formula occurrence'
            similarity=SequenceMatcher(None,a,b,autojunk=False).ratio()
            if similarity<0.75:continue
            evidence=0.20 if contexts[g['block_id']] and contexts[g['block_id']]==observed_context else 0
            row.append({'gold_index':gi,'observed_index':oi,'similarity':similarity,'score':similarity+evidence,'context':contexts[g['block_id']]})
        row.sort(key=lambda x:(-x['score'],x['observed_index']))
        # Two exact observed copies are a duplicate, which remains a critical
        # unexpected node. Equal repeated gold without landmarks is ambiguous.
        repeated=sum(kind(x)==kind(g) and (formula_key(x)==formula_key(g) if g['type']=='formula' and formula_key(g) is not None else key(x)==key(g)) for x in gs)>1
        if repeated and len(row)>1 and row[0]['score']-row[1]['score']<0.15:
            ambiguous.append(g['block_id']);ambiguous_observations.update(c['observed_index'] for c in row);candidates.append([])
        else:candidates.append(row)
    scores={(c['gold_index'],c['observed_index']):c['score'] for row in candidates for c in row}
    dp=[[0.0]*(len(os)+1) for _ in range(len(gs)+1)]
    for i in range(1,len(gs)+1):
        for j in range(1,len(os)+1):dp[i][j]=max(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]+scores.get((i-1,j-1),-100))
    main=[];i=len(gs);j=len(os)
    while i and j:
        if (i-1,j-1) in scores and abs(dp[i][j]-dp[i-1][j-1]-scores[i-1,j-1])<1e-8:main.append((i-1,j-1));i-=1;j-=1
        elif dp[i-1][j]>=dp[i][j-1]:i-=1
        else:j-=1
    main.reverse()
    free=[];usedg=set();usedo=set()
    for c in sorted((x for row in candidates for x in row),key=lambda x:(-x['score'],x['gold_index'],x['observed_index'])):
        if c['gold_index'] not in usedg and c['observed_index'] not in usedo:
            free.append((c['gold_index'],c['observed_index']));usedg.add(c['gold_index']);usedo.add(c['observed_index'])
    return gs,os,{'method':'monotone-DP-v1 + independent-greedy-free-v1','threshold':0.75,'margin':0.15,'candidates':candidates,'main_pairs':main,'free_pairs':free,'ambiguous':ambiguous,'ambiguous_observed_indices':sorted(ambiguous_observations)}


def score(gold,observations,audit,format='control-markdown',verification=None):
    denominators=counts(gold);gs,os,alignment=align(gold,observations,format)
    mapping=dict(alignment['free_pairs']);matched=set(mapping.values());ambiguous=set(alignment['ambiguous'])
    attributed=unknown_attribution(gs,os,mapping,ambiguous);unresolved_indices={i for i,j in attributed}
    missing=[g['block_id'] for i,g in enumerate(gs) if i not in mapping and i not in unresolved_indices and g['block_id'] not in ambiguous]
    unexpected=[o['observation_id'] for i,o in enumerate(os) if i not in matched and i not in alignment['ambiguous_observed_indices'] and o['type']!='unknown']
    unknown=[o['observation_id'] for o in os if o['type']=='unknown' or o.get('unknown')]
    failures=[];unavailable=[];details=[];composition=[]
    # Unknown content is kept in the denominator but cannot establish which
    # block is missing. Other independently known losses still take precedence.
    if missing:failures.append({'metric':'blocks','missing':missing})
    if unexpected:failures.append({'metric':'blocks','unexpected':unexpected})
    ambiguous_extra=max(0,len(set(alignment['ambiguous_observed_indices'])-matched)-len(ambiguous))
    if ambiguous_extra:failures.append({'metric':'blocks','reason':'UNEXPECTED_AMBIGUOUS_OCCURRENCES','count':ambiguous_extra})
    if ambiguous:unavailable.append({'metric':'alignment','reason':'AMBIGUOUS','blocks':sorted(ambiguous)})
    if unknown:unavailable.append({'metric':'reader','reason':'UNAVAILABLE','observations':unknown,'attribution_rule':'exact-visible-unique-v1','attributed':[{'block_id':gs[i]['block_id'],'observation_id':os[j]['observation_id']} for i,j in attributed]})
    if audit.get('status')!='COMPLETE':
        failures.append({'metric':'bundle','reason':audit.get('reason',audit.get('status'))})
    note_map={g.get('note_id',g.get('anchor')):os[mapping[i]].get('note_id') for i,g in enumerate(gs) if g['type']=='note' and i in mapping}
    observed_note_ids=[o.get('note_id') for o in os if o['type']=='note']
    duplicates=[k for k,v in Counter(observed_note_ids).items() if v>1]
    mapped_ids=[v for v in note_map.values() if v is not None]
    if duplicates or len(mapped_ids)!=len(set(mapped_ids)):
        failures.append({'metric':'note','reason':'NON_BIJECTIVE_OR_DUPLICATE_ID','duplicate_ids':duplicates,'mapped_ids':mapped_ids})
    anchors={a for o in observations['nodes'] for a in o.get('anchors',[])}
    totals={'text_ref_codepoints':0,'text_errors':0,'raw_text_errors':0,'word_errors':0,'word_ref':0,'numbers_expected':denominators['numbers'],'numbers_exact':0,'formulas_expected':denominators['formulas'],'formulas_exact':0,'cells_expected':denominators['cells'],'cells_exact':0,'figures_expected':denominators['figures'],'figures_exact':0,'references_expected':denominators['references'],'references_exact':0}
    def inline_image_reference(expected_ref,actual_ref,local,unresolved,location):
        if expected_ref.get('kind')!='image':return
        image=next((v for v in audit.get('assets',[]) if v['uri']==actual_ref['target']),None)
        asset=expected_ref.get('asset')
        if not asset:
            unresolved.append({'metric':'inline_image','reason':'No manual inline asset content reference','location':location});return
        if not image or not image.get('content_verified') or image.get('expected_asset')!=asset:local.append({'metric':'inline_image','expected_asset':asset,'location':location,'audit':image})
    for i,g in enumerate(gs):
        if i not in mapping:
            text=expected_text(g,format)
            if g['type'] not in ('formula','table','figure'):
                normalized=soft(text) if g['type'] not in ('code','literal_text') else text
                totals['text_ref_codepoints']+=len(normalized);totals['word_ref']+=len(normalized.split())
                if i not in unresolved_indices and g['block_id'] not in ambiguous:
                    totals['text_errors']+=len(normalized);totals['raw_text_errors']+=len(text);totals['word_errors']+=len(normalized.split())
                composition.append({'category':'expected_omission','block_id':g['block_id'],'expected_type':g['type'],'text':normalized,'reference_words':len(normalized.split()),'known_word_edits':len(normalized.split()) if i not in unresolved_indices and g['block_id'] not in ambiguous else 0,'status':'UNAVAILABLE' if i in unresolved_indices or g['block_id'] in ambiguous else 'KNOWN','role_candidates':role_evidence(normalized,gs)})
            continue
        o=os[mapping[i]];local=[];unresolved=[]
        if g['type']=='table':o=table_projection(g,o)
        if o.get('unknown'):unresolved.append({'metric':'inline_reader','reason':'UNSUPPORTED','occurrences':o['unknown']})
        for residue in o.get('residual_contents',[]):
            # Typed known additions remain discriminating; raw unsupported
            # syntax alone contributes no invented text or formula.
            if known_text(residue).strip() or residue.get('references') or any(f['math']['fully_supported'] for f in residue.get('formulas',[])):
                local.append({'metric':'html_residual','reason':residue['reason'],'raw':residue['raw'],'known_text':known_text(residue),'location':residue['location']})
        detail={'block_id':g['block_id'],'observation_id':o['observation_id'],'offset':o['offset'],'expected_type':g['type'],'actual_type':o['type']}
        if g['type'] not in ('formula','table','figure'):
            expected=expected_text(g,format);actual=o.get('text','');allow_soft=g['type'] not in ('code','literal_text')
            es,as_= (soft(expected),soft(actual)) if allow_soft else (expected,actual)
            dist=levenshtein(es,as_);raw=levenshtein(expected,actual);words=levenshtein(es.split(),as_.split())
            totals['text_ref_codepoints']+=len(es);totals['word_ref']+=len(es.split())
            if dist is None or words is None:unresolved.append({'metric':'text','reason':'comparison cell limit'})
            else:
                if not textual_unknown(o):
                    totals['text_errors']+=dist;totals['raw_text_errors']+=raw if raw is not None else dist;totals['word_errors']+=words
                    if dist:local.append({'metric':'text','expected':expected,'actual':actual,'edits':dist,'raw_edits':raw})
            composition.append({'category':'aligned_block_edits','block_id':g['block_id'],'observation_id':o['observation_id'],'expected_type':g['type'],'actual_type':o['type'],'expected_text':es,'actual_text':as_,'reference_words':len(es.split()),'known_word_edits':words if dist is not None and words is not None and not textual_unknown(o) else 0,'status':'UNAVAILABLE' if dist is None or words is None or textual_unknown(o) else 'KNOWN','offset':o['offset']})
            measurable=not textual_unknown(o) and dist is not None and words is not None
            detail['text']={'reference_codepoints':len(es),'edits':dist if measurable else None,'raw_edits':raw if measurable else None,'softwrap_only':allow_soft,'CER':dist/len(es) if measurable and es else None,'availability':'AVAILABLE' if measurable else 'UNAVAILABLE'}
        # Occurrences are attached to a typed text/cell coordinate. Compare
        # lexical occurrence order within each value, never float or global bag.
        observed_values=[o.get('body_text',o.get('text',''))]+[c for row in o.get('cells',[]) for c in row]
        expected_by_index={}
        for n in g.get('numbers',[]):expected_by_index.setdefault(n['text_index'],[]).append(n)
        for index,expected_ns in expected_by_index.items():
            expected_ns=sorted(expected_ns,key=lambda n:n['offset'])
            value=observed_values[index] if index<len(observed_values) else ''
            actual_ns=[{'lexeme':m[0],'offset':m.start()} for m in re.finditer(NUMBER,value)]
            exact=sum(n['lexeme']==a['lexeme'] for n,a in zip(expected_ns,actual_ns));totals['numbers_exact']+=exact
            if [n['lexeme'] for n in expected_ns]!=[a['lexeme'] for a in actual_ns]:local.append({'metric':'numbers','text_index':index,'expected':expected_ns,'actual':actual_ns})
        for fidx,f in enumerate(g.get('formulas',[])):
            actual=o.get('formulas',[])[fidx] if fidx<len(o.get('formulas',[])) else None
            if actual is None:local.append({'metric':'formula','formula_id':f['formula_id'],'reason':'MISSING'});continue
            m=actual['math']
            if not m['fully_supported']:unresolved.append({'metric':'formula','formula_id':f['formula_id'],'reason':m['reason'],'raw':m['raw'],'offset':m['offset']});continue
            diff=differences(f['structure_expected'],m['structure']);detail.setdefault('formulas',[]).append({'formula_id':f['formula_id'],'raw_different':f['reference_latex']!=m['raw'],'diff':diff,'observed':m})
            if diff:local.append({'metric':'formula','formula_id':f['formula_id'],'diff':diff})
            else:totals['formulas_exact']+=1
        if len(o.get('formulas',[]))>len(g.get('formulas',[])):local.append({'metric':'formula','reason':'UNEXPECTED','count':len(o['formulas'])-len(g.get('formulas',[]))})
        if g['type']=='table':
            observed_cells=o.get('cells',[]);expected_cells=g['cells'];cells=[]
            for ri,row in enumerate(expected_cells):
                for ci,value in enumerate(row):
                    present=ri<len(observed_cells) and ci<len(observed_cells[ri]);actual=observed_cells[ri][ci] if present else None
                    exact=present and actual==value;totals['cells_exact']+=exact;cells.append({'row':ri,'col':ci,'expected':value,'present':present,'actual':actual,'exact':exact,'observed_source_coordinate':o.get('cell_contents',[])[ri][ci].get('location') if ri<len(o.get('cell_contents',[])) and ci<len(o['cell_contents'][ri]) else None})
            detail['cells']=cells
            projection=o['header_projection'];detail['header']=projection
            if projection['actual_rows']!=projection['expected_rows']:local.append({'metric':'table_header','reason':'SEMANTIC_HEADER_MISMATCH',**projection})
            expected_grid=g.get('cell_contents',[])
            for ri,row in enumerate(o.get('cell_contents',[])):
                for ci,content in enumerate(row):
                    expected_content=expected_grid[ri][ci] if ri<len(expected_grid) and ci<len(expected_grid[ri]) else {}
                    # Known non-text content is never made invisible by text==''.
                    for field in ('references','formulas'):
                        actual_inline=content.get(field,[])
                        expected_inline=expected_content.get(field,[])
                        if len(actual_inline)!=len(expected_inline):local.append({'metric':'cell_inline','row':ri,'col':ci,'component':field,'expected':len(expected_inline),'actual':len(actual_inline)})
                        for expected_leaf,actual_leaf in zip(expected_inline,actual_inline):
                            if field=='references':inline_image_reference(expected_leaf,actual_leaf,local,unresolved,{'kind':'cell','row':ri,'col':ci})
                            if field=='references' and expected_leaf.get('target') and (expected_leaf.get('kind'),expected_leaf['target'])!=(actual_leaf['kind'],actual_leaf['target']):local.append({'metric':'cell_inline_reference','row':ri,'col':ci,'expected':expected_leaf,'actual':actual_leaf})
                            if field=='formulas' and expected_leaf.get('structure_expected') and actual_leaf['math']['fully_supported'] and differences(expected_leaf['structure_expected'],actual_leaf['math']['structure']):local.append({'metric':'cell_inline_formula','row':ri,'col':ci,'diff':differences(expected_leaf['structure_expected'],actual_leaf['math']['structure'])})
            header_rows=g.get('header_rows',1)
            detail['body_cells']={'expected':sum(len(r) for r in expected_cells[header_rows:]),'exact':sum(c['exact'] for c in cells if c['row']>=header_rows)}
            if [len(r) for r in observed_cells]!=[len(r) for r in expected_cells] or any(not c['exact'] for c in cells):local.append({'metric':'cells','expected_shape':[len(r) for r in expected_cells],'actual_shape':[len(r) for r in observed_cells],'diff':[c for c in cells if not c['exact']]})
            # Absent and empty captions are text-equivalent; neither may hide
            # an addition. Typed components and unknowns are compared below.
            if (o.get('caption') or '')!=(g.get('caption') or ''):local.append({'metric':'table_caption','expected':g.get('caption'),'actual':o.get('caption'),'policy':'absent-empty-text-equivalent-v1'})
        if g['type'] in ('table','figure'):
            expected_caption=g.get('caption_content',{});actual_caption=o.get('caption_content') or {}
            for field in ('references','formulas'):
                expected_leaf=expected_caption.get(field,[]);actual_leaf=actual_caption.get(field,[])
                if len(expected_leaf)!=len(actual_leaf):local.append({'metric':'caption_inline','component':field,'expected':len(expected_leaf),'actual':len(actual_leaf)})
                if field=='references':
                    for er,ar in zip(expected_leaf,actual_leaf):
                        inline_image_reference(er,ar,local,unresolved,{'kind':'caption'})
                        if er.get('target') and (er.get('kind'),er['target'])!=(ar['kind'],ar['target']):local.append({'metric':'caption_inline_reference','expected':er,'actual':ar})
        if g['type']=='figure':
            image=next((x for x in audit.get('assets',[]) if x['uri']==o.get('asset_uri')),None)
            exact=bool(image and image.get('content_verified') and image.get('expected_asset')==g['asset'] and o.get('caption')==g['caption'])
            totals['figures_exact']+=exact;detail['asset_audit']=image
            if not exact:local.append({'metric':'figure','expected_asset':g['asset'],'caption_expected':g['caption'],'caption_actual':o.get('caption'),'audit':image})
        actual_refs=[r for r in o.get('references',[]) if r['kind']!='image'];used=set()
        for ref in g.get('references',[]):
            target=note_map.get(ref['target'],ref['target']) if ref['kind']=='note_call' else ref['target']
            candidate=next((j for j,r in enumerate(actual_refs) if j not in used and r['target']==target and (r['kind']==ref['kind'] or ref['kind']=='backlink' and r['kind']=='link')),None)
            if candidate is None:local.append({'metric':'reference','expected':ref,'mapped_target':target,'actual':actual_refs})
            else:used.add(candidate);totals['references_exact']+=1
            if ref['kind']=='note_call' and ref['target'] not in note_map:local.append({'metric':'note','reason':'ORPHAN','target':ref['target']})
            if ref['target'].startswith('#') and ref['target'][1:] not in anchors:local.append({'metric':'reference','reason':'BROKEN_ANCHOR','target':ref['target']})
        if len(actual_refs)!=len(used):local.append({'metric':'reference','reason':'UNEXPECTED','actual':[r for j,r in enumerate(actual_refs) if j not in used]})
        if g.get('anchor') and g['type']!='note' and g['anchor'] not in o.get('anchors',[]):local.append({'metric':'anchor','expected':g['anchor'],'actual':o.get('anchors')})
        if local:failures.extend({'block_id':g['block_id'],'observation_id':o['observation_id'],**x} for x in local)
        if unresolved:unavailable.extend({'block_id':g['block_id'],**x} for x in unresolved)
        detail['status']='FAIL' if local else 'UNAVAILABLE' if unresolved else 'PASS';details.append(detail)
    for j,o in enumerate(os):
        if j not in matched and o['type'] not in ('formula','table','figure') and j not in {b for a,b in attributed} and j not in alignment['ambiguous_observed_indices']:
            uncertain=bool(textual_unknown(o));value=soft(known_text(o) if uncertain else o.get('text',''))
            words=[v for v in value.split() if any(c.isalnum() for c in v)] if uncertain else value.split()
            totals['text_errors']+=len(value);totals['raw_text_errors']+=len(value) if uncertain else len(o.get('text',''));totals['word_errors']+=len(words)
            if uncertain and words:failures.append({'metric':'text','reason':'STRUCTURAL_UNMATCHED_KNOWN_TEXT','observation_id':o['observation_id'],'known_text':value,'conditional_word_edits':len(words)})
            composition.append({'category':'unmatched_observed_insertion','observation_id':o['observation_id'],'actual_type':o['type'],'text':value,'reference_words':0,'known_word_edits':len(words),'status':'CONDITIONAL' if uncertain else 'KNOWN','offset':o['offset'],'role_candidates':role_evidence(value,gs)})
    resolved=concordant=discordant=0
    for i in range(len(gs)):
        for j in range(i+1,len(gs)):
            if i in mapping and j in mapping:
                resolved+=1
                if mapping[i]<mapping[j]:concordant+=1
                else:discordant+=1
    pair_denom=len(gs)*(len(gs)-1)//2
    if discordant:failures.append({'metric':'order','inversions':discordant})
    pages=[]
    for page in sorted({g['page'] for g in gs if 'page' in g}):
        indices=[i for i,g in enumerate(gs) if g.get('page')==page];found=sum(i in mapping for i in indices)
        pages.append({'page':page,'expected_blocks':len(indices),'found':found,'missing_page':found==0 and not any(i in unresolved_indices or gs[i]['block_id'] in ambiguous for i in indices),'coverage':found/len(indices),'complete':found==len(indices) and all(next((d['status'] for d in details if d['block_id']==gs[i]['block_id']),'MISSING')=='PASS' for i in indices)})
    verified=bool(verification and verification.get('status')=='VERIFIED')
    if not verified:unavailable.append({'metric':'gold','reason':(verification or {}).get('status','PROVISIONAL')})
    if denominators['formulas']:unavailable.append({'metric':'formula_output_typography_and_semantics','reason':'UNAVAILABLE: separate human output judgments not assigned','expected':denominators['formulas']})
    text_unavailable=[g['block_id'] for i,g in enumerate(gs) if g['type'] not in ('formula','table','figure') and (i in unresolved_indices or g['block_id'] in ambiguous or i in mapping and textual_unknown(os[mapping[i]]))]
    text_reasons=[{'block_id':b,'scope':'whole-block textual CER/WER','reason':'unresolved/ambiguous expected textual block'} for b in text_unavailable]
    for j,o in enumerate(os):
        occurrences=textual_unknown(o)
        if occurrences:
            text_reasons.append({'observation_id':o['observation_id'],'block_ids':[gs[i]['block_id'] for i,k in [*mapping.items(),*attributed] if k==j],'scope':'observed whole-block textual CER/WER','reason':'unsupported observed textual content','occurrences':occurrences,'known_text':known_text(o),'offset':o['offset']})
    for u in unavailable:
        if u['metric']=='text':text_reasons.append({'block_id':u.get('block_id'),'scope':'whole-block textual CER/WER','reason':u['reason']})
    if text_reasons:unavailable.append({'metric':'text_complete','reason':'UNMEASURABLE_TEXT_DOMAIN','rule':'text-domain-unknown-v1','provenance':text_reasons})
    assert sum(c['known_word_edits'] for c in composition)==totals['word_errors']
    assert sum(c['reference_words'] for c in composition)==totals['word_ref']
    quality='FAIL' if failures else 'INDETERMINATE' if unavailable else 'PASS'
    result={'schema_version':1,'metrics_version':VERSION,'observations_sha256':digest(observations),'document_id':gold['document_id'],'quality':quality,'automatic_gates':'FAIL' if failures else 'INDETERMINATE' if any(u['metric'] not in ('gold','formula_output_typography_and_semantics') for u in unavailable) else 'PASS','score_exit':3 if failures else 5 if unavailable else 0,'denominators':denominators,'totals':totals,'CER':totals['text_errors']/totals['text_ref_codepoints'] if totals['text_ref_codepoints'] and not text_reasons else None,'WER':totals['word_errors']/totals['word_ref'] if totals['word_ref'] and not text_reasons else None,'text_metric_availability':{'status':'UNAVAILABLE' if text_reasons else 'AVAILABLE','rule':'text-domain-unknown-v1','scope':'non-table/non-figure/non-formula whole-block text, including code literal','blocks':text_unavailable,'reasons':text_reasons,'numerators':'counts conditional on structural matching; full expected denominators retained'},'alignment':alignment,'blocks':{'expected':len(gs),'observed':len(os),'matched':len(mapping),'missing':missing,'unexpected':unexpected,'ambiguous':sorted(ambiguous),'unresolved_attributed':[gs[i]['block_id'] for i in sorted(unresolved_indices)],'recall':len(mapping)/len(gs) if gs else None,'precision':len(mapping)/len(os) if os else None},'order':{'expected_pairs':pair_denom,'resolved_pairs':resolved,'concordant':concordant,'inversions':discordant,'unresolved_pairs':pair_denom-resolved},'pages':pages,'failures':failures,'unavailable':unavailable,'details':details,'threshold_scope':'Zero loss only for this controlled synthetic corpus; not universal OCR threshold'}
    result['text_metric_availability']['numerators']='totals are counts conditional on structural matching, NOT guaranteed bounds; full expected denominators retained'
    result['text_metric_availability']['bound_rule']='global-text-completion-bound-v1'
    for entry in composition:
        entry['conditional_word_edits']=entry['known_word_edits']
        entry['measure']='structural conditional count'
        entry['guaranteed_bound']=False
    result['WER_decomposition']={'version':'wer-structural-components-v2','measure':'conditional on recorded structural alignment; NOT a guaranteed lower bound','reference_words':totals['word_ref'],'known_word_edits':totals['word_errors'],'conditional_word_edits':totals['word_errors'],'complete':not bool(text_reasons),'categories':{c:sum(v['known_word_edits'] for v in composition if v['category']==c) for c in ('aligned_block_edits','expected_omission','unmatched_observed_insertion')},'entries':composition,'role_rule':'exact text diagnostic candidates; no alignment or counting changes','overlap':'caption/header errors may also fail structural gates; never sum gates as independent losses'}
    result['conditional_text_counts']={'version':'structural-text-counts-v1','char_errors':totals['text_errors'],'word_errors':totals['word_errors'],'reference_codepoints':totals['text_ref_codepoints'],'reference_words':totals['word_ref'],'guaranteed_bound':False,'correspondence':'recorded free_pairs/unknown attribution; omission and insertion may describe the same visible content'}
    result['text_lower_bound']=text_lower_bound(gs,os,format,text_reasons,totals)
    return result


def evaluate_exit(run_exit,score_exit):
    if 4 in (run_exit,score_exit):return 4
    if run_exit==130:return 130
    if run_exit==2:return 2
    if score_exit==3:return 3
    if score_exit==5:return 5
    return 0


def batch(records):
    success=sum(r['conversion']=='SUCCESS' and r['bundle']=='COMPLETE' for r in records)
    return {'attempts':len(records),'successful':success,'conversion_batch':'INTERRUPTED' if any(r['conversion']=='INTERRUPTED' for r in records) else 'COMPLETE' if records and success==len(records) else 'PARTIAL' if success else 'FAILED','quality':'FAIL' if any(r['quality']=='FAIL' for r in records) else 'PASS' if records and all(r['quality']=='PASS' for r in records) else 'INDETERMINATE','errors':[r['attempt_id'] for r in records if r['conversion']!='SUCCESS' or r['bundle']!='COMPLETE'],'coverage':'RECORDS_PRESENT only; expected manifest required for execution_complete'}
