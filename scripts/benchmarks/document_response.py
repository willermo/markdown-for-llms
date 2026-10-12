"""Receipt V v2 documentale: conserva testo ricevuto, nessuna firma atomica."""
from datetime import datetime,timezone
from pathlib import Path
import re

from data import descriptor,digest,digest_bytes,load,write_new
from packet import verify_packet
from safety import regular


def decision(value):
    normalized=' '.join(value.casefold().split()).strip(' .!')
    if re.search(r'non riesco|non so|non posso|forse|sembra|dubbi|incerto',normalized):return 'AMBIGUOUS'
    if re.search(r'\bmanc\w*|\bdifferenz\w*|\bcambi\w*|non corrisponde|tranne|errat\w*',normalized):return 'REJECTED'
    if normalized in ('corrisponde','tutto corrisponde','corrisponde completamente','ho confrontato tutto: corrisponde'):return 'ACCEPTED'
    return 'PENDING'


def explicit_all_documents(value):
    """Finite complete declaration; qualifiers never expand its scope.

    Normalize whitespace only for interpretation. Keep the received bytes and
    the single collective statement intact, including an attached Italian
    article (``idocumenti``). Vague approval, subsets and exceptions do not match.
    """
    normalized=' '.join(value.casefold().split())
    return re.fullmatch(
        r'tutti (?:i ?)?documenti corrispondono[.!]?'
        r'(?: ad un esame visivo, valido le conversioni verificate[.!]?)?',
        normalized,
    ) is not None


def classify(packet,raw,packet_sha,author,received_at,receipt_method):
    if packet.get('schema_version')!=2 or packet.get('kind')!='document-comparison-v2':raise ValueError('Document response needs packet v2')
    date=datetime.fromisoformat(received_at)
    if date.tzinfo is None or not author.strip() or not receipt_method.strip():raise ValueError('Actual receipt metadata required')
    aliases={}
    for pair in packet['identities']:
        for name in (pair['document_id'],pair['document_id'].split('-')[0]):
            if name.casefold() in aliases:raise ValueError('Duplicate document alias')
            aliases[name.casefold()]=pair['document_id']
    text=raw.decode('utf-8');statements=[];unattributed=[]
    if explicit_all_documents(text):
        statements.append({'document_ids':[p['document_id'] for p in packet['identities']],'text':text,'original_line':text,'attribution':'explicit_all_documents_statement'})
    for line in ([] if statements else text.splitlines()):
        if not line.strip():continue
        match=re.match(r'^\s*(?:[-*]\s+)?([CK][0-9]{2}(?:-[A-Za-z0-9-]+)?)\s*:\s*(.*)$',line,re.I)
        if match:
            ident=aliases.get(match[1].casefold())
            if ident:statements.append({'document_id':ident,'text':match[2],'original_line':line})
            else:unattributed.append(line)
        elif statements:
            statements[-1]['text']+='\n'+line;statements[-1]['original_line']+='\n'+line
        else:unattributed.append(line)
    documents=[]
    for pair in packet['identities']:
        rows=[r for r in statements if r.get('document_id')==pair['document_id'] or pair['document_id'] in r.get('document_ids',[])]
        state='PENDING' if not rows else 'STALE' if packet_sha!=packet['packet_sha256'] else 'AMBIGUOUS' if len(rows)>1 else 'ACCEPTED' if rows[0].get('attribution')=='explicit_all_documents_statement' else decision(rows[0]['text'])
        documents.append({'document_id':pair['document_id'],'canonical_sha256':pair['canonical_sha256'],'gold_sha256':pair['gold_sha256'],'pair_sha256':pair['pair_sha256'],'corpus_revision':pair['corpus_revision'],'scope':pair['scope'],'status':state,'response_text':[r['text'] for r in rows],'differences':[r['text'] for r in rows] if state=='REJECTED' else [],'explicit_complete_visible_document':state=='ACCEPTED'})
        if rows and rows[0].get('attribution')=='explicit_all_documents_statement':
            documents[-1]['response_attribution']='projection_of_single_explicit_all_documents_statement'
    states=[d['status'] for d in documents]
    status='STALE' if packet_sha!=packet['packet_sha256'] else 'REJECTED' if 'REJECTED' in states else 'AMBIGUOUS' if 'AMBIGUOUS' in states or unattributed else 'ACCEPTED' if all(s=='ACCEPTED' for s in states) else 'PENDING'
    return {'schema_version':2,'kind':'human_document_response_as_received','granularity':'document','packet_sha256_received':packet_sha,'packet_sha256_current':packet['packet_sha256'],'original_received_sha256':digest_bytes(raw),'receipt':{'author':author,'received_at':received_at,'method':receipt_method,'limits':'caller attributes author and receipt time from actual message; no cryptographic authentication, no per-property signature or observation duration'},'statements':statements,'unattributed_text':unattributed,'documents':documents,'status':status}


def import_document_response(root,response,packet_sha,output,author,received_at=None,receipt_method='actual response receipt at import; source message time unavailable'):
    packet=verify_packet(root);raw=regular(response.parent,response.name)
    received_at=received_at or datetime.now(timezone.utc).isoformat()
    result=classify(packet,raw,packet_sha,author,received_at,receipt_method)
    output.mkdir(exist_ok=False);(output/'received.txt').write_bytes(raw)
    result.update(source=descriptor(response),packet=descriptor(root/'packet.json'),imported_utc=datetime.now(timezone.utc).isoformat())
    result['response_sha256']=digest(result);write_new(output/'response.json',result)
    print('V_DOCUMENT_'+result['status']);return 0 if result['status']=='ACCEPTED' else 5


def validate_response(path):
    """Reclassify original bytes, compare every classified field and envelope."""
    value=load(path)
    if value.get('schema_version')!=2 or value.get('granularity')!='document':raise ValueError('Response v2 schema')
    if digest({k:v for k,v in value.items() if k!='response_sha256'})!=value.get('response_sha256'):raise ValueError('Response metadata STALE')
    packet_path=Path(value['packet']['path'])
    if descriptor(packet_path)!=value['packet']:raise ValueError('Response packet STALE')
    packet=verify_packet(packet_path.parent);raw=regular(path.parent,'received.txt')
    r=value['receipt'];classified=classify(packet,raw,value['packet_sha256_received'],r['author'],r['received_at'],r['method'])
    for key,expected in classified.items():
        if value.get(key)!=expected:raise ValueError('Response contract mismatch: '+key)
    return value
