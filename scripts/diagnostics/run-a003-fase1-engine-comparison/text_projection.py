"""Riproduce annotazioni finite E01; non riconosce Markdown o AST del motore."""
from array import array
import hashlib
import re
import string
import unicodedata

VERSION = 'a003-text-projection-v6'
CLASSES = {'CONTENT', 'SYNTAX', 'DOMAIN_OTHER', 'UNRESOLVED'}


def checked_span(raw, span, boundaries):
    if not isinstance(span, list) or len(span) != 2:
        raise ValueError('Context span')
    a, b = span
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(raw) or a not in boundaries or b not in boundaries:
        raise ValueError('Context span/UTF-8 boundary')
    return a, b


def raw_boundaries(raw):
    offset = 0
    boundaries = {0}
    for ch in raw.decode('utf-8', errors='strict'):
        offset += len(ch.encode('utf-8'))
        boundaries.add(offset)
    return boundaries


def finite_contexts(raw, annotation, boundaries):
    """Verifica soltanto intervalli dichiarati; nessun parsing/riscrittura globale."""
    contexts = annotation.get('contexts', [])
    for ctx in contexts:
        a, b = checked_span(raw, ctx['span'], boundaries)
        value = raw[a:b].decode('utf-8')
        if ctx['kind'] == 'uri':
            if not value.lower().startswith(('https://', 'http://', 'mailto:')) or any(ord(x) < 32 or ord(x) == 127 or x in '<>' for x in value) or (any(x.isspace() for x in value) and (ctx.get('fragmented') is not True or any(x.isspace() and x != ' ' for x in value))):
                raise ValueError('Finite URI context')
        elif ctx['kind'] == 'html-text':
            opening, closing = ctx['opening'], ctx['closing']
            checked_span(raw, opening, boundaries)
            checked_span(raw, closing, boundaries)
            left = raw[slice(*opening)].decode('utf-8')
            right = raw[slice(*closing)].decode('utf-8')
            if opening[1] != a or closing[0] != b or not re.fullmatch(r'<(span|p|em|strong|b|i|sup|sub)>', left) or right != '</' + left[1:-1] + '>':
                raise ValueError('Unproved finite HTML context')
            if '<' in value or '>' in value:
                raise ValueError('Unsupported nested HTML context')
        elif ctx['kind'] != 'prose':
            raise ValueError('Unsupported operation context')
    return contexts


def escaped_at(raw, i):
    j = i
    while j and raw[j-1:j] == b'\\':
        j -= 1
    return (i-j) % 2 == 1


def fence_regions(raw):
    """Vista finita dei prefissi; coordinate sempre sul raw, anche nei contenitori.

    La continuazione di un marker lista è la sua larghezza in spazi. Prefissi
    diversi/non qualificabili non chiudono il fence: il suffisso resta protetto.
    Solo i wrapper esterni brevi conservano la rimozione T01 già supportata.
    """
    lines = list(re.finditer(rb'[^\n]*\n|[^\n]+$', raw))
    opening = re.compile(rb'((?:[ \t]*(?:>[ \t]?|(?:[-+*]|[0-9]{1,9}[.)])[ \t]+))*[ \t]*)(`{3,}|~{3,})([^\r\n]*)(?:\r?\n|$)')
    regions = []; i = 0
    while i < len(lines):
        line = lines[i]; match = opening.fullmatch(line[0])
        if not match:
            i += 1; continue
        prefix, marker, info = match.groups()
        external = len(prefix) <= 3 and not prefix.strip(b' ')
        continuation = re.sub(rb'(?:[-+*]|[0-9]{1,9}[.)])[ \t]+', lambda m: b' '*len(m[0]), prefix)
        qualified = line[0].endswith(b'\n') and (marker[:1] != b'`' or b'`' not in info)
        close_prefix = rb' {0,3}' if external else re.escape(continuation)
        close_pattern = re.compile(close_prefix + re.escape(marker[:1]) + rb'{' + str(len(marker)).encode() + rb',}[ \t]*(?:\r?\n|$)')
        j = i+1
        while j < len(lines) and not close_pattern.fullmatch(lines[j][0]):
            j += 1
        end = lines[j].end() if j < len(lines) else len(raw)
        wrappers = [[line.start(), line.end()], [lines[j].start(), end]] if j < len(lines) and qualified and external else []
        # Solo i fence con ambito raw comprovato possono escludere delimiter.
        # Indent >=4 e liste numerate che interrompono prosa restano stime.
        proved = qualified and j < len(lines) and (external or bool(prefix.strip()) and not prefix.startswith((b'    ', b'\t')))
        numbered = re.search(rb'([0-9]{1,9})[.)] +', prefix)
        if proved and numbered:
            try:
                ordered_list_position(raw, dict(span=[line.start()+numbered.start(), line.start()+numbered.end()]), allow_fence=False)
            except ValueError:
                proved = False
        regions.append(dict(span=[line.start(), end], kind='code', wrappers=wrappers, inline_opaque=proved))
        i = j+1
    return regions


def inline_regions(raw, ch, kind, opaque):
    """Una passata con domini opachi espliciti, senza consumare altre famiglie."""
    regions = []; i = 0
    def enclosing(pos):
        return max((r['span'][1] for r in opaque if r['span'][0] <= pos < r['span'][1]), default=pos)
    while i < len(raw):
        end_opaque = enclosing(i)
        if end_opaque > i:
            i = end_opaque; continue
        if raw[i:i+1] != ch or escaped_at(raw, i):
            i += 1; continue
        j = i+1
        while raw[j:j+1] == ch:
            j += 1
        end = len(raw)
        for match in re.finditer(re.escape(ch)+b'+', raw[j:]):
            a, b = j+match.start(), j+match.end()
            if enclosing(a) > a:
                continue
            if raw[a:b] == raw[i:j] and (kind == 'code' or not escaped_at(raw, a)):
                end = b; break
        regions.append(dict(span=[i, end], kind=kind)); i = end
    return regions


def qualified_domains(raw, annotation, contexts):
    """Unione finita monotona di domini raw e interpretazioni inline concorrenti.

    Le stime raw e i dichiarativi proteggono ma non revocano letture indipendenti.
    Solo ambiti raw qualificati possono escludere delimiter; due chiusure finite
    conservano anche le letture con stime opache, senza revocare alcuno span.
    Non è una grammatica generale: aperture senza chiusura proteggono il suffisso.
    """
    regions = fence_regions(raw)
    independent = [*inline_regions(raw, b'`', 'code', []), *inline_regions(raw, b'$', 'math', [])]
    def add(a, b, kind, inline_opaque=False):
        regions.append(dict(span=[a, b], kind=kind, inline_opaque=inline_opaque))
    def starts_outside(a, kinds):
        return not any(r['kind'] in kinds and r['span'][0] < a < r['span'][1] for r in independent)
    for opening, closing, kind in [(b'\\(', b'\\)', 'math'), (b'\\[', b'\\]', 'math')]:
        i = 0
        while (i := raw.find(opening, i)) >= 0:
            end = raw.find(closing, i+len(opening))
            end = end+len(closing) if end >= 0 else len(raw)
            add(i, end, kind, starts_outside(i, {'code','math'})); i = end
    # Profilo circoscritto: code indentato solo a inizio/dopo blank.
    offset = 0; blank_before = True; indent_start = None
    for line in raw.splitlines(keepends=True):
        blank = not line.strip(b' \t\r\n')
        indented = bool(re.match(rb'(?: {4}| {0,3}\t)', line))
        if indent_start is not None and not blank and not indented:
            add(indent_start, offset, 'code', True); indent_start = None
        if indent_start is None and indented and not blank and blank_before:
            indent_start = offset
        offset += len(line); blank_before = blank
    if indent_start is not None:
        add(indent_start, len(raw), 'code', True)
    for match in re.finditer(rb'<(?:code|pre)\b[^>]*(?:>|$)', raw, re.I):
        tag = re.match(rb'<(code|pre)', match[0], re.I)[1]
        close = re.search(rb'</'+tag+rb'\s*>', raw[match.end():], re.I)
        add(match.start(), match.end()+close.end() if close else len(raw), 'code', bool(close) and starts_outside(match.start(), {'code','math'}))
    for match in re.finditer(rb'(?:[a-z][a-z0-9+.-]*://|mailto:|www\.)[^\s<>\[\]]+', raw, re.I):
        add(*match.span(), 'uri', starts_outside(match.start(), {'code','math'}))
    fixed = list(regions)
    for ctx in contexts:
        if ctx['kind'] == 'uri':
            add(*ctx['span'], 'uri')
    for segment in annotation['segments']:
        if segment.get('channel') in {'code', 'literal', 'math', 'formula', 'cell', 'cells'} or segment.get('domain') in {'code', 'literal', 'math', 'cell', 'uri'}:
            add(*segment['span'], 'declared-preserved')
    families = ((b'`', 'code'), (b'$', 'math'))
    # Due chiusure indipendenti: le stime raw aggiungono protezione, mai
    # sostituiscono le letture con soli fence comprovati. I dichiarativi
    # aggiungono span ma non sono prove per escludere delimiter raw.
    bases = [[r for r in fixed if r.get('inline_opaque')], fixed]
    for base in bases:
        known = {kind: {} for _, kind in families}
        def retain(kind, found):
            changed = False
            for region in found:
                key = tuple(region['span'])
                if key not in known[kind]:
                    known[kind][key] = region; regions.append(region); changed = True
            return changed
        for ch, kind in families:
            retain(kind, inline_regions(raw, ch, kind, base))
        changed = True
        while changed:
            changed = False
            previous = {kind: list(values.values()) for kind, values in known.items()}
            for ch, kind in families:
                other = previous['math' if kind == 'code' else 'code']
                for mask in [other, *[[region] for region in other]]:
                    changed |= retain(kind, inline_regions(raw, ch, kind, base+mask))
    return regions


def preserved_boundary(segment, regions):
    a, b = segment['span']
    if any(a < c['span'][1] and b > c['span'][0] for c in regions):
        raise ValueError('Operation inside preserved code/math/URI/raw domain')


def operation_context(segment, contexts, kind):
    a, b = segment['span']
    if not any(c['kind'] == kind and c['span'][0] <= a < b <= c['span'][1] for c in contexts):
        raise ValueError('Operation context absent')


def finite_destination(raw, start):
    """Conclusione del solo target nudo bilanciato o <...>, senza titoli."""
    i = start
    if raw[i:i+1] == b'<':
        i += 1
        while i < len(raw) and raw[i:i+1] != b'>':
            if raw[i:i+1] == b'<' or raw[i] <= 32 or raw[i] == 127:
                raise ValueError('Unsupported angle destination')
            i += 1
        if i == start+1 or raw[i:i+2] != b'>)':
            raise ValueError('Unclosed/ambiguous angle destination')
        if any(x.isspace() or unicodedata.category(x).startswith('C') for x in raw[start+1:i].decode('utf-8')):
            raise ValueError('Destination whitespace/control')
        return i+1
    depth = 0
    while i < len(raw):
        ch = raw[i]
        if ch <= 32 or ch == 127 or ch in (60, 62, 92):
            raise ValueError('Destination whitespace/control/unsupported escape')
        if ch == 40:
            depth += 1
        elif ch == 41:
            if depth == 0:
                if i == start:
                    raise ValueError('Empty destination')
                value = raw[start:i].decode('utf-8')
                if any(x.isspace() or unicodedata.category(x).startswith('C') for x in value):
                    raise ValueError('Destination whitespace/control')
                return i
            depth -= 1
        i += 1
    raise ValueError('Unclosed finite destination')


def padding_boundary(raw, segment, annotation):
    a, b = segment['span']
    left = raw[raw.rfind(b'\n', 0, a)+1:a]
    end = raw.find(b'\n', b)
    end = end if end >= 0 else len(raw)
    right = raw[b:end].removesuffix(b'\r')
    if not left.strip(b' ') or not right.strip(b' '):
        return
    # Solo suffisso strutturale già comprovato da finite_relations, a fine riga.
    suffix = [s for s in annotation['segments'] if b <= s['span'][0] < end]
    if suffix and suffix[0]['span'][0] == b and suffix[-1]['span'][1] == end and all(s['operation'] == 'T05' or s['operation'] == 'T04' and s.get('role') == 'declaration' for s in suffix):
        return
    raise ValueError('T00 internal significant padding; unit boundary is not evidence')


def fence_boundary(raw, segment, region):
    """Solo wrapper esterno di un fence chiuso, payload interamente preservato."""
    return segment['span'] in region.get('wrappers', [])


def block_position(raw, segment, annotation):
    """Prefisso reale di riga: indent breve e wrapper precedenti qualificati."""
    a = segment['span'][0]
    start = raw.rfind(b'\n', 0, a)+1
    cursor = start
    while cursor < a and raw[cursor:cursor+1] == b' ':
        cursor += 1
    if cursor-start > 3:
        raise ValueError('T01 unsupported block indentation')
    for previous in annotation['segments']:
        x, y = previous['span']
        if x < cursor or y > a:
            continue
        if x != cursor or previous['operation'] != 'T01' or previous.get('wrapper') not in {'quote', 'list'}:
            break
        cursor = y
    if cursor != a:
        raise ValueError('T01 not at proved raw block position')


def ordered_list_position(raw, segment, *, allow_fence=True):
    """1–9 cifre; numero diverso da 1 solo in un blocco già comprovato.

    Continuazioni di paragrafo/di contenitore non qualificate restano dati.
    Il canale strutturale conserva il marker raw, senza normalizzare il numero.
    """
    a = segment['span'][0]
    value = raw[slice(*segment['span'])]
    match = re.fullmatch(rb'([0-9]{1,9})[.)] ', value)
    if not match:
        if re.fullmatch(rb'[-+*] ', value):
            return
        raise ValueError('T01 unsupported list marker length/form')
    end = raw.find(b'\n', segment['span'][1])
    end = end if end >= 0 else len(raw)
    nonempty = bool(raw[segment['span'][1]:end].strip(b' \t\r'))
    if int(match[1]) == 1 and nonempty:
        return
    start = raw.rfind(b'\n', 0, a)+1
    prefix = raw[start:a]
    if allow_fence and any(r.get('inline_opaque') and r['span'][1] == start for r in fence_regions(raw[:start])):
        return
    # La vista di contenitore è esatta: non inventa coordinate né riscrive raw.
    prior = raw[:start].splitlines()
    while prior:
        line = prior.pop()
        if prefix:
            if not line.strip(b' >\t\r'):
                return
            if not line.startswith(prefix):
                raise ValueError('T01 unqualified list container continuation')
            line = line[len(prefix):]
        if not line.strip(b' \t\r'):
            return
        if re.match(rb' {0,3}#{1,6} +\S', line):
            return
        previous = re.match(rb' {0,3}(?:([-+*])|([0-9]{1,9})[.)]) +\S', line)
        if not previous:
            raise ValueError('T01 ordered number interrupts raw paragraph')
        if previous[1] or int(previous[2]) == 1:
            return
        # Le righe numerate precedenti devono risalire a un inizio qualificato.
    return


def metadata_position(raw, segment):
    a, b = segment['span']
    if segment.get('kind') == 'blank-layout':
        value = raw[a:b]
        if not value or not re.fullmatch(rb'(?:\r?\n)+', value):
            raise ValueError('T08 invalid blank layout')
        if raw[a-1:a] in (b'\r', b'\n') or raw[b:b+1] in (b'\r', b'\n'):
            raise ValueError('T08 split raw layout run')
        # Una sola fine riga interna è soft-wrap, mai metadata.
        if value.count(b'\n') < 2 and raw[:a].strip() and raw[b:].strip():
            raise ValueError('T08 internal line break is not a blank separator')
    elif segment.get('kind') == 'empty-anchor':
        if (a and raw[a-1:a] not in b' \t\r\n') or (b < len(raw) and raw[b:b+1] not in b' \t\r\n'):
            raise ValueError('T08 anchor lacks raw separation')


def emphasis_flanking(raw, segment):
    a, b = segment['span']
    before = raw[:a].decode('utf-8')[-1:]
    after = raw[b:].decode('utf-8')[:1]
    marker = segment['raw'][0]
    if before == marker or after == marker or escaped_at(raw, a):
        raise ValueError('T03 split/escaped delimiter run')
    whitespace = lambda ch: not ch or ch in '\t\n\f\r' or unicodedata.category(ch) == 'Zs'
    punctuation = lambda ch: bool(ch) and (ch in string.punctuation or unicodedata.category(ch)[0] in 'PS')
    left = not whitespace(after) and (not punctuation(after) or whitespace(before) or punctuation(before))
    right = not whitespace(before) and (not punctuation(before) or whitespace(after) or punctuation(after))
    opens = left and (marker == '*' or not right or punctuation(before))
    closes = right and (marker == '*' or not left or punctuation(after))
    if not (opens if segment.get('side') == 'open' else closes if segment.get('side') == 'close' else False):
        raise ValueError('T03 unproved delimiter flanking')
    return opens, closes


FINITE_TAG = rb'</?(?:span|em|strong|b|i|sup|sub|p|a)(?:\s+(?:id|name)="[A-Za-z0-9_-]+")?\s*>|<br ?/?>'


def emphasis_block_boundaries(raw):
    """Confini finiti sul raw, indipendenti da T01/T08 o nomi delle unità."""
    points = []
    for line in re.finditer(rb'[^\n]*\n|[^\n]+$', raw):
        value = line[0].removesuffix(b'\n').removesuffix(b'\r')
        if not value.strip(b' \t') or re.match(rb' {0,3}(?:#{1,6}[ \t]+\S|>[ \t]?|(?:[-+*]|1[.)])[ \t]+\S|`{3,}|~{3,}|</?p(?:>|[ \t]))', value):
            points.append(line.start())
    return points


def emphasis_label_scope(pair, relations):
    """Entrambi i marker nella label, o entrambi fuori dall'intero link."""
    opening, closing = pair
    for relation in relations.values():
        if relation['kind'] not in {'link', 'image'}:
            continue
        a, b = relation['label_span']
        if opening[0] < b and closing[1] > a:
            inside = a <= opening[0] and closing[1] <= b
            enclosing = opening[1] <= relation['segments'][0][0] and closing[0] >= relation['segments'][-1][1]
            if not (inside or enclosing):
                return False
    return True


def raw_emphasis_pairs(raw, regions):
    """Accoppiamento raw di run interi; KEEP/unità/pair_id non assegnano ruoli.

    Chiusura prima dell'apertura per run a doppio ruolo, opener più recente.
    Consumo parziale di run, incroci e competizione non qualificata sono rifiutati.
    """
    stack = []; pairs = set(); previous_end = 0
    boundaries = emphasis_block_boundaries(raw)
    tags = [m.span() for m in re.finditer(FINITE_TAG, raw)]
    for match in re.finditer(rb'\*+|_+', raw):
        a, b = match.span()
        if any(previous_end <= point <= a for point in boundaries):
            stack.clear()
        previous_end = b
        if any(x <= a and b <= y for x,y in tags):
            continue
        if escaped_at(raw, a) or any(a < r['span'][1] and b > r['span'][0] for r in regions):
            continue
        value = match[0].decode()
        if len(value) > 2:
            stack.clear(); continue
        roles = []
        for side in ('open', 'close'):
            try:
                emphasis_flanking(raw, dict(span=[a,b],raw=value,side=side))
                roles.append(side)
            except ValueError:
                pass
        closed = False
        if 'close' in roles and stack:
            opener = stack[-1]
            # Regola dei multipli di tre per ruoli concorrenti, senza run parziali.
            compatible = not (('close' in opener['roles'] or 'open' in roles) and
                              (len(opener['raw'])+len(value)) % 3 == 0 and
                              (len(opener['raw']) % 3 or len(value) % 3))
            if opener['raw'] == value and compatible:
                stack.pop(); pairs.add((opener['span'], (a,b))); closed = True
            elif any(s['raw'] == value for s in stack):
                stack.clear()
        if not closed and 'open' in roles:
            stack.append(dict(span=(a,b),raw=value,roles=roles))
    return pairs


def qualify_operations(raw, annotation, contexts, regions):
    """Gate integrale prima di costruire emissioni; nessun matching col gold."""
    pairs = {}
    stack = []
    for s in annotation['segments']:
        if s['operation'] == 'T01':
            block_position(raw, s, annotation)
            if s.get('wrapper') == 'list':
                ordered_list_position(raw, s)
            if s.get('wrapper') == 'fence' and not any(fence_boundary(raw, s, r) for r in regions):
                raise ValueError('T01 unproved external fence wrapper')
        if s['operation'] == 'T08':
            metadata_position(raw, s)
        if s['operation'] not in {'T00', 'T02', 'T03', 'T07'}:
            if s['operation'] != 'KEEP':
                a,b = s['span']
                blocked = [r for r in regions if r['kind'] != 'uri' and a < r['span'][1] and b > r['span'][0]]
                if blocked and not (s['operation'] == 'T01' and s.get('wrapper') == 'fence' and
                                    all(r['kind'] == 'code' and fence_boundary(raw,s,r) for r in blocked)):
                    raise ValueError('Syntax operation inside preserved code/math/literal raw domain')
            continue
        preserved_boundary(s, regions)
        if s['operation'] in {'T02', 'T07'}:
            operation_context(s, contexts, 'prose' if s['operation'] == 'T02' else 'html-text')
        if s['operation'] == 'T00' and s.get('mode') == 'padding':
            padding_boundary(raw, s, annotation)
        if s['operation'] == 'T00' and s.get('mode') == 'soft-wrap':
            a, b = s['span']
            if not a or b == len(raw) or raw[a-1:a] in (b'\n', b'\r', b'\\') or raw[a-2:a] == b'  ' or raw[b:b+1] in (b'\n', b'\r'):
                raise ValueError('T00 unproved soft-wrap/hard-break')
        if s['operation'] == 'T03':
            if s['raw'] not in {'*', '**', '_', '__'}:
                raise ValueError('T03 invalid delimiter')
            emphasis_flanking(raw, s)
            pairs.setdefault(s['pair_id'], []).append(s)
            if s.get('side') == 'open':
                stack.append(s['pair_id'])
            elif s.get('side') != 'close' or not stack or stack.pop() != s['pair_id']:
                raise ValueError('Crossed/unpaired emphasis intervals')
    raw_regions = qualified_domains(raw, {'segments': []}, [])
    relations = finite_relations(raw, annotation)
    # Le destinazioni sono già comprovate da finite_relations, non dal nome
    # del dominio dichiarato: marker al loro interno non sono run di prosa.
    targets = [dict(span=s['span'], kind='reference-target') for s in annotation['segments']
               if s['operation'] == 'T04' and s.get('role') == 'destination']
    raw_pairs = raw_emphasis_pairs(raw, raw_regions+targets)
    uncertain_tags = [m.span() for m in re.finditer(rb'</?[A-Za-z][^<>]*(?:>|$)', raw) if not re.fullmatch(FINITE_TAG, m[0])]
    for pair in pairs.values():
        if len(pair) != 2 or pair[0]['raw'] != pair[1]['raw'] or [s['side'] for s in pair] != ['open','close'] or pair[0]['span'][1] >= pair[1]['span'][0]:
            raise ValueError('Unpaired/empty emphasis')
        candidate = (tuple(pair[0]['span']), tuple(pair[1]['span']))
        if candidate not in raw_pairs or not emphasis_label_scope(candidate, relations) or any(candidate[0][0] < y and candidate[1][1] > x for x,y in uncertain_tags):
            raise ValueError('T03 unproved pair against competing raw delimiter runs')
    if stack:
        raise ValueError('Unpaired emphasis')


def finite_relations(raw, annotation):
    relations = annotation.get('relations', {})
    grouped = {}
    for segment in annotation['segments']:
        if segment['operation'] in {'T04', 'T05'}:
            grouped.setdefault(segment.get('relation'), []).append(segment)
    if set(grouped) != set(relations):
        raise ValueError('Reference registry missing/unused')
    for ident, segments in grouped.items():
        r = relations[ident]
        if r.get('channel') != 'references' or r.get('segments') != [s['span'] for s in segments]:
            raise ValueError('Reference channel/spans')
        if r.get('declarations', []) != [s['raw'] for s in segments if s.get('role') == 'declaration']:
            raise ValueError('Reference declaration differs from raw')
        if r['kind'] in {'link', 'image'}:
            main = [s for s in segments if s.get('role') != 'declaration']
            if len(main) != 4 or [s['raw'] for s in main[::3]] != [('![' if r['kind'] == 'image' else '['), ')'] or main[1]['raw'] != '](' or [s.get('role') for s in main] != ['delimiter', 'delimiter', 'destination', 'delimiter']:
                raise ValueError('Finite link delimiters/roles')
            if main[1]['span'][1] != main[2]['span'][0] or main[2]['span'][1] != main[3]['span'][0] or r['label_span'] != [main[0]['span'][1], main[1]['span'][0]] or r['target'] != main[2]['raw'] or not r['target']:
                raise ValueError('Finite link label/target')
            if finite_destination(raw, main[2]['span'][0]) != main[2]['span'][1]:
                raise ValueError('Destination absorbs prose beyond raw construct')
            label = raw[slice(*r['label_span'])].decode('utf-8')
            if r['label'] != label:
                raise ValueError('Reference label')
            for s in annotation['segments']:
                if r['label_span'][0] <= s['span'][0] < r['label_span'][1] and s['operation'] not in {'KEEP', 'T02', 'T03', 'T06', 'T07'}:
                    raise ValueError('Label content discarded')
        elif r['kind'] == 'anchor':
            if len(segments) != 1 or segments[0].get('role') != 'declaration':
                raise ValueError('Anchor declaration')
        elif r['kind'] == 'note':
            if len(segments) != 1 or segments[0]['operation'] != 'T05' or segments[0]['raw'] != '[^' + r['id'] + ']':
                raise ValueError('Note marker association')
            definition = r['definition_span']
            if raw[slice(*definition)].decode('utf-8') != '[^' + r['id'] + ']: ' or not any(s['span'] == definition and s['operation'] == 'T01' and s.get('wrapper') == 'note' for s in annotation['segments']):
                raise ValueError('Note definition absent')
        else:
            raise ValueError('Unsupported finite relation')
        for s in segments:
            if s.get('role') == 'declaration' and not re.fullmatch(r'(?:\[\])?\{#[A-Za-z0-9_.#-]+(?: \.noteref)?\}', s['raw']):
                raise ValueError('Unsupported anchor declaration')
    return relations


def project(raw, annotation):
    text = raw.decode('utf-8', errors='strict')
    if annotation['version'] != VERSION or annotation['sha256'] != hashlib.sha256(raw).hexdigest() or annotation['bytes'] != len(raw):
        raise ValueError('Raw-map identity STALE')
    boundaries = raw_boundaries(raw)
    cursor = 0
    for segment in annotation['segments']:
        a, b = checked_span(raw, segment['span'], boundaries)
        if a != cursor or raw[a:b].decode('utf-8') != segment['raw'] or segment['classification'] not in CLASSES:
            raise ValueError('Raw-map gap/overlap/content/classification')
        cursor = b
    if cursor != len(raw):
        raise ValueError('Incomplete raw coverage')
    contexts = finite_contexts(raw, annotation, boundaries)
    relations = finite_relations(raw, annotation)
    regions = qualified_domains(raw, annotation, contexts)
    qualify_operations(raw, annotation, contexts, regions)
    cursor = 0
    pairs = {}
    units = {}
    records = []
    unresolved = set()
    for segment in annotation['segments']:
        a, b = segment['span']
        if type(a) is not int or type(b) is not int or a != cursor or b <= a or a not in boundaries or b not in boundaries:
            raise ValueError('Raw-map gap/overlap/UTF-8 boundary')
        cursor = b
        original = raw[a:b].decode('utf-8')
        if original != segment['raw'] or segment['classification'] not in CLASSES:
            raise ValueError('Raw-map content/classification')
        op = segment['operation']
        unit = segment['unit_id']
        units.setdefault(unit, [])
        emitted = original
        if op == 'KEEP':
            if segment['classification'] == 'SYNTAX':
                raise ValueError('Syntax requires explicit operation')
        elif op == 'T00':
            mode = segment['mode']
            if segment.get('domain') != 'prose':
                raise ValueError('T00 only qualified prose layout')
            if mode == 'soft-wrap' and original in {'\n', '\r\n'}:
                emitted = ' '
            elif mode == 'crlf' and original == '\r\n':
                emitted = '\n'
            elif mode == 'padding' and original and set(original) == {' '}:
                emitted = ''
            else:
                raise ValueError('T00 unsupported')
        elif op == 'T01':
            patterns = {'heading': r'#{1,6} ', 'quote': r'> ', 'list': r'(?:[-+*]|[0-9]{1,9}[.)]) ',
                        'caption': r': ', 'note': r'\[\^[A-Za-z0-9_-]+\]: ', 'fence': r'(?:`{3,}|~{3,})(?:[^\r\n]*)\r?\n'}
            if not re.fullmatch(patterns[segment['wrapper']], original) or not segment.get('evidence'):
                raise ValueError('T01 wrapper/evidence')
            emitted = ''
        elif op == 'T02':
            if segment.get('domain') != 'prose' or len(original) != 2 or original[0] != '\\' or original[1] not in string.punctuation:
                raise ValueError('T02 invalid escape/domain')
            emitted = original[1]
        elif op == 'T03':
            if original not in {'*', '**', '_', '__'} or segment.get('domain') != 'prose':
                raise ValueError('T03 invalid delimiter/domain')
            pairs.setdefault(segment['pair_id'], []).append((original, segment['side'], unit, a))
            emitted = ''
        elif op in {'T04', 'T05'}:
            if not segment.get('relation') or not segment.get('evidence'):
                raise ValueError('Reference relationship/evidence missing')
            if op == 'T05' and not re.fullmatch(r'\[\^[A-Za-z0-9_-]+\]', original):
                raise ValueError('T05 non-marker')
            if op == 'T04' and not segment.get('role') in {'delimiter', 'destination', 'declaration'}:
                raise ValueError('T04 syntax role')
            emitted = ''
        elif op == 'T06':
            allowed = r'</?(?:span|em|strong|b|i|sup|sub|p|a)(?:\s+(?:id|name)="[A-Za-z0-9_-]+")?\s*>'
            if original in {'<br>', '<br/>', '<br />'}:
                emitted = '\n'
            elif re.fullmatch(allowed, original) and segment.get('evidence'):
                emitted = ''
            else:
                raise ValueError('T06 unsupported/active HTML')
        elif op == 'T07':
            if segment.get('domain') != 'html-text':
                raise ValueError('T07 outside HTML text')
            named = {'&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&apos;': "'"}
            if original in named:
                emitted = named[original]
            elif re.fullmatch(r'&#(?:[0-9]+|x[0-9a-fA-F]+);', original):
                value = int(original[3:-1], 16) if original.startswith('&#x') else int(original[2:-1])
                if value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
                    raise ValueError('T07 invalid Unicode')
                emitted = chr(value)
            else:
                raise ValueError('T07 unknown entity')
        elif op == 'T08':
            if segment.get('kind') == 'page' and segment.get('paginate') is True:
                wanted = '\n\n{' + str(segment['page_id']) + '}' + '-' * 48 + '\n\n'
                if original != wanted:
                    raise ValueError('T08 page marker')
            elif segment.get('kind') == 'blank-layout' and set(original) <= {'\n', '\r'} and segment.get('evidence'):
                pass
            elif segment.get('kind') == 'empty-anchor' and re.fullmatch(r'\[\]\{#[A-Za-z0-9_-]+\}', original) and segment.get('evidence'):
                pass
            else:
                raise ValueError('T08 unsupported metadata')
            emitted = ''
        else:
            raise ValueError('Unknown operation')
        if segment['classification'] == 'UNRESOLVED':
            if op != 'KEEP':
                raise ValueError('Unresolved raw must remain intact')
            unresolved.add(unit)
        if segment['classification'] == 'DOMAIN_OTHER':
            if not segment.get('channel') or not segment.get('evidence') or op != 'KEEP':
                raise ValueError('Domain content requires preserved channel/evidence')
        if segment.get('emit') != emitted:
            raise ValueError('Declared emission differs')
        units[unit].append(emitted)
        records.append(dict(segment, emitted=emitted))
    if cursor != len(raw):
        raise ValueError('Incomplete raw coverage')
    for pair in pairs.values():
        if len(pair) != 2 or pair[0][:1] != pair[1][:1] or [p[1] for p in pair] != ['open', 'close'] or pair[1][3] <= pair[0][3]+len(pair[0][0]):
            raise ValueError('Unpaired emphasis')
    stack = []
    for segment in annotation['segments']:
        if segment['operation'] != 'T03':
            continue
        if segment['side'] == 'open':
            stack.append(segment['pair_id'])
        elif not stack or stack.pop() != segment['pair_id']:
            raise ValueError('Crossed emphasis intervals')
    return {'units': {k: ''.join(v) for k, v in units.items()}, 'unresolved': sorted(unresolved), 'operations': records, 'codepoints': len(text), 'references': relations}


def edits(expected, observed):
    n, m = len(expected), len(observed)
    if max(n, m) > 32768 or n * m > 4000000:
        return {'status': 'UNAVAILABLE_CAP', 'n': n, 'm': m, 'cells': n * m}
    rows = [array('I', range(m + 1))]
    for i, a in enumerate(expected, 1):
        row = array('I', [i])
        for j, b in enumerate(observed, 1):
            row.append(min(rows[-1][j-1] + (a != b), rows[-1][j] + 1, row[-1] + 1))
        rows.append(row)
    i, j = n, m
    operations = []
    while i or j:
        if i and j and rows[i][j] == rows[i-1][j-1] + (expected[i-1] != observed[j-1]):
            operations.append({'op': 'match' if expected[i-1] == observed[j-1] else 'substitution', 'expected_index': i-1, 'observed_index': j-1})
            i -= 1
            j -= 1
        elif i and rows[i][j] == rows[i-1][j] + 1:
            operations.append({'op': 'deletion', 'expected_index': i-1, 'observed_index': j})
            i -= 1
        else:
            operations.append({'op': 'insertion', 'expected_index': i, 'observed_index': j-1})
            j -= 1
    d = rows[n][m]
    return {'status': 'MEASURED' if n else 'UNAVAILABLE_ZERO_DENOMINATOR' if m else 'NOT_APPLICABLE',
            'n': n, 'm': m, 'distance': d, 'ratio': d/n if n else None, 'edits': list(reversed(operations)),
            'tie_break': 'diagonal,deletion,insertion'}


def measure(expected, observed):
    tokens_e = [t for t in expected.split(' ') if t]
    tokens_o = [t for t in observed.split(' ') if t]
    return {'expected': expected, 'observed': observed, 'expected_tokens': tokens_e, 'observed_tokens': tokens_o,
            'CER': edits(expected, observed), 'WER': edits(tokens_e, tokens_o)}


def extra_class(unit, observed, results, raw):
    """Classe solo da locatori raw e due vicini MATCHED concordi, senza gold matching."""
    classes = {oid:row['class'] for row in results if row['status'] == 'MATCHED' and row['expected_kind'] == 'prose'
               and not row['type_discrepancy'] and row['measurement'] != 'UNAVAILABLE'
               for oid in row['observed_ids']}
    bounds = raw_boundaries(raw) if raw is not None else set()
    def located(value):
        spans = value.get('raw_spans', [])
        if not spans or raw is None:
            return None
        previous = -1
        for span in spans:
            a, b = checked_span(raw, span, bounds)
            if a < previous:
                raise ValueError('Unordered observed raw locator')
            previous = b
        return spans[0][0], spans[-1][1]
    where = located(unit)
    if where is None:
        return None, dict(status='UNAVAILABLE',reason='raw locators absent',candidate_classes=[])
    if ''.join(raw[slice(*span)].decode('utf-8') for span in unit['raw_spans']) != unit['text']:
        raise ValueError('EXTRA content differs from raw locators')
    left = []; right = []
    for value in observed.values():
        if value['id'] == unit['id']:
            continue
        position = located(value)
        if position is None:
            continue
        if position[0] < where[1] and position[1] > where[0]:
            raise ValueError('EXTRA raw locator overlaps observed content')
        if value['id'] in classes:
            if position[1] <= where[0]: left.append((position[1],value['id']))
            if position[0] >= where[1]: right.append((position[0],value['id']))
    l = max(left)[1] if left else None
    r = min(right)[1] if right else None
    candidates = [classes[x] for x in (l,r) if x and isinstance(classes[x],dict)]
    proof = dict(left_observed=l,right_observed=r,raw_spans=unit['raw_spans'],candidate_classes=candidates,
                 rule='nearest MATCHED prose neighbors on both raw sides, identical full class')
    if l and r and isinstance(classes[l],dict) and classes[l] == classes[r]:
        return classes[l], dict(proof,status='ATTRIBUTED')
    return None, dict(proof,status='UNAVAILABLE',reason='absent or discordant neighbor classes')


def text_aggregates(assessment):
    """E01.d: extra al numeratore, gold invariato; incertezza non è zero."""
    groups = {}
    def key(cls): return (cls.get('page'),cls.get('native'),cls.get('language'))
    for row in assessment['units']:
        cls = row['class']
        if row['expected_kind'] != 'prose' or not isinstance(cls,dict):
            continue
        v = groups.setdefault(key(cls), dict(CER=dict(distance=0,n=0),WER=dict(distance=0,n=0),unavailable=False))
        m = row['measurement']
        if m == 'UNAVAILABLE':
            v['unavailable'] = True
            v['CER']['n'] += row['expected_characters']
            v['WER']['n'] += len([x for x in row['expected'].split(' ') if x])
        else:
            for metric in ('CER','WER'):
                if 'distance' not in m[metric]: v['unavailable'] = True
                else: v[metric]['distance'] += m[metric]['distance']
                v[metric]['n'] += m[metric]['n']
    for extra in assessment['extras']:
        if extra['kind'] != 'prose': continue
        if extra['attribution']['status'] == 'ATTRIBUTED':
            k = key(extra['class'])
            if k not in groups: raise ValueError('EXTRA attributed outside expected classes')
            for metric in ('CER','WER'):
                m = extra['measurement'][metric]
                if 'distance' not in m: groups[k]['unavailable'] = True
                else: groups[k][metric]['distance'] += m['distance']
        else:
            candidates = extra['attribution'].get('candidate_classes', [])
            affected = {key(cls) for cls in candidates} if candidates else set(groups)
            for k in affected:
                if k in groups: groups[k]['unavailable'] = True
    classes = []
    for (page,native,language),v in groups.items():
        metrics = {}
        for name in ('CER','WER'):
            m = v[name]; n = m['n']
            status = 'UNAVAILABLE' if v['unavailable'] else 'MEASURED' if n else 'UNAVAILABLE_ZERO_DENOMINATOR' if m['distance'] else 'NOT_APPLICABLE'
            metrics[name] = dict(m,status=status,ratio=m['distance']/n if n and not v['unavailable'] else None)
        classes.append(dict(page=page,native=native,language=language,metrics=metrics))
    return classes


def native_text_verdict(assessment):
    """Gate native0 fini: un extra non diventa PASS a denominatore zero."""
    if any(extra['attribution']['status'] == 'UNAVAILABLE' for extra in assessment['extras'] if extra['kind'] == 'prose'):
        return 'UNAVAILABLE'
    if any(extra['status'] == 'FAIL_EXTRA' for extra in assessment['extras'] if extra['kind'] == 'prose'):
        return 'FAIL'
    classes = text_aggregates(assessment)
    if not classes or any(m['status'].startswith('UNAVAILABLE') for cls in classes for m in cls['metrics'].values()):
        return 'UNAVAILABLE'
    return 'FAIL' if any(m['distance'] for cls in classes for m in cls['metrics'].values()) else 'PASS'


def assess(expected_units, observed_units, alignment, *, raw=None):
    """Tutti i denominatori restano presenti; falso code/math non esenta prosa."""
    expected = {u['id']: u for u in expected_units}
    observed = {u['id']: u for u in observed_units}
    if len(expected) != len(expected_units) or len(observed) != len(observed_units):
        raise ValueError('Duplicate unit IDs')
    if set(alignment) != set(expected):
        raise ValueError('Missing alignment entries')
    used = set()
    results = []
    for eid, gold in expected.items():
        row = alignment[eid]
        status = row['status']
        if status not in {'MATCHED', 'MISSING', 'AMBIGUOUS'}:
            raise ValueError('Alignment status')
        ids = row.get('observed_ids', [])
        if status == 'MISSING' and ids or status == 'MATCHED' and not ids:
            raise ValueError('Alignment cardinality')
        if len(ids) != len(set(ids)) or any(i not in observed or i in used for i in ids):
            raise ValueError('Observed unit duplicated/unknown')
        used.update(ids)
        values = [observed[i] for i in ids]
        discrepant = any(u['kind'] != gold['kind'] for u in values)
        unresolved = status == 'AMBIGUOUS' or any(u.get('unresolved') for u in values)
        text = ''.join(u['text'] for u in values)
        result = {'expected_id': eid, 'status': status, 'observed_ids': ids, 'type_discrepancy': discrepant,
                  'expected_kind': gold['kind'], 'class': gold.get('class'), 'expected_characters': len(gold['text']),
                  'critical_missing': status == 'MISSING'}
        if unresolved:
            result.update(measurement='UNAVAILABLE', expected=gold['text'], observed=text)
        else:
            result.update(measurement=measure(gold['text'], text), critical_type_loss=discrepant)
        results.append(result)
    extras = []
    for i, u in observed.items():
        if i in used: continue
        cls, attribution = extra_class(u, observed, results, raw) if u['kind'] == 'prose' else (u.get('class'),dict(status='SEPARATE_DOMAIN'))
        extras.append(dict(observed_id=i,kind=u['kind'],**{'class':cls},measurement=measure('',u['text']),attribution=attribution,
                           status='FAIL_EXTRA' if u['text'] else 'EMPTY_EXTRA'))
    return {'version': VERSION, 'units': results, 'extras': extras,
            'gold_char_denominator': sum(len(u['text']) for u in expected.values()),
            'human_assessment': 'PENDING', 'author': 'AI implementatore; finite annotations, no candidate AST'}
