"""Microcasi review Claude r014: raw, premesse AST condizionali e attesi manuali.

Congelati prima di qualunque esecuzione (freeze-receipt.json). Nessun helper del
prodotto: posizioni attese ricavate dal raw con str.find nel checker host.
Ogni caso ha due rami: l'AST Pandoc 3.1.3 sceglie soltanto la sintassi
(HTML reale oppure owner inerte); sicurezza/asset restano attesi manuali.
La previsione statica V14 deriva dalla lettura del reader (non eseguito prima).
"""

CASES = [
    # --- stato del blocco dopo un opener div non chiuso (F60/F61, C03) ---
    dict(id='U1-unclosed-div-tilde-script',
         raw='::: x\n~~~\n<script>x</script>\n~~~\n\nEnd.\n',
         tag='<script', premise='Pandoc: div non chiuso non e Div; ~~~ non interrompe il paragrafo ::: x; script HTML reale',
         if_active=dict(detail_prefix='HTML attivo/sconosciuto: script', visits=1, audit='INCOMPLETE/UNSAFE_REFERENCE'),
         if_inert=dict(detail_prefix=None, visits=0, audit='COMPLETE'),
         static_v14=dict(detail=None, visits=0, audit='COMPLETE', note='div_at azzera paragraph_open; fence code; nodo unknown unclosed div senza resource_syntax')),
    dict(id='U2-unclosed-div-tilde-img-safe',
         raw='::: x\n~~~\n<img src="p.png">\n~~~\n\nEnd.\n',
         tag='<img', premise='come U1, img sicura HTML reale',
         if_active=dict(detail_prefix=None, visits=1, audit='COMPLETE con asset p.png oppure INCOMPLETE con residuo resource_syntax', asset='p.png'),
         if_inert=dict(detail_prefix=None, visits=0, audit='COMPLETE'),
         static_v14=dict(detail=None, visits=0, audit='COMPLETE', assets=[])),
    dict(id='U3-unclosed-div-heading-quoted-html',
         raw='::: x\n# H {k="<img src=../x.png>"}\n\nEnd.\n',
         tag='<img', premise='div non chiuso: # H e testo del paragrafo (blank_before_header); img nel valore e RawInline reale',
         if_active=dict(detail_prefix="Path evasivo: '../x.png'", visits=1, audit='INCOMPLETE/UNSAFE_REFERENCE'),
         if_inert=dict(detail_prefix=None, visits=0, audit='COMPLETE'),
         static_v14=dict(detail=None, visits=0, audit='COMPLETE', note='heading dopo div con metadata accettati opachi')),
    dict(id='C1-control-closed-div-tilde-script',
         raw='::: x\n~~~\n<script>x</script>\n~~~\n:::\n\nEnd.\n',
         tag='<script', premise='controllo: div chiuso, CodeBlock nel Div, script inerte',
         if_active=dict(detail_prefix='HTML attivo/sconosciuto: script', visits=1, audit='INCOMPLETE/UNSAFE_REFERENCE'),
         if_inert=dict(detail_prefix=None, visits=0, audit='COMPLETE'),
         static_v14=dict(detail=None, visits=0, audit='COMPLETE')),
    # --- heading ATX con spazi iniziali (stato/heading, C03/C04) ---
    dict(id='H1-indented-heading-tilde-script',
         raw=' # H\n~~~\n<script>x</script>\n~~~\n\nEnd.\n',
         tag='<script', premise='se Pandoc non accetta ATX indentata, " # H" e paragrafo e ~~~ non lo interrompe',
         if_active=dict(detail_prefix='HTML attivo/sconosciuto: script', visits=1, audit='INCOMPLETE/UNSAFE_REFERENCE'),
         if_inert=dict(detail_prefix=None, visits=0, audit='COMPLETE'),
         static_v14=dict(detail=None, visits=0, audit='COMPLETE')),
    dict(id='H2-indented-heading-quoted-html',
         raw=' # H {k="<img src=../x.png>"}\n\nEnd.\n',
         tag='<img', premise='se " # H" e paragrafo, img nel valore e RawInline reale',
         if_active=dict(detail_prefix="Path evasivo: '../x.png'", visits=1, audit='INCOMPLETE/UNSAFE_REFERENCE'),
         if_inert=dict(detail_prefix=None, visits=0, audit='COMPLETE'),
         static_v14=dict(detail=None, visits=0, audit='COMPLETE')),
    # --- token di attributo accettati dal reader, non provati da AST (F61, C04) ---
    dict(id='G1-class-plus-quoted-html',
         raw='See [a]{.c+ k="<img src=../x.png>"} end.\n',
         tag='<img', premise='se Pandoc rifiuta la classe .c+ il gruppo non e Span e img e RawInline reale',
         if_active=dict(detail_prefix="Path evasivo: '../x.png'", visits=1, audit='INCOMPLETE/UNSAFE_REFERENCE'),
         if_inert=dict(detail_prefix=None, visits=0, audit='COMPLETE'),
         static_v14=dict(detail=None, visits=0, audit='COMPLETE')),
    dict(id='G2-class-underscore-quoted-html',
         raw='See [a]{._c k="<img src=../x.png>"} end.\n',
         tag='<img', premise='se Pandoc richiede lettera iniziale nella classe, ._c rifiuta il gruppo',
         if_active=dict(detail_prefix="Path evasivo: '../x.png'", visits=1, audit='INCOMPLETE/UNSAFE_REFERENCE'),
         if_inert=dict(detail_prefix=None, visits=0, audit='COMPLETE'),
         static_v14=dict(detail=None, visits=0, audit='COMPLETE')),
    dict(id='G3-id-plus-quoted-html',
         raw='See [a]{#a+b k="<img src=../x.png>"} end.\n',
         tag='<img', premise='se Pandoc rifiuta + nell id il gruppo non e Span',
         if_active=dict(detail_prefix="Path evasivo: '../x.png'", visits=1, audit='INCOMPLETE/UNSAFE_REFERENCE'),
         if_inert=dict(detail_prefix=None, visits=0, audit='COMPLETE'),
         static_v14=dict(detail=None, visits=0, audit='COMPLETE')),
    dict(id='G4-control-valid-class-quoted-html',
         raw='See [a]{.c k="<img src=../x.png>"} end.\n',
         tag='<img', premise='controllo: gruppo valido, valore quoted inerte',
         if_active=dict(detail_prefix="Path evasivo: '../x.png'", visits=1, audit='INCOMPLETE/UNSAFE_REFERENCE'),
         if_inert=dict(detail_prefix=None, visits=0, audit='COMPLETE'),
         static_v14=dict(detail=None, visits=0, audit='COMPLETE')),
]

if __name__ == '__main__':
    import json, hashlib
    print(json.dumps([dict(id=c['id'], raw_sha256=hashlib.sha256(c['raw'].encode()).hexdigest(), bytes=len(c['raw'].encode())) for c in CASES], indent=1))
