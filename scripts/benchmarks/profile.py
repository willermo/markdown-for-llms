"""Profilo writer chiuso E05; bitset completo Pandoc Markdown 3.1.3."""
from pathlib import Path
from data import descriptor, digest

ENABLED=frozenset('all_symbols_escapable backtick_code_blocks blank_before_blockquote blank_before_header bracketed_spans escaped_line_breaks fenced_code_attributes fenced_code_blocks fenced_divs footnotes header_attributes implicit_figures inline_code_attributes intraword_underscores link_attributes pipe_tables raw_attribute raw_html raw_tex space_in_atx_header table_captions tex_math_dollars'.split())


def config(format,engine):
    path=Path(__file__).with_name('pandoc-markdown-extensions-3.1.3.txt')
    inventory=[line[1:] for line in path.read_text().splitlines()]
    if len(inventory)!=len(set(inventory)) or not ENABLED.issubset(inventory) or len(ENABLED)!=22:raise ValueError('Extension inventory invalid')
    writer='markdown'+''.join(('+' if n in ENABLED else '-')+n for n in sorted(inventory))
    options=['--from='+{'tex':'latex','docx':'docx','epub':'epub','rtf':'rtf','txt':'markdown'}.get(format,format),'--to='+writer,'--wrap=none','--markdown-headings=atx','--data-dir=data','--resource-path=source','--extract-media=media']
    files=[Path(__file__).with_name(name) for name in ('profile.py','safety.py','observations.py','math_structure.py','adapters.py','legacy_child.py','literal_child.py','audit.py','metrics.py','benchmark.py','resources.py','pandoc_guard.py')]
    result={'schema_version':1,'profile':'pandoc-md-bench-v1','protocol':'conversion-fidelity-v1','engine':engine,'input_format':format,'pandoc':descriptor('/usr/bin/pandoc'),'pandoc_version_required':'3.1.3','extension_inventory':descriptor(path),'enabled':sorted(ENABLED),'disabled':sorted(set(inventory)-ENABLED),'writer':writer,'ordered_options':options,'helpers':[descriptor(p) for p in files],'safety':{'network':'R required; no fetch','tex_compile':False,'include_allowlist':['sections/one.tex','sections/two.tex'],'file_bytes':33554432,'stream_bytes':1048576,'engine_timeout_s':60,'sampled_tree_rss_stop_bytes':6442450944},'sandbox_flag':'omitted; R no-network and source validation; no universal filesystem guarantee'}
    result['config_sha256']=digest(result);return result
