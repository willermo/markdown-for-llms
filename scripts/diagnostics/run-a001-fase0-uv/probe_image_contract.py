#!/usr/bin/env python3
"""Diagnostica V8 fidata su stdin: nessun startup ordinario/peso/font."""
import asyncio
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import os
import json
from pathlib import Path
import socket
import sys
import tempfile
from unittest.mock import patch

# Il driver imposta EXPECTED con JSON host fidato, prima di questo sorgente.

def main(expected):
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError('Richiesto -I -B')
    ns=os.readlink('/proc/self/ns/net')
    if ns==expected['host_netns'] or os.listdir('/sys/class/net')!=['lo']:
        raise ValueError('Runtime non isolato dal namespace host')
    daemon_paths=['/run/docker.sock','/var/run/docker.sock','/run/systemd/private','/run/dbus/system_bus_socket','/run/user/1000/systemd/private']
    if any(Path(x).exists() for x in daemon_paths):raise ValueError('Socket host visibile')
    left,right=socket.socketpair();left.close();right.close()
    with tempfile.TemporaryDirectory() as scratch:
        local=socket.socket(socket.AF_UNIX);local.bind(str(Path(scratch)/'synthetic.sock'));local.close()
    isolation={'status':'PASS','netns':ns,'host_netns':expected['host_netns'],'interfaces':['lo'],
               'host_daemons_absent':daemon_paths,'AF_UNIX_positive':True}
    distribution=importlib.metadata.distribution('markdown-for-llms')
    origins={}
    records={r[0]:r for r in csv.reader(io.StringIO(distribution.read_text('RECORD')))}
    for item in expected['modules']:
        p=Path(distribution.locate_file(item['name']));b=p.read_bytes()
        if not p.is_relative_to('/opt/venv') or len(b)!=item['bytes'] or hashlib.sha256(b).hexdigest()!=item['sha256']:
            raise ValueError('Modulo immagine diverso da S: '+item['name'])
        r=records[item['name']]
        if r[1]!='sha256='+base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b'=').decode() or int(r[2])!=len(b):
            raise ValueError('RECORD installato difforme')
        spec=importlib.util.find_spec(item['name'][:-3])
        if spec.origin!=str(p):raise ValueError('Spec difforme')
        origins[item['name']]=str(p)
    attempts=[]
    def deny(*args,**kwargs):
        attempts.append(repr(args[1:])[:200]);raise RuntimeError('Network denied by V8')
    with patch.object(socket.socket,'connect',deny),patch.object(socket.socket,'connect_ex',deny),patch.object(socket.socket,'sendto',deny):
        import torch,marker,surya,bs4,filetype,weasyprint
        import marker.converters as converters
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.renderers.markdown import MarkdownOutput
        from marker.providers.registry import provider_from_ext,provider_from_filepath
        from marker.settings import settings as marker_settings
        from surya.settings import settings as surya_settings
        import marker_api_server as server
        from fastapi import UploadFile,HTTPException
        packages={d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()}
        if packages['marker-pdf']!='1.10.2' or packages['surya-ocr']!='0.17.1' or not torch.__version__.startswith('2.7.1'):
            raise ValueError('Versioni motore diverse dai candidati')
        if torch.version.cuda is not None or any(n.startswith('nvidia-') for n in packages):raise ValueError('Profilo CPU contiene CUDA')
        if any(n in packages for n in ['pytest','pytest-cov','pytest-mock','black','flake8','mypy']):raise ValueError('Dev nel runtime')
        inspect.signature(PdfConverter).bind(artifact_dict={},processor_list=None,renderer=None)
        if not callable(create_model_dict):raise ValueError('Contratto models mancante')
        expected_providers={'pdf':'PdfProvider','epub':'EpubProvider','docx':'DocumentProvider','pptx':'PowerPointProvider',
                            'xlsx':'SpreadSheetProvider','html':'HTMLProvider','png':'ImageProvider'}
        providers={ext:provider_from_ext('synthetic.'+ext).__name__ for ext in expected_providers}
        if providers!=expected_providers:raise ValueError('Registry full difforme')
        with tempfile.TemporaryDirectory() as scratch:
            h=Path(scratch)/'x.html';h.write_text('<html><body>Test sintetico</body></html>')
            if provider_from_filepath(str(h)).__name__!='HTMLProvider':raise ValueError('Provider HTML reale mancante')
            pdf=weasyprint.HTML(string='<html><body><p>Città 123,45 E = mc²</p></body></html>').write_pdf()
            if not pdf.startswith(b'%PDF-') or len(pdf)<500:raise ValueError('Rendering native incompleto')
            output=MarkdownOutput(markdown='# Risultato\n\n123,45 $E=mc^2$ [1]',images={},metadata={})
            def fake(path):
                if Path(path).read_bytes()!=b'synthetic input':raise ValueError('Input wrapper difforme')
                return output
            fake.page_count=2
            async def invoke():
                upload=UploadFile(filename='x.pdf',file=io.BytesIO(b'synthetic input'))
                with patch.object(server,'get_converter',return_value=fake):return await server.convert_document(upload)
            response=asyncio.run(invoke());body=json.loads(response.body)
            if body['markdown']!=output.markdown or body['page_count']!=2 or body['images']!={}:raise ValueError('Contratto wrapper difforme')
            async def failure():
                def fail(path):raise RuntimeError('synthetic failure')
                with patch.object(server,'get_converter',return_value=fail):
                    try:await server.convert_document(UploadFile(filename='bad.pdf',file=io.BytesIO(b'bad')))
                    except HTTPException as error:return error.status_code
                return 0
            if asyncio.run(failure())!=500:raise ValueError('Errore wrapper non propagato')
        # Tutte le dipendenze dei processor sostituite prima del costruttore.
        resolved=[]
        def resolve(self,cls):resolved.append(cls.__name__);return object()
        with patch.object(converters,'download_font') as font,patch.object(PdfConverter,'resolve_dependencies',resolve):
            constructed=PdfConverter(artifact_dict={},processor_list=None,renderer=None)
            if font.call_count!=1 or not resolved or not constructed.processor_list:raise ValueError('Constructor mock incompleto')
        if attempts:raise ValueError('Tentativi di rete rilevati')
        if (marker_settings.FONT_PATH!='/data/fonts/GoNotoCurrent-Regular.ttf' or surya_settings.MODEL_CACHE_DIR!='/data/cache/models' or
                any(not str(p).startswith('/data/fonts/') for p in surya_settings.RECOGNITION_RENDER_FONTS.values())):
            raise ValueError('Cache/font diversi dai path configurati')
        return {'schema':1,'status':'PASS_OFFLINE_CONTRACT_ONLY','S_id':expected['S_id'],'S_sha256':expected['S_sha256'],
                'origins':origins,'versions':packages,'R_D4':isolation,'torch_cuda':torch.version.cuda,'providers':providers,
                'native_pdf_bytes':len(pdf),'wrapper_content':body,'constructor_resolved':resolved,
                'patches':['socket.connect/connect_ex/sendto','BaseConverter.download_font','PdfConverter.resolve_dependencies','server.get_converter'],
                'network_attempts':attempts,'subcases':['installed_hashes_RECORD','imports_versions_CPU','signature','providers_full',
                                                      'weasyprint_native','MarkdownOutput_wrapper','wrapper_error','constructor_mock','cache_font_paths'],
                'limits':'Nessuna inferenza, pesi/font, startup ordinario o prova GPU'}


if __name__=='__main__':
    try:print(json.dumps(main(EXPECTED),sort_keys=True))
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error':type(exc).__name__+': '+str(exc)}));raise SystemExit(2)
