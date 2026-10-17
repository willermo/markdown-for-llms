"""Validate raster gallery candidates without importing Marker."""
from pathlib import Path
import json
import warnings

from ..files import open_file, read_json, write_json


def main(args):
    from PIL import Image
    config = read_json(Path('/scratch'), 'raster-request.json')
    Image.MAX_IMAGE_PIXELS = config['max_pixels']
    warnings.simplefilter('error', Image.DecompressionBombWarning)
    results = []
    for item in config['artifacts']:
        result = dict(path=item['path'], gallery=False, reason='Unsupported raster format')
        try:
            with open_file(Path('/scratch/original'), item['path']) as stream:
                with Image.open(stream, formats=['PNG', 'JPEG']) as image:
                    width, height = image.size
                    if width < 1 or height < 1 or max(width, height) > config['max_dimension'] or width * height > config['max_pixels']:
                        raise ValueError('Image pixel/dimension limit')
                    image.verify()
                stream.seek(0)
                with Image.open(stream, formats=['PNG', 'JPEG']) as image:
                    image.load()
                    result.update(gallery=True, reason=None, mime='image/png' if image.format == 'PNG' else 'image/jpeg', width=width, height=height)
        except Exception as exc:
            result['reason'] = 'Gallery rejected: ' + type(exc).__name__
        results.append(result)
    write_json(Path('/scratch'), 'raster-result.json', results)
    return 0
