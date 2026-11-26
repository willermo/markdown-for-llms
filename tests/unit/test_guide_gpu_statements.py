"""Statements of the guide and of `.env.example` that the tranche B of MFL023 made false or left out (MFL023 B10, C-DB-NB1
of the review of the diff B): the route marker-server runs in the archive on the GPU presets of the instance; the floor
of readiness of `mdf convert` is the same on RunPod and on Modal (`engine.servers.DEFAULTS`); `runpod_endpoint.py
delete` removes the endpoint and, by the public REST v2, its template; the worker reads the keys of the GPU sessions by name from
the env file of the user, a private file (since MFL051 the one `.env`, mounted in the containers)."""
from pathlib import Path

from markdown_for_llms.engine import servers

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "how-to" / "convert-documents.md"
ENV_EXAMPLE = ROOT / ".env.example"


def one_line(path):
    return " ".join(path.read_text().split())


def test_the_route_in_the_archive():
    text = one_line(GUIDE)
    assert "the archive of the product refuses it (`route_not_in_product`" not in text
    assert "the archive of the product refuses it with `route_not_in_product` in this version" not in text
    assert "only without a `gpu_preset`" in text


def test_the_floor_of_readiness_on_both_providers():
    floor = servers.DEFAULTS["runpod"]["ready_within"]
    assert servers.DEFAULTS["modal"]["ready_within"] == floor
    text = one_line(GUIDE)
    assert "1200 s on Modal" not in text
    assert "%d s on RunPod and on Modal by default" % floor in text


def test_the_delete_of_an_endpoint_and_its_template():
    text = one_line(GUIDE)
    assert "`templateId`" in text
    assert "`template: forbidden`" in text


def test_the_keys_of_the_gpu_sessions_in_env_example():
    text = one_line(ENV_EXAMPLE)
    # MFL051: one env file, mounted in the containers; the statement keeps its three parts (by name, from the file
    # and never an environment, a private file that is not a link)
    assert "The worker and the `mdf` tool read a key by name from this file each time they use it" in text
    assert "no key is ever in the environment of a container" in text
    assert "`make up` gives this file mode 0600; a link, or a file of another user, is refused" in text
