import importlib.util
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_api_reference_is_up_to_date():
    spec = importlib.util.spec_from_file_location("gen_api", DOCS / "gen_api.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert (DOCS / "api.md").read_text() == gen.generate(), "run: python docs/gen_api.py"


def test_every_public_object_is_documented():
    import inspect

    import driftfdr

    missing = [name for name in driftfdr.__all__ if not inspect.getdoc(getattr(driftfdr, name))]
    assert not missing, missing
