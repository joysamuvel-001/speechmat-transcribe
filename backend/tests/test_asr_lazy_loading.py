import importlib
import sys


def test_asr_model_is_not_loaded_on_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", None)

    sys.modules.pop("transcription.asr", None)
    module = importlib.import_module("transcription.asr")

    assert module._asr_model is None

    module._load_model()
