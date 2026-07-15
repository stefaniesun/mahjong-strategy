import pytest


torch = pytest.importorskip("torch")

from learning.device import resolve_device


def test_resolve_device_supports_cpu_and_auto():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_resolve_device_rejects_unknown_requests():
    with pytest.raises(ValueError, match="auto, cpu, or cuda"):
        resolve_device("metal")


def test_explicit_cuda_requires_an_available_device():
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(ValueError, match="CUDA was requested"):
            resolve_device("cuda")
