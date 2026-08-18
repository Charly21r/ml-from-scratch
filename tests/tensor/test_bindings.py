"""Tests for the compiled C++ extension module.

This layer tests behaviour visible from Python, checked against NumPy as
ground truth. Internals not exposed through the bindings (stride math, error
paths, invariants) are tested in ``forge/csrc/tests/`` instead.
"""

import pytest


def test_extension_imports():
    """The compiled extension is importable.

    Mostly a build smoke test: if CMake stops producing the module, or the
    editable install goes stale, this fails before anything more interesting
    gets a chance to.
    """
    from forge import _cpp

    assert _cpp.hello() == "forge_cpp loaded"


@pytest.mark.skip(reason="Tensor is not exposed through pybind11 yet")
def test_tensor_roundtrips_with_numpy():
    """Placeholder for now.

    ``forge_cpp.Tensor`` should round-trip with NumPy, and every op should
    match its NumPy equivalent within float32 tolerance.
    """
