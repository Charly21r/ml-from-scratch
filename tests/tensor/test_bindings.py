"""Tests for the compiled C++ extension module.

This layer tests behaviour visible from Python, checked against NumPy as
ground truth. Internals not exposed through the bindings (stride math, error
paths, invariants) are tested in ``forge/csrc/tests/`` instead.
"""

import numpy as np
import pytest

from forge import _cpp


def test_extension_imports():
    """The compiled extension is importable and exposes Tensor.

    Mostly a build smoke test: if CMake stops producing the module, or the
    editable install goes stale, this fails before anything more interesting
    gets a chance to.
    """
    assert hasattr(_cpp, "Tensor")


class TestTensorClass:
    """The py::class_ binding surface."""

    def test_construct_from_shape(self):
        t = _cpp.Tensor([2, 3])
        assert t.shape == [2, 3]
        assert t.strides == [3, 1]
        assert t.numel() == 6
        assert t.ndim() == 2

    def test_construct_from_data(self):
        t = _cpp.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3])
        assert t.numel() == 6
        assert t[[1, 0]] == 3.0

    def test_getitem_and_setitem(self):
        t = _cpp.Tensor([2, 3])
        t[[1, 2]] = 7.5
        assert t[[1, 2]] == 7.5

    def test_shape_is_read_only(self):
        """Assigning shape would leave strides stale, so it is not exposed."""
        t = _cpp.Tensor([2, 3])
        with pytest.raises(AttributeError):
            t.shape = [3, 2]

    def test_out_of_range_raises(self):
        t = _cpp.Tensor([2, 3])
        with pytest.raises(ValueError, match="Index out of range"):
            _ = t[[2, 0]]

    def test_repr(self):
        assert repr(_cpp.Tensor([2, 3])) == "Tensor(shape=[2, 3])"


class TestNumpyInterop:
    """Conversion in both directions, with numpy as the correctness oracle."""

    def test_from_numpy_preserves_shape_and_values(self):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        t = _cpp.from_numpy(a)
        assert t.shape == [2, 3]
        assert t.numel() == 6
        assert t[[1, 2]] == pytest.approx(5.0)

    def test_to_numpy_round_trips(self):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        out = _cpp.to_numpy(_cpp.from_numpy(a))
        assert out.dtype == np.float32
        assert out.shape == (2, 3, 4)
        np.testing.assert_allclose(out, a)

    def test_to_numpy_returns_a_copy(self):
        """The numpy array must not alias the Tensor's buffer."""
        t = _cpp.Tensor([0.0, 1.0, 2.0, 3.0], [2, 2])
        out = _cpp.to_numpy(t)
        out[0, 0] = 99.0
        assert t[[0, 0]] == 0.0

    def test_non_contiguous_input_is_handled(self):
        """A transposed view is not C-contiguous; forcecast copies it."""
        a = np.arange(6, dtype=np.float32).reshape(2, 3).T
        assert not a.flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(_cpp.to_numpy(_cpp.from_numpy(a)), a)

    def test_float64_input_is_cast(self):
        a = np.arange(6, dtype=np.float64).reshape(2, 3)
        np.testing.assert_allclose(_cpp.to_numpy(_cpp.from_numpy(a)), a)

    def test_scalar_round_trips(self):
        a = np.array(3.5, dtype=np.float32)
        t = _cpp.from_numpy(a)
        assert t.shape == []
        assert t.numel() == 1
        np.testing.assert_allclose(_cpp.to_numpy(t), a)
