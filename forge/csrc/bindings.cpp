// pybind11 bindings exposing the C++ Tensor to Python as forge._cpp.
//
// The module name here (_cpp) must match the target name in CMakeLists.txt.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // py::array_t, buffer_info
#include <pybind11/stl.h>    // std::vector <-> Python list, automatically

#include "tensor.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

// Copy a numpy array into a new Tensor.
//
// `c_style | forcecast` makes pybind11 hand over a C-contiguous float32 array,
// converting (and therefore copying) if the caller passed something else: a
// transposed view, a float64 array. Tensor cannot yet describe a non-contiguous
// layout -- it has no offset field -- so converting up front keeps this correct
// at the cost of a silent copy.
static Tensor from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info info = arr.request();
    std::vector<int> shape(info.shape.begin(), info.shape.end());
    const float* ptr = static_cast<float*>(info.ptr);
    std::vector<float> data(ptr, ptr + info.size);

    return Tensor(data, shape);
}

// Copy a Tensor into a new numpy array.
//
// Tensor strides count elements; numpy strides count bytes, hence the scaling
// below. Returning a copy keeps ownership simple -- the array never aliases the
// Tensor's buffer, so neither can outlive the other. A zero-copy view would
// need py::buffer_protocol plus a lifetime guarantee.
static py::array_t<float> to_numpy(const Tensor& t) {
    std::vector<py::ssize_t> shape(t.shape.begin(), t.shape.end());
    std::vector<py::ssize_t> strides;
    strides.reserve(t.strides.size());

    for (int s: t.strides) {
        strides.push_back(static_cast<py::ssize_t>(s) * sizeof(float));
    }

    py::array_t<float> out(shape, strides);

    const std::vector<float>& data = *t.data;
    std::copy(data.begin(), data.end(), out.mutable_data());

    return out;
}

PYBIND11_MODULE(_cpp, m) {
    m.doc() = "Forge C++ tensor core.";

    // --- Tensor class ------------------------------------------------------
    py::class_<Tensor>(m, "Tensor")
        // Two constructors. pybind11 picks between them by argument count.
        .def(py::init<std::vector<int>>(), py::arg("shape"))
        .def(py::init<std::vector<float>, std::vector<int>>(),
             py::arg("data"), py::arg("shape"))

        // read-only, NOT read-write: letting Python assign shape directly
        // would leave strides stale and silently corrupt the tensor.
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)

        .def("numel", &Tensor::numel)
        .def("ndim", &Tensor::ndim)

        // Element access. Taking `const Tensor&` here selects the const
        // overload of at(); the setter takes `Tensor&` and uses the reference
        // returned by the non-const overload.
        .def("__getitem__",
             [](const Tensor& t, std::vector<int> idx) { return t.at(idx); })
        .def("__setitem__",
             [](Tensor& t, std::vector<int> idx, float value) { t.at(idx) = value; })

        .def("__repr__", [](const Tensor& t) {
            std::ostringstream os;
            os << "Tensor(shape=[";
            for (size_t i = 0; i < t.shape.size(); ++i) {
                os << (i ? ", " : "") << t.shape[i];
            }
            os << "])";
            return os.str();
        });

    // --- numpy interop -----------------------------------------------------
    m.def("from_numpy", &from_numpy, py::arg("array"),
          "Copy a numpy array into a new Tensor.");
    m.def("to_numpy", &to_numpy, py::arg("tensor"),
          "Copy a Tensor into a new numpy array.");
}
