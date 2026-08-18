#include "tensor.h"
#include <stdexcept>


Tensor::Tensor(std::vector<int> shape) : shape(std::move(shape)) {
    compute_strides();
    data = std::make_shared<std::vector<float>>(numel(), 0.0f);
}


Tensor::Tensor(std::vector<float> data, std::vector<int> shape) : shape(std::move(shape)) {
    compute_strides();
    if ((int)data.size() != numel()) {
        throw std::invalid_argument("data size does not match shape");
    }
    this->data = std::make_shared<std::vector<float>>(std::move(data));
}


int Tensor::numel() const {
    int result = 1;
    
    for (int dim: this->shape) {
        result *= dim;
    }

    return result;
}


void Tensor::compute_strides() {
    strides.resize(shape.size());

    int stride = 1;

    for (int i = static_cast<int>(shape.size()) -1; i>=0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}


int Tensor::flat_offset(const std::vector<int>& indices) const {
    // check that its shape is valid
    if (indices.size() != shape.size()) {
        throw std::invalid_argument("Number of indices does not match tensor dimensions");
    }

    int offset = 0;

    for (size_t i=0; i < indices.size(); i++) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::invalid_argument("Index out of range in tensor.");
        }
        offset += indices[i] * strides[i];
    }

    return offset;
}


float& Tensor::at(const std::vector<int>& indices) {
    int offset = flat_offset(indices);
    return (*data)[offset];
}


float Tensor::at(const std::vector<int>& indices) const {
    int offset = flat_offset(indices);
    return (*data)[offset];
}
