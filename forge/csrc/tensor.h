#pragma once
#include <vector>
#include <memory>

class Tensor {
public:
    // The three things a tensor is
    std::shared_ptr<std::vector<float>> data;
    std::vector<int> shape;
    std::vector<int> strides;

    // Construct from shape (allocates storage, zeros out)
    explicit Tensor(std::vector<int> shape);

    // Construct from existing data + shape (you compute strides)
    Tensor(std::vector<float> data, std::vector<int> shape);

    // Element access using the stride formula
    float& at(std::vector<int> indices);
    float  at(std::vector<int> indices) const;

    int  ndim()    const { return shape.size(); }
    int  numel()   const; // product of shape dims

private:
    void compute_strides(); // row-major: strides[i] = product(shape[i+1:])
};
