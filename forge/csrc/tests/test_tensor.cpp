// Unit tests for the C++ Tensor class.
//
// These cover the memory model and internals -- strides, index math, error
// paths. Behaviour that is visible from Python belongs in tests/tensor/
// instead, checked against NumPy as ground truth.

#include "third_party/doctest.h"

#include "tensor.h"

#include <stdexcept>
#include <vector>

TEST_CASE("numel is the product of the shape") {
    CHECK(Tensor(std::vector<int>{2, 3}).numel() == 6);
    CHECK(Tensor(std::vector<int>{4}).numel() == 4);
    CHECK(Tensor(std::vector<int>{2, 3, 4}).numel() == 24);

    SUBCASE("a scalar has an empty shape and one element") {
        // The product of no numbers is 1.
        CHECK(Tensor(std::vector<int>{}).numel() == 1);
    }
}

TEST_CASE("strides are row-major") {
    SUBCASE("2D") {
        Tensor a(std::vector<int>{2, 3});
        CHECK(a.strides == std::vector<int>{3, 1});
    }

    SUBCASE("3D") {
        Tensor a(std::vector<int>{2, 3, 4});
        CHECK(a.strides == std::vector<int>{12, 4, 1});
    }

    SUBCASE("1D has unit stride") {
        Tensor a(std::vector<int>{5});
        CHECK(a.strides == std::vector<int>{1});
    }

    SUBCASE("scalar has no strides") {
        Tensor a(std::vector<int>{});
        CHECK(a.strides.empty());
    }
}

TEST_CASE("shape constructor zero-fills") {
    Tensor a(std::vector<int>{2, 3});
    REQUIRE(a.data->size() == 6);
    for (float v : *a.data) {
        CHECK(v == 0.0f);
    }
}

TEST_CASE("data constructor rejects a size mismatch") {
    CHECK_THROWS_AS(Tensor(std::vector<float>{1, 2, 3}, std::vector<int>{2, 3}),
                    std::invalid_argument);
}

TEST_CASE("at maps indices onto the flat buffer") {
    // Values 0..5 laid out row-major:
    //     [[0, 1, 2],
    //      [3, 4, 5]]
    Tensor a(std::vector<float>{0, 1, 2, 3, 4, 5}, std::vector<int>{2, 3});

    SUBCASE("reads") {
        CHECK(a.at({0, 0}) == 0.0f);
        CHECK(a.at({0, 2}) == 2.0f);
        CHECK(a.at({1, 0}) == 3.0f);
        CHECK(a.at({1, 2}) == 5.0f);
    }

    SUBCASE("returns a writable reference into the buffer") {
        a.at({1, 1}) = 99.0f;
        CHECK(a.at({1, 1}) == 99.0f);
        // Confirm the write landed in the right flat slot, not just that
        // reading it back agrees with writing it.
        CHECK((*a.data)[4] == 99.0f);
    }

    SUBCASE("the const overload reads the same values") {
        const Tensor& c = a;
        CHECK(c.at({1, 2}) == 5.0f);
    }
}

TEST_CASE("at validates its indices") {
    Tensor a(std::vector<float>{0, 1, 2, 3, 4, 5}, std::vector<int>{2, 3});

    SUBCASE("wrong number of indices") {
        CHECK_THROWS_AS(a.at({0}), std::invalid_argument);
        CHECK_THROWS_AS(a.at({0, 0, 0}), std::invalid_argument);
    }

    SUBCASE("index out of range") {
        CHECK_THROWS_AS(a.at({2, 0}), std::invalid_argument);
        CHECK_THROWS_AS(a.at({0, 3}), std::invalid_argument);
    }

    SUBCASE("negative index") {
        // No Python-style negative indexing at this layer.
        CHECK_THROWS_AS(a.at({-1, 0}), std::invalid_argument);
    }
}

TEST_CASE("copies share storage") {
    // Consequence of holding data in a shared_ptr: copying a Tensor makes a
    // view, not a clone. This is what will make transpose and slicing free,
    // but it is a sharp edge until clone() exists.
    Tensor a(std::vector<float>{0, 1, 2, 3, 4, 5}, std::vector<int>{2, 3});
    Tensor b = a;
    b.at({0, 0}) = 42.0f;
    CHECK(a.at({0, 0}) == 42.0f);
}
