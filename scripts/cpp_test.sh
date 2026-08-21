#!/usr/bin/env bash
# Fast inner-loop C++ tests: compiles straight with g++, no CMake.
#
# Use this while iterating. For the full picture (and what CI runs), use:
#     cmake -S . -B build/tests -DCMAKE_BUILD_TYPE=Debug
#     cmake --build build/tests && ctest --test-dir build/tests --output-on-failure
#
# Any arguments are forwarded to the test binary, e.g.:
#     ./scripts/cpp_test.sh --test-case="*strides*"
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p build

g++ -std=c++17 -g -Wall -Wextra -fsanitize=address,undefined \
    -I forge/csrc -I forge/csrc/tests \
    -o build/test_cpp \
    forge/csrc/tests/doctest_main.cpp \
    forge/csrc/tests/test_tensor.cpp \
    forge/csrc/tensor.cpp

./build/test_cpp "$@"
