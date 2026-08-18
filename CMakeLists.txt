cmake_minimum_required(VERSION 3.18)
project(forge_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sources making up the tensor library, shared by the Python extension and
# the C++ test binary.
set(FORGE_CORE_SOURCES
    forge/csrc/tensor.cpp
)

# --- Python extension module ------------------------------------------------

find_package(pybind11 REQUIRED)

pybind11_add_module(_cpp
    forge/csrc/bindings.cpp
    ${FORGE_CORE_SOURCES}
)

target_include_directories(_cpp PRIVATE forge/csrc/)
target_compile_options(_cpp PRIVATE -Wall -Wextra)

set_target_properties(_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/forge"
)

# scikit-build-core needs an install rule to place the module in the wheel.
install(TARGETS _cpp DESTINATION forge)

# --- C++ tests --------------------------------------------------------------
#
# Skipped during wheel builds: scikit-build-core only wants the extension, and
# building tests there would slow every `pip install` down.
option(FORGE_BUILD_TESTS "Build the C++ unit tests" ON)

if(FORGE_BUILD_TESTS AND NOT SKBUILD)
    enable_testing()

    add_executable(test_cpp
        forge/csrc/tests/doctest_main.cpp
        forge/csrc/tests/test_tensor.cpp
        ${FORGE_CORE_SOURCES}
    )

    target_include_directories(test_cpp PRIVATE forge/csrc/ forge/csrc/tests/)
    target_compile_options(test_cpp PRIVATE -Wall -Wextra)

    # AddressSanitizer/UBSan on the test binary catches memory errors that
    # would otherwise be silent corruption. Debug builds only.
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(test_cpp PRIVATE -fsanitize=address,undefined)
        target_link_options(test_cpp PRIVATE -fsanitize=address,undefined)
    endif()

    add_test(NAME cpp_tensor COMMAND test_cpp)
endif()
