cmake_minimum_required(VERSION 3.18)
project(forge_cpp)

find_package(pybind11 REQUIRED)

pybind11_add_module(_cpp
    forge/csrc/bindings.cpp
)

target_include_directories(_cpp PRIVATE forge/csrc/)

set_target_properties(_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/forge"
)