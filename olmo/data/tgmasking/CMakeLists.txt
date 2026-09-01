cmake_minimum_required(VERSION 3.15...3.29)
project(tg_mask LANGUAGES CXX)

# Use the active Conda Python instead of hard-coding an environment path and
# CPython minor version.  Python3_EXECUTABLE can also be passed explicitly.
if(NOT Python3_EXECUTABLE AND DEFINED ENV{CONDA_PREFIX})
    set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
endif()
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pathlib, torch; print(pathlib.Path(torch.__file__).parent)"
    RESULT_VARIABLE TORCH_DISCOVERY_STATUS
    OUTPUT_VARIABLE Torch_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import torch; print(int(torch.compiled_with_cxx11_abi()))"
    RESULT_VARIABLE TORCH_ABI_STATUS
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    RESULT_VARIABLE PYBIND11_DISCOVERY_STATUS
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT TORCH_DISCOVERY_STATUS EQUAL 0 OR NOT TORCH_ABI_STATUS EQUAL 0)
    message(FATAL_ERROR
        "Could not import torch with ${Python3_EXECUTABLE}; activate the target "
        "Conda environment or pass -DPython3_EXECUTABLE=/path/to/python")
endif()
if(NOT PYBIND11_DISCOVERY_STATUS EQUAL 0)
    message(FATAL_ERROR "Could not import pybind11 with ${Python3_EXECUTABLE}")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKE_DIR}" NO_DEFAULT_PATH)

# mask.cpp is a CPU-only extension (ATen tensor ops only, no CUDA kernels).
# Link the torch libraries directly instead of find_package(Torch): this torch
# wheel is CUDA-enabled, so find_package(Torch) would otherwise require a CUDA
# toolkit (nvcc) that is not present on this machine.
set(TORCH_INCLUDE
    "${Torch_DIR}/include"
    "${Torch_DIR}/include/torch/csrc/api/include")
set(TORCH_LIB_DIR "${Torch_DIR}/lib")
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_LIB_DIR}")
find_library(TORCH_LIBRARY       torch       PATHS "${TORCH_LIB_DIR}")
find_library(TORCH_CPU_LIBRARY   torch_cpu   PATHS "${TORCH_LIB_DIR}")
find_library(C10_LIBRARY         c10         PATHS "${TORCH_LIB_DIR}")

foreach(TORCH_COMPONENT
        TORCH_PYTHON_LIBRARY TORCH_LIBRARY TORCH_CPU_LIBRARY C10_LIBRARY)
    if(NOT ${TORCH_COMPONENT})
        message(FATAL_ERROR "Missing ${TORCH_COMPONENT} under ${TORCH_LIB_DIR}")
    endif()
endforeach()

# Let the built module find libtorch*.so at runtime without LD_LIBRARY_PATH.
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${TORCH_LIB_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

pybind11_add_module(tg_mask mask.cpp)
target_include_directories(tg_mask PRIVATE ${TORCH_INCLUDE})
target_compile_definitions(
    tg_mask PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_link_directories(tg_mask PRIVATE ${TORCH_LIB_DIR})
target_link_libraries(tg_mask PRIVATE
    ${TORCH_LIBRARY}
    ${TORCH_CPU_LIBRARY}
    ${C10_LIBRARY}
    ${TORCH_PYTHON_LIBRARY}
    pybind11::module)

set_target_properties(tg_mask PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
