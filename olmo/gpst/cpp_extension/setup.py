"""Build the C++ chart-table backend (`cppbackend`) for GPST.

This is a CPU-only extension (no CUDA ops — all GPU work lives in PyTorch).
Build with::

    python olmo/gpst/cpp_extension/setup.py build_ext --inplace

which produces ``cppbackend.cpython-<ver>-<arch>.so`` next to this file.
The resulting module is imported by ``olmo.gpst.data_structure.py_backend``.

Ported verbatim from ant-research/StructuredLM_RTDT setup.py.
"""
import glob
import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, library_paths

include_dirs = os.path.dirname(os.path.abspath(__file__))
source_files = glob.glob(os.path.join(include_dirs, "*.cpp"))
torch_library_dirs = library_paths()

setup(
    name="cppbackend",
    ext_modules=[
        CppExtension(
            "cppbackend",
            sources=source_files,
            include_dirs=[include_dirs],
            # CppExtension supplies link-time library paths but does not embed
            # them. Without RUNPATH, importing cppbackend before torch fails to
            # find libc10.so/libtorch*.so.
            runtime_library_dirs=torch_library_dirs if sys.platform.startswith("linux") else [],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
