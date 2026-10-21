"""Report dlopen's actual error, including errors ld on the shared library cannot explain."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import dace
from dace.codegen.compiled_sdfg import ReloadableDLL


@pytest.fixture
def compiled_stub(tmp_path):
    stub = tmp_path / "stub.so"
    source = Path(dace.__file__).parent / "codegen/tools/dacestub.cpp"
    subprocess.run([os.environ.get("CXX", "g++"), "-shared", "-fPIC", "-fopenmp", str(source),
                    "-ldl", "-o", str(stub)], check=True)
    return stub


@pytest.mark.skipif(sys.platform != "linux", reason="Linux loader diagnostics and GCC stub build")
def test_missing_library_reports_the_dynamic_loader_error(tmp_path, compiled_stub):
    missing = tmp_path / "missing_kernel.so"
    dll = ReloadableDLL(str(missing), libstub_path=str(compiled_stub))
    with pytest.raises(RuntimeError, match="missing_kernel.so.*cannot open shared object file"):
        dll.load()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux loader diagnostics and GCC stub build")
def test_existing_library_with_an_unresolved_symbol_reports_that_symbol(tmp_path, compiled_stub):
    source = tmp_path / "kernel.cpp"
    source.write_text('extern "C" void absent_dependency();\nextern "C" void kernel() { absent_dependency(); }\n')
    library = tmp_path / "kernel.so"
    subprocess.run([os.environ.get("CXX", "g++"), "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    dll = ReloadableDLL(str(library), libstub_path=str(compiled_stub))
    with pytest.raises(RuntimeError, match="undefined symbol: absent_dependency"):
        dll.load()
