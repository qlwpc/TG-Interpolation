"""Lazy CPU-only extension; compile once per source/Python ABI, before workers start."""

import functools
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sysconfig
import tempfile


@functools.lru_cache(None)
def native_builder(required=False):
    try:
        import fcntl
        import pybind11

        source = Path(__file__).with_name("tg_layout.cpp")
        identity = (
            source.read_bytes() + sysconfig.get_config_var("SOABI").encode() + pybind11.__version__.encode()
        )
        cache = (
            Path(tempfile.gettempdir()) / f"olmo-tg-{os.getuid()}" / hashlib.sha256(identity).hexdigest()[:20]
        )
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / ("_tg_layout_cpu" + sysconfig.get_config_var("EXT_SUFFIX"))
        with (cache / "build.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                temporary = target.with_suffix(".tmp.so")
                cmd = shlex.split(os.environ.get("CXX", "c++")) + [
                    "-O3",
                    "-shared",
                    "-std=c++17",
                    "-fPIC",
                    str(source),
                    "-I" + pybind11.get_include(),
                    "-I" + sysconfig.get_path("include"),
                    "-o",
                    str(temporary),
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                temporary.replace(target)
        spec = importlib.util.spec_from_file_location("_tg_layout_cpu", target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build
    except (ImportError, OSError, subprocess.SubprocessError) as exc:
        if required:
            raise RuntimeError(
                "Cannot build TG CPU extension (requires pybind11 and a C++17 compiler)"
            ) from exc
        logging.getLogger(__name__).warning(
            "TG CPU extension unavailable; using Python layout builder: %s", exc
        )
        return None
