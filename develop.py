#! python
import subprocess
import os
import sys
import shutil

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    subprocess.check_call(
        ["python", "./setup_pybind.py", "develop", "--build-type", ("Debug" if len(sys.argv) > 1 and sys.argv[1] == "debug" else "Release" )], cwd=PROJECT_DIR
    )
    shutil.copytree(os.path.join(PROJECT_DIR, "_skbuild/linux-x86_64-3.9/cmake-install/"),
                    PROJECT_DIR, symlinks=False, dirs_exist_ok=True)

    subprocess.check_call(
        ["python", "-m", "mypy.stubgen", "--include-private", "-o", PROJECT_DIR, "-m", "dedup_mod.cmp_func",
         "-m", "dedup_mod.util.comparator", "-m", "dedup_mod.method.method"], cwd=PROJECT_DIR
    )
