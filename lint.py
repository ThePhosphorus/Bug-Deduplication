#! python
import subprocess
import os

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    subprocess.check_call(
        ["python", "-m", "mypy"], cwd=PROJECT_DIR
    )
