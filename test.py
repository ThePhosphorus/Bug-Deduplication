#! python
import subprocess
import os


PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
DEDUP_MOD_TEST_DIR = os.path.join(PROJECT_DIR, "dedup_mod/tests")

if __name__ == "__main__":
    subprocess.check_call(
        ["python", "-m", "pytest", "tests/", "--junitxml=./out/tests/pytest.xml"], cwd=PROJECT_DIR
    )

    # Find all cpp tests
    tests = [each for each in os.listdir(
        DEDUP_MOD_TEST_DIR) if each.startswith("test_")]


    os.environ['PYTHONPATH'] = 'C:\\Users\\Adem Aber Aouni\\.conda\\envs\\bug_dedup_pybind\\python.exe'
    for test in tests:
        print(test)
        subprocess.check_call(
            [os.path.join(DEDUP_MOD_TEST_DIR, test)], cwd=PROJECT_DIR)
