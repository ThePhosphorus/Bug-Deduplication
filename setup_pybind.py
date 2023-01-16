import os
import shutil
from skbuild import setup
from setuptools import find_packages


# Setup dedup_mod Folder
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
DEDUP_MOD_FOLDER_PATH = os.path.join(PROJECT_DIR, "dedup_mod")
SRC_DEDUP_MOD_FOLDER_PATH = os.path.join(PROJECT_DIR, "dedup_mod_pybind")
if os.path.exists(DEDUP_MOD_FOLDER_PATH):
    shutil.rmtree(DEDUP_MOD_FOLDER_PATH)

shutil.copytree(SRC_DEDUP_MOD_FOLDER_PATH,
                DEDUP_MOD_FOLDER_PATH, symlinks=False)


setup(name='dedup_mod_pybind', packages=find_packages(include=["dedup_mod",
      "data", "experiments", "models", "trainers"]), cmake_install_dir="dedup_mod/")
