# code adapted from https://github.com/rgl-epfl/cholespy/blob/main/setup.py

import sys

try:
    from skbuild import setup
    import nanobind
except ImportError:
    print("The preferred way to invoke 'setup.py' is via pip, as in 'pip "
          "install .'. If you wish to run the setup script directly, you must "
          "first install the build dependencies listed in pyproject.toml!",
          file=sys.stderr)
    raise

setup(
    name="miwireframe",
    version="1.0.0",
    description="Utility functions for wireframe rendering.",
    author="Kenji Tojo",
    license="MIT",
    packages=["miwireframe"],
    package_dir={"": "src"},
    cmake_install_dir="src/miwireframe",
    include_package_data=True,
    python_requires=">=3.8"
)