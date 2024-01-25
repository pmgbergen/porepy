"""Set-up file for PorePy for installations usins ``pip install .``"""

# import os
# from glob import glob
from setuptools import find_packages, setup

# Defining porepy source packages and a map
# (subpackage) -> directory (relative to setup.py)
src_packages = find_packages("src")
package_dir = dict([(p, f'src/{p.replace(".", "/")}') for p in src_packages])

# Adding the examples as a subpackage and adding its directory (not found in src)
packages = src_packages + ["porepy.examples"]
package_dir["porepy.examples"] = "examples"


with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="porepy",
    url="https://github.com/pmgbergen/porepy",
    version="1.8.0",
    license="GPL",
    keywords=["porous media simulation fractures deformable"],
    install_requires=required,
    description="Simulation tool for fractured and deformable porous media",
    maintainer="Eirik Keilegavlen",
    maintainer_email="Eirik.Keilegavlen@uib.no",
    platforms=["Linux", "Windows", "Mac OS-X"],
    package_data={
        "porepy": [
            "py.typed",
            "applications/md_grids/gmsh_file_library/**/*.csv",
            "applications/md_grids/gmsh_file_library/**/*.geo",
        ],
    },
    packages=packages,
    package_dir=package_dir,
    # NOTE this is left for future reference: Relevant when unpackaged py-files are
    # added to src/ (free-standing modules)
    # py_modules=[
    #     os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")
    # ],
    zip_safe=False,
)
