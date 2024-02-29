"""Set-up file for PorePy for installations usins ``pip install .``"""
from setuptools import find_packages, setup


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
    packages=find_packages("src"),
    package_dir={"": "src"},
    zip_safe=False,
)
