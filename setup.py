import os.path
from glob import glob

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="porepy",
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
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")
    ],
    zip_safe=False,
)
