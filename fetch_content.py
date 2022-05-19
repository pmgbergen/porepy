import importlib
import subprocess
from pathlib import Path

import requests


def fetch_polyhedron_file():

    # Makes http request for external file
    file_request = requests.get(
        "https://raw.githubusercontent.com/mdickinson/polyhedron/master/polyhedron.py"
    )

    try:
        importlib.import_module("porepy")
    except ImportError:
        raise ImportError(
            """Cannot import porepy.
                  Install it via 'pip install .' or 'pip install --user -e .' commands
                  Read documentation for further instructions.
                  """
        )
    finally:
        # Fetches local porepy installation directory (platform-independent via subprocess)
        command = "pip show porepy"
        # To understand the logic in the lines below, run the command and have a look at
        # the output.
        raw_ouput = subprocess.getoutput(command)
        # Drops 'Location: '
        begin = raw_ouput.find("Location:") + 10
        # Drops end of line and also '/src', since we want the file located in the top
        # directory of PorePy (this subtracting 1 + 4 characters).
        end = raw_ouput.find("Requires:") - 5
        # Extracts porepy location
        site_packages_directory = raw_ouput[begin:end]
        print("Writing robust_point_in_polyhedron.py in :", site_packages_directory)
        print(
            f"If {site_packages_directory} is not in your PYTHONPATH, please move the file or append the PYTHONPATH."
        )

        # Saves file data to local copy (platform-independent via pathlib)
        with open(
            Path(site_packages_directory) / Path("robust_point_in_polyhedron.py"), "wb"
        ) as file:
            file.write(file_request.content)


# Fetches external file
fetch_polyhedron_file()
