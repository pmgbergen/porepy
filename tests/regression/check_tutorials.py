import glob
import os
import sys

import pytest

TUTORIAL_FILENAMES = glob.glob("tutorials/*.ipynb")


@pytest.mark.parametrize("tutorial_path", TUTORIAL_FILENAMES)
def test_run_tutorials(tutorial_path: str):
    """We run the tutorial and check that it didn't raise any error.
    This assumes we run pytest from the porepy directory.

    Notice: the current file is now called "check_tutorials.py". Its name doesn't start
    from "test_", so it won't be automatically run by the command "pytest ."

    """
    new_file = tutorial_path[:-6] + ".py"

    # This command might fail in github actions.
    cmd_convert = "jupyter nbconvert --to script " + tutorial_path
    os.system(cmd_convert)
    remove_plots(new_file)

    cmd_run = "python " + str(new_file)
    status = os.system(cmd_run)

    assert status == 0

    # Removing the generated source file after the assertion. If the test fails, it is
    # useful to keep it in order to see what went wrong there.
    os.remove(new_file)


def remove_plots(filename: str):
    """Matplotlib opens a new windows for each figure in the interactive mode.
    Here, we prevent it by setting the noninteractive matplotlib backend.
    "template" backend is a dummy backend that does nothing.

    """
    with open(filename) as f:
        content = f.readlines()

    with open(filename, "w") as f:
        f.write("import matplotlib; matplotlib.use('template')\n")
        f.writelines(content)


if __name__ == "__main__":
    try:
        filenames = [sys.argv[1]]
    except IndexError:
        filenames = TUTORIAL_FILENAMES
    for tutorial_path in filenames:
        test_run_tutorials(tutorial_path=tutorial_path)
