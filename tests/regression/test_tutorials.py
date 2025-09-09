import os
import sys
from pathlib import Path

import pytest

TUTORIAL_FILENAMES = [fn.resolve() for fn in Path("tutorials").glob("*.ipynb")]


@pytest.mark.tutorials
@pytest.mark.parametrize("tutorial_path", TUTORIAL_FILENAMES)
def test_run_tutorials(tutorial_path: Path):
    """We run the tutorial and check that it didn't raise any error.
    This assumes we run pytest from the porepy directory.

    """
    new_file = tutorial_path.with_suffix(".py")

    # This command might fail in github actions.
    cmd_convert = "jupyter-nbconvert --to script " + str(tutorial_path)
    status = os.system(cmd_convert)
    if status != 0:
        raise RuntimeError(
            ".ipynb file is not converted to .py file. Is jupyter-nbconvert available?"
        )
    edit_imports(new_file)

    cmd_run = "python " + str(new_file)
    status = os.system(cmd_run)

    assert status == 0

    # Removing the generated source file after the assertion. If the test fails, it is
    # useful to keep it in order to see what went wrong there.
    new_file.unlink()


def edit_imports(filename: Path):
    """Matplotlib opens a new window for each figure in the interactive mode.
    Here, we prevent it by setting the noninteractive matplotlib backend.
    "template" backend is a dummy backend that does nothing.
    Also, we cd to the tutorials directory.

    """
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()

    with open(filename, "w", encoding="utf-8") as f:
        f.write("import os; os.chdir('./tutorials')\n")
        f.write("import matplotlib; matplotlib.use('template')\n")
        f.writelines(content)


if __name__ == "__main__":
    try:
        filenames = [Path(sys.argv[1])]
    except IndexError:
        filenames = TUTORIAL_FILENAMES
    for tut_path in filenames:
        test_run_tutorials(tutorial_path=tut_path)
