"""A script to insert the docstrings of the implementations to the protocol.

Only inserts the docstrings for those protocol methods that have the "# AUTODOC" comment
in their body, and satisfy these requirements:

* Have exactly one implementation within the models/ folder, non-recursively.
* Have some docstring (possibly empty) to be replaced with the new one.

In case if any of the requirements is not satisfied, the error will be raised and no
changes will be applied.

"""

import ast
from collections import defaultdict
from pathlib import Path
import os
import sys


PROTOCOLS_DEFAULT_PATH = Path("src/porepy/models/protocol.py")


def extract_implementation_docstrings(source: str) -> dict[str, str]:
    """Extract functions and their docstrings from the implementation file."""
    tree = ast.parse(source)
    functions_with_docstrings = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            if docstring:
                functions_with_docstrings[node.name] = docstring
    return functions_with_docstrings


def find_autodoc_functions(source: str) -> dict[str, ast.FunctionDef]:
    """Find functions with the # AUTODOC flag in the declaration file."""
    tree = ast.parse(source)
    source_lines = source.split("\n")
    autodoc_functions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # The comment can be the last line of the function, and the end_lineno will
            # not account for it. To prevent it, one more line is added. Also, line
            # numeration starts from 1.
            function_source = source_lines[node.lineno - 1 : node.end_lineno + 1]
            for line in function_source:
                if "# AUTODOC" in line:
                    autodoc_functions[node.name] = node
                    break
    return autodoc_functions


def insert_docstrings(
    protocols_source: str,
    autodoc_functions: dict[str, ast.FunctionDef],
    impl_docstrings: dict[str, str],
) -> list[str]:
    """Insert docstrings into the declaration file for functions that have # AUTODOC."""
    line_offset = 0

    protocols_source = protocols_source.split("\n")

    for func_name, node in autodoc_functions.items():
        if func_name not in impl_docstrings:
            break
        docstring = impl_docstrings[func_name]

        # Prepare the docstring with proper indentation
        docstrine_lines_old = docstring.split("\n")
        docstring_lines = []
        indent = " " * node.body[0].col_offset  # Assuming 4 spaces indentation
        for i, line in enumerate(docstrine_lines_old):
            if i == 0:
                line = f'{indent}"""{line}'
            elif line != "":
                line = f"{indent}{line}"

            if i == len(docstrine_lines_old) - 1:
                line = f'{line}\n\n{indent}"""'
            docstring_lines.append(line)

        # Insert the docstring instead of the old one.
        if not isinstance(node.body[0].value, ast.Constant):
            raise ValueError(
                "This is likely not a docstring, please check:\n"
                f"{node.body[0].value.value}"
            )
        protocols_source[node.body[0].lineno - 1 + line_offset] = "\n".join(
            docstring_lines
        )
        for lineno in range(node.body[0].lineno, node.body[0].end_lineno):
            del protocols_source[lineno + line_offset]
            line_offset -= 1
    return protocols_source


def collect_implementation_files(protocols_path: Path):
    """Collects the implementation files from the protocol's directory."""
    models_dir = protocols_path.parent
    implementation_files = []
    for filename in os.listdir(models_dir):
        path = models_dir / filename
        if filename == protocols_path.name:
            continue
        if not os.path.isfile(path):
            continue
        implementation_files.append(path)
    return implementation_files


def update_protocol_docstrings(protocols_path: Path):
    """Inserts the docstrings to the protocol methods from their implementations.

    A protocol method must be marked with "# AUTODOCK" comment.

    """
    # Mapping of method names to the docstrings of their implementations.
    impl_docstrings: dict[str, str] = {}
    # Counts how many implementations of each method did we encounter.
    impl_functions_counter: dict[str, int] = defaultdict(lambda: 0)
    # Collect all the docstrings of the implementations and count them.
    for impl_filepath in collect_implementation_files(protocols_path):
        with open(impl_filepath, encoding="utf8") as f:
            impl_source = f.read()
        tmp = extract_implementation_docstrings(impl_source)
        impl_docstrings |= tmp
        for function_name in tmp:
            impl_functions_counter[function_name] += 1

    with open(protocols_path, encoding="utf8") as f:
        protocols_source = f.read()
    # Mapping of the "# AUTODOC" methods in the protocol to their AST nodes.
    autodoc_functions = find_autodoc_functions(protocols_source)

    # Each "# AUTODOC" method must have only one implementation.
    for function_name in autodoc_functions:
        counter = impl_functions_counter[function_name]
        if counter == 0:
            raise ValueError(f"No implementation of {function_name} found.")
        if counter >= 2:
            raise ValueError(f"Multiple implementations of {function_name} found.")

    # Insert the docstrings into the protocol file
    updated_lines = insert_docstrings(
        protocols_source, autodoc_functions, impl_docstrings
    )

    # Write the updated file
    with open(protocols_path, "w", encoding="utf8") as f:
        f.write("\n".join(updated_lines))
    print(f"Updated docstrings in {protocols_path}.")


if __name__ == "__main__":
    try:
        protocols_path = sys.argv[1]
    except IndexError:
        print(f"Assuming protocols path: {PROTOCOLS_DEFAULT_PATH}.")
        protocols_path = PROTOCOLS_DEFAULT_PATH
    update_protocol_docstrings(protocols_path)
