# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup -------------------------------------------------------------------------------

import os
import sys
# rel path to porepy source code
sys.path.insert(0, os.path.abspath("../src"))
# rel path to example documentation
sys.path.append(os.path.abspath("."))

# -- Project information ----------------------------------------------------------------------

project = "PorePy"
release = "1.5"
version = "docs/alpha"

author = ("Eirik Keilegavlen, "
          + "Ivar Stefansson, "
          + "Jakub W. Both, "
          + "Jhabriel Varela, "
          + "Omar Duran, "
          + "Veljko Lipovac"
)

copyright = ("2022 UiB Center for Modeling of Coupled Subsurface Dynamics, "
             +"GNU LESSER GENERAL PUBLIC LICENSE Version 3, 29 June 2007."
)

# -- General configuration --------------------------------------------------------------------

# Name of the root document or "homepage"
root_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

# Removes the module name space in front of classes and functions
# i.e. porepy.ad.Scalar() -> Scalar()
add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output ------------------------------------------------------------------

# The theme to use for HTML and HTML Help pages. Currently set to theme of Python 2 docs
html_theme = "nature"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]

# Customize the html theme here. Supported  customization depends on the chosen HTML theme,
# but unknown entries will cause no error when compiling.
html_theme_options = {
    # "rightsidebar": "false",
    # "relbarbgcolor": "black",
    # "externalrefs": "true",
    # "bodyfont": "Arial",
    # "headfont": "Arial",
}

html_short_title = "PorePy"
html_split_index = True
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False

# -- Autodoc Settings -------------------------------------------------------------------------

# autoclass concatenates docs strings from init and class.
autoclass_content = "class"  # class-both-init

# Display the signature next to class name
autodoc_class_signature = "mixed"  # mixed-separated

# orders the members of an object group wise, e.g. private, special or public methods
autodoc_member_order = "groupwise"  # alphabetical-groupwise-bysource

# type hints will be shortened: porepy.grids.grid.Grid -> Grid
autodoc_typehints_format = "short"

# default configurations for all autodoc directives
autodoc_default_options = {
    "members": True,
    "special-members": False,
    "private-members": False,
    "show-inheritance": True,
    "no-value": False
}

# -- Napoleon Settings ------------------------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Intersphinx Settings ---------------------------------------------------------------------

intersphinx_mapping = {
    'python': ("https://docs.python.org/3", None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None)
}

# -- Todo Settings ----------------------------------------------------------------------------

todo_include_todos = True
todo_emit_warnings = False
todo_link_only = False

# -- Viewcode Settings ------------------------------------------------------------------------
