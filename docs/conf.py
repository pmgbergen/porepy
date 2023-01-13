# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- path setup ------------------------------------------------------------------------

import os
import sys
# rel path to porepy source code
sys.path.insert(0, os.path.abspath("../src"))
# rel path to example documentation
sys.path.append(os.path.abspath("."))

# -- project information ---------------------------------------------------------------

project = "PorePy"
release = "1.6.0"
version = "1.6.0"

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

# -- general configuration -------------------------------------------------------------

# Name of the root document or "homepage"
root_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    # 'sphinx_autodoc_typehints',  # this moves type links from signature to docstring
    'sphinx.ext.intersphinx',
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

porepy_type_alias_map = {
    'ArrayLike': 'ArrayLike',
    'npt.ArrayLike': 'ArrayLike',
    'NDArray': 'NDArray',
    'DTypeLike': 'DTypeLike',
    'ExampleArrayLike': 'ExampleArrayLike',
}

# -- options for HTML output -----------------------------------------------------------

# The theme to use for HTML and HTML Help pages. Currently set to theme of Python 2 docs
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_short_title = "PorePy"
html_split_index = True
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_baseurl = 'https://pmgbergen.github.io/porepy/'

# relative path to project logo, to be displayed on docs webpage
# html_logo = ''

# theme-specific customization, read the docs of respective theme
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# -- napoleon settings -----------------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True  ### NOTE: Part of type hint position switch
napoleon_preprocess_types = True
napoleon_type_aliases = porepy_type_alias_map
napoleon_attr_annotations = True

# -- autodoc settings ------------------------------------------------------------------

# autoclass concatenates docs strings from init and class.
autoclass_content = "class"  # class-both-init

# Display the signature next to class name
autodoc_class_signature = "mixed"  # mixed-separated

# orders the members of an object group wise, e.g. private, special or public methods
autodoc_member_order = "groupwise"  # alphabetical-groupwise-bysource

# Avoid double appearance of documentation if child member has no docs
autodoc_inherit_docstrings = False

# do not evaluate default arguments, leave as is
autodoc_preserve_defaults = True

# type hints will be shortened: porepy.grids.grid.Grid -> Grid
autodoc_typehints_format = "short"

# uses type hints in signatures for e.g. linking (default)
autodoc_typehints = "description"  # signature, description
### NOTE: Part of type hint position switch

# display types in signature which are documented, or all (by default)
# all, documented, documented_params
autodoc_typehints_description_target = 'all'

# default configurations for all autodoc directives
autodoc_default_options = {
    "members": True,
    "special-members": False,
    "private-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "no-value": False
}

# Used to shorten the parsing of type hint aliases
# NOTE: As of now, hyperlinking to type aliases is an open issue
# see https://github.com/sphinx-doc/sphinx/issues/10785
# NOTE: There is a temporary work-around for custom type aliases
# https://github.com/pandas-dev/pandas/issues/33025#issuecomment-699636759
# may be of interest.
autodoc_type_aliases = porepy_type_alias_map

# -- autodoc typehints settings --------------------------------------------------------

# shorten namespace to only contain class name
typehints_fully_qualified = False
# adds stub documentation to undocumented types
always_document_param_types = False
# If True, adds the :rtype: directive after :returns:
typehints_document_rtype = True
# separates rtype from return, combines them otherwise
# see napoleon_use_rtype, must not be in conflict
typehints_use_rtype = True
# auto-generates formatting for default values in signature
typehints_defaults = 'comma'  # comma, braces-after
# Optional[Union] -> Union
# Unions containing None are always Optional
simplify_optional_unions = False

# -- intersphinx settings --------------------------------------------------------------

intersphinx_mapping = {
    'python3': ("https://docs.python.org/3", None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None)
}

# -- todo settings ---------------------------------------------------------------------

todo_include_todos = True
todo_emit_warnings = False
todo_link_only = False
