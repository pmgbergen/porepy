""" Wrapper to read configuration file for PorePy.

The config file contains settings that are specific for each user's setup, and
is therefore not shipped with the library itself.

To create the config file, create a file porepy_config.py, and place it
somewhere in your $PYTHONPATH (to see that it works, open a python interpreted
and write 'import porepy_config' , this should work without errors).

The file porepy_config.py should contain a dictionary called 'config', that
should contain

config = {'gmsh_path': path/to/gmsh/executable,
          'num_processors': 4 
}

The variables needed to change the config file may change as PorePy is further
developed. If a new key is introduced, without the local config file being
updated, this will raise a KeyError.

"""


def read():
    """ Read configuration file, located somewhere in the PYTHONPATH.

    Returns 
    -------
    dictionary
        See module level comments for details.

    Raises
    ------
    ImportError 
        If the file porepy_config is not found in PYTHONPATH


    """
    try:
        import porepy_config
    except ImportError:
        s = "Could not load configuration file for PorePy.\n"
        s += "To see instructions on how to generate this file, confer help\n"
        s += "for this module (for instance, type \n"
        s += "  porepy.utils.read_config? \n"
        s += "in ipython)"
        raise ImportError(s)

    return porepy_config.config
