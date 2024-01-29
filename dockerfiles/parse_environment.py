"""THIS SCRIPT IS CURRENTLY NOT IN USE, BUT IS KEPT FOR LEGACY REASONS

This script is used to modify a Conda environment.yml file
so that it is compatible with installing the environment inside the
Docker container with a PorePy installation.

For the moment, the script does the following modifications to the
specified environment file:
  1) The name of the environment is changed to pp_env (this is
     necessary for the Docker image to work)
  2) If no Python version is provided, it set to 3.10.

Example: To enforce Python 3.9 (instead of the default 3.10),
modify the environment.yml file (found in 
${POREPY_DIR}/dockerfiles/environment.yml) to read

name: pp_env
dependencies:
   python: 3.9

"""
from typing import Dict
import collections.abc
import yaml


# We do two kinds of updates: Some are forced (values in
# the original file are overridden), others set default
# values if not provided.


# Source: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
def forced_update(orig: Dict, mandatory: Dict) -> Dict:
    for k, v in mandatory.items():
        if isinstance(v, collections.abc.Mapping):
            orig[k] = forced_update(orig.get(k, {}), v)
        else:
            orig[k] = v
    return orig


def amend_dict(orig: Dict, defaults: Dict) -> Dict:
    for k, v in defaults.items():
        if isinstance(v, collections.abc.Mapping):
            orig[k] = amend_dict(orig.get(k, {}), v)
        else:
            if k not in orig:
                orig[k] = v
    return orig


# Mandatory data: Environment name should be pp_env
mandatory_data = {"name": "pp_env"}
# Default data:
#   * Python version is set to 3.10 if not specified
default_data = {"dependencies": {"python": "3.10"}}

# Load environment file
with open("environment.yml") as file:
    data = yaml.safe_load(file)

# First do the forced updates
forced_update(data, mandatory_data)
# The the amendments
amend_dict(data, default_data)

# Write to file again
with open("environment.yml", "w") as f:
    yaml.dump(data, f, sort_keys=False)

# Extra edit: It turned out conda required a particular format
# for the .yml file. EK has no idea why, but the following did
# the job
s = ""
with open("environment.yml") as f:
    # Open (the already modified) environment file, change
    # the line specifying python version.
    for line in f:
        if "python" in line:
            line = line.replace("python", "- python")
            line = line.replace(": ", "=")
            s += line
        else:
            s += line

# Write the modified file. Now we should be done.
with open("environment.yml", "w") as f:
    f.write(s)
