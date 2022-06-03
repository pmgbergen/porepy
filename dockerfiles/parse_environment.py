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
#   * conda-forge is added to the list of environments
default_data = {"dependencies": {"python": "3.10"}}

with open("environment.yml") as file:
    data = yaml.safe_load(file)

forced_update(data, mandatory_data)

amend_dict(data, default_data)

with open("environment.yml", "w") as f:
    yaml.dump(data, f, sort_keys=False)

s = ""
with open("environment.yml") as f:
    for line in f:
        if "python" in line:
            line = line.replace("python", "- python")
            line = line.replace(": ", "=")
            s += line
        else:
            s += line

with open("environment.yml", "w") as f:
    f.write(s)
