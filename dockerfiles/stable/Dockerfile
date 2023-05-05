# Docker image to install the stable version of PorePy (e.g.,
# the latest push to the branch main).
#
# To access the docker image:
#
#    docker pull porepy/stable:latest
#
# To run (sharing the current directory with the container):
#
#    docker run -ti -v $(pwd):/home/porepy/shared porepy/stable:latest
#
# To save a modified container, from host:
#
#    docker commit porepy TAG_FOR_NEW_IMAGE
#    docker run -ti TAG_FOR_NEW_IMAGE  # will contain modifications.
#
# For more information, see the docker documentation.

# Use the base PorePy image; most of the heavy lifting is done from there.
FROM porepy/base

## Download PorePy from github
# Make sure we are in the home directory
WORKDIR ${POREPY_HOME}
# Clone from Github
RUN git clone https://github.com/pmgbergen/porepy.git ${POREPY_SRC}

# Enter source directory
WORKDIR ${POREPY_SRC}
# Make sure we're on the main branch
RUN git checkout main

# Do the install
RUN pip install --user -e .

# Add PorePy home to the pythonpath. This may or may not be necessary.
ENV PYTHONPATH $POREPY_HOME:$PYTHONPATH

# Run tests to check that everything works.
# NOTE: When run as a GH action (presumably after a PR to the develop branch
# was accepted, e.g., when the tests have already been run on GH), this should
# only fail if the docker image itself is flawed.
WORKDIR ${POREPY_SRC}/tests
RUN pytest
WORKDIR ${POREPY_HOME}

