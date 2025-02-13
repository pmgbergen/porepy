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
# Make sure we're on the main branch
RUN git checkout main && git pull

# Run tests to check that everything works.
# NOTE: When run as a GH action (presumably after a PR to the develop branch
# was accepted, e.g., when the tests have already been run on GH), this should
# only fail if the docker image itself is flawed.
WORKDIR ${POREPY_HOME}/tests
RUN pytest
WORKDIR ${POREPY_HOME}
