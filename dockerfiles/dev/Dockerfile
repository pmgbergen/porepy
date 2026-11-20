# Docker image to install the development version of PorePy (e.g.,
# the latest push to the branch develop).
#
# To access the docker image:
#
#    docker pull porepy/dev:latest
#
# To run (sharing the current directory with the container):
#
#    docker run -ti -v $(pwd):/home/porepy/shared porepy/dev:latest
#
# To save a modified container, from host:
#
#    docker commit porepy TAG_FOR_NEW_IMAGE
#    docker run -ti TAG_FOR_NEW_IMAGE  # will contain modifications.
#
# For more information, see the docker documentation.
# Docker image to install the development version of PorePy.

# Use the base PorePy image; most of the heavy lifting is done from there.
FROM porepy/base

## Download PorePy from github
# Make sure we are in the home directory
WORKDIR ${POREPY_HOME}
# Make sure we're on the develop branch
RUN git checkout develop && git pull

# Run tests to check that everything works.
# NOTE: When run as a GH action (presumably after a PR to the develop branch
# was accepted, e.g., when the tests have already been run on GH), this should
# only fail if the docker image itself is flawed.
# Run on a single thread to make results (e.g., from the pypardiso solver)
# deterministic. The variables are set for this command only, so that the image itself
# is not restricted to a single thread.
WORKDIR ${POREPY_HOME}/tests
RUN OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    pytest
WORKDIR ${POREPY_HOME}
