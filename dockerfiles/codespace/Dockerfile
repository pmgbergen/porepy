# Docker image dedicated to running PorePy on GitHub codespaces.
#
# The image is (for the moment) based on the develop branch of PorePy.

# Use the base PorePy image; most of the heavy lifting is done from there.
FROM porepy/base

# Make sure we are in the home directory
WORKDIR ${POREPY_HOME}
# Make sure we're on the develop branch
RUN git checkout develop && git pull