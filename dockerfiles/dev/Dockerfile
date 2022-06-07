# Docker image to install the development version of PorePy.

# Use the base PorePy image; most of the heavy lifting is done from there.
FROM porepy/base

# Install PorePy dependencies, as specified in the PorePy repository.
RUN conda install -c conda-forge --file ${POREPY_SRC}/requirements-dev.txt

# Install PorePy
WORKDIR ${POREPY_SRC}
# Make sure we're on the develop branch - this should not really be necessary
RUN git checkout develop

RUN pip install --user -e .

# Get external content
RUN python fetch_content.py

ENV PYTHONPATH $POREPY_HOME:$PYTHONPATH

# Run tests to check that everything works.
# FIXME: Is this really necessary?
WORKDIR /home/porepy/pp/tests
RUN pytest


