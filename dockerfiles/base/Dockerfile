# This is the base image for PorePy installations. It is not intended for direct use,
# instead, use the stable or development versions.
# To build:
#
#    docker build . --tag porepy
#
# To run (sharing the current directory with the container)
#
#    docker run -ti -v $(pwd):/home/porepy/shared porepy:latest
#
# To save a modified container, from host:
#
#    docker commit porepy TAG_FOR_NEW_IMAGE
#    docker run -ti TAG_FOR_NEW_IMAGE  # will contain modifications.
#

# Base the Docker image on phusion, since this apparently gives a version of
# Ubuntu better suited for running inside a container.
FROM phusion/baseimage:jammy-1.0.0

MAINTAINER PorePy Development Team

ENV PYTHON_VERSION 3.10

# Adding wget and bzip2
RUN apt-get update && yes|apt-get upgrade
RUN apt-get install -y wget vim bzip2 git gcc libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 ffmpeg libgl1-mesa-glx libsm6 libxext6 

# Add sudo
RUN apt-get -y install sudo
RUN adduser --disabled-password porepy
RUN adduser porepy sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER porepy

# Set up a directory intended for sharing with the host
# Name of home directory for the user porepy
ENV POREPY_HOME /home/porepy
RUN touch $POREPY_HOME/.sudo_as_admin_successful && \
    mkdir $POREPY_HOME/shared
VOLUME /home/porepy/shared

# .. and the directory for the PorePy source
ENV POREPY_SRC=${POREPY_HOME}/pp


### Set up the conda environment

USER porepy

# Location for the conda install
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp $POREPY_HOME/.conda \
    && rm -rf /tmp/miniconda.sh

# Add the conda binary to path
ENV PATH /home/porepy/.conda/bin:$PATH

# Prepare Conda to work with bash
RUN conda update conda && \
    conda init bash

# Updating Anaconda packages

USER porepy

# Create the Conda environment, we call this pp_env
RUN conda create --name pp_env python=${PYTHON_VERSION}

RUN conda init bash

# Activate the new Conda environment when logging into the container
# Add activation to the .bashrc
# Source: https://stackoverflow.com/questions/64323539/docker-run-interactive-with-conda-environment-already-activated
RUN echo "conda activate pp_env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Add conda-forge as a channel
RUN conda config --add channels conda-forge

# Install PorePy dependencies, as specified in the PorePy repository

# Get the PorePy requirements-dev file (we'll install the full development
# version, the overhead in doing so compared to just the run requirements
# is not too big)
ENV TMP_DIR /tmp
WORKDIR ${TMP_DIR}
RUN wget https://raw.githubusercontent.com/pmgbergen/porepy/develop/requirements-dev.txt

# For some reason, which is completely incomprehenible to EK, the conda install
# of gmsh fails: Conda will list it as being installed, but 'import gmsh' in python
# still fails. As a workaround, we do a pip install of gmsh instead. This is unfortunate,
# since updates like 'conda update --all' will not update all packages, likely contrary
# to the user's expectation, but it will have to do.
# Remove gmsh from the list of requirements for PorePy.
RUN sed -i '/gmsh/d' requirements-dev.txt 

# Activating the conda environment inside the Docker container is
# not straightforward, since every RUN command is executed in a separate
# shell. Use a workaround:
# Main source: https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["conda", "run", "-n", "pp_env", "/bin/bash", "-c"]
RUN conda env list

# Install all requirements
RUN conda install -c conda-forge --file requirements-dev.txt

# Pip install gmsh, since we skipped it from the conda install
RUN pip install gmsh

WORKDIR ${POREPY_HOME}

# Copy file to set home permissions. The path assumes the docker build is invoked
# from the top PorePy directory, e.g.,
#   docker build  --tag porepy/base:latest --file dockerfiles/base/Dockerfile .
#
COPY ./dockerfiles/base/set-home-permissions.sh /etc/my_init.d/set-home-permissions.sh

USER root
RUN chmod +x /etc/my_init.d/set-home-permissions.sh
ENTRYPOINT ["/sbin/my_init","--quiet","--","/sbin/setuser","porepy","/bin/bash","-l","-c"]
CMD ["/bin/bash","-i"]

