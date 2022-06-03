# To build:
#
#    docker build . --tag porepy
#
# To run (sharing the current directory with the container)
#
#    docker run -ti -v $(pwd):/home/porepy/ porepy:latest
#
# To save a modified container, from host:
#
#    docker commit porepy TAG_FOR_NEW_IMAGE
#    docker run -ti TAG_FOR_NEW_IMAGE  # will contain modifications.
#

FROM continuumio/miniconda3:latest

MAINTAINER PorePy Development Team

# Adding wget and bzip2
RUN apt-get update && yes|apt-get upgrade && apt-get install -y wget vim 
RUN apt-get install -y wget bzip2 git gcc libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 

# Add sudo
RUN apt-get -y install sudo

# Updating Anaconda packages
RUN conda update conda
RUN conda update --all

# Add user ubuntu with no password, add to sudo group
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN adduser --disabled-password porepy
RUN adduser porepy sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ubuntu
WORKDIR /home/ubuntu/
RUN chmod a+rwx /home/ubuntu/

USER porepy
ENV POREPY_HOME /home/porepy
RUN touch $POREPY_HOME/.sudo_as_admin_successful && \
    mkdir $POREPY_HOME/shared
VOLUME /home/porepy/shared

# Print something nice on entry.
RUN ["echo", "Clone PorePy"]

# Name of top directory for porepy installation
ENV POREPY_HOME=/home/porepy
# .. and the directory for the PorePy source
ENV POREPY_SRC=${POREPY_HOME}/pp

### Set up the conda environment

WORKDIR ${POREPY_HOME}


# Copy the file environment.yml from host into the container if it exists.
# Copying the dockerfile is necessary to avoid errors environment.yml
# does not exist, and the * and / in hte below expression are also necessary.
# Source: https://redgreenrepeat.com/2018/04/13/how-to-conditionally-copy-file-in-dockerfile/
COPY dockerfile environment.yml* ${POREPY_HOME}/

# Modify the environment file to:
#  1) Ensure that the environment name is pp_env; if not, below commands
#     are invalid.
#  2) Set a version of Python, if not specified.

# First, install pyyaml. This we do with pip since installing with
# conda at this stage gave strange error messages (installation will only
# have effect in the conda base environment).
RUN pip install pyyaml

# TODO: Put parse_environment.py in Dockerfiles, move the git pull
# to above this line, copy the parsing file (if necessary).
RUN conda init bash
COPY parse_environment.py .
RUN python parse_environment.py

# Prepare Conda to work with bash

RUN cat ./environment.yml

# Create the Conda environment, we call this pp_env
# NOTE: While it would have been preferrable to set the environment name
# in as a variable, combining this with the activation commands below
# was not worth the effort. Instead, we enforce the name pp_env when parsing
# the environment.yml file, and take for granted that is the name of the
# environment below.
RUN conda env create -f environment.yml

ENV PATH /home/porepy/.conda/bin:$PATH

# Activating the conda environment inside the Docker container is
# not straightforward, since every RUN command is executed in a separate
# shell. Use a workaround:
# Main source: https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["conda", "run", "-n", "pp_env", "/bin/bash", "-c"]

# Activate the new Conda environment when logging into the container
# First, make conda commands available for bash
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Next add activation to the .bashrc
# Source: https://stackoverflow.com/questions/64323539/docker-run-interactive-with-conda-environment-already-activated
RUN echo "conda activate pp_env" >> ~/.bashrc

ENV PYTHONPATH $POREPY_HOME

USER porepy

### Add various packages that may be useful for development
#RUN conda install -c conda-forge spyder jupyterlab 


# Clone PorePy from GitHub
#WORKDIR ${POREPY_HOME}
#RUN git clone https://github.com/pmgbergen/porepy.git ${POREPY_SRC}

#WORKDIR ${POREPY_SRC}

# FIXME: Change to main, or develop, later on
#RUN git checkout docker

# Install PorePy dependencies, as specified in the PorePy repository
#RUN conda install -c conda-forge --file requirements-dev.txt

# Install PorePy
#RUN pip install --user -e .

# Get external content
#RUN python fetch_content.py

# WORKDIR /home/porepy/pp/tests
# RUN pytest

USER porepy


