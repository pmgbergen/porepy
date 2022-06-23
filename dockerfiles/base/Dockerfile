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

ENV TMP_DIR /tmp

# Name of top directory for porepy installation
ENV POREPY_HOME=/home/porepy
# .. and the directory for the PorePy source
ENV POREPY_SRC=${POREPY_HOME}/pp

## Pull PorePy 
# Clone PorePy from GitHub
RUN git clone https://github.com/pmgbergen/porepy.git ${POREPY_SRC}
WORKDIR ${POREPY_SRC}/dockerfiles

### Set up the conda environment

# Copy the file environment.yml from host into the container if it exists.
# Copying the dockerfile is necessary to avoid errors environment.yml
# does not exist, and the * and / in hte below expression are also necessary.
# Source: https://redgreenrepeat.com/2018/04/13/how-to-conditionally-copy-file-in-dockerfile/

# First make sure there is a file environment (if this is empty, the below
# parsing will assign sensible values)
RUN touch ${TMP_DIR}/environment.yml

# Then copy the environment file from the host if it exists
COPY environment.yml* ${TMP_DIR}/

# Copy the parse_environment file from the PorePy directory to /tmp
RUN cp ${POREPY_SRC}/dockerfiles/parse_environment.py ${TMP_DIR}
WORKDIR ${TMP_DIR}

# Modify the environment file to:
#  1) Ensure that the environment name is pp_env; if not, below commands
#     are invalid.
#  2) Set a version of Python, if not specified.

# First, install pyyaml. This we do with pip since installing with
# conda at this stage gave strange error messages (installation will only
# have effect in the conda base environment).
RUN pip install pyyaml

# Prepare Conda to work with bash
RUN conda init bash

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

# Add conda-forge as a channel
RUN conda config --add channels conda-forge
