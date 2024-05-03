# This is the base Dockerfile for creating a PorePy installation. It is not intended for
# direct use, instead use the development or stable images (both available on
# DockerHub).
#
# The setup of the Docker image is under development and may change in the future.

# Base the Docker image on the official Python image.
# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Since using non-root users may lead to permission issues, we use root for now. See
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user for more.

ENV HOME /workdir
ENV POREPY_HOME=${HOME}/porepy

# Create and move to code directory. This is where PorePy will be installed.
WORKDIR ${HOME}

# Install missing packages. Most of these are needed for gmsh.
RUN apt-get update 
RUN apt-get install -y build-essential vim git libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 ffmpeg libgl1-mesa-glx libsm6 libxext6 python3-tk

# Update pip
RUN pip install --upgrade pip

# Fetch PorePy from GitHub to code directory
RUN git clone https://github.com/pmgbergen/porepy.git ${POREPY_HOME}
# Move to PorePy sub-directory
WORKDIR ${POREPY_HOME}

# Install pypardiso. This will also install mkl and some other packages, all of which
# are useful for achieving good performance.
RUN pip install pypardiso

# Install dependencies
RUN pip install -r ./requirements.txt

# Move back to code directory
WORKDIR ${HOME}

