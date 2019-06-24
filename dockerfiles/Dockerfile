# In part based on http://www.science.smith.edu/dftwiki/index.php/Tutorial:_Docker_Anaconda_Python_--_4

# We will use Ubuntu for our image
FROM ubuntu:18.04
MAINTAINER EirikKeilegavlen <Eirik.Keilegavlen@uib.no>

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade && apt-get install -y wget

# Adding wget and bzip2
RUN apt-get install -y wget bzip2 git gcc libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1

# Add sudo
RUN apt-get -y install sudo

# Add user ubuntu with no password, add to sudo group
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

WORKDIR /home/porepy
#USER root

RUN wget http://gmsh.info/bin/Linux/gmsh-4.3.0-Linux64.tgz && \
     tar xvf gmsh-4.3.0-Linux64.tgz && \
     rm gmsh-4.3.0-Linux64.tgz

RUN echo "config={\"gmsh_path\":\"/home/porepy/gmsh-4.3.0-Linux64/bin/gmsh\"} " > porepy_config.py

# Anaconda installing
RUN cd /tmp  && \
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O conda.sh  && \
    chmod +x conda.sh && \
    bash conda.sh -b -p /home/porepy/.conda && \
    rm /tmp/*

# Set path to conda
ENV PATH /home/porepy/.conda/bin:$PATH

ENV PYTHONPATH /home/porepy


# Updating Anaconda packages
RUN conda update conda
RUN conda update --all
RUN conda install -c anaconda python=3.7

# Set up conda to work with conda activate

#RUN conda activate pp

RUN git clone https://github.com/pmgbergen/porepy.git pp
#	cd pp && \
#	echo "config = {\"gmsh_path\":\"/usr/local/bin/gmsh\"} " > porepy_config.py &&\
#    /bin/bash -c -l 'echo "PYTHONPATH=/home/porepy/pp:$PYTHONPATH">>~/.profile'  &&\
#     /bin/bash -c -l 'source ~/.profile' && \
#    python setup.py install 

WORKDIR /home/porepy/pp
RUN conda install numpy=1.16.3 scipy=1.2.1 networkx=2.3 sympy=1.4 cython=0.29.7 numba=0.43.1 matplotlib=3.0.3 pytest=4.5.0 pytest-cov=2.6.1 pytest-runner=4.4 jupyter=1.0.0
# Vtk should be install from conda-forged (not all dependencies are installed otherwise):
RUN conda install -c conda-forge vtk
RUN pip install meshio==2.3.8 shapely==1.6.4.post2 shapely[vectorized]==1.6.4.post2

RUN python setup.py install

WORKDIR /home/porepy

RUN wget https://raw.githubusercontent.com/keileg/polyhedron/master/polyhedron.py
RUN mv polyhedron.py robust_point_in_polyhedron.py


