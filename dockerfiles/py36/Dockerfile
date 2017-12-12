FROM phusion/baseimage:0.9.22
MAINTAINER dbaroli <davide.baroli@uni.lu>

# Get Ubuntu updates
USER root
RUN apt-get update && \
    apt-get upgrade -y -o Dpkg::Options::="--force-confold" && \
    apt-get -y install locales sudo && \
    echo "C.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set locale environment
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8

COPY set-home-permission.sh /etc/my_init.d/set-home-permission.sh
RUN useradd -m -s /bin/bash -G sudo,docker_env porepy && \
    echo "porepy:docker" | chpasswd && \
    echo "porepy ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    touch /etc/service/syslog-forwarder/down && \
    echo "cat /home/porepy/WELCOME" >> /home/porepy/.bashrc && \
    chmod +x /etc/my_init.d/set-home-permission.sh && \
    ldconfig
RUN echo "/usr/local/lib/python3/dist-packages" >> /usr/local/lib/python3.5/dist-packages/debian-ubuntu-sitepath-fix.pth
RUN touch /etc/service/syslog-forwarder/down

USER porepy
ENV POREPY_HOME /home/porepy
RUN touch $POREPY_HOME/.sudo_as_admin_successful && \
    mkdir $POREPY_HOME/shared
VOLUME /home/porepy/shared

# Print something nice on entry.
COPY WELCOME $POREPY_HOME/WELCOME

WORKDIR /home/porepy
ENTRYPOINT ["sudo","/sbin/my_init","--quiet","--","sudo","-u","porepy","/bin/bash","-l","-c"]
CMD ["/bin/bash","-i"]

USER root

WORKDIR /tmp

ENV GMSH_VERSION 2.11.0
RUN wget http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64.tgz &&\
    tar xf gmsh-${GMSH_VERSION}-Linux64.tgz && \
    cp gmsh-${GMSH_VERSION}-Linux/bin/gmsh /usr/local/bin/gmsh && \
    rm -rf /tmp/*

RUN cd /tmp  && \
    wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O conda.sh  && \
    chmod +x conda.sh && \
    bash conda.sh -b -p /home/porepy/.conda && \
    rm /tmp/*

RUN id porepy &&\
     chown -R porepy:porepy /home


USER porepy


ENV PATH=/home/porepy/.conda/bin:$PATH


RUN echo "PATH=/home/porepy/.conda/bin:$PATH" >> ~/.profile && \
     /bin/bash -c -l 'source  ~/.profile'

RUN  hash -r  && \
    conda config --set always_yes yes --set changeps1 no  && \
   conda update -q conda   && \
    conda info -a  && \
    conda create --yes -n porepy python="3.6"
RUN /bin/bash -c -l 'source activate porepy'

ENV PATH /home/porepy/.conda/envs/porepy/bin:$PATH
ENV CONDA_DEFAULT_ENV porepy
ENV CONDA_PREFIX /home/porepy/.conda/envs/porepy



RUN conda install --yes numpy scipy matplotlib pip nose sip llvmlite numba &&\
    conda install -c https://conda.anaconda.org/clinicalgraphics vtk &&\
    pip install setuptools && \
    pip install numpy-stl && \
    pip install coveralls && \
    pip install coverage &&\
    pip install cython  &&\
    pip install jupyter ipython pdbpp 


WORKDIR $POREPY_HOME
RUN  mkdir -p $POREPY_HOME/.config/matplotlib 
COPY matplotlibsrc $POROPY_HOME/.config/matplotlib/matplotlibrc


COPY dependencies $POROPY_HOME/.dependencies
RUN pip install --no-cache-dir -r $POROPY_HOME/.dependencies
RUN /bin/bash -c -l 'echo "source activate porepy">>~/.profile' &&\
    /bin/bash -c -l 'echo "PYTHONPATH=/home/porepy/conda/envs/porepy/lib/python3.6/site-packages:$PYTHONPATH">>~/.profile' 

RUN  git clone https://github.com/pmgbergen/porepy.git porepy-src &&\
    cd porepy-src &&\
    echo "config = {\"gmsh_path\":\"/usr/local/bin/gmsh\"} " > porepy_config.py &&\
    /bin/bash -c -l 'echo "PYTHONPATH=/home/porepy/porepy-src:$PYTHONPATH">>~/.profile'  &&\
     /bin/bash -c -l 'source ~/.profile' && \
    python setup.py install 
