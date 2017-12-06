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

USER root
ENTRYPOINT ["sudo","/sbin/my_init","--quiet","--","sudo","-u","porepy","/bin/bash","-l","-c"]
#ENTRYPOINT ["/sbin/my_init","--quiet","--","/sbin/setuser","porepy","/bin/bash","-l","-c"]
CMD ["/bin/bash","-i"]

RUN id porepy
RUN chown -R porepy:porepy /home

RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs \
        -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install curl && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update -y; apt-get install -y --force-yes --fix-missing --no-install-recommends curl git git-lfs unzip tree subversion vim cmake bison g++ gfortran openmpi-bin pkg-config wget \
    libpcre3-dev bison flex swig libglu1-mesa libxcursor-dev libxft-dev libxinerama-dev libmed1v5 libpng12-0 liboce-foundation10 liboce-modeling10  pyqt5-dev-tools  bash-completion  &&\
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /tmp

ENV GMSH_VERSION 2.11.0
RUN wget http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64.tgz &&\
    tar xf gmsh-${GMSH_VERSION}-Linux64.tgz && \
    cp gmsh-${GMSH_VERSION}-Linux/bin/gmsh /usr/local/bin/gmsh && \
    rm -rf /tmp/*
USER root

RUN cd /tmp  && \
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh  && \
    chmod +x miniconda.sh && \
    bash miniconda.sh -b -p /usr/local/miniconda && \
    rm /tmp/*
ENV PATH=/usr/local/miniconda/bin:$PATH
RUN chown -R porepy:porepy /usr/local/miniconda

RUN echo "PATH=/usr/local/miniconda/bin:$PATH" >> ~/.profile
RUN /bin/bash -c 'source  ~/.profile'

RUN  hash -r  && \
    conda config --set always_yes yes --set changeps1 no  && \
    conda update -q conda  
RUN  conda info -a  && \
    conda create --yes -n porepy python="3.5";

RUN /bin/bash -c 'source  activate porepy'  
RUN    conda install --yes numpy scipy matplotlib pip nose &&\
       conda install --yes -c clinicalgraphics vtk=7.1.0 && \
    pip install setuptools && \
    pip install enum34 && \
    pip install numpy-stl && \
    pip install coveralls && \
    pip install coverage &&\
    pip install numba cython  &&\
    pip install jupyter ipython pdb 

RUN  mkdir -p $POREPY_HOME/.config/matplotlib 
COPY matplotlibsrc $POROPY_HOME/.config/matplotlib/matplotlibrc


COPY dependencies.txt $POROPY_HOME/.dependencies
RUN pip install --no-cache-dir -r $POROPY_HOME/.dependencies

USER porepy
WORKDIR $POREPY_HOME
RUN /bin/bash -c ' export PATH=/usr/local/miniconda/bin:$PATH  && source activate porepy'

RUN  git clone https://github.com/pmgbergen/porepy.git $POREPY_HOME/porepy-src &&\
    cd porepy-src &&\
    echo "config = {\"gmsh_path\":\"/usr/local/bin/gmsh\"} " > porepy_config.py &&\
    python setup.py install 

 
   
	
   
	

