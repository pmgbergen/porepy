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
USER root
ENTRYPOINT ["/sbin/my_init","--quiet","--","/sbin/setuser","porepy","/bin/bash","-l","-c"]
CMD ["/bin/bash","-i"]

WORKDIR /tmp

RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs \
        -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install curl && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get -y install \
        bison \
        ccache \
        cmake \
        doxygen \
        flex \
        g++ \
        gfortran \
        git \
        git-lfs \ 
        libboost-filesystem-dev \
        libboost-iostreams-dev \
        libboost-math-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-timer-dev \
        libeigen3-dev \
        liblapack-dev \
        libmpich-dev \
        libopenblas-dev \
        libpcre3-dev \
        libhdf5-mpich-dev \
        libgmp-dev \
        libcln-dev \
        libmpfr-dev \       
	wget \
	libfltk1.3-dev \
        libfreetype6-dev \
        libgl1-mesa-dev \
        liblapack-dev \
        libxi-dev \
        libxmu-dev \
        mesa-common-dev \
	build-essential \
        bash-completion  &&\
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt-get -qq update && \
    apt-get -y --with-new-pkgs \
        -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
        python-dev python3-dev \
        python-flufl.lock python3-flufl.lock \
        python-numpy python3-numpy \
        python-ply python3-ply \
        python-pytest python3-pytest \
        python-scipy python3-scipy \
        python-six python3-six \
        python-subprocess32 \
        python-urllib3  python3-urllib3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install setuptools
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    pip3 install --no-cache-dir setuptools && \
    pip3 install --no-cache-dir jupyter && \
    pip3 install --no-cache-dir sympy && \
    pip3 install --no-cache-dir matplotlib && \
    rm -rf /tmp/*

RUN wget http://prdownloads.sourceforge.net/tcl/tcl8.6.6-src.tar.gz && tar -zxvf tcl8.6.6-src.tar.gz &&\
    cd tcl8.6.6/unix && ./configure && make && make install &&\
    rm -rf /tmp/*


RUN wget http://prdownloads.sourceforge.net/tcl/tk8.6.6-src.tar.gz && tar -zxvf tk8.6.6-src.tar.gz &&\
    cd tk8.6.6/unix && ./configure && make && make install &&\
    rm -rf /tmp/*


RUN wget http://www.vtk.org/files/release/7.0/VTK-7.0.0.tar.gz && tar -zxvf VTK-7.0.0.tar.gz &&\
    cd VTK-7.0.0 &&\
    mkdir build &&\
    cd build  &&\
    cmake \
   -DCMAKE_BUILD_TYPE:STRING=Release \
   -DBUILD_TESTING:BOOL=OFF \
   -DVTK_WRAP_PYTHON:BOOL=ON \
   -DVTK_WRAP_PYTHON_SIP:BOOL=ON \
   -DVTK_WRAP_TCL:BOOL=ON \
   -DVTK_PYTHON_VERSION:STRING=3 \
   -DVTK_USE_TK:BOOL=ON \
    .. &&\
    make && \
    cd Wrapping/Python &&\
    make && \
    make install &&\
    rm -rf /tmp/*

ENV GMSH_VERSION 2.11.0
RUN wget http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64.tgz &&\
    tar xf gmsh-${GMSH_VERSION}-Linux64.tgz && \
    cp gmsh-${GMSH_VERSION}-Linux/bin/gmsh /usr/local/bin/gmsh && \
    rm -rf /tmp/*

WORKDIR $POROPY_HOME
RUN chwon -R poropy:poropy $POROPY_HOME

USER poropy
RUN  mkdir -p $POROPY_HOME/.config/matplotlib &&\
     git clone https://github.com/pmgbergen/porepy.git $POREPY_HOME/porepy-src &&\
     cd porepy-src &&\
     pip3 install -r requirements-dev.txt && \
     echo "config = {\"gmsh_path\":\"/usr/local/bin/gmsh\"} " > porepy_config.py && \
     python setup.py install 
RUN  python setup.py test 
 
COPY matplotlibrc $POROPY_HOME/.config/matplotlib/matplotlibrc
      
	

