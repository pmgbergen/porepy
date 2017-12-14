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
ENTRYPOINT ["sudo","/sbin/my_init","--quiet","--","sudo","-u","porepy","/bin/bash","-l","-c"]
CMD ["/bin/bash","-i"]

