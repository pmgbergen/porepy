#!/bin/bash
#
# ACKNOWLEDGEMENT: This file is heavily inspired by the Docker configuration
# for the fenics project, see in particular
#
#   https://bitbucket.org/fenics-project/docker/src/master/dockerfiles/base/set-home-permissions.sh
#
# To avoid issues with file permissions in shared folders (/home/porepy/shared)
# it is useful if the id of the user porepy (which is the default user in the
# Docker container) coincides with that of the user on the host. To that end,
# we let the user pass --env arguments to the docker run command that (by the
# below command) will reassign the user and group ids of the user porepy (in
# the container). Specifically, if docker is ran with the command
#
#    docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) ...
#
# file permissions should be okay.
#
# From comments on the internet, it seems this is most relevant if the host
# runs Linux, while on Mac and Windows, shared 
# User can pass e.g. --env HOST_UID=1003 so that UID in the container matches
# with the UID on the host. This is useful for Linux users, Mac and Windows
# already do transparent mapping of shared volumes.
if [ "$HOST_UID" ]; then
    usermod -u $HOST_UID porepy
fi
if [ "$HOST_GID" ]; then
    groupmod -g $HOST_GID porepy
fi
