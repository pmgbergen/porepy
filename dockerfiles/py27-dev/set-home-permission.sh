#!/bin/bash
# User can pass e.g. --env HOST_UID=1003 so that UID in the container matches
# with the UID on the host. This is useful for Linux users, Mac and Windows
# already do transparent mapping of shared volumes.
if [ "$HOST_UID" ]; then
    usermod -u $HOST_UID porepy
fi
if [ "$HOST_GID" ]; then
    groupmod -g $HOST_GID porepy
fi
# This makes sure that all files in /home/porepy are accessible by the user
# fenics. We exclude the folder ~/shared to reduce IO out to the host. Docker
# for Mac, Docker for Windows and the UID/GID trick above should mean that file
# permissions work seamlessly now.
find /home/porepy -maxdepth 1 | sed "1d" | grep -v "/home/porepy/shared" | xargs chown -R porepy:porepy 2> /dev/null || true
