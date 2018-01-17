# Docker Advanced

To facilitate the devoloping, using the text editor,version control and other tools already installed on your computers,
it is possible to share files from the host into the container:

```bash
>  docker run -ti -v $(pwd):/home/porepy/shared  pmgbergen/porepylib:py27
```
To allow the X11 forwarding in the container, on Linux system just run:

```bash
>  docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  -v $(pwd):/home/porepy/shared  pmgbergen/porepylib:py27
```
It is also available a docker container based on python 3.6, just running:
```bash
>  docker run -ti  docker.io/pmgbergen/porepylib:py36
```
For Windows system, you need to install Cygwin/X version and running the command in Cygwin terminal. While for mac system, you need to install xquartz. 

# For Developing/ enhance Docker
If you would like to compile Docker for developing porpose. You could associate this github repo with docker cloud service for deployment. Alternatively, on you own machine on terminal (Linux) or on Docker terminal (Mac/Win) you just run:
```bash
> cd  dockerfiles/py36 && docker build . --tag porepy:develop
```
The tag of your container will be "porepy" and the version "develop".
# Q&A 
To clean the possible cache created by docker you just run:
```bash
> docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q)"
```
This command will stop all running container and remove them from your cache. When you exit from the container you need carelly to write exit in terminal in place of close by brute force the terminal.
The following command allow you delete all orphan image that you have create. 

```bash
> docker rmi -f $(docker images | grep "<none>" | awk "{print \$3}")"
```
