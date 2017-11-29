# Docker Advanced

To facilitate the devoloping, using the text editor,version control and other tools already installed on your computers,
it is possible to share files from the host into the container:

```bash
>  docker run -ti -v $(pwd):/home/porepy/shared  pmgbergen/porepy:py27
```
To allow the X11 forwarding in the container, on Linux system just run:

```bash
>  docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  -v $(pwd):/home/porepy/shared  pmgbergen/porepy:py27
```
It is also available a docker container based on python 3.6, just running:
```bash
>  docker run -ti  docker.io/pmgbergen/porepy:py36
```
For Windows system, you need to install Cygwin/X version and running the command in Cygwin terminal. While for mac system, you need to install xquartz. 
# For Developing/ enhance Docker
If you would like to compile Docker for developing porpose. You could associate this github repo with docker cloud service for deployment. Alternatively, on you own machine on terminal (Linux) or on Docker terminal (Mac/Win) you just run:
```bash
> cd  dockerfiles && docker build . --tag porepy:develop
```
The tag of your container will be "porepy" and the version "develop".
