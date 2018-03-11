#!/usr/bin/env bash

# Requirements
# Nvidia Docker 2: https://github.com/NVIDIA/nvidia-docker

docker build . -t octmps:latest

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run \
    -it \
    --runtime=nvidia \
	-v $XSOCK:$XSOCK:rw \
	-v $XAUTH:$XAUTH:rw \
	--privileged \
    --device=/dev/dri/card0:/dev/dri/card0 \
	-e DISPLAY=$DISPLAY \
	-e XAUTHORITY=$XAUTH \
    octmps:latest \
    bash

# Sample Usage
# python ./src/python/octmps_main.py \
#                    --input-opt-json-file ./data/input/ellipsoid_and_two_spheres/input_opt.json \
#                    --input-bias-json-file ./data/input/ellipsoid_and_two_spheres/input_bias.json \
#                    --input-mesh-file ./data/input/ellipsoid_and_two_spheres/ellipsoid_and_two_spheres_60_degree.mesh \
#                    --visualize