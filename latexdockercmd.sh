#!/bin/sh
# based on https://github.com/blang/latex-docker
IMAGE=timeq
exec docker run --rm -i --user="$(id -u):$(id -g)" --net=none -v "$PWD":/data "$IMAGE" "$@"
