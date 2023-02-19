#!/bin/bash

# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -evx

# Setting venv name for Jupyterlab as parent directory name
SCRIPT_DIR=$(cd `dirname $0` && pwd)

# References
TAG="latest"
PROJECT_NAME='pytorch-research'
DOCKERFILE_LOCATION=${SCRIPT_DIR}/../container/dockerfile
CONTEXT=${SCRIPT_DIR}/../

# Building image
docker build  -f "${DOCKERFILE_LOCATION}" \
              --rm \
              -t "${PROJECT_NAME}":${TAG} \
              ${CONTEXT}