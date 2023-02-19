#!/bin/bash

# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -evx

# Setting venv name for Jupyterlab as parent directory name
SCRIPT_DIR=$(cd `dirname $0` && pwd)

docker run \
        -it \
        --rm \
        --gpus all \
        -v "${SCRIPT_DIR}/../":"/app" \
        pytorch-research:latest \
        bash
