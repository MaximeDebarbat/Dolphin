FROM nvcr.io/nvidia/tensorrt:23.01-py3

WORKDIR /app

# disable bytecode generation
ENV PYTHONDONTWRITEBYTECODE 1

# send the python output directly to the terminal without first buffering it
ENV PYTHONUNBUFFERED 1

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY setup.py setup.py

RUN python3 -m pip install --upgrade pip

RUN apt-get update
RUN apt-get install libgl1 -y
