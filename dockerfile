FROM nvcr.io/nvidia/tensorrt:23.01-py3

WORKDIR /app

# disable bytecode generation
ENV PYTHONDONTWRITEBYTECODE 1

# send the python output directly to the terminal without first buffering it
ENV PYTHONUNBUFFERED 1

COPY setup.py setup.py

RUN python3 -m pip install --upgrade pip
RUN pip3 install -e .

RUN apt-get update
RUN apt-get install libgl1 -y