FROM python:latest

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY docs/requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip3 install sphinx sphinx_rtd_theme
