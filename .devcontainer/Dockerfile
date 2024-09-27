ARG PY_VER
ARG DISTRO
FROM mcr.microsoft.com/devcontainers/python:${PY_VER}-${DISTRO}

RUN \
    # dev setup
    apt update && \
    apt-get install sudo git bash-completion graphviz default-mysql-client s3fs procps -y && \
    pip install --no-cache-dir --upgrade black faker ipykernel pip && \
    apt-get clean

COPY ./ /tmp/element-array-ephys/

RUN \
    # pipeline dependencies
    apt-get install gcc g++ ffmpeg libsm6 libxext6 -y && \
    pip install --no-cache-dir -e /tmp/element-array-ephys[elements,tests] && \
    # clean up
    rm -rf /tmp/element-array-ephys && \
    apt-get clean

USER root
ENV DJ_HOST db
ENV DJ_USER root
ENV DJ_PASS simple

ENV EPHYS_ROOT_DATA_DIR /workspaces/element-array-ephys/example_data
ENV DATABASE_PREFIX neuro_