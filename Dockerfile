# FROM python:3.10-buster
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
RUN rm -f /etc/apt/sources.list.d/cuda.list && \
    rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80
RUN --mount=type=cache,target=/var/cache/apt set -eux; \
apt-get update -qq; \
apt-get install -qqy --no-install-recommends curl; \
rm -rf /var/lib/apt/lists/*; \
TINI_VERSION=v0.19.0; \
TINI_ARCH="$(dpkg --print-architecture)"; \
curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
chmod +x /sbin/tini
ENTRYPOINT ["/sbin/tini", "--"]
ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:$PATH"
RUN --mount=type=cache,target=/var/cache/apt apt-get update -qq && apt-get install -qqy --no-install-recommends \
        make \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        git \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*
RUN curl -s -S -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash && \
        git clone https://github.com/momo-lab/pyenv-install-latest.git "$(pyenv root)"/plugins/pyenv-install-latest && \
        pyenv install-latest "3.10" && \
        pyenv global $(pyenv install-latest --print "3.10") && \
        pip install "wheel<1"
WORKDIR /workspace
COPY  requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 7860
COPY context_builder.py .
COPY app.py .
CMD ["python","app.py"]