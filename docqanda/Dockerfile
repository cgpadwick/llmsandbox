ARG FUNCTION_DIR="/function"

FROM --platform=linux/x86-64 ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
  apt-get install -y \
  g++ \
  make \
  cmake \
  unzip \
  libcurl4-openssl-dev \
  python3 \
  python3-pip \
  python3-dev

RUN pip3 install awslambdaric

ARG FUNCTION_DIR

RUN mkdir -p ${FUNCTION_DIR}

ADD . ${FUNCTION_DIR}

WORKDIR ${FUNCTION_DIR}

RUN pip3 install -r ${FUNCTION_DIR}/lambda.requirements.txt --target "${FUNCTION_DIR}"

RUN python3 lambda.py

ENTRYPOINT ["/usr/bin/python3", "-m", "awslambdaric"]

CMD ["lambda.handler"]