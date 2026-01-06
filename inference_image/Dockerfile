FROM tensorflow/tensorflow:2.16.1

WORKDIR /opt/program

RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir pandas==2.2.2

COPY predict.py /opt/program/predict.py
