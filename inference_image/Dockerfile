FROM tensorflow/tensorflow:2.16.1

WORKDIR /opt/program

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      numpy==1.26.4 \
      pandas==2.2.2 \
      keras==3.11.3

COPY predict.py /opt/program/predict.py