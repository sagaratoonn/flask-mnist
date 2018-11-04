FROM tensorflow/tensorflow:latest-py3

RUN pip3 install -q flask
RUN pip3 install -q flasgger
RUN pip3 install -q jsonify
COPY ./app.py ./
COPY ./numbers ./numbers

EXPOSE 5000

ENTRYPOINT ["python3", "./app.py"]
