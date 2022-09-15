FROM debian:11

RUN apt-get update && apt-get -y install python3 python3-pip

COPY . /opt/app

RUN python3 -m pip install -r /opt/app/requirements.txt

CMD FLASK_APP="./opt/app/app.py" flask run --host=0.0.0.0