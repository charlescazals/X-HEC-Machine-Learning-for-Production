FROM debian:11

RUN apt-get update && apt-get -y install python3 python3-pip

COPY . /opt/app

RUN python3 -m pip install -r /opt/app/requirements.txt

CMD gunicorn --workers 9 --bind 0.0.0.0:5000 opt.app.app:app