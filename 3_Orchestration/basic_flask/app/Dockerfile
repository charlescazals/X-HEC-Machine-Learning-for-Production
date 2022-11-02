FROM python:3.9

COPY . /opt/app

RUN pip3 install -r /opt/app/requirements.txt

CMD gunicorn --workers 9 --bind 0.0.0.0:5000 opt.app.app:app