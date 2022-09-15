FROM python:3.9

COPY . /opt/app

RUN pip3 install -r /opt/app/requirements.txt

CMD FLASK_APP="./opt/app/app.py" flask run --host=0.0.0.0