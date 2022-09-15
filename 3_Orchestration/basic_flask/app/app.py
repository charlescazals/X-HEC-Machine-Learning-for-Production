import os

from flask import Flask, request, jsonify
import random
import pandas as pd
from joblib import dump, load
from sqlalchemy import create_engine
from sklearn.ensemble import GradientBoostingClassifier


app = Flask(__name__)

ROOT_APP_FOLDER = "/opt/app"
DATA_FILEPATH = os.path.join(ROOT_APP_FOLDER, 'data', 'cs-training.csv')
MODEL_FILEPATH = os.path.join(ROOT_APP_FOLDER, 'models', 'model.joblib')

DB_NAME = 'database'
DB_USER = 'myusername'
DB_PASSWORD = 'mysecretpassword'
DB_HOST = 'my-postgres-db'
DB_PORT = '5432'
# Connect to the database
POSTGRES_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
db = create_engine(POSTGRES_URL)

ALL_FIELDS = [
    "id", "serious_dlq_in_2_yrs", "revolving_utilization_of_unsecured_lines", "age",
    "number_of_time_30_59_days_past_due_not_worse", "debt_ratio", "monthly_income",
    "number_of_open_credit_lines_and_loans", "number_of_times_90_days_late",
    "number_real_estate_loans_or_lines", "number_of_time_60_89_days_past_due_not_worse", "number_of_dependents"
]


@app.route('/', methods=['GET'])
def index():
    return "Welcome to the basic Flask app."


@app.route('/predict', methods=['POST'])
def get_scores():
    payload = request.json
    input_df = pd.DataFrame(payload)
    input_df.fillna(-1, inplace=True)

    if not os.path.exists(MODEL_FILEPATH):
        train_and_save_model()

    model = load(MODEL_FILEPATH)
    predictions = model.predict_proba(input_df)
    scores = [prediction[1] for prediction in predictions]

    return jsonify({'scores': scores})


def train_and_save_model(model_filepath: str = MODEL_FILEPATH):
    with db.connect() as connection:
        train = pd.read_sql_query(
            sql=f"SELECT {', '.join(ALL_FIELDS)} FROM credits_history", con=connection, index_col='id'
        )
    train.fillna(-1, inplace=True)
    X = train.drop('serious_dlq_in_2_yrs', axis=1)
    y = train['serious_dlq_in_2_yrs']
    gbm = GradientBoostingClassifier()
    gbm.fit(X, y)
    dump(gbm, model_filepath)


@app.route('/train', methods=['GET'])
def train_model():
    try:
        train_and_save_model()
        return jsonify({'status': 'success', 'message': 'Model successfully updated'})

    except Exception as e:
        return str(e)


@app.route('/records/<record_id>')
def get_record(record_id: int):
    try:
        result = db.execute(f"SELECT {', '.join(ALL_FIELDS)} FROM credits_history WHERE ID = '{record_id}'")
        records = [list(row) for row in result]
        if len(records) == 0:
            return jsonify({'status': 'error', 'message': f'Id {record_id} not found'})
        else:
            return jsonify({'status': 'success', record_id: records[0]})

    except Exception as e:
        return str(e)


@app.route('/records', methods=['POST'])
def add_records():
    records = request.json
    formatted_records = [f"({','.join([str(val) for val in record])})" for record in records]  # Eg. ["(1,0,NULL)", ...]
    try:
        db.execute(f"INSERT INTO credits_history ({', '.join(ALL_FIELDS)}) VALUES {','.join(formatted_records)};")
        return jsonify({'status': 'success', 'message': f'Successfully added {len(formatted_records)} record(s)'})

    except Exception as e:
        return str(e)
