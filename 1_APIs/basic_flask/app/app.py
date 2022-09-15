import os

from flask import Flask, request, jsonify
import random
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier


app = Flask(__name__)

DATA_FILEPATH = os.path.join('app', 'data', 'cs-training.csv')
MODEL_FILEPATH = os.path.join('app', 'models', 'model.joblib')


@app.route('/', methods=['GET'])
def index():
    return "Welcome to the basic Flask app."


# 0. Random

@app.route('/predict', methods=['GET'])
def get_score():
    score = random.uniform(0, 1)
    return jsonify({"score": score})


# 1.0 Univariate threshold

# @app.route('/predict', methods=['GET'])
# def get_score():
#     debt_ratio = request.args.get('debt_ratio', default=1, type=float)
#     return jsonify({"score": debt_ratio})


# 1.1 Univariate threshold, batch

# @app.route('/predict', methods=['POST'])
# def get_scores():
#     customer_infos = request.json
#
#     scores = []
#     for customer in customer_infos:
#         customer_id = customer["id"]
#         score = customer["debt_ratio"]
#         scores.append({"id": customer_id, "score": score})
#
#     return jsonify({"scores": scores})


# 2. Trained model

# def train_and_save_model(model_filepath):
#     train = pd.read_csv(DATA_FILEPATH, index_col=0)
#     train.fillna(-1, inplace=True)
#     X = train.drop('SeriousDlqin2yrs', axis=1)
#     y = train['SeriousDlqin2yrs']
#     gbm = GradientBoostingClassifier()
#     gbm.fit(X, y)
#     dump(gbm, model_filepath)
#
#
# @app.route('/train', methods=['GET'])
# def train_model():
#     try:
#         train_and_save_model(MODEL_FILEPATH)
#         return jsonify(({'status': 'success', 'message': 'Model successfully updated'}))
#
#     except Exception as e:
#         return str(e)
#
#
# @app.route('/predict', methods=['POST'])
# def get_scores():
#     payload = request.json
#     input_df = pd.DataFrame(payload)
#     input_df.fillna(-1, inplace=True)
#
#     if not os.path.exists(MODEL_FILEPATH):
#         train_and_save_model(MODEL_FILEPATH)
#
#     model = load(MODEL_FILEPATH)
#     predictions = model.predict_proba(input_df)
#     scores = [prediction[1] for prediction in predictions]
#
#     return jsonify({'scores': scores})