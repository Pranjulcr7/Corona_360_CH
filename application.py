from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict', methods=["Get", "Post"])
def predict():
    """
    ---
    parameters:

    responses:

    """
    body = request.get_json()
    score = float(body.get("score"))
    total_contact_duration = float(body.get("total_contact_duration"))
    contact_level = int(body.get("contact_level"))
    total_contacts = float(body.get("total_contacts"))
    symptom_1 = int(body.get("symptom_1"))
    symptom_2 = int(body.get("symptom_2"))
    symptom_3 = int(body.get("symptom_3"))
    symptom_4 = int(body.get("symptom_4"))
    symptom_5 = int(body.get("symptom_5"))
    gender = int(body.get("Gender"))
    age = int(body.get("Age"))

    data = pd.DataFrame(
        data=[
            [score, total_contact_duration, contact_level, total_contacts, symptom_1, symptom_2,
             symptom_3, symptom_4, symptom_5, gender, age]
        ],
        columns=['score', 'total_contact_duration', 'contact_level', 'total_contacts', 'symptom_1', 'symptom_2',
                 'symptom_3', 'symptom_4', 'symptom_5', 'Gender', 'Age']
    )
    prediction = classifier.predict(data)
    confidence = classifier.predict_proba(data)
    response = {
        "prediction" : str(prediction[0]),
        "confidence" : str(
            confidence[0][prediction[0]]
        )
    }
    return jsonify(response)


# @app.route('/predict_file', methods=["POST"])
# def predict_file():
#     """

#     ---
#     parameters:

#     responses:

#     """
#     df_test = pd.read_csv(request.files.get("file"))
#     print(df_test.head())
#     prediction = classifier.predict(df_test)

#     return str(list(prediction))


if __name__ == '__main__':
    app.run()