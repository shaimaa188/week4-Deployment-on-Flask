import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

# create flask app
app = Flask(__name__)

# load the pickle model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html", prediction_text = "The Species of the flower is {}".format(prediction))



@app.route("/predict_api", methods=["POST"])
def predict_api():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({"Predection": list(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
