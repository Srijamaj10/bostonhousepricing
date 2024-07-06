import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load your model and scaler
with open('regmodel.pkl', 'rb') as f:
    regmodel = pickle.load(f)

with open('scaling.pkl', 'rb') as f:
    scalar = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    try:
        new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
        output = regmodel.predict(new_data)[0]
        return jsonify({"prediction": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_input = scalar.transform(np.array(data).reshape(1, -1))
        output = regmodel.predict(final_input)[0]
        return render_template("home.html", prediction_text="The house price prediction is {}".format(output))
    except Exception as e:
        return render_template("home.html", prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
