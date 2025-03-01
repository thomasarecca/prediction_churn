from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data["features"]])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
