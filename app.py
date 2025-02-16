from flask import Flask, request, jsonify
import joblib


app = Flask(__name__)

# Load your best model (pickle or joblib)
with open("best_model.joblib", "rb") as f:
    model = joblib.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Example: data['features'] is a list of features
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
