from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='static')
CORS(app)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['Product_ID'],
        data['Product_Category'],
        data['Quantity_Sold'],
        data['Promotions'],
        data['Day'],
        data['Month']
    ]])
    prediction = model.predict(features)
    return jsonify({'Predicted_Sales': float(prediction[0])})

# ✅ Add this route to serve index.html
@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
