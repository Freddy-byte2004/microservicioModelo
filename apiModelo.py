from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo entrenado
model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Recibe JSON
    features = np.array(data['features']).reshape(1, -1)
    probability = model.predict_proba(features)[0][1]  # Clase positiva
    return jsonify({'probabilidad_diabetes': float(probability)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)