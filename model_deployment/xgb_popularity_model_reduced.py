#!/usr/bin/python

import joblib
import pandas as pd
from flask import Flask
from flask_restx import Api, Resource, fields
import numpy as np

# 1. Cargar el modelo XGBoost entrenado
try:
    xgb_model = joblib.load('xgb_popularity_model_reduced.pkl')
except FileNotFoundError:
    print("Error: No se encontró el archivo del modelo XGBoost (xgb_popularity_model.pkl). Asegúrate de que esté en el mismo directorio o proporciona la ruta correcta.")
    exit()

# 2. Inicializa la aplicación Flask
app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Song Popularity Prediction API',
    description='API para predecir la popularidad de canciones usando un modelo XGBoost.'
)

# 3. Define el espacio de nombres para las predicciones
ns = api.namespace('predict', description='Predicciones de Popularidad de Canciones')

# 4. Define la estructura de los datos de entrada esperados (las 5 variables más comunes)
song_features = api.model('Custom5Features', {
    'play_count': fields.Float(required=True, description='Conteo de reproducciones'),
    'duration_ms': fields.Integer(required=True, description='Duración de la canción en milisegundos'),
    'release_year': fields.Integer(required=True, description='Año de lanzamiento'),
    'danceability': fields.Float(required=True, description='Bailabilidad de la canción'),
    'tempo': fields.Float(required=True, description='Tempo de la canción en BPM')
})

# 5. Define el modelo de respuesta de la API
prediction_response = api.model('PredictionResponse', {
    'popularity': fields.Float(description='Predicción de popularidad de la canción')
})

# 6. Define el recurso para realizar predicciones
@ns.route('/')
class SongPopularityPrediction(Resource):
    @ns.expect(song_features)
    @ns.marshal_with(prediction_response)
    def post(self):
        '''Realiza una predicción de popularidad de la canción utilizando las 5 variables más importantes.'''
        data = api.payload
        print(f"Datos recibidos: {data}")  # Imprime los datos de entrada
        try:
            # Crear un DataFrame con los datos de entrada (asegurándose del orden)
            input_df = pd.DataFrame([data])
            # Asegúrate de que el orden de las columnas coincide con el entrenamiento del modelo
            prediction = xgb_model.predict(input_df[['play_count', 'duration_ms', 'release_year', 'danceability', 'tempo']])[0]
            print(f"Predicción: {prediction}")  # Imprime la predicción antes de devolverla
            return {'popularity': float(prediction)}, 200
        except Exception as e:
            error_message = str(e)
            print(f"Error durante la predicción: {error_message}")  # Imprime el mensaje de error
            return {'message': f'Error al realizar la predicción: {error_message}', 'popularity': None}, 500

# 7. Ejecuta la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
