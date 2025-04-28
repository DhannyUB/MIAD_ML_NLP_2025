import joblib
import pandas as pd
from flask import Flask
from flask_restx import Api, Resource, fields
import numpy as np

# 1. Cargar el modelo XGBoost entrenado
try:
    xgb_model = joblib.load('xgb_popularity_model_rest.pkl')
except FileNotFoundError:
    print("Error: No se encontró el archivo del modelo XGBoost (xgb_popularity_model.pkl).")
    exit()

# 2. Inicializa la aplicación Flask
app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='MIAD 202512 | Song Popularity Prediction API | ADFS Team',
    description='''
    API para predecir la popularidad de canciones en Spotify usando un modelo XGBoost.
    
    ### Descripción
    Esta API utiliza un modelo de machine learning para predecir la popularidad de una canción (escala 0-100) basándose en cinco características principales.
    
    ### Características del modelo (ordenadas por importancia):
    1. play_count (51.43%): Número de reproducciones [≥ 0]
    2. release_year (37.99%): Año de lanzamiento [1900-2025]
    3. tempo (3.65%): Velocidad de la canción en BPM [0-250]
    4. duration_ms (3.63%): Duración en milisegundos [≥ 10000]
    5. danceability (3.30%): Índice de bailabilidad [0-1]
    
    ### Ejemplo de uso:
    ```json
    {
        "play_count": 1000000,
        "release_year": 2023,
        "tempo": 120,
        "duration_ms": 180000,
        "danceability": 0.8
    }
    ```
    
    ### Notas:
    - La predicción de popularidad siempre estará en el rango [0-100]
    - Todas las características son requeridas
    - Los valores deben estar dentro de los rangos especificados
    '''
)

# 3. Define el espacio de nombres para las predicciones
ns = api.namespace('predict', description='Predicciones de Popularidad de Canciones')

# 4. Define la estructura de los datos de entrada esperados (las 5 variables más comunes)
song_features = api.model('Custom5Features', {
    'play_count': fields.Float(required=True, description='Conteo de reproducciones - Rango: >= 0'),
    'release_year': fields.Integer(required=True, description='Año de lanzamiento - Rango: [1900, 2025]'),
    'tempo': fields.Float(required=True, description='Tempo de la canción en BPM - Rango: [0, 250]'),
    'duration_ms': fields.Integer(required=True, description='Duración de la canción en milisegundos - Rango: >= 10000'),
    'danceability': fields.Float(required=True, description='Bailabilidad de la canción - Rango: [0, 1]')
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
        '''Realiza una predicción de popularidad de la canción utilizando las 5 variables.'''
        data = api.payload
        print(f"Datos recibidos: {data}")  
        try:
            input_df = pd.DataFrame([data])
            prediction = xgb_model.predict(input_df[['play_count', 'release_year', 'tempo', 'duration_ms', 'danceability']])[0]
            # Ajustar la predicción al rango [0,100]
            prediction = np.clip(prediction, 0, 100)
            print(f"Predicción: {prediction}")  # Imprime la predicción antes de devolverla
            return {'popularity': float(prediction)}, 200
        except Exception as e:
            error_message = str(e)
            print(f"Error durante la predicción: {error_message}")  
            return {'message': f'Error al realizar la predicción: {error_message}', 'popularity': None}, 500

# 7. Ejecuta la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
