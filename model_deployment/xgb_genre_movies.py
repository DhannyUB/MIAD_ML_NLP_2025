from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from datetime import datetime

# Ensure NLTK resources are downloaded
nltk_resources = ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4', 'stopwords']
for res in nltk_resources:
    try:
        nltk.data.find(f'corpora/{res}')
    except LookupError:
        nltk.download(res)
stop_words_nltk = set(stopwords.words('english'))

# --- Preprocessing Functions ---
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    return {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    text = re.sub(r'\W+', ' ', str(text).lower())
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens if token not in stop_words_nltk]

# --- Load Models and Preprocessors ---
try:
    loaded_model = joblib.load('movie_genre_model.pkl')
    loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_mlb = joblib.load('multilabelbinarizer.pkl')
except FileNotFoundError as e:
    print(f"Error loading model or preprocessor: {e}")
    exit()

# --- Flask App Setup ---
app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='MIAD 202512 | Movie Genre Prediction API | ADFS Team',
    description='''
    **Movie Genre Prediction API**

    This API uses a machine learning model to predict the most likely genres of a movie based on its plot and release year.

    ### Description
    The model was trained using TF-IDF features from movie plots, normalized release year, and keyword-based genre flags. It outputs probabilities for multiple genres.

    ### Input Fields:
    - `title` (string): Title of the movie
    - `plot` (string): Plot or synopsis of the movie
    - `year` (integer): Release year (between 1900 and current year)

    ### Output:
    - 'interpretation': A summary of the most probable genre prediction
    - `genres_probability`: Dictionary of predicted probabilities for each genre


    ### Example Input:
    ```json
    {
        "title": "Inception",
        "plot": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea. If he succeeds, it will be the perfect crime, but a dangerous enemy anticipates Cobb's every move.",
        "year": 2010
    }
    ```

    ### Notes:
    - The model supports multi-label classification (a movie can belong to multiple genres).
    - All fields are required.
    '''
)
ns = api.namespace('predict', description='Genre Predictions')

genre_keywords = {
    'Action': 'Action', 'Adventure': 'Adventure', 'Animation': 'Animation', 'Biography': 'Biography',
    'Comedy': 'Comedy', 'Crime': 'Crime', 'Documentary': 'Documentary', 'Drama': 'Drama',
    'Family': 'Family', 'Fantasy': 'Fantasy', 'Film-Noir': 'Film-Noir', 'History': 'History',
    'Horror': 'Horror', 'Music': 'Music', 'Musical': 'Musical', 'Mystery': 'Mystery',
    'News': 'News', 'Romance': 'Romance', 'Sci-Fi': 'Sci-Fi', 'Short': 'Short',
    'Sport': 'Sport', 'Thriller': 'Thriller', 'War': 'War', 'Western': 'Western'
}

movie_input_features = api.model('MovieInput', {
    'title': fields.String(required=True, description='Movie title'),
    'plot': fields.String(required=True, description='Movie plot'),
    'year': fields.Integer(required=True, description='Release year')
})


prediction_response = api.model('PredictionResponse', {
    'interpretation': fields.String(description='Message most probable genre'), 
    #'highest_probability_genre': fields.String(description='Genre with the highest probability'),
    #'highest_probability': fields.Float(description='Highest probability'),
    'genres_probability': fields.Raw(description='Predicted probabilities by genre'),
})

@ns.route('/')
class MovieGenrePrediction(Resource):
    @ns.expect(movie_input_features)
    @ns.marshal_with(prediction_response)
    def post(self):
        '''Predicts the probabilities of movie genres.'''
        data = api.payload
        plot = data.get('plot', '').strip()
        year = data.get('year')

        # Input validation
        if not plot:
            return {'message': 'The "plot" field cannot be empty.'}, 400
        if not isinstance(year, int) or not (1900 <= year <= datetime.now().year):
            return {'message': f'The "year" field must be between 1900 and {datetime.now().year}.'}, 400

        try:
            lemmatized_plot = lemmatize_text(plot)
            tfidf_input = loaded_vectorizer.transform([' '.join(lemmatized_plot)])
            numeric_input = loaded_scaler.transform(np.array([[year]]))
            numeric_sparse = csr_matrix(numeric_input)

            input_has_features = pd.DataFrame([
                {f'has_{genre}': 1 if keyword.lower() in plot.lower() else 0
                 for genre, keyword in genre_keywords.items()}
            ])
            input_has_sparse = csr_matrix(input_has_features.values)

            model_input = hstack([input_has_sparse, numeric_sparse, tfidf_input])

            probabilities = loaded_model.predict_proba(model_input)
            genre_probabilities = {
                loaded_mlb.classes_[i]: float(probabilities[i][0][1])
                for i in range(len(loaded_mlb.classes_))
            }

            sorted_probabilities = dict(sorted(genre_probabilities.items(), key=lambda item: item[1], reverse=True))
            highest_probability_genre = next(iter(sorted_probabilities))
            highest_probability = sorted_probabilities[highest_probability_genre]


            interpretation = (
                f"The genre of the movie is most likely '{highest_probability_genre}' "
                f"with a probability of {highest_probability:.0%}."
            )


            return {
                #'highest_probability_genre': highest_probability_genre,
                #highest_probability': highest_probability,
                'genres_probability': sorted_probabilities,
                'interpretation': interpretation
            }, 200

        except Exception as e:
            #import traceback
            #traceback.print_exc()  
            return {'message': f'Error predicting: {str(e)}'}, 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
