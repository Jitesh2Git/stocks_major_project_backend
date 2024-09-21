from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nrclex import NRCLex
import numpy as np  # Use numpy instead of pandas
import pickle

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize FastAPI
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3000",
    "https://soft-stock.netlify.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading of models
model_files = {
    'XGB_close_classifier': "general_data_models/XGB_close_classifier.pkl",
    'XGB_trade_classifier': "general_data_models/XGB_trade_classifier.pkl",
    'xgb_regressor_trade': "general_data_models/xgb_regressor_trade.pkl",
    'xgb_regressor_close': "general_data_models/xgb_regressor_close.pkl",
}

models = {}

def load_model(model_name, model_path):
    if model_name not in models:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                models[model_name] = pickle.load(file)
                print(f"Loaded {model_name} successfully.")
        else:
            raise HTTPException(status_code=500, detail=f"{model_name} model not found.")
    return models[model_name]

# NLTK downloads (Lazy download, ensure the downloads happen once)
nltk.data.path.append("/nltk_data")  # Add path to avoid repeated downloads
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Initialize components
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# Initialize NER model lazily
ner_pipeline = None

def load_ner_pipeline():
    global ner_pipeline
    if not ner_pipeline:
        ner_pipeline = pipeline('ner', model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
    return ner_pipeline

# Define input data model
class InputData(BaseModel):
    headline: str
    ticker: int

# Ticker mappings
ticker_to_company = {
    0: 'AAPL',
    1: 'AMZN',
    2: 'MSFT',
    3: 'NVDA',
    4: 'TSLA'
}

# Helper functions
def preprocess_text(text):
    tokens = tokenizer.tokenize(text.lower())
    tagged_tokens = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_pos_tag(tag)) for token, tag in tagged_tokens if token not in stop_words]
    return " ".join(lemmatized_tokens)

def get_pos_tag(tag):
    return {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }.get(tag[0], wordnet.NOUN)  # Default to NOUN

def extract_features(text, ticker):
    # Load the NER pipeline lazily
    ner_pipeline = load_ner_pipeline()
    
    # Extract various text features
    blob = TextBlob(text).sentiment
    entities = ner_pipeline(text)
    emotion_scores = NRCLex(text).raw_emotion_scores

    tokens = text.split()
    positive_word_count = sum(1 for word in tokens if sia.polarity_scores(word)['compound'] > 0.05)
    negative_word_count = sum(1 for word in tokens if sia.polarity_scores(word)['compound'] < -0.05)

    features = {
        'Vader_sentiment_score': sia.polarity_scores(text)['compound'],
        'Blob_polarity': blob.polarity,
        'BlobSubjectivity': blob.subjectivity,
        'positive_word_count': positive_word_count,
        'negative_word_count': negative_word_count,
        'person_count': sum(1 for entity in entities if entity['entity_group'] == 'PER'),
        'organization_count': sum(1 for entity in entities if entity['entity_group'] == 'ORG'),
        'location_count': sum(1 for entity in entities if entity['entity_group'] == 'LOC'),
        'anger': emotion_scores.get('anger', 0),
        'anticipation': emotion_scores.get('anticipation', 0),
        'disgust': emotion_scores.get('disgust', 0),
        'fear': emotion_scores.get('fear', 0),
        'joy': emotion_scores.get('joy', 0),
        'sadness': emotion_scores.get('sadness', 0),
        'trust': emotion_scores.get('trust', 0),
        'Ticker': ticker
    }

    return np.array([list(features.values())])

# Asynchronous request handling
@app.post("/predict")
async def predict(data: InputData):
    try:
        preprocessed_text = preprocess_text(data.headline)
        features = extract_features(preprocessed_text, data.ticker)

        # Load models when required
        trade_classifier = load_model('XGB_trade_classifier', model_files['XGB_trade_classifier'])
        close_classifier = load_model('XGB_close_classifier', model_files['XGB_close_classifier'])
        trade_regressor = load_model('xgb_regressor_trade', model_files['xgb_regressor_trade'])
        close_regressor = load_model('xgb_regressor_close', model_files['xgb_regressor_close'])

        # Perform predictions
        prob_trade = float(trade_classifier.predict_proba(features)[0, 1])
        prob_close = float(close_classifier.predict_proba(features)[0, 1])
        pred_trade = float(trade_regressor.predict(features)[0])
        pred_close = float(close_regressor.predict(features)[0])

        return {
            "probability_trade": prob_trade,
            "probability_close": prob_close,
            "predicted_trade": pred_trade,
            "predicted_close": pred_close
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

