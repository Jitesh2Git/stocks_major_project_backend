# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pandas as pd
# import pickle
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# from nltk.tokenize import TreebankWordTokenizer
# from transformers import pipeline
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from textblob import TextBlob
# from nrclex import NRCLex
# import os
# import warnings

# # Suppress specific warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Initialize FastAPI
# app = FastAPI()

# # CORS configuration
# origins = [
#     "http://localhost:3000",
#     "https://soft-stock.netlify.app"
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load models efficiently
# model_files = {
#     'XGB_close_classifier': "general_data_models/XGB_close_classifier.pkl",
#     'XGB_trade_classifier': "general_data_models/XGB_trade_classifier.pkl",
#     'xgb_regressor_trade': "general_data_models/xgb_regressor_trade.pkl",
#     'xgb_regressor_close': "general_data_models/xgb_regressor_close.pkl",
# }

# models = {}

# for model_name, model_path in model_files.items():
#     if os.path.exists(model_path):
#         with open(model_path, 'rb') as file:
#             models[model_name] = pickle.load(file)
#             print(f"Loaded {model_name} successfully.")
#     else:
#         raise HTTPException(status_code=500, detail=f"{model_name} model not found.")

# # Initialize NLTK and Transformers
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

# # Specify the model and aggregation strategy to avoid warnings
# ner_pipeline = pipeline('ner', model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
# sia = SentimentIntensityAnalyzer()
# lemmatizer = WordNetLemmatizer()
# tokenizer = TreebankWordTokenizer()
# stop_words = set(stopwords.words('english'))

# # Define input data model
# class InputData(BaseModel):
#     headline: str
#     ticker: int

# # Mapping ticker numbers to company names
# ticker_to_company = {
#     0: 'AAPL',
#     1: 'AMZN',
#     2: 'MSFT',
#     3: 'NVDA',
#     4: 'TSLA'
# }

# # Helper functions
# def preprocess_text(text):
#     tokens = tokenizer.tokenize(text.lower())
#     tagged_tokens = nltk.pos_tag(tokens)
#     lemmatized_tokens = [lemmatizer.lemmatize(token, get_pos_tag(tag)) for token, tag in tagged_tokens if token not in stop_words]
#     return " ".join(lemmatized_tokens)

# def get_pos_tag(tag):
#     return {
#         'J': wordnet.ADJ,
#         'V': wordnet.VERB,
#         'N': wordnet.NOUN,
#         'R': wordnet.ADV
#     }.get(tag[0], wordnet.NOUN)  # Return NOUN as the default

# def extract_features(text, ticker):
#     # Extract various text features
#     blob = TextBlob(text).sentiment
#     entities = ner_pipeline(text)
#     emotion_scores = NRCLex(text).raw_emotion_scores

#     tokens = text.split()
#     positive_word_count = sum(1 for word in tokens if sia.polarity_scores(word)['compound'] > 0.05)
#     negative_word_count = sum(1 for word in tokens if sia.polarity_scores(word)['compound'] < -0.05)

#     inputdf = pd.DataFrame([{
#         'Vader_sentiment_score': sia.polarity_scores(text)['compound'],
#         'Blob_polarity': blob.polarity,
#         'BlobSubjectivity': blob.subjectivity,
#         'positive_word_count': positive_word_count,
#         'negative_word_count': negative_word_count,
#         'person_count': sum(1 for entity in entities if entity['entity_group'] == 'PER'),
#         'organization_count': sum(1 for entity in entities if entity['entity_group'] == 'ORG'),
#         'location_count': sum(1 for entity in entities if entity['entity_group'] == 'LOC'),
#         'anger': emotion_scores.get('anger', 0),
#         'anticipation': emotion_scores.get('anticipation', 0),
#         'disgust': emotion_scores.get('disgust', 0),
#         'fear': emotion_scores.get('fear', 0),
#         'joy': emotion_scores.get('joy', 0),
#         'sadness': emotion_scores.get('sadness', 0),
#         'trust': emotion_scores.get('trust', 0),
#         'Ticker': ticker
#     }])

#     return inputdf

# @app.post("/predict")
# def predict(data: InputData):
#     try:
#         preprocessed_text = preprocess_text(data.headline)
#         features_df = extract_features(preprocessed_text, data.ticker)
#         features_df_numeric = features_df.select_dtypes(include=['int64', 'float64'])

#         # Perform predictions
#         prob_trade = float(models['XGB_trade_classifier'].predict_proba(features_df_numeric)[0, 1])
#         prob_close = float(models['XGB_close_classifier'].predict_proba(features_df_numeric)[0, 1])
#         pred_trade = float(models['xgb_regressor_trade'].predict(features_df_numeric)[0])
#         pred_close = float(models['xgb_regressor_close'].predict(features_df_numeric)[0])

#         return {
#             "probability_trade": prob_trade,
#             "probability_close": prob_close,
#             "predicted_trade": pred_trade,
#             "predicted_close": pred_close
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/")
# async def root():
#     return {"message": "Healthy"}

# @app.get("/")
# def read_root():
#     return {"message": "Service is running"}


from flask import Flask, request, jsonify
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nrclex import NRCLex
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Flask app
app = Flask(__name__)

# Load models
model_files = {
    'XGB_close_classifier': "general_data_models/XGB_close_classifier.pkl",
    'XGB_trade_classifier': "general_data_models/XGB_trade_classifier.pkl",
    'xgb_regressor_trade': "general_data_models/xgb_regressor_trade.pkl",
    'xgb_regressor_close': "general_data_models/xgb_regressor_close.pkl",
}

models = {}

for model_name, model_path in model_files.items():
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            models[model_name] = pickle.load(file)
            print(f"Loaded {model_name} successfully.")
    else:
        raise Exception(f"{model_name} model not found.")

# Initialize NLTK and Transformers
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Specify the model and aggregation strategy to avoid warnings
ner_pipeline = pipeline('ner', model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))

# Mapping ticker numbers to company names
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
    }.get(tag[0], wordnet.NOUN)  # Return NOUN as the default

def extract_features(text, ticker):
    # Extract various text features
    blob = TextBlob(text).sentiment
    entities = ner_pipeline(text)
    emotion_scores = NRCLex(text).raw_emotion_scores

    tokens = text.split()
    positive_word_count = sum(1 for word in tokens if sia.polarity_scores(word)['compound'] > 0.05)
    negative_word_count = sum(1 for word in tokens if sia.polarity_scores(word)['compound'] < -0.05)

    inputdf = pd.DataFrame([{
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
    }])

    return inputdf

# Define routes for Flask
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        headline = data['headline']
        ticker = data['ticker']

        preprocessed_text = preprocess_text(headline)
        features_df = extract_features(preprocessed_text, ticker)
        features_df_numeric = features_df.select_dtypes(include=['int64', 'float64'])

        # Perform predictions
        prob_trade = float(models['XGB_trade_classifier'].predict_proba(features_df_numeric)[0, 1])
        prob_close = float(models['XGB_close_classifier'].predict_proba(features_df_numeric)[0, 1])
        pred_trade = float(models['xgb_regressor_trade'].predict(features_df_numeric)[0])
        pred_close = float(models['xgb_regressor_close'].predict(features_df_numeric)[0])

        return jsonify({
            "probability_trade": prob_trade,
            "probability_close": prob_close,
            "predicted_trade": pred_trade,
            "predicted_close": pred_close
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Service is running"})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 
