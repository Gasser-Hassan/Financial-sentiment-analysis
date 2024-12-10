from flask import Flask, jsonify, request
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
from scipy.sparse import hstack
import re
from flask_cors import CORS
from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key='a96eabfb357e4ea8b342c583297df626')
app = Flask(__name__)
CORS(app)
def fetch_latest_headlines():
    topics = ['economy', 'finance', 'stocks', 'investment']
    all_headlines = []
    for topic in topics:
        results = newsapi.get_everything(
            q=topic,
            language='en',
            domains='wsj.com,bloomberg.com,bbc.com,theverge.com,reuters.com',
            sort_by='relevancy'
        )
        for article in results['articles']:
            all_headlines.append(article['title'])
    # Create DataFrame from headlines
    return pd.DataFrame(all_headlines, columns=['headlines'])

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
@app.route('/api/predict', methods=['POST'])

def predict_single_headline():
    try:
        # Get the headline from the request
        data = request.json
        headline = data.get('headline', '')

        if not headline:
            return jsonify({'error': 'Headline is required'}), 400

        # Preprocess the headline
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', headline.lower())
        tokenized_text = word_tokenize(cleaned_text)
        stop_words = set(stopwords.words('english'))
        filtered_text = [word for word in tokenized_text if word not in stop_words]
        lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]
        processed_text = ' '.join(lemmatized_text)

        # Transform using TF-IDF
        X_transformed = vectorizer.transform([processed_text])
        vader_features = [sia.polarity_scores(processed_text)['pos'],
                          sia.polarity_scores(processed_text)['neg'],
                          sia.polarity_scores(processed_text)['neu'],
                          sia.polarity_scores(processed_text)['compound']]
        X_combined = hstack([X_transformed, [vader_features]])

        # Predict sentiment
        prediction = int(model.predict(X_combined)[0])

        # Map prediction to sentiment description
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment_description = sentiment_map.get(prediction, "Unknown")

        return jsonify({
            'headline': headline,
            'prediction': prediction,
            'sentiment_description': sentiment_description
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/sentiment', methods=['GET'])
def calculate_sentiment():
    try:
        # Load the API headlines
        api_data = fetch_latest_headlines()

        # Preprocess the headlines (similar to your workflow)
        api_data['cleaned_text'] = api_data['headlines'].str.lower()
        api_data['cleaned_text'] = api_data['cleaned_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
        api_data['tokenized_text'] = api_data['cleaned_text'].apply(word_tokenize)
        stop_words = set(stopwords.words('english'))
        api_data['filtered_text'] = api_data['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])
        api_data['lemmatized_text'] = api_data['filtered_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
        api_data['processed_text'] = api_data['lemmatized_text'].apply(lambda x: ' '.join(x))
        # Transform using TF-IDF
        X_api = vectorizer.transform(api_data['processed_text'])
        api_data['vader_pos'] = api_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['pos'])
        api_data['vader_neg'] = api_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['neg'])
        api_data['vader_neu'] = api_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['neu'])
        api_data['vader_compound'] = api_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
        api_vader_features = api_data[['vader_pos', 'vader_neg', 'vader_neu', 'vader_compound']].values
        api_X_combined = hstack([X_api, api_vader_features])
        # Predict sentiments
        predictions = model.predict(api_X_combined)

        # Compute the sentiment score
        weights = {0: -1, 1: 0.5, 2: 1}  # Adjust weights as needed
        sentiment_counts = Counter(predictions)
        sentiment_counts = {int(k): v for k, v in sentiment_counts.items()}
        weighted_sum = sum(weights[sentiment] * count for sentiment, count in sentiment_counts.items())
        total_sentiments = sum(sentiment_counts.values())
        final_sentiment_score = weighted_sum / total_sentiments
        if final_sentiment_score >= 0.8:
                    sentiment_description = "Very Optimistic"
        elif 0.5 <= final_sentiment_score < 0.8:
                    sentiment_description = "Optimistic"
        elif 0.1 <= final_sentiment_score < 0.5:
                    sentiment_description = "Slightly Optimistic"
        elif -0.1 <= final_sentiment_score < 0.1:
                    sentiment_description = "Neutral"
        elif -0.5 <= final_sentiment_score < -0.1:
                    sentiment_description = "Slightly Pessimistic"
        elif -0.8 <= final_sentiment_score < -0.5:
                    sentiment_description = "Pessimistic"
        else:
                    sentiment_description = "Very Pessimistic"

        return jsonify({
                    'sentiment_score': round(final_sentiment_score, 2),
                    'sentiment_description': sentiment_description,
                    'sentiment_counts': dict(sentiment_counts)
                })
    except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)