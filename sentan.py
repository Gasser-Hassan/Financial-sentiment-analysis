import pandas as pd
#reading the file and naming the headers in the dataframe 
data = pd.read_csv('/Users/gasser/Downloads/archive/all-data.csv',encoding='latin1', header=None)
data.columns = ['sentiment', 'headline']
api_data = pd.read_csv('/Users/gasser/Desktop/senti/apiheadlines.csv')
new_data = pd.read_csv('/Users/gasser/Downloads/data.csv')
new_data.columns = ['headline', 'sentiment'] 
print(new_data.head())
#converting headlines to lowercase and removing special characters
data['cleaned_text'] = data['headline'].str.lower()
api_data['cleaned text'] = api_data['headlines'].str.lower()
new_data['cleaned_text'] = new_data['headline'].str.lower()

import re
data['cleaned_text'] = data['cleaned_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
api_data['cleaned text'] = api_data['cleaned text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
new_data['cleaned_text'] = new_data['cleaned_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
#tokenizing the words
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
data['tokenized_text'] = data['cleaned_text'].apply(word_tokenize)
api_data['tokenized_text'] = api_data['cleaned text'].apply(word_tokenize)
new_data['tokenized_text'] = new_data['cleaned_text'].apply(word_tokenize)
#removing stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['filtered_text'] = data['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])
api_data['filtered_text'] = api_data['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])
new_data['filtered_text'] = new_data['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])
#lemmatizing the words
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
data['lemmatized_text'] = data['filtered_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
api_data['lemmatized_text'] = api_data['filtered_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
new_data['lemmatized_text'] = new_data['filtered_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
# Join words back into a single string
data['processed_text'] = data['lemmatized_text'].apply(lambda x: ' '.join(x))
api_data['processed_text'] = api_data['lemmatized_text'].apply(lambda x: ' '.join(x))
new_data['processed_text'] = new_data['lemmatized_text'].apply(lambda x: ' '.join(x))
combined_data = pd.concat([data, new_data], ignore_index=True)

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

vectorizor = TfidfVectorizer(ngram_range=(1,2),max_features= 5000)
X = vectorizor.fit_transform(combined_data['processed_text'])
X_api = vectorizor.transform(api_data['processed_text'])

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizor, f)

#using sentiment analyzer to improve feature engineering
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
combined_data['vader_pos'] = combined_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['pos'])
combined_data['vader_neg'] = combined_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['neg'])
combined_data['vader_neu'] = combined_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['neu'])
combined_data['vader_compound'] = combined_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
api_data['vader_pos'] = api_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['pos'])
api_data['vader_neg'] = api_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['neg'])
api_data['vader_neu'] = api_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['neu'])
api_data['vader_compound'] = api_data['processed_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
#combining features
from scipy.sparse import hstack
api_vader_features = api_data[['vader_pos', 'vader_neg', 'vader_neu', 'vader_compound']].values
vader_features = combined_data[['vader_pos', 'vader_neg', 'vader_neu', 'vader_compound']].values
api_X_combined = hstack([X_api, api_vader_features])
X_combined = hstack([X, vader_features])
#converting sentiment to numerical format 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(combined_data['sentiment'])
data.to_csv('/Users/gasser/Desktop/senti/processed_data.csv', index=False)

#splitting testing, validation and training data
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(X_combined,y, test_size=0.2, random_state=42)

#the MODEL
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np 
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter

smote_tomek = SMOTETomek(random_state=42)
X_train_resample,y_train_resample = smote_tomek.fit_resample(X_train,y_train)
param_distributions = {
    'n_estimators': [100, 500, 1000, 1500],  # Number of trees
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size shrinkage
    'max_depth': [3, 6, 9],  # Maximum depth of a tree
    'subsample': [0.6, 0.8, 1.0],  # Fraction of samples used per tree
    'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features used per tree
    'gamma': [0, 1, 5],  # Minimum loss reduction required for a split
    'min_child_weight': [1, 5, 10],  # Minimum sum of instance weight in a child
}
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=1,
    random_state=42
)
'''random_search.fit(X_train,y_train)
print('best parameters:', random_search.best_params_)
print('best score:',random_search.best_score_ )
best_params = random_search.best_params_'''
final_model = XGBClassifier(
    subsample=0.8,
    n_estimators=500
    ,min_child_weight=1
    ,max_depth=3
    ,learning_rate=0.1
    ,gamma=1
    ,colsample_bytree=0.8
    , random_state=42)
final_model.fit(X_train_resample,y_train_resample)
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
trial_model = XGBClassifier(subsample=0.8,
    n_estimators=500
    ,min_child_weight=1
    ,max_depth=3
    ,learning_rate=0.1
    ,gamma=1
    ,colsample_bytree=0.8
    , random_state=42)

trial_model.fit(X_train,y_train)
trial_preds = trial_model.predict(X_valid)
final_predictions = final_model.predict(X_valid)
classification_report_final = classification_report(y_valid, final_predictions)
print('classification report after resample:', classification_report_final)
print('classification report without resample: ', classification_report(y_valid,trial_preds))
print("distribution before resample:",Counter(y_train))
print('distribution after resampling', Counter(y_train_resample))
api_preds = final_model.predict(api_X_combined)
print(api_preds)
#sentiment score
weights = {0:-1,1:0.5,2:1}
sentiment_counts = Counter(api_preds)
weighted_sum = sum(weights[sentiment] * count for sentiment, count in sentiment_counts.items())
total_sentiments = sum(sentiment_counts.values())
final_sentiment_score = weighted_sum / total_sentiments
print('final sentiment score:', final_sentiment_score)