import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

airlines = pd.read_csv('Tweets.csv')

features = airlines['text'].values
labels = airlines['airline_sentiment'].values

def preprocess(features):
    processed_features = []
    for sentence in range(0,len(features)):
        processed_feature = re.sub(r'\W',' ',str(features[sentence]))    # Remove all the special characters
        processed_feature = re.sub(r'\s+[a-zA-z]\s+',' ',processed_feature)    # remove all single characters
        processed_feature = re.sub(r'\^[a-zA-z]\s+',' ', processed_feature)    # Remove single characters from the start
        processed_feature = re.sub(r'\s+', ' ',processed_feature,flags=re.I)    # Substituting multiple spaces with single space
        processed_feature = re.sub(r'^b\s+','',processed_feature)     # Removing prefixed 'b'
        processed_feature = processed_feature.lower()    # Convert to lowercase
        processed_features.append(processed_feature)
    return processed_features

processed_features = preprocess(features)

vec = TfidfVectorizer(max_features=2500, min_df=7,max_df = 0.8,stop_words=stopwords.words('english'))
processed_features = vec.fit_transform(processed_features).toarray()

X_train, X_test,y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

# Using Random Forest
classifier = RandomForestClassifier(n_estimators=200, random_state=0)
classifier.fit(X_train,y_train)
predict = classifier.predict(X_test)

# Using Support Vector Machine
svcclassifier = SVC()
svcclassifier.fit(X_train,y_train)
prediction = svcclassifier.predict(X_test)

# Random Forest F1-score
print('Random Forest F1-score')
print(classification_report(y_test,predict))
# SVM F1-score
print('SVM F1-score')
print(classification_report(y_test,prediction))
# Random Forest Accuracy score
print('Random Forest Accuracy score')
print(accuracy_score(y_test, predict))
# SVM Accuracy score
print('SVM Accuracy score')
print(accuracy_score(y_test, prediction))
