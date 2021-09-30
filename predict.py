import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("test_data.csv")

x = df["text"]

model = pickle.load(open("model.pkl", 'rb'))
cv = pickle.load(open("vectorizer.pkl", 'rb'))

print(model.predict(cv.transform(x)))
