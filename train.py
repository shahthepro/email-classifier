import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

import pickle

df = pd.read_csv("training_data.csv", usecols = ['class','text'])

# Define dimensions of the data
x = df["text"]
y = df["class"]

# Split data for training and testing
x_train,y_train = x[0:3000],y[0:3000]
x_test,y_test = x[3000:3050],y[3000:3050]

# Vectorize column
cv = CountVectorizer()  
features = cv.fit_transform(x_train.values.astype('U'))

# Create and train model
model = GridSearchCV(svm.SVC(), {})

model.fit(features,y_train)

# Save trained model
pickle.dump(model, open("model.pkl", 'wb'))
pickle.dump(cv, open("vectorizer.pkl", 'wb'))

# Print accuracy of the trained model
print(model.score(cv.transform(x_test),y_test))