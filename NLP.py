import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# delimiter means that the deciding factor is a tab
# quoting means that we decide to ignore the quote as deciding parameter

# cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
# loop for all the 1000 lines
for i in range(0, 1000):
  # to ignore all the letter except space, small and capital letters
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  # to lowercase all the letter
  review = review.lower()
  # to split str in array
  review = review.split()
  ps = PorterStemmer()
  # 'stopwords' to remove all the useless words like this, the
  # 'PorterStemmer' to trim all the words like change the past tense into present
  review = [ps.stem(word) for word in review if not word in (stopwords.words('english'))]
  # to join the array and make it a str
  review = ' '.join(review)
  corpus.append(review)

  
  
from sklearn.feature_extraction.text import CountVectorizer
# 'max_features = 1500' beacuse currently our bag of words contains 1565 coulumn 
#  some are useless like 'Rick' which apears only once,
#  so we will remove some words 
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
