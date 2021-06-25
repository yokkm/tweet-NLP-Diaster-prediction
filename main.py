import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams, bigrams, trigrams
import re

pd.set_option('display.max_colwidth', None)

import warnings
warnings.filterwarnings('ignore')
# ---------------------------------- #

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# load and read data
# train
train = pd.read_csv('../input/nlp-getting-started/train.csv')
train = train.drop_duplicates(subset='text', keep='first', inplace=False)
train = train.drop(['id','keyword','location'], axis=1)

#test
test = pd.read_csv('../input/nlp-getting-started/test.csv')
test = test.drop_duplicates(subset='text', keep='first', inplace=False)
test = test.drop(['id','keyword','location'], axis=1)

#remove link
from urllib.parse import urlparse
def remove_link(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
# remove links
train['text'] = [' '.join(y for y in x.split() if not remove_link(y)) for x in train['text']]

# call stopwords
nltk.download('stopwords')
corpus = []

def clean_text(df,column):
    stop = stopwords.words('english')
    
    # to lower all column
    df['clean'] = df[column].str.lower()
    # --- extract hastag --- #
    df['hash_tag'] = df['clean'].str.findall(r'#.*?(?=\s|$)')
    # remove stop words
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    #--- regular expression keeping only letters --- #
    df['text'] = df['text'].str.replace('[^\w\s]','')
    
    return df

df = clean_text(train,'text')
df = df[['clean','target']]


# Get shape
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df['clean'])
X_train_counts.shape


#get training data set
X = X_train_counts.toarray()
y = df.iloc[:, 1].values

seed = 10
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# #--------- XGBOOST Classifier -----------#
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#given accuracy Accuracy: 78.01%

#--------- Random Forest -----------#
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)
y_pred_RF = classifier.predict(X_test)
accuracy_RF = accuracy_score(y_test, y_pred_RF)
print("Accuracy: %.2f%%" % (accuracy_RF * 100.0))
#given accuracy Accuracy: 79.08%
