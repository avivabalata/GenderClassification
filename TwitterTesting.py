


import nltk
nltk.download()
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import csv

# clean
import re
import string
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via', '__']
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    # r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    # r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    s = re.sub(r'[^\x00-\x7f]*',r'',s)
    return tokens_re.findall(s)

def cleanAndNormalizeText(data):
    tokens = tokenize(data)
    tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    # tokenize
    # tokenize = word_tokenize(data)
    # remove stop words
    filterText = [w for w in tokens if w not in stop]
    filterText = [w for w in filterText if not len(w) <= 1]
    # stem
    ps = PorterStemmer()
    for i in range(len(filterText) - 1):
        if len(filterText[i]) > 1:
            try:
                filterText[i] = ps.stem(filterText[i])
            except Exception as e:
                filterText[i] = filterText[i]

    return filterText

terms_male = []
terms_female = []
terms_brand = []
genderClassData = []
with open('C:\\Users\\abalata\\Downloads\\gender-classifier-DFE-791531.csv\\gender-classifier-DFE-791531.csv','r', encoding="utf8" ,errors="ignore") as csvfile:
    reader = csv.reader(csvfile)
    count = 0
    for row in reader:
        if count != 0:
            text = row[10]+' '+row[13]+' '+row[18]+' '+row[19]
            cleantext = cleanAndNormalizeText(text)
            genderClassData.append([row[0], row[5], ' '.join(cleantext)])
            for token in cleantext:
                if token not in stop:
                    if row[5] == 'male':
                        terms_male.append(token)
                    else:
                        if row[5] == 'female':
                            terms_female.append(token)
                        else:
                            if row[5] == 'brand':
                                terms_brand.append(token)
        count = count+1


Table = pd.DataFrame(genderClassData, columns=['ID', 'gender', 'text'])
# print(Table)


classDistribution = []
for c in ['male', 'female', 'brand']:
    classDistribution.append([c, Table["gender"].value_counts()[c]])
pd.DataFrame(classDistribution, columns=['Gender', '# of Samples'])
pd

from collections import Counter
import pprint


count_male = Counter()
count_male.update(terms_male)
pp = pprint.PrettyPrinter()
print("Male most common:",sum(count_male.values()))
pp.pprint(count_male.most_common(20))


count_female = Counter()
count_female.update(terms_female)
print("Female most common:",sum(count_female.values()))
pp.pprint(count_female.most_common(20))

count_brand = Counter()
count_brand.update(terms_brand)
print("Brand most common:",sum(count_brand.values()))
pp.pprint(count_brand.most_common(20))



from sklearn.model_selection import train_test_split
x = Table.drop(['gender'], axis=1)
y = Table[["gender"]]
print(x)
print(y)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)
xTrain = xTrain.apply(lambda x: " ".join(x), axis=1)
xTest = xTest.apply(lambda x: " ".join(x), axis=1)
xTrain = xTrain.tolist()
xTest = xTest.tolist()
yTrain = yTrain['gender'].tolist()
yTest = yTest['gender'].tolist()


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# The classifying process
def classify(parameters, feature, ml):
            pipeline = Pipeline(steps=[('vect', feature), ('clf', ml)])
            gs_clf = GridSearchCV(pipeline, parameters, n_jobs=1)
            gs_clf = gs_clf.fit(xTrain, yTrain)
            prediction = gs_clf.predict(xTest)
            accuracy = metrics.accuracy_score(yTest, prediction)
            return [accuracy, gs_clf.best_params_]

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

#create the model
embedding_vecor_length = 32
max_review_length = 1000
model = Sequential()
model.add(Embedding(10, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=5, batch_size=128)



results = []
scores = model.evaluate(xTest, yTest, verbose=0)
results.append(['****', '***', '***', (scores[1]*100)])

parameters = {'vect__max_df': (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0), 'clf__alpha': (0.001, 0.01, 0.1, 1.0)} # understand why this numbers

# Using classify with Bag of Words and SVM algorithm
ans = classify(parameters, CountVectorizer(), SGDClassifier())
results.append(['Bag of Words', 'SVM', ans[1], ans[0]])

# Using classify with Bag of Words and NB algorithm
ans = classify(parameters, CountVectorizer(), MultinomialNB())
results.append(['Bag of Words', 'Naïve Bayes',  ans[1], ans[0]])

# Using classify with tf-idf and SVM algorithm
ans = classify(parameters, TfidfVectorizer(sublinear_tf=True, stop_words='english'), SGDClassifier())
results.append(['TF-IDF', 'SVM', ans[1], ans[0]])

# Using classify with tf-idf and NB algorithm
ans = classify(parameters, TfidfVectorizer(sublinear_tf=True, stop_words='english'), MultinomialNB())
results.append(['TF-IDF', 'Naïve Bayes', ans[1], ans[0]])

pd.DataFrame(results, columns=['feature extract', 'machine learning algorithm', 'parameters', 'accuracy'])

print(results)












#
# import sys
# print(sys.executable)
# print(sys.path)
#
# # START
# import tweepy
# auth = tweepy.OAuthHandler("YzWFDST42FGUpQrqnFd6cT85u", "Ois5i7dau00xpluXjFPuU3Uu20UjDxd9KXbQjd7duXEO370iu8")
# auth.set_access_token("1089235299862040576-gxsMbZg9nQgbooxCpsJ33ETDKb6IM5", "bSg2xKANcbIYhxEpL8OMwZIY7gXXbkc2wUkStxMSbU4K7")
# api = tweepy.API(auth)
#
#
# # PLACE
# places = api.geo_search(query="USA", granularity="country")
# place_id = places[0].id
# print(place_id)
#
# # clean
# import nltk
# nltk.download()
# import pandas as pd
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
#
#
# def cleanAndNormalizeText(data):
#     # tokenize
#     tokenize = word_tokenize(data)
#     # remove stop words
#     filterText = [w for w in tokenize if w not in stop_words]
#     filterText = [w for w in filterText if not len(w) <= 1]
#
#     # stem
#     ps = PorterStemmer()
#     for i in range(len(filterText)-1):
#         if len(filterText[i]) > 1:
#             try:
#                 filterText[i] = ps.stem(filterText[i])
#             except Exception as e:
#                 filterText[i] = filterText[i]
#
#     return ' '.join(filterText)
#
# # get tweets
# # CAN CHANGE THE QUERY WITH HASHTAG
# # MAYBE NEED STREAMING PU IN ANOTHER WAY?
#
#
# tweetsData = []
# for tweet in tweepy.Cursor(api.search, q="place:%s" % place_id).items(15000):
#     text = cleanAndNormalizeText(tweet.text)
#     tweetsData.append([tweet.id, text])
# trainTable = pd.DataFrame(tweetsData, columns=['ID', 'text'])
# print(trainTable)
