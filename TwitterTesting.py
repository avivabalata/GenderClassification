# import keras
# import nltk
#nltk.download()
# import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import csv
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import pprint
import tweepy
import json
#
import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics



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
    s = re.sub(r'[^\x00-\x7f]*',r'',str(s))
    return tokens_re.findall(s)

def cleanAndNormalizeText(data):
    tokens = tokenize(data)
    tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
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

locations = []
terms_male = []
terms_female = []
terms_brand = []
genderClassData = []
with open('C:\\Users\\abalata\\Downloads\\gender-classifier-DFE-791531.csv\\gender-classifier-DFE-791531.csv','r', encoding="utf8" ,errors="ignore") as csvfile:
    reader = csv.reader(csvfile)
    count = 0
    for row in reader:
        if count != 0:
            text = row[10]+' '+row[19]
            locations.append(row[25])
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


# class distribution
classDistribution = []
for c in ['male', 'female', 'brand']:
    classDistribution.append([c, Table["gender"].value_counts()[c]])
classDis = pd.DataFrame(classDistribution, columns=['Gender', '# of Samples'])
print(classDis)
count = set(classDis['# of Samples'])
his = classDis.hist(column="# of Samples", bins=len(count))
plt.title('Class Distribution')
plt.xlabel('count')
plt.ylabel('gender')
# plt.show(his)



count_country = Counter()
count_country.update(locations)
print(count_country.most_common(2))

count_male = Counter()
count_male.update(terms_male)
pp = pprint.PrettyPrinter()
print("Male most common:",sum(count_male.values()))
pp.pprint(count_male.most_common(50))


count_female = Counter()
count_female.update(terms_female)
print("Female most common:",sum(count_female.values()))
pp.pprint(count_female.most_common(50))

count_brand = Counter()
count_brand.update(terms_brand)
print("Brand most common:",sum(count_brand.values()))
pp.pprint(count_brand.most_common(50))

Table = Table.loc[Table['gender'].isin(['female','male'])]
x = Table['text']
y = Table["gender"]


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)
# xTrain = xTrain.apply(lambda x: " ".join(x), axis=1)
# xTest = xTest.apply(lambda x: " ".join(x), axis=1)
# xTrain = xTrain.tolist()
# xTest = xTest.tolist()
# yTrain = yTrain['gender'].tolist()
# yTest = yTest['gender'].tolist()

xTrainVector = TfidfVectorizer.fit_transform(xTrain)
xTestVector = TfidfVectorizer.fit_transform(xTest)

token = Tokenizer(num_words=3000)
token.fit_on_texts(x)
dic = token.word_index

# prepare to deep learning model
Train = []
Test = []
for data in xTrain:
    Train.append([dic[word] for word in kpt.text_to_word_sequence(data)])
for data in xTest:
    Test.append([dic[word] for word in kpt.text_to_word_sequence(data)])
Train = np.asarray(Train)
Test = np.asarray(Test)
xTrainMatx = token.sequences_to_matrix(Train, mode='binary')
xTestMatx = token.sequences_to_matrix(Test, mode='binary')
lb_make = LabelEncoder()
trainNum = lb_make.fit_transform(yTrain)
testNum = lb_make.fit_transform(yTest)
yTrainMatx = keras.utils.to_categorical(trainNum,3)
yTestMatx = keras.utils.to_categorical(testNum,3)


embedding_vecor_length = 32
max_review_length = 47016
model = Sequential()
model.add(Embedding(3000, 32))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.add(Dense(512, input_shape=(3000,), activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(256, input_shape=(3000,), activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, input_shape=(3000,), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(xTrainMatx, yTrainMatx,batch_size=128,epochs=5,validation_data=(xTestMatx, yTestMatx), verbose=2)
results = []
scores = model.evaluate(xTestMatx, yTestMatx, verbose=0)
results.append(['****', '***', '***', scores[1]])

xTrain = xTrain.apply(lambda x: " ".join(x))
xTest = xTest.apply(lambda x: " ".join(x))
xTrain = xTrain.tolist()
xTest = xTest.tolist()
yTrain = yTrain.tolist()
yTest = yTest.tolist()
def classify(parameters, feature, ml):
            gs_clf = Pipeline([('vect', feature), ('clf', ml)])
            # gs_clf = GridSearchCV(pipeline, parameters, n_jobs=1)
            gs_clf = gs_clf.fit(xTrain, yTrain)
            prediction = gs_clf.predict(xTest)
            accuracy = metrics.accuracy_score(yTest, prediction)
            return accuracy

parameters = {'vect__max_df': (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0), 'clf__alpha': (0.001, 0.01, 0.1, 1.0)} # understand why this numbers

# Using classify with Bag of Words and SVM algorithm
ans = classify(parameters, CountVectorizer(), SGDClassifier())
results.append(['Bag of Words SVM', ans])

# Using classify with Bag of Words and NB algorithm
ans = classify(parameters, CountVectorizer(), MultinomialNB())
results.append(['Bag of Words', 'Naïve Bayes',  ans[1], ans[0]])

# Using classify with tf-idf and SVM algorithm
ans = classify(parameters, TfidfVectorizer(sublinear_tf=True, stop_words='english'), SGDClassifier())
results.append(['TF-IDF', 'SVM', ans[1], ans[0]])

# Using classify with tf-idf and NB algorithm
ans = classify(parameters, TfidfVectorizer(sublinear_tf=True, stop_words='english'), MultinomialNB())
results.append(['TF-IDF Naïve Bayes', ans])

accuracy = pd.DataFrame(results, columns=['machine learning algorithm', 'accuracy'])

print(accuracy)



























# # START
auth = tweepy.OAuthHandler("YzWFDST42FGUpQrqnFd6cT85u", "Ois5i7dau00xpluXjFPuU3Uu20UjDxd9KXbQjd7duXEO370iu8")
auth.set_access_token("1089235299862040576-gxsMbZg9nQgbooxCpsJ33ETDKb6IM5", "bSg2xKANcbIYhxEpL8OMwZIY7gXXbkc2wUkStxMSbU4K7")
api = tweepy.API(auth,  wait_on_rate_limit=True)

# # get tweets
class MyStreamListener(tweepy.StreamListener):
    num_tweets = 0
    def on_data(self, data):
        if self.num_tweets < 15000:
                try:
                    with open('tweetsFromTwitter.json', 'a') as f:
                        f.write(data)
                        self.num_tweets += 1
                        return True
                except BaseException as e:
                    print("Error on_data: %s" % str(e))
                    return True
        else:
            return False

    def on_error(self, status):
        print(status)
        return True

    def _init_(self, api=None):
        super(MyStreamListener, self)._init_()
        self.num_tweets = 0

# geo location is: -162.8,28.2,-64.4,71.6 (US & CANADA)
myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
myStream.filter(locations=[-6.42, 49.86, 1.76, 55.81])


# read add tweets from twitter
tweetsTwitter = []
for row in open('tweetsFromTwitter.json','r'):
    try:
        tweet = json.loads(row)
        cleanTweet = cleanAndNormalizeText(tweet['text'])
        tweetsTwitter.append(cleanTweet)
    except:
        continue

# most popular terms from tweeter
terms = []
count = Counter()
for tweet in tweetsTwitter:
    for token in tweet:
        terms.append(token)
count.update(terms)
print(count.most_common(20))

predictions = []
for tweet in tweetsTwitter:
    if len(tweet) > 0:
        gs_clf = Pipeline([('vect', TfidfVectorizer(sublinear_tf=True, stop_words='english'), ), ('clf', MultinomialNB())])
        # gs_clf = GridSearchCV(pipeline, parameters, n_jobs=1)
        gs_clf = gs_clf.fit(xTrain, yTrain)
        prediction = gs_clf.predict(tweet)
        predictions.append([tweet, prediction])

Pred = pd.DataFrame(predictions,columns=['Tweet','Gender'])
print(Pred)
