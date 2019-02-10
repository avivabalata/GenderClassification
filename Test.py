
import nltk
#nltk.download()
import os
# import pandas as pd
# import numpy
# from os import listdir
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#import matplotlib.pyplot as plt
import tweepy

#######################################################################
auth = tweepy.OAuthHandler("YzWFDST42FGUpQrqnFd6cT85u", "Ois5i7dau00xpluXjFPuU3Uu20UjDxd9KXbQjd7duXEO370iu8")
auth.set_access_token("1089235299862040576-gxsMbZg9nQgbooxCpsJ33ETDKb6IM5", "bSg2xKANcbIYhxEpL8OMwZIY7gXXbkc2wUkStxMSbU4K7")

api = tweepy.API(auth)
public_tweets = api.home_timeline()
user = api.get_user('twitter')
# print(user.screen_name)
# print(user.followers_count)
# for friend in user.friends():
#    print(friend.screen_name)
# for tweet in public_tweets:
#     print(tweet.text)


######################################################## download data
# import requests
# data_url = "https://www.kaggle.com/crowdflower/twitter-user-gender-classification/downloads/gender-classifier-DFE-791531.csv"
# kaggle_info = {'UserName': "avevanes", 'Password': "zqPwpz0t?"}
# r = requests.get(data_url)
# r = requests.post(r.url, data=kaggle_info)
# f = open('gender-classifier-DFE-791531.csv','w')
# for chunk in r.iter_content(chunk_size=512 * 1024):
#     if chunk:
#         f.write(chunk)
# f.close()
#########################################################

stop_words = stopwords.words('english')
def cleanAndNormalizeText(data):
    # tokenize
    tokenize = word_tokenize(data)
    # remove stop words
    filterText = [w for w in tokenize if w not in stop_words]
    filterText = [w for w in filterText if not len(w) <= 1]

    # stem
    ps = PorterStemmer()
    for i in range(len(filterText)-1):
        if len(filterText[i]) > 1:
            try:
                filterText[i] = ps.stem(filterText[i])
            except Exception as e:
                filterText[i] = filterText[i]

    return ' '.join(filterText)
#########################################################

path = 'C:\\Users\\abalata\\Downloads\\gender-classifier-DFE-791531.csv'
Folder = os.listdir(path)
#for file in Folder:
fileReader = open(path + '\\gender-classifier-DFE-791531.csv', mode='r')
text = fileReader.read().lower()
fileReader.close()
Text = cleanAndNormalizeText(text)

# import csv
#
# with open('C:\\Users\\abalata\\Downloads\\gender-classifier-DFE-791531.csv\\\gender-classifier-DFE-791531.csv', 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         print(row)