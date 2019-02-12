

# START
import tweepy
auth = tweepy.OAuthHandler("YzWFDST42FGUpQrqnFd6cT85u", "Ois5i7dau00xpluXjFPuU3Uu20UjDxd9KXbQjd7duXEO370iu8")
auth.set_access_token("1089235299862040576-gxsMbZg9nQgbooxCpsJ33ETDKb6IM5", "bSg2xKANcbIYhxEpL8OMwZIY7gXXbkc2wUkStxMSbU4K7")
api = tweepy.API(auth)


# PLACE
places = api.geo_search(query="USA", granularity="country")
place_id = places[0].id
print(place_id)

# clean
import nltk
nltk.download()
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
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

# get tweets
# CAN CHANGE THE QUERY WITH HASHTAG
# MAYBE NEED STREAMING PU IN ANOTHER WAY?

tweetsData = []
for tweet in tweepy.Cursor(api.search, q="place:%s" % place_id).items(15000):
    text = cleanAndNormalizeText(tweet.text)
    tweetsData.append([tweet.id, text])
trainTable = pd.DataFrame(tweetsData, columns=['ID', 'text'])
print(trainTable)