import tweepy 

import numpy as np # untuk mengolah data skalar dan vector
import pandas as pd # untuk mengolah data tabular
from wordcloud import WordCloud # untuk visualisasi wordcloud

# liibrary nlp
import gensim # untuk pemodelan bahasa 
from gensim.models import Word2Vec # untuk pemodelan bahasa
from elang.plot.utils import plot2d, plotNeighbours # untuk visualisasi 
from elang.word2vec.utils import cleansing

#library deeplearning
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# library pembantu
import pickle # untuk membaca file binary
from tqdm import tqdm # untuk melihat progress
import re # untuk pencarian pola dalam teks
import helper # 

try:
    lang_model = Word2Vec.load('w2v_model')
except:
    lang_model = None

def userlist2df(user_list):

    userid = []
    username = []
    bio = []
    location = []
    following = []
    followers = []
    favorites = []
    lists = []
    tweets = []
    created_at = []
    protected = []
    verified = []

    for user in user_list:
        userid.append(user.id)
        username.append(user.screen_name)
        bio.append(user.description)
        location.append(user.location)
        following.append(user.friends_count)
        followers.append(user.followers_count)
        favorites.append(user.favourites_count)
        lists.append(user.listed_count)
        tweets.append(user.statuses_count)
        created_at.append(user.created_at)
        protected.append(user.protected)
        verified.append(user.verified)

    dataframe = pd.DataFrame(
        list(zip(userid, username, bio, location, following, followers, favorites, lists, tweets ,created_at, protected, verified)),
        columns=['userid', 'username', 'bio', 'loc', 'following', 'followers', 'favorites', 'lists', 'tweets', 'created_at', 'protected', 'verified'])

    return dataframe

def tweetlist2df(tweet_list):
    data = []
    for tweet in tqdm(tweet_list):
        
        if (x:=tweet.entities['user_mentions']) != []:
            mentions = [user['screen_name'] for user in x]
        else:
            mentions = []

        if (x:=tweet.entities['hashtags']) != []:
            hashtags = [tag['text'] for tag in x]
        else:
            hashtags = []
        
    
        text = tweet.full_text
        id = tweet.id
        userid = tweet.author.id
        username = tweet.author.screen_name
        created_at = tweet.created_at
        likes = tweet.favorite_count # could be 0
        retweets = tweet.retweet_count
        isquote = tweet.is_quote_status
        is_retweet = True if hasattr(tweet, 'retweeted_status') else False 
        reply_to = tweet.in_reply_to_status_id # could be None
        geo = tweet.geo
        place = tweet.place
        data.append([text, id, userid, username, created_at, likes, retweets, isquote, is_retweet, mentions, hashtags, reply_to, geo, place])
    dataframe = pd.DataFrame(data, columns=['text','id', 'userid', 'username', 'created_at', 'likes', 'retweets', 'isquote', 'is_retweet', 'mentions','hashtags','reply_to', 'geo', 'place' ])
    return dataframe

def clean_teks(teks):
    teks = teks.encode('ascii', 'ignore').decode('ascii') #menghilangkan emoji
    teks = cleansing.remove_stopwords_id(teks) # menghilangkan stopwords
    
    # Hilangakan tertawa     
    haha_pattern = '(?:a{0,2}h{1,2}a{0,2}){2,}h?'
    hehe_pattern = '(?:e{0,2}h{1,2}e{0,2}){2,}h?'
    hihi_pattern = '(?:i{0,2}h{1,2}i{0,2}){2,}h?'
    wkwk_pattern = r'\b[wk]*(?:wk|kw)[wk]*\b'
    
    #hilangkan link
    link_pattern = r'http\S+'
    
    #hilangkan tanda baca
    punct_pattern = r'[^\w\s]'
    
    teks = re.sub(f'{haha_pattern}|{hehe_pattern}|{hihi_pattern}|{wkwk_pattern}|{link_pattern}|{punct_pattern}', '', teks)
    
    teks = cleansing.remove_stopwords_id(teks)
    return teks

def word2token(word):
    try:
        return lang_model.wv.vocab[word].index
    except KeyError:
        return 0
    
def separate_rows(df, column):
    res = pd.DataFrame()
    
    for i, row in tqdm(df.iterrows()):
        items = row[column]
        n_items = len(items)
        if n_items == 0:
            row[column] = np.nan
            res= res.append(row)
        elif n_items == 1:
            row[column] = row[column][0]
            res= res.append(row)
        else:
            data = [row]*n_items
            for i in range(n_items):
                data[i][column] = items[i]
            res = res.append(data)
            
    return res


