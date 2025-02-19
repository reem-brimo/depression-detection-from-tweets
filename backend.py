from flask import Blueprint, render_template, request
import pandas as pd
from datetime import datetime
import numpy as np
import tensorflow 
import numpy 
import pickle 
import joblib
import os
import tweepy, csv, re
import json
import re
import preprocessor as p
import contractions
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


second = Blueprint("second", __name__, static_folder="static", template_folder="template")


@second.route("/search")
def search():
    return render_template("search.html")


class GetInput:


    def __init__(self):
        self.tweets = []
        self.tweetText = []

    def get_tweets(self,user_name):

        
        data= pd.read_csv("static/test.csv")
        timeline = pd.DataFrame()
        timeline = data.loc[data['userName'] == user_name]
        return timeline

    def pre_processing(self, tweet):
 
        p.set_options(p.OPT.URL)
        tweet = p.clean(tweet) #Remove http links
        
        #expandContractions
        tweet = contractions.fix(tweet)

        tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)       #replace consecutive non-ASCII characters with a space
        tweet = re.sub("&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6});", "",tweet) #remove html garbage
        tweet = re.sub("@([a-zA-Z0-9_]{1,15})","",tweet)
        tweet = re.sub("([0-9])","",tweet)
        tweet = re.sub("#","", tweet)
        tweet=re.sub(r'(.)\1+', r'\1\1', tweet) 

        #remove punctuation characters
        filters='"$%&\'()*+-/,.:;<=>[]^`{|}~_' # just removed ?!
        translate_dict = dict((c, " ") for c in filters)
        translate_map = str.maketrans(translate_dict)
        tweet = tweet.translate(translate_map)

        # convert text to lowercase and remove repetations
        tweet = tweet.strip().lower()
        tweet = re.sub(r'\\n',' ',tweet)
        tweet = re.sub(r'\\t',' ',tweet)
        tweet = re.sub(r'\\r',' ',tweet)
        tweet = re.sub(' +',' ',tweet)
        tweet = re.sub('\?+','?',tweet)
        tweet = re.sub('\!+','!',tweet)
        return tweet
    
    def tokenize_padd(self, tweets):
        max = 62         
            
        tokenizer = Tokenizer() 
        # tokenizing based on "texts".
        # This step generates the word_index and map each word to an integer other than 0.
        tokenizer.fit_on_texts(tweets)

        # generating sequence based on tokenizer's word_index.
        # Each sentence will now be represented by combination of numericals
        seq = tokenizer.texts_to_sequences(tweets)

        word_index = tokenizer.word_index
        # padding each numerical representation of sentence to have fixed length.

        pad_x = pad_sequences(seq,max)
        #total_words = len(word_index) + 1
        
        return pad_x
    
    def get_user_timeline(self, user_name):
        # authenticating
        auth = tweepy.OAuthHandler(keys["consumer_key"], keys["consumer_secret"])
        auth.set_access_token(keys["access_key"], keys["access_secret"])
        api = tweepy.API(auth)

        # searching for tweets
        return tweepy.Cursor(api.user_timeline, screen_name=username, tweet_mode="extended").items()

    def predictResult(self, timeline, start, end):
  
        analyzed_tweets = pd.DataFrame(columns=[ "Tweet", "Datetime", "state"])
        depressed = 0
        # process start and end 
        start = datetime.strptime(start,'%Y-%m-%d')
        end = datetime.strptime(end,'%Y-%m-%d')

        date = []
        tweets = []
        
        timeline = timeline.reset_index()
        for index, row in timeline.iterrows():
            datet = datetime.strptime(row['created_at'],'%d %b %H:%M %Y')
            if  datet >= start and datet <= end:       
                date.append(datet)
                clean_tweet = self.pre_processing(row["tweet"])
                clean_tweet = clean_tweet[:62]
                tweets.append(clean_tweet)
            elif datet < start:
                break                   

        #self.tweetText = self.tokenize_padd(tweets)
        self.tweetText = self.tokenize_padd("i am so happy")

        from keras.models import load_model
        model = load_model('../static/Cnn.h5')
    
        print(model)

        # compile and evaluate loaded model
        model.compile(loss='MSE',optimizer='Adamax',metrics=['accuracy'])

        self.tweetText = numpy.array(self.tweetText)

        #ndarray
        prediction = model.predict(self.tweetText)
        prediction  = np.round(prediction.flatten())
 
        for i in prediction :
            if i == 1:
                depressed = depressed + 1


        print (prediction)

        depressed = depressed >= len(prediction)/2

        analyzed_tweets["state"] = prediction
        analyzed_tweets["Datetime"] = date
        analyzed_tweets["Tweet"] = tweets
        return  analyzed_tweets,depressed

  


@second.route('/results', methods=['POST','GET'])
def results():
    user_name = request.form.get('user_name')
    Start = request.form.get('from')
    End = request.form.get('to')
    result = GetInput()
    #timeline = result.get_user_timeline(user_name)
    timeline = result.get_tweets(user_name)
    table,depressed = result.predictResult(timeline, Start, End)

    return render_template('results.html', tables=[table.to_html()], titles=table.columns.values, depressed = depressed)





