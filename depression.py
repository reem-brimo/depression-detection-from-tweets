import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import CountVectorizer
import os.path
import sqlite3 as db
import datetime

#pd.set_option('mode.chained_assignment', None)
file = "selected_data.json"
SAMPLE = 500
Debug = 1

def connect():
      con = db.connect('database.db')
      return con
def sql_table(cursorObj):
    cursorObj.execute("CREATE TABLE users(user_id integer PRIMARY KEY AUTOINCREMENT,name text unique)")
    cursorObj.execute("CREATE TABLE tweets(tweet_id integer PRIMARY KEY AUTOINCREMENT, tweet text, date date,length integer)")
    cursorObj.execute('''CREATE TABLE tweet_user(user_id integer not null,tweet_id integer not null)''')
    cursorObj.execute('''CREATE TABLE tweet_mentioned(tweet_id integer not null,mentioned_id integer not null)''')

def add_to_user(name,cursorObj):
     cursorObj.execute('''insert or ignore into users(name) values(?)''',(name,))
     result = cursorObj.execute('''select user_id from users where name = ?''',(name,) ).fetchone()
     return result[0]
def add_to_tweets(tweet,date,len,cursorObj):
     cursorObj.execute('''insert into tweets(tweet,date,length) values(?,?,?)''',(tweet,date,len))
     id = cursorObj.lastrowid
     return id

def add_to_tweet_mentioned(t_id,m_id,cursorObj):
     cursorObj.execute('''insert into tweet_mentioned(tweet_id,mentioned_id) values(?,?)''',(t_id,m_id))

def add_to_user_tweet(u_id,t_id,cursorObj):
     cursorObj.execute('''insert into tweet_user(user_id,tweet_id) values(?,?)''',(u_id,t_id))

      
def extract_mention(text):
    result = re.findall("@([a-zA-Z0-9_]{1,15})", text)
    text = re.sub("@([a-zA-Z0-9_]{1,15})","",text)
    text = re.sub("[_@]","",text)
    text = re.sub(' +',' ',text)
    return text,result

def process(text,username,date):
            tweet,mentioned = extract_mention(text)
            u_id = add_to_user(username,cursorObj)
            length = len(tweet.split())
            t_id = add_to_tweets(tweet,date,length,cursorObj)      
            add_to_user_tweet(u_id,t_id,cursorObj)
            for m in mentioned:
                m_id = add_to_user(m,cursorObj)
                add_to_tweet_mentioned(t_id,m_id,cursorObj)


def pre_processing(tweet):
    tweet = re.sub(r"(?:\|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = re.sub("&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6});", "",tweet) #remove html garbage
    tweet = tweet.replace("#", "") #Remove hashtag sign but keep the text
    tweet = re.sub(r"\'", "", tweet) 
   
    #replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?[\\]^`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    tweet = tweet.translate(translate_map)

    # convert text to lowercase
    tweet = tweet.strip().lower()
   
    return tweet

if Debug == 1:
        if os.path.exists("database.db"):
            os.remove("database.db")
    
        data_raw = pd.read_csv('../Data/training.1600000.processed.noemoticon.csv',encoding="iso-8859-1", header=None, usecols=[0,1,2,4,5], delimiter=",")
        data_raw.columns = ["label", "tweetId", "date", "username", "text"]


        data = data_raw[data_raw['label'] == 0]

        #words by montigomry to select tweets by
        with open("../input data/words.txt") as file:
            words = file.read().split()


        con = connect()
        cursorObj = con.cursor()
        sql_table(cursorObj)

        start = datetime.datetime.now()

        #selecting tweets
        data = data[data['text'].str.contains(r'\b|\b'.join(words))]
       
        # pre process tweets in vectorizing technique
        vect_pre_process = np.vectorize(pre_processing)
        data['text'] =  vect_pre_process(data['text'])
       
        #populating the database
        vect_process = np.vectorize(process)
        vect_process(data['text'],data['username'],data['date'])

        #number of usage of every word by every user (ranking)
        vectorizer = CountVectorizer()

        vectorizer.fit_transform(words)

        inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
        vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

        count_df = pd.DataFrame(
        data = vectorizer.transform(data['text']).toarray(),
        index = data['username'],
        columns = vocabulary
        )
        

        trans  = dict((c, 'sum') for c in words)

        count_df = count_df.groupby(count_df.index).agg(trans)
        count_df.to_sql("ranking",con= con,index=True)

        end = datetime.datetime.now()

        print('Duration:{}'.format(end-start)) 
        

        con.commit()
        cursorObj.close()
        con.close()
        #with open("file1.json", "w") as outfile: 
        # json.dump(dictionary,outfile)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)
        

#import nltk
#from nltk.tokenize import TweetTokenizer
#from nltk.tag import pos_tag
#from nltk.stem.wordnet import WordNetLemmatizer

#def tokenize(part_data): 
#    tk = TweetTokenizer(reduce_len = True)
#    X = part_data['text'].tolist()
#    data = []
#    for x in X:
#        data.append(tk.tokenize(x))
#    return data
#    #end tokenize
#def lemmatize_sentence(tokens):
#    lemmatizer = WordNetLemmatizer()
#    lemmatized_sentence = []
#    for word, tag in pos_tag(tokens):
#        # First, we will convert the pos_tag output tags to a tag format that
#        # the WordNetLemmatizer can interpret
#        # In general, if a tag starts with NN, the word is a noun and if it
#        # stars with VB, the word is a verb.
#        if tag.startswith('NN'):
#            pos = 'n'
#        elif tag.startswith('VB'):
#            pos = 'v'
#        else:
#            pos = 'a'
#        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
#    return lemmatized_sentence
       
