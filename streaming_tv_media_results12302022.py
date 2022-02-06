#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode()
#from jupyter_plotly_dash import JupyterDash
import plotly.graph_objects as go 
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
#from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, dcc
#from dash import html
import dash_html_components as html
#from dash import Input, Output, State, dcc, html
#import app
#from app import streaming_tv_media_results_1262022
#import server
#from dash import Input, Output, State, dcc
#from jupyterlab_dash import AppViewer
from datetime import date
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
import csv 
import random
from gnewsclient import gnewsclient
import plotly.express as px
from datetime import datetime
import pandas as pd 
import numpy as np 
from time import time
import os
import locale
import string
import tweepy as tw
from os import path
from PIL import Image
import matplotlib.pyplot as plt
import re
#from bs4 import BeautifulSoup
from tqdm import tqdm
from textblob import TextBlob
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()
#stopit = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
stopit = nltk.download('stopwords')
from nltk import corpus
from nltk import word_tokenize
lemmatizer = nltk.stem.WordNetLemmatizer()
from nltk.stem.snowball import SnowballStemmer
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,classification_report
from scikitplot.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib
from autocorrect import Speller
from autocorrect.word_count import count_words
from spellchecker import SpellChecker
from textblob.en import Spelling    
from PyDictionary import PyDictionary
dictionary=PyDictionary()
from os import path
from PIL import Image
import matplotlib.pyplot as plt
from english_words import english_words_set
import wordsegment
from nltk.corpus import words
nltk.download('words')
words_me = set(nltk.corpus.words.words())
import time 
import pkg_resources
#from symspellpy import SymSpell, Verbosity
from spellchecker import SpellChecker
from nltk.chunk import conlltags2tree
from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import warnings
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[ ]:


my_api_key = "Hzbb0sXPSUX5yFXRozdmdeqSX"
my_api_secret = "j2kIHynH5tqE4RV6lMC602K4HQKbTlVbTiguFj7R4UgeAHVpVV"
# authenticate
auth = tw.OAuthHandler(my_api_key, my_api_secret)
api = tw.API(auth, wait_on_rate_limit=False)


# In[ ]:


usernetflix = "netflix"
netflix = api.user_timeline(screen_name=usernetflix, 
                           count=200,
                           include_rts = True,
                           tweet_mode = 'extended')
netflix_df = pd.DataFrame()


# In[ ]:


for tweet in netflix:
    hashtags = []
    try:
        for hashtag in tweet.entities["hashtags"]:
            hashtags.append(hashtag["text"])
        text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
    except:
        pass
    netflix_df = netflix_df.append(pd.DataFrame({'user_name': tweet.user.name, 
                                               'user_location': tweet.user.location,\
                                               'user_description': tweet.user.description,
                                               'user_verified': tweet.user.verified,
                                               'date': tweet.created_at,
                                               'text': text, 
                                               'hashtags': [hashtags if hashtags else None],
                                               're_tweet': tweet.retweet,
                                               'source': tweet.source}))
    netflix_df = netflix_df.reset_index(drop=True)


# In[ ]:


userhulu = "hulu"
hulu = api.user_timeline(screen_name=userhulu, 
                           count=200,
                           include_rts = True,
                           tweet_mode = 'extended')
hulu_df = pd.DataFrame()


# In[ ]:


for tweet in hulu:
    hashtags = []
    try:
        for hashtag in tweet.entities["hashtags"]:
            hashtags.append(hashtag["text"])
        text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
    except:
        pass
    hulu_df = hulu_df.append(pd.DataFrame({'user_name': tweet.user.name, 
                                               'user_location': tweet.user.location,\
                                               'user_description': tweet.user.description,
                                               'user_verified': tweet.user.verified,
                                               'date': tweet.created_at,
                                               'text': text, 
                                               'hashtags': [hashtags if hashtags else None],
                                               're_tweet': tweet.retweet,
                                               'source': tweet.source}))
    hulu_df = hulu_df.reset_index(drop=True)


# In[ ]:


usertubi = "tubi"
tubi = api.user_timeline(screen_name=usertubi, 
                           count=200,
                           include_rts = True,
                           tweet_mode = 'extended')
tubi_df = pd.DataFrame()


# In[ ]:


for tweet in tubi:
    hashtags = []
    try:
        for hashtag in tweet.entities["hashtags"]:
            hashtags.append(hashtag["text"])
        text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
    except:
        pass
    tubi_df = tubi_df.append(pd.DataFrame({'user_name': tweet.user.name, 
                                               'user_location': tweet.user.location,\
                                               'user_description': tweet.user.description,
                                               'user_verified': tweet.user.verified,
                                               'date': tweet.created_at,
                                               'text': text, 
                                               'hashtags': [hashtags if hashtags else None],
                                               're_tweet': tweet.retweet,
                                               'source': tweet.source}))
    tubi_df = tubi_df.reset_index(drop=True)


# In[ ]:


def preprocess(word):
    word=str(word)
    word = word.lower()
    word=word.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', word)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopit]
    return " ".join(filtered_words)


# In[ ]:


netflix_df = netflix_df.fillna({'hashtags':' '})
hulu_df = hulu_df.fillna({'hashtags':' '})
tubi_df = tubi_df.fillna({'hashtags':' '})


#netflix_df['english_text'] = netflix_df['text'].map(lambda s:preprocess(s))
#hulu_df['english_text'] = hulu_df['text'].map(lambda s:preprocess(s))
#tubi_df['english_text'] = tubi_df['text'].map(lambda s:preprocess(s))

netflix_df['english_text']= netflix_df['text'].map(str)
hulu_df['english_text']= hulu_df['text'].map(str)
tubi_df['english_text']= tubi_df['text'].map(str)



#netflix_df['english_text']= netflix_df['english_text'].map(str)
#hulu_df['english_text']= hulu_df['english_text'].map(str)
#tubi_df['english_text']= tubi_df['english_text'].map(str)



spell = Speller(fast = True)
netflix_df['english_text'] = [' '.join([spell(i) for i in x.split()]) for x in netflix_df['english_text']]
hulu_df['english_text'] = [' '.join([spell(i) for i in x.split()]) for x in hulu_df['english_text']]
tubi_df['english_text'] = [' '.join([spell(i) for i in x.split()]) for x in tubi_df['english_text']]


netflix_df['tokens_text'] = netflix_df['english_text'].apply(word_tokenize)
hulu_df['tokens_text'] = hulu_df['english_text'].apply(word_tokenize)
tubi_df['tokens_text'] = tubi_df['english_text'].apply(word_tokenize)


# In[ ]:


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

netflix_df['lemma_text'] = netflix_df['english_text'].apply(lemmatize_text)
hulu_df['lemma_text'] = hulu_df['english_text'].apply(lemmatize_text)
tubi_df['lemma_text'] = tubi_df['english_text'].apply(lemmatize_text)


# In[ ]:


netflixexplain = netflix_df
huluexplain = hulu_df
tubiexplain = tubi_df


# In[ ]:


netflixexplain['words'] = netflixexplain['english_text']
huluexplain['words'] = huluexplain['english_text']
tubiexplain['words'] = tubiexplain['english_text']


netflixexplain['word_count'] = netflixexplain['words'].apply(lambda x: len(str(x).split(" ")))
huluexplain['word_count'] = huluexplain['words'].apply(lambda x: len(str(x).split(" ")))
tubiexplain['word_count'] = tubiexplain['words'].apply(lambda x: len(str(x).split(" ")))



netflixexplain['char_count'] = netflixexplain['words'].str.len()
huluexplain['char_count'] = huluexplain['words'].str.len()
tubiexplain['char_count'] = tubiexplain['words'].str.len()


# In[ ]:


from textblob import Word
netflixexplain['words'] = netflixexplain['words'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
huluexplain['words'] = huluexplain['words'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
tubiexplain['words'] = tubiexplain['words'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[ ]:


netflixexplain['sentiment'] = netflixexplain['words'].apply(lambda x: TextBlob(x).sentiment[0] )
huluexplain['sentiment'] = huluexplain['words'].apply(lambda x: TextBlob(x).sentiment[0] )
tubiexplain['sentiment'] = tubiexplain['words'].apply(lambda x: TextBlob(x).sentiment[0] )


# In[ ]:


conditions = [
    (netflixexplain['sentiment'] == 0.0),
    (netflixexplain['sentiment'] <= -0.1),
    (netflixexplain['sentiment'] >= -0.1)]
   
values = ['Netural', 'Negative', 'Positive',]

netflixexplain['results'] = np.select(conditions, values)


# In[ ]:


conditions = [
    (huluexplain['sentiment'] == 0.0),
    (huluexplain['sentiment'] <= -0.1),
    (huluexplain['sentiment'] >= -0.1)]
   
values = ['Netural', 'Negative', 'Positive',]

huluexplain['results'] = np.select(conditions, values)


# In[ ]:


conditions = [
    (tubiexplain['sentiment'] == 0.0),
    (tubiexplain['sentiment'] <= -0.1),
    (tubiexplain['sentiment'] >= -0.1)]
   
values = ['Netural', 'Negative', 'Positive',]

tubiexplain['results'] = np.select(conditions, values)


# In[ ]:


netflixexplain['hashtagscloud'] = netflixexplain['hashtags']
#netflixexplain['words']

#netflixexplainvalues = netflixexplain['hashtagscloud'].value_counts()

huluexplain['hashtagscloud'] = huluexplain['hashtags']
#huluexplainvalues = huluexplain['hashtagscloud'].value_counts()

tubiexplain['hashtagscloud'] = tubiexplain['hashtags']
#tubiexplainvalues = tubiexplain['hashtagscloud'].value_counts()


# In[ ]:


mediadata = pd.concat([netflixexplain,huluexplain,tubiexplain],ignore_index=True)


# In[ ]:


twitterdata = mediadata[['user_name','date','english_text','lemma_text','words','word_count','char_count','sentiment','results','hashtagscloud']]


# In[ ]:


netflixfig = px.bar(netflixexplain, y="results", orientation='h', title='Netflix Sentiments', hover_data=["source", "words", "hashtags"],  color_discrete_sequence=["#FB0D0D"])
netflixfig.update_layout(title_font_color="#FB0D0D")
hulufig = px.bar(huluexplain, y="results", orientation='h', title='Hulu Sentiments', hover_data=["source", "words", "hashtags"], color_discrete_sequence=["#109618"])
hulufig.update_layout(title_font_color="#109618")
tubifig = px.bar(tubiexplain, y="results", orientation='h', title='Tubi Sentiments', hover_data=["source", "words", "hashtags"], color_discrete_sequence=["#FE00CE"])
tubifig.update_layout(title_font_color="#FE00CE")


# In[ ]:


netflixhashwords = netflixexplain['hashtagscloud'].astype(str)
netflix_wordcloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(netflixhashwords))
fig_netflix = px.imshow(netflix_wordcloud, title="Netflix Hashtags")
fig_netflix.update_xaxes(visible=False)
fig_netflix.update_yaxes(visible=False)
fig_netflix.update_layout(title_font_color="#FB0D0D")
#fig_netflix.show()

huluhashwords = huluexplain['hashtagscloud'].astype(str)
hulu_wordcloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(huluhashwords))
fig_hulu = px.imshow(hulu_wordcloud, title="Hulu Hashtags")
fig_hulu.update_xaxes(visible=False)
fig_hulu.update_yaxes(visible=False)
fig_hulu.update_layout(title_font_color="#109618")
#fig_hulu.show()

tubihashwords = tubiexplain['hashtagscloud'].astype(str)
tubi_wordcloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(tubihashwords))
fig_tubi = px.imshow(tubi_wordcloud, title="Tubi Hashtags")
fig_tubi.update_xaxes(visible=False)
fig_tubi.update_yaxes(visible=False)
fig_tubi.update_layout(title_font_color="#FE00CE")
#fig_tubi.show()


# In[ ]:


netflixwords = netflixexplain['lemma_text'].astype(str)
netflixwordscloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(netflixwords))
fig_netflixwords = px.imshow(netflixwordscloud, title="Netflix Twitter Words")
fig_netflixwords.update_xaxes(visible=False)
fig_netflixwords.update_yaxes(visible=False)
fig_netflixwords.update_layout(title_font_color="#FB0D0D")
#fig_netflixwords.show()

huluwords = huluexplain['lemma_text'].astype(str)
huluwordscloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(huluwords))
fig_huluwords = px.imshow(huluwordscloud, title="Hulu Twitter Words")
fig_huluwords.update_xaxes(visible=False)
fig_huluwords.update_yaxes(visible=False)
fig_huluwords.update_layout(title_font_color="#109618")
#fig_huluwords.show()

tubiwords = tubiexplain['lemma_text'].astype(str)
tubiwordscloud = WordCloud(background_color='white', height=275, width=400).generate(' '.join(tubiwords))
fig_tubiwords = px.imshow(tubiwordscloud, title="Tubi Twitter Words")
fig_tubiwords.update_xaxes(visible=False)
fig_tubiwords.update_yaxes(visible=False)
fig_tubiwords.update_layout(title_font_color="#FE00CE")
#fig_tubiwords.show()


# In[ ]:


netflixsfig = px.bar(netflixexplain, y="source", orientation='h', title='Netflix Source Sentiments', hover_data=[ "words", "hashtags"],  color_discrete_sequence=["#FB0D0D"])
netflixsfig.update_layout(title_font_color="#FB0D0D")
hulusfig = px.bar(huluexplain, y="source", orientation='h', title='Hulu Source Sentiments', hover_data=[ "words", "hashtags"], color_discrete_sequence=["#109618"])
hulusfig.update_layout(title_font_color="#109618")
tubisfig = px.bar(tubiexplain, y="source", orientation='h', title='Tubi Source Sentiments', hover_data=[ "words", "hashtags"], color_discrete_sequence=["#FE00CE"])
tubisfig.update_layout(title_font_color="#FE00CE")


# In[ ]:


netflixclient = gnewsclient.NewsClient(language='hindi', location='india', topic='Business', max_results=5)
huluclient = gnewsclient.NewsClient(language='english', location='usa', topic='HULU', max_results=5)
tubiclient = gnewsclient.NewsClient(language='english', location='usa', topic='TUBI', max_results=5)


# In[ ]:


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# In[ ]:


app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Social Media', children=[
            dbc.Row(dbc.Col(html.H4('Streaming TV Media Twitter Results'))), 
                dbc.Row([dbc.Col([dcc.Graph(figure=netflixsfig,  style={'height':300, 'width':400}),],width={'size': 4, 'offset': 0, 'order': 1}), 
             dbc.Col([dcc.Graph(figure=hulusfig,  style={'height':300, 'width':400}), ],width={'size': 4, 'offset': 0, 'order': 2}),      
             dbc.Col([dcc.Graph(figure=tubisfig,  style={'height':300, 'width':400}),], width={'size': 4, 'offset': 0, 'order': 3}), ]),  
            
             dbc.Row([dbc.Col([dcc.Graph(figure=netflixfig, style={'height':300, 'width':400}),],width={'size': 4, 'offset': 0, 'order': 1}), 
             dbc.Col([dcc.Graph(figure=hulufig, style={'height':300, 'width':400}), ],width={'size': 4, 'offset': 0, 'order': 2}),      
             dbc.Col([dcc.Graph(figure=tubifig, style={'height':300, 'width':400}),], width={'size': 4, 'offset': 0, 'order': 3}), ]),  
            
             dbc.Row([dbc.Col([dcc.Graph(figure=fig_netflix, style={'height':300, 'width':400}),],width={'size': 4, 'offset': 0, 'order': 1}), 
             dbc.Col([dcc.Graph(figure=fig_hulu, style={'height':300, 'width':400}), ],width={'size': 4, 'offset': 0, 'order': 2}),      
             dbc.Col([dcc.Graph(figure=fig_tubi, style={'height':300, 'width':400}),], width={'size': 4, 'offset': 0, 'order': 3}), ]),  
            
             dbc.Row([dbc.Col([dcc.Graph(figure=fig_netflixwords, style={'height':300, 'width':400}),],width={'size': 4, 'offset': 0, 'order': 1}), 
             dbc.Col([dcc.Graph(figure=fig_huluwords, style={'height':300, 'width':400}), ],width={'size': 4, 'offset': 0, 'order': 2}),      
             dbc.Col([dcc.Graph(figure=fig_tubiwords, style={'height':300, 'width':400}),], width={'size': 4, 'offset': 0, 'order': 3}), ])
        ]),
        dcc.Tab(label='Resume', children=[
          dbc.Row(dbc.Col(html.H4('Naeemah Aliya Small'))), 
                dbc.Row([dbc.Col([dcc.Graph(figure=netflixsfig,  style={'height':300, 'width':400}),],width={'size': 4, 'offset': 0, 'order': 1}), 
             dbc.Col([dcc.Graph(figure=hulusfig,  style={'height':300, 'width':400}), ],width={'size': 4, 'offset': 0, 'order': 2}),      
             dbc.Col([dcc.Graph(figure=tubisfig,  style={'height':300, 'width':400}),], width={'size': 4, 'offset': 0, 'order': 3}), ]),  
            
             dbc.Row([dbc.Col([dcc.Graph(figure=netflixfig, style={'height':300, 'width':400}),],width={'size': 4, 'offset': 0, 'order': 1}), 
             dbc.Col([dcc.Graph(figure=hulufig, style={'height':300, 'width':400}), ],width={'size': 4, 'offset': 0, 'order': 2}),      
             dbc.Col([dcc.Graph(figure=tubifig, style={'height':300, 'width':400}),], width={'size': 4, 'offset': 0, 'order': 3}), ]),  
            
             dbc.Row([dbc.Col([dcc.Graph(figure=fig_netflix, style={'height':300, 'width':400}),],width={'size': 4, 'offset': 0, 'order': 1}), 
             dbc.Col([dcc.Graph(figure=fig_hulu, style={'height':300, 'width':400}), ],width={'size': 4, 'offset': 0, 'order': 2}),      
             dbc.Col([dcc.Graph(figure=fig_tubi, style={'height':300, 'width':400}),], width={'size': 4, 'offset': 0, 'order': 3}), ]),  
            
             dbc.Row([dbc.Col([dcc.Graph(figure=fig_netflixwords, style={'height':300, 'width':400}),],width={'size': 4, 'offset': 0, 'order': 1}), 
             dbc.Col([dcc.Graph(figure=fig_huluwords, style={'height':300, 'width':400}), ],width={'size': 4, 'offset': 0, 'order': 2}),                   
                      dbc.Col([dcc.Graph(figure=fig_tubiwords, style={'height':300, 'width':400}),], width={'size': 4, 'offset': 0, 'order': 3}), ]) ])]),])    
  


# In[ ]:


#app.run_server(mode="inline", host="localhost",port=8055)
#app.run_server(mode="external")
#dev_tools_hot_reload=False
#if __name__ == '__main__':
#    app.run_server(debug=False)

#if __name__ == '__main__':
#    app.run_server(dev_tools_hot_reload=False)
#if __name__ == "__main__":
#    app.run_server(debug=True)
#app.run_server(mode='inline')
if __name__ == '__main__':
    app.run_server(host='0.0.0.0')


# In[ ]:




