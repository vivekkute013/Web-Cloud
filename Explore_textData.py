# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:51:57 2022

@author: Vivek
"""

import pandas as pd

data = pd.read_csv("D:\Machine Learning\dataset_files\Exploring Text Data/tweets.csv", encoding='ISO-8859-1')

''' Generating Word Frequency'''
def text_gen(text):
    #will store the list of words
    word_list = []
    # Loop over all tweets and extrat words in word list
    for i in text.split():
        word_list.extend(i)
        
    #create word frequency using word list
    word_freq = pd.Series(word_list).value_counts()
    
    print(word_freq[:20])
    
    return word_freq

print(text_gen(data.text.str))

''' EDA - Exploratoey Data Analysis using WORD Clouds
Now that you have succesfully created a frequency table, 
you can use that to create multiple visualizations in the form of word clouds. 
Sometimes, the quickest way to understand the context of the text data is u
sing a word cloud of top 100-200 words. Let's see how to create that in Python.'''

import matplotlib.pyplot as plt
from wordcloud import WordCloud

#Generate word frequencies
word_freq = text_gen(data.text.str)

#Generate word cloud
wc = WordCloud(width=400, height=300, max_words = 100, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12,8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

'''Few things to Note:-

There is noise in the form of "RT" and "&amp" which can be removed from the word frequency.

Stop words like "the", "in", "to", "of" etc. are obviously ranking among the top frequency words but these are just constructs of the English language and are not specific to the people's tweets.

Words like "demonetization" have occured multiple times. The reason for this is that the current text is not Normalized so words like "demonetization", "Demonetization" etc. are all considered as different words.
'''
# Text Cleaning
#You have already learnt how to utilize Regex to do text cleaning, 
#that is precisely what we are doing here.

import re

def clean_text(text):
    #remove RT
    text = re.sub('RT','',text)
    # Fix &
    text = re.sub('&amp','&',text)
    #Remove punctuations
    text = re.sub(r'[?!.;:,#@-]', '', text)
    # convert to lower case to maintain consistency
    text = text.lower()
    
    return text

'''Stop words Removal
WordCloud provides its own stopwords list. You can have a look at it by-'''

from wordcloud import STOPWORDS
#print(STOPWORDS)

text = data.text.apply(lambda x: clean_text(x))
word_freq = text_gen(text.str)*100
word_freq = word_freq.drop(labels=STOPWORDS, errors = 'ignore')

 
#Generate word cloud
wc = WordCloud(width=450, height=330, max_words=200, background_color='white').generate_from_frequencies(word_freq)
 
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

    
    









