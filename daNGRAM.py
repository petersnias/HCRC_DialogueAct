'''
Note this code using n-grams to model the priors of dialogue acts within the HCRC Map Task Corpus
Author:  Peters, Nia S.
Code Ref:  https://www.kaggle.com/nicapotato/explore-the-spooky-n-grams-wordcloud-bayes
Last Updated: 4 Jan 2020

'''

# Packages
import os
import numpy as np
import pandas as pd
import nltk
import random
import string as str

# Pre-Processing
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import *

# Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import matplotlib.pyplot as plt

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
#matplotlib inline
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns

# N- Grams
from nltk.util import ngrams
from collections import Counter

# Topic Modeling
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Word 2 Vec
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

# Models
import datetime
from nltk import naivebayes

import warnings
warnings.filterwarnings("ignore")

tokenizer = RegexpTokenizer(r'\w+')


def preprocessing(data):
    txt = data.str.lower().str.cat(sep=' ') #1
    words = tokenizer.tokenize(txt) #2
    words = [w for w in words] #3
    #words = [w for w in words]  # 3, don't get rid of STOPWORDS
    #words = [ps.stem(w) for w in words] #4
    return words

## Helper Functions
def get_ngrams(text, n):
    n_grams = ngrams((text), n)
    return [ ' '.join(grams) for grams in n_grams]

def gramfreq(text,n,num):
    # Extracting bigrams
    result = get_ngrams(text,n)
    # Counting bigrams
    result_count = Counter(result)

    # Converting to the result to a data frame
    df = pd.DataFrame.from_dict(result_count, orient='index')
    df = df.rename(columns={'index':'words', 0:'frequency'}) # Renaming index column name
    return df.sort_values(["frequency"],ascending=[0])[:num]


def gramprop(text,n,num):
    # Extracting bigrams
    result = get_ngrams(text,n)
    # Counting bigrams
    result_count = Counter(result)
    result_prop = Pmf(result_count)
    result_prop.normalize()

    # Converting to the result to a data frame
    df = pd.DataFrame.from_dict(result_prop, orient='index')
    df = df.rename(columns={'index':'words', 0:'proportion'}) # Renaming index column name
    return df.sort_values(["proportion"],ascending=[0])[:num]


def gram_table(sentences, gram, length):
    out = pd.DataFrame(index=None)
    for i in gram:
        table = pd.DataFrame(gramfreq(preprocessing(sentences),i,length).reset_index())
        table.columns = ["{}-Gram".format(i),"Occurence{}".format(i)]
        out = pd.concat([out, table], axis=1)
    return out

def gram_tableProp(sentences, gram, length):
    out = pd.DataFrame(index=None)
    for i in gram:
        table = pd.DataFrame(gramprop(preprocessing(sentences),i,length).reset_index())
        table.columns = ["{}-Gram".format(i),"Occurence{}".format(i)]
        out = pd.concat([out, table], axis=1)
    return out

class Pmf(Counter):
    """A Counter with probabilities."""

    def normalize(self):
        """Normalizes the PMF so the probabilities add to 1."""
        total = float(sum(self.values()))
        for key in self:
            self[key] /= total

    def __add__(self, other):
        """Adds two distributions.

        The result is the distribution of sums of values from the
        two distributions.

        other: Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for key1, prob1 in self.items():
            for key2, prob2 in other.items():
                pmf[key1 + key2] += prob1 * prob2
        return pmf

    def __hash__(self):
        """Returns an integer hash value."""
        return id(self)

    def __eq__(self, other):
        return self is other

    def render(self):
        """Returns values and their probabilities, suitable for plotting."""
        return zip(*sorted(self.items()))


dataDF= pd.read_csv('./data/Maptask_bigdata.csv')
output = dataDF['move']
ngramTable = gram_table(output, gram=[1, 2, 3, 4], length=100)
ngramTable.to_csv("HCRC_DialogueAct_NGRAM.csv")

dataDF= pd.read_csv('./data/Maptask_bigdata.csv')
output = dataDF['move']
ngramTable = gram_tableProp(output, gram=[1, 2, 3, 4], length=100)
ngramTable.to_csv("HCRC_DialogueAct_NGRAM_DIST.csv")


dataDF= pd.read_csv('./data/check_moves.csv')
output = dataDF['pos_tags']
ngramTable = gram_table(output, gram=[1, 2, 3, 4], length=50)
ngramTable.to_csv("CHECK_POS_NGRAM.csv")

dataDF= pd.read_csv('./data/check_moves.csv')
output = dataDF['pos_tags']
ngramTable = gram_tableProp(output, gram=[1, 2, 3, 4], length=50)
ngramTable.to_csv("CHECK_POS_NGRAM_DIST.csv")