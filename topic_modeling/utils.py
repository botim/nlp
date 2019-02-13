import inspect
import logging
import os
import re

import pandas as pd
from nltk import bigrams, trigrams

_current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
_stopwords_filepath = os.path.join(_current_dir, "..", "stopwords.txt")
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

with open(_stopwords_filepath, encoding="utf-8") as f:
    logger.info("Loading stopwords from {}".format(_stopwords_filepath))
    STOPWORDS = set(f.readlines())
    STOPWORDS = {w.replace("\n", "") for w in STOPWORDS}
    logger.info("Loading stopwords from {} -- DONE".format(_stopwords_filepath))


def preprocess(input_str, remove_user_mentions=True, use_bigrams_phrase=False, use_trigrams_phrase=False):
    PUNCTUATION = {",", ".", "!", "?", ";", "-", "*", "&", "|",":" , '(', ')'}
    result = input_str
    #remove links
    result = " ".join([t for t in result.split() if not t.startswith("http")])
    mentions = re.findall("@[a-zA-Z0-9_.]*", result)
    #remove user name (@UserName)
    if remove_user_mentions:
        for mention in mentions:
            result = result.replace(mention, "")
    #remove punctuation
    for sign in PUNCTUATION:
        result = result.replace(sign, " ")
    #remove '
    result = result.replace("'", "")
    #remove stopwords
    for stopword in STOPWORDS:
        result = result.replace(" {}".format(stopword), "")
        result = result.replace("{} ".format(stopword), "")
    #create unigram, bigrams and trigrams
    unigram = [w for w in result.split() if len(w)>1]
    features = unigram
    if use_bigrams_phrase:
        bigrams_phrase = [b[0]+" "+b[1] for b in bigrams(unigram)]
        features += bigrams_phrase
    if use_trigrams_phrase:
        trigrams_phrase =[b[0]+" "+b[1]+" "+b[2] for b in trigrams(unigram)]
        features += trigrams_phrase
    return features


def load_tweets_file(filepath):
    logger.info("loading tweets from file {}".format(filepath))
    tweets = pd.read_csv(filepath)
    # set 'Text column as text
    logger.debug("setting type str to 'Text' Column")
    tweets["Text"] = tweets["Text"].astype(str)
    # set text column as datetime
    logger.debug("setting type datetime to 'Date Created' Column")
    tweets["Date Created"] = pd.to_datetime(tweets["Date Created"])
    # add column only the hour of 'Date Created'
    logger.debug("adding 'Hour' column to dataframe")
    hours = []
    for d in tweets["Date Created"]:
        hours.append(d.hour)
    tweets["Hour"] = hours
    return tweets
