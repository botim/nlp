import numpy as np
from gensim.models import FastText
from .models import Tweet

MODEL_FILE = "fb_comments_ft_50epoch.model" #TODO - fill w/ real path
ft = FastText.load(MODEL_FILE)
Session = None #TODO - fill this w/ get_session() from wherever utils library


def get_embeddings(tweet_id):
    session = Session()
    tweet = session.query(Tweet).get(tweet_id)
    if not tweet:
        raise Exception("Invalid tweet_id: {}".format(tweet_id))
    text = tweet.text
    tokens = text.split()
    token_vectors = [ft[token] for token in tokens]
    vectors_np_arr = np.array(token_vectors)
    return vectors_np_arr
