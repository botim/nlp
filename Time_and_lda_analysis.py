import pandas as pd
import datetime as dt
from gensim.corpora.dictionary import Dictionary
from gensim import models

from topic_modeling.utils import preprocess, load_tweets_file

pd.set_option('display.max_colwidth', -1)

# Parameters
# value between 0.1 to 1. filter days when there were fewer tweets than this quantile
quantile_filter = 0.5
num_topics = 2
filter_words_that_appeared_less_than = 5
use_bigrams_phrase = True
use_trigrams_phrase = True
remove_user_name = True
tweets_filepath = 'tweets_israeli-girls.csv'

# load tweets
tweets = load_tweets_file(tweets_filepath)

# find tweets min and max 'Date Created'
max_date = tweets['Date Created'].max()
max_date_limit = dt.datetime(max_date.year, max_date.month, max_date.day) + dt.timedelta(days=1)
min_date = tweets['Date Created'].min()
min_date_limit = dt.datetime(min_date.year, min_date.month, min_date.day)

dic_tweet_per_day = dict()
start_date = min_date_limit
while start_date < max_date_limit:
    end_date = start_date + dt.timedelta(days=1)
    current = tweets[tweets['Date Created'].between(start_date, end_date)]
    if len(current) > 0:
        dic_tweet_per_day[start_date] = current
    start_date = end_date

dic_counts = {'Date': [], 'Count': []}
for k in dic_tweet_per_day.keys():
    dic_counts['Date'].append(k)
    count = len(dic_tweet_per_day[k])
    dic_counts['Count'].append(count)

count_df = pd.DataFrame(dic_counts)
filter_value = count_df['Count'].quantile(quantile_filter)

dic_for_data_fram = {'Date': [], 'Words': []}

start_date = min_date_limit
while start_date < max_date_limit:
    end_date = start_date + dt.timedelta(days=1)
    current = tweets[tweets['Date Created'].between(start_date, end_date)]
    if len(current) >= filter_value:
        df = current

        processed_docs = df['Text'].map(preprocess)

        dictionary = Dictionary(processed_docs)
        dictionary.filter_extremes(no_below=filter_words_that_appeared_less_than, keep_n=100000)

        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        if len(dictionary) > 0:
            print(start_date)
            lda_model = models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=2)

            for idx, topic in lda_model.print_topics(-1):
                print(topic)
                dic_for_data_fram['Date'].append(start_date)
                dic_for_data_fram['Words'].append(topic)

    dic_tweet_per_day[start_date] = current
    start_date = end_date

pd.DataFrame(dic_for_data_fram).to_csv('./suspicion_bots_topics.csv', index=False)
