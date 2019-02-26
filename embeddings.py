import re
import shelve

from gensim.models.fasttext import FastText
from gensim.models.callbacks import CallbackAny2Vec
from pprint import pprint
from multiprocessing import Pool
from topic_modeling.utils import preprocess


class FTCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f"Epoch #{self.epoch} start")

    def on_epoch_end(self, model):
        print()
        print(f"Epoch #{self.epoch} end")
        self.epoch += 1



if __name__ == '__main__':
    pool = Pool(4)
    print("STARTING")
    with shelve.open("facebook.db") as f:
        db = f.get("data")
    print("DB Loaded")
    heb_regex = re.compile("[א-ת]+")
    sentences = [doc["_source"]["Data"] for doc in db if doc["_source"]["Data"]]
    sentences = pool.map(preprocess, [s for s in sentences if heb_regex.findall(s)])
    print("EXTRACTED HEBREW SENTENCES")
    # pprint(sentences[:5])

    print("STARTING FASTTEXT TRAINING")
    cb = FTCallback()
    ft = FastText.load("comments_heb_fasttext.model")
    pprint(ft.similar_by_vector(ft["פאשיסט"]))
    ft = FastText(sentences, max_vocab_size=100000, word_ngrams=2, min_count=5, sg=1, iter=50, callbacks=(cb,))
    print("SAVING FASTTEXT MODEL")
    for word in ["נתניהו", "זועבי", "מניאק", "סמולני", "פשיסט"]:
        print(f"MOST SIMILAR TO {word}")
        print("-------------")
        pprint(ft.similar_by_word(word))
        print("-------------")
    ft.save("comments_heb_fasttext_50epoch.model")
    print("DONE")
