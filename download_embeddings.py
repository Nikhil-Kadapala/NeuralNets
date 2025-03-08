import gensim
import gensim.downloader
import pickle

wv = gensim.downloader.load('glove-wiki-gigaword-50', quiet=True)
with open("glove_embeddings.pkl", "wb") as f:
    pickle.dump(wv, f)