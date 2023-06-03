from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

corpus = api.load('wiki-english-20171001')  # download the corpus and return it opened as an iterable
model = Word2Vec(corpus, vector_size=768)

