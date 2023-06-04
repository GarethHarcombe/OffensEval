from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

print("loading file...")
corpus = api.load('wiki-english-20171001')  # download the corpus and return it opened as an iterable
print("file loaded")
model = Word2Vec(corpus, vector_size=768, workers=20, sg=1)
print("model trained???")
model.save("word2vec.model")


# Store just the words + their trained embeddings.
word_vectors = model.wv
word_vectors.save("word2vec.wordvectors")

# Load back with memory-mapping = read-only, shared across processes.
# wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')