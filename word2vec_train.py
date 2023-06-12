from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
import string
import nltk
from smart_open import smart_open
import sentencepiece as spm


class WikiCorpus:

    def __init__(self, corpus) -> None:
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __iter__(self):
        for article in self.corpus:
            for section_text in article["section_texts"]:
                text_p = "".join([char for char in section_text.lower() if char not in (string.punctuation + "—–”") and not char.isnumeric()])
                yield " ".join(nltk.word_tokenize(text_p))

def write_data_to_txt(corpus):
    wiki_corpus = iter(WikiCorpus(corpus))
    with smart_open('wiki-english-20171001.txt', 'w', encoding="utf-8") as outfile:
        iterating = True
        while iterating:
            try:
                outfile.write(next(wiki_corpus))
            except StopIteration:
                iterating = False


def train_embedding(corpus):
    model = Word2Vec(WikiCorpus(corpus), vector_size=768, workers=20, sg=1, epochs=1)
    print("model trained")
    print(len(model.wv.index_to_key))

    # Store just the words + their trained embeddings.
    word_vectors = model.wv
    word_vectors.save("word2vec.wordvectors")



if __name__ == "__main__":
    # print("loading file...")
    # corpus = api.load('wiki-english-20171001')  # download the corpus and return it opened as an iterable
    # print("file loaded")
    
    # write_data_to_txt(corpus)

    spm.SentencePieceTrainer.train(input='wiki-english-20171001.txt', model_prefix='m', vocab_size=100_000)

    # train_embedding(corpus)
