import indexer as idx
import nltk
from nltk.stem import RSLPStemmer
nltk.download('punkt')
nltk.download('rslp')


class Searcher():

    def __init__(self, index):
        self.index = index
        self.max_docs_to_show = 10
        self._stemmer = RSLPStemmer()


    def search(self, query):
        query = nltk.word_tokenize(query, language='portuguese')
        query = self._stemmer.stem(query)
        for i in range(self.max_docs_to_show):
            return self.index


    def where(self, query_params):
        pass




if __name__=="__main__":
    indexer =  idx.Indexer("elo7_recruitment_dataset_100.csv")
    searcher = Searcher(indexer)
