import nltk
from nltk.stem import RSLPStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')
import string
from num2words import num2words
import pandas as pd


class Indexer():

    def __init__(self, dataset_path):
        """Dataset according to the elo7 format. Fields are
        listed below. the labels [doc] and [eval] indicante
        whether the information is related to the document
        or the evaluation of the trained model.
        - product_id        [doc]
        - seller_id         [doc]
        - query             [eval]
        - search_page       [eval]
        - position          [eval]
        - title             [doc, text]
        - concatenated_tags [doc, text]
        - creation_date     [doc]
        - price             [doc]
        - weight            [doc]
        - express_delivery  [doc]
        - minimum_quantity  [doc]
        - view_counts       [doc]
        - order_counts      [doc]
        - category          [doc]
        """
        dataset = pd.read_csv(dataset_path)
        self.documents = self._drop_fields(dataset)
        self.index = {}
        self.stop_words = self._generate_stop_words()
        self.stemmer = RSLPStemmer()
        self.create_inverted_index()


    def create_inverted_index(self):
        """
        Creates a simple index from stemmed words in 
        the corpus. 
        """
        for doc in self.documents:
            self._index_document(doc, 'title')
            self._index_document(doc, 'concatenated_tags')


    def _index_document(self, doc, category):
        """
        Add a single document to the index
        """
        text = doc[category].lower()
        text = nltk.word_tokenize(text, language='portuguese')
        text = self._number_to_word(text)
        text = [self.stemmer.stem(w) for w in text if\
                (w not in self.stop_words) and (len(w) > 1)]
        for word in text:
            if word not in self.index.keys():
                self.index[word] = set()
            self.index[word].add(doc['product_id'])

        
    def _number_to_word(self, tokens):
        """
        Given a list of tokens, it converts the
        numeric to written representation.
        """
        new_tokens = []
        for t in tokens:
            try:
                word = num2words(t, lang='pt_BR')
                word = nltk.word_tokenize(word, language='portuguese')
                new_tokens += word
            except:
                new_tokens.append(t)
        return new_tokens

    
    def _generate_stop_words(self):
        punctuation = string.punctuation
        punctuation += "+-/'\\"
        stop_words  = nltk.corpus.stopwords.words('portuguese')
        return list(punctuation) + stop_words


    def _drop_fields(self, dataset):
        """
        Removes the fields from the dataset that do not
        belong to the actual document to be indexed.
        """
        fields = ['query', 'search_page', 'position']
        dataset = dataset.drop(fields, axis=1)
        return dataset.to_dict(orient='records')


    def store_index(self):
        pass
        


if __name__=="__main__":
    indexer = Indexer("elo7_recruitment_dataset_100.csv")