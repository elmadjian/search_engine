import os
import nltk
from nltk.stem import RSLPStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')
import string
from num2words import num2words
import pandas as pd
import unidecode
import numpy as np


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
        documents = self._drop_fields(dataset)
        self.features = ['view_counts', 'order_counts']
        max_features = self._calculate_max_features(dataset)
        self.inverted_index = {}
        self.tfidf_index = None
        self.word_idx = None
        self.doc_idx = None
        self.inv_doc_freq = {}
        self.term_freq = {}
        self.documents = {}
        self.stop_words = self._generate_stop_words()
        self.stemmer = RSLPStemmer()
        if os.path.exists(dataset_path[:-4]+'_processed.npz'):
            print('>>> Preprocessed index found. Loading...')
            self._load_indexes(dataset_path)
        else:
            self.create_indexes(documents, max_features)
            self._store_indexes(dataset_path)


    def create_indexes(self, documents, max_features):
        """
        Creates an inverted index for fast document retrieval
        and a TF-IDF matrix of the form N x (V+F) for ranking, where:
        N is the number of documents and
        V is the vocabulary extracted from the corpus.
        F is the set of non-textual added features
        """
        print(">>> Processing documents and creating inverted index...")
        features = {doc['product_id']: [] for doc in documents}
        for doc in documents:
            text = self.preprocess(doc['title'])
            text += self.preprocess(doc['concatenated_tags'])
            tf = self._count_frequency(text, doc['product_id'])
            self.term_freq[doc['product_id']] = tf
            self.documents[doc['product_id']] = doc
            for feat in self.features:
                if np.isnan(doc[feat]):
                    doc[feat] = 0
                features[doc['product_id']].append(1+doc[feat]/max_features[feat])
        print(">>> Creating sparse TF-IDF matrix...")
        n_doc = len(documents)
        for word in self.inv_doc_freq.keys():
            n_word = self.inv_doc_freq[word]
            self.inv_doc_freq[word] = 1+np.log(n_doc/n_word)
        self.word_idx = {w:i for i,w in enumerate(self.inv_doc_freq.keys())}
        self.doc_idx  = {d:i for i,d in enumerate(self.term_freq.keys())}
        self.tfidf_index = self._create_tfidf_index(features)     


    def get_document_ids(self, words):
        """
        Given a list of words, return a list of product ids
        in which at least the occurance of one of the words
        is observed.
        """ 
        products = []    
        for word in words:
            if word in self.inverted_index.keys():
                products += self.inverted_index[word]
        return products


    def get_documents(self, prod_ids):
        """
        Returns the list of products of the dataset
        indexed by 'product_id'
        """
        return self.documents


    def get_vector_doc(self, product_id):
        """
        Returns the tf*idf vector associated to a product
        """
        row = self.doc_idx[product_id]
        return self.tfidf_index[row]


    def get_number_features(self):
        """
        Returns the number of selected features for the model
        """
        return len(self.features)


    def preprocess(self, doc):
        """
        Performs a series of NLP errands:
        Tokenization, converting number to words, stemming, stop word removal
        """
        if not type(doc) == str:
            return []
        text = doc.lower()
        text = unidecode.unidecode(text)
        text = nltk.word_tokenize(text, language='portuguese')
        text = self._number_to_word(text)
        text = [self.stemmer.stem(w) for w in text if\
                (w not in self.stop_words) and (len(w) > 1)]
        return text


    def _count_frequency(self, text, doc_id):
        """
        Calculates and return the term frequency (TF) of a given document.
        It also updates global document frequency
        """
        n_words = len(text)
        words_doc = {w:0 for w in set(text)}
        for word in text: #building inv index and TF
            words_doc[word] += 1
            if word not in self.inverted_index.keys():
                self.inverted_index[word] = []
            self.inverted_index[word].append(doc_id)
        for word in words_doc.keys(): #normalizing TF, counting DF
            words_doc[word] /= n_words
            if word not in self.inv_doc_freq.keys():
                self.inv_doc_freq[word] = 0
            self.inv_doc_freq[word] += 1
        return words_doc


    def _create_tfidf_index(self, features):
        """
        Creates a sparse matrix where rows are indexed by documents
        and columns are the vocabulary. The matrix stores the
        tf*idf values found while building the index.
        """
        rows = len(self.term_freq.keys())
        cols = len(self.inv_doc_freq.keys())
        matrix = np.zeros((rows, cols))
        n_feat = len(self.features)
        feat_mat = np.zeros((rows, n_feat))
        for i, doc in enumerate(self.term_freq.keys()):
            for j, word in enumerate(self.inv_doc_freq.keys()):
                if word in self.term_freq[doc].keys():
                    tf = self.term_freq[doc][word]
                    idf = self.inv_doc_freq[word]
                    matrix[i,j] = tf*idf
            for k in range(n_feat):
                feat_mat[i,k] = features[doc][k]
        matrix = np.hstack((matrix, feat_mat))
        return matrix

        
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
        """
        Creates a list of stop words in Portuguese
        """
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


    def _calculate_max_features(self, dataset):
        features = {f:0 for f in self.features}
        for feature in features:
            features[feature] = dataset[feature].max()
        return features


    def _store_indexes(self, datapath):
        index_path = datapath[:-4]+'_processed.npz'
        print('>>> Saving processed indexes for fast loading...')
        np.savez(index_path, 
                 inverted_index=self.inverted_index,
                 tfidf_index=self.tfidf_index,
                 word_idx=self.word_idx,
                 doc_idx=self.doc_idx,
                 inv_doc_freq=self.inv_doc_freq,
                 term_freq=self.term_freq,
                 documents=self.documents)


    def _load_indexes(self, datapath):
        index_path = datapath[:-4]+'_processed.npz'
        indexes = np.load(index_path, allow_pickle=True)
        self.inverted_index = indexes['inverted_index'].item()
        self.tfidf_index = indexes['tfidf_index']
        self.word_idx = indexes['word_idx'].item()
        self.doc_idx = indexes['doc_idx'].item()
        self.inv_doc_freq = indexes['inv_doc_freq'].item()
        self.term_freq = indexes['term_freq'].item()
        self.documents = indexes['documents'].item()



if __name__=="__main__":
    datapath = os.path.abspath("../data/elo7_recruitment_dataset_100.csv")
    indexer = Indexer(datapath)