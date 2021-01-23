import os
import indexer as idx
import numpy as np


class Searcher():

    def __init__(self, index):
        self.index = index
        self.word_idx = index.word_idx
        self.doc_idx = index.doc_idx
        self.tfidf_index = index.tfidf_index


    def search(self, query, prods_to_show=10, **kwargs):
        """
        """
        query = self.index.preprocess(query)
        query_vec = self._query_vector(query)
        products = self.index.get_document_ids(query)
        similarities = self._cosine_similarity_docs(query_vec,products)
        ranking = dict(sorted(similarities.items(), 
                              key=lambda item: item[1], reverse=True))
        ranking = list(ranking.keys())
        if kwargs:
            ranking = self._filter_by_params(ranking, kwargs)
        return ranking[:prods_to_show]


    def _query_vector(self, query):
        """
        Given a preprocessed query, it returns a
        transformed tf*idf associated vector
        """
        vocab = len(self.word_idx.keys())
        query_vec = np.zeros((vocab,))
        for word in query:
            if word in self.word_idx.keys():
                query_vec[self.word_idx[word]] += 1
        query_vec = query_vec/np.sum(query_vec)
        for word in query:
            idf = 1
            if word in self.index.inv_doc_freq.keys():
                idf = self.index.inv_doc_freq[word]
            query_vec[self.word_idx[word]] *= idf
        return query_vec    


    def _cosine_similarity_docs(self, query_vec, docs):
        """
        Calculates the cosine similarity between the
        query transformed to the vector space (query_vec)
        and a list of relevant product ids (docs).
        """
        similarities = {doc:0 for doc in docs}
        for doc in docs:
            doc_vec = self.index.get_vector_doc(doc)
            dot = np.dot(query_vec, doc_vec)
            norm1 = np.linalg.norm(query_vec)
            norm2 = np.linalg.norm(doc_vec)
            similarities[doc] = dot/(norm1 * norm2)
        return similarities


    def _filter_by_params(self, ranking, query_params):
        """
        Filter the ranked output of a search by selected parameters.
        Valid parameters:
        - seller_id (int)
        - title (str)
        - price_min (float)
        - price_max (float)
        - weight_min (int)
        - weight_max (int)
        - express_delivery (bool)
        - min_quantity (int)
        - category (str)
        """
        products = self.index.get_documents(ranking)
        for param in query_params.keys():
            if param == 'seller_id':
                ranking = [i for i in ranking if products[i]['seller_id']\
                           == query_params['seller_id']]
            elif param == 'title':
                ranking = [i for i in ranking if products[i]['title']\
                           == query_params['title']]
            elif param == 'price_min':
                ranking = [i for i in ranking if products[i]['price']\
                           >= query_params['price_min']]
            elif param == 'price_max':
                ranking = [i for i in ranking if products[i]['price']\
                           <= query_params['price_max']]
            elif param == 'weight_min':
                ranking = [i for i in ranking if products[i]['weight']\
                           >= query_params['weight_min']]
            elif param == 'weight_max':
                ranking = [i for i in ranking if products[i]['weight']\
                           <= query_params['weight_max']]
            elif param == 'express_delivery':
                ranking = [i for i in ranking if products[i]['express_delivery']]
            elif param == 'min_quantity':
                ranking = [i for i in ranking if products[i]['min_quantity']\
                           >= query_params['min_quantity']]
            elif param == 'category':
                ranking = [i for i in ranking if products[i]['category']\
                           == query_params['category']]
        return ranking




if __name__=="__main__":
    datapath = os.path.abspath("../data/elo7_recruitment_dataset_100.csv")
    indexer =  idx.Indexer(datapath)
    searcher = Searcher(indexer)
    print(searcher.search("mandala croche", category="Decoração"))
    print(searcher.search("lembrancinha", category="Decoração"))
