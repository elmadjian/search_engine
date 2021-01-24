import unittest
import src.indexer as idx
import os
import pandas as pd
import string

class TestIndexer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)
        self.datapath = os.path.abspath("data/elo7_recruitment_dataset_10.csv")
        self.indexer = idx.Indexer(self.datapath)
  
    def test_drop_fields(self):
        dataset = pd.read_csv(self.datapath)
        dataset = self.indexer._drop_fields(dataset)
        found_key = False
        for prod in dataset:
            if ('query' or 'search_page' or 'position') in prod.keys():
                found_key = True
                break
        self.assertEqual(found_key, False)


    def test_calculate_max_features(self):
        dataset = pd.read_csv(self.datapath)
        expected = [1178, 109]
        feat = self.indexer._calculate_max_features(dataset)
        self.assertEqual(list(feat.values()), expected)

    
    def test_store_indexes(self):
        test_name = self.datapath + '_test_.csv'
        self.indexer._store_indexes(test_name)
        val = os.path.exists(test_name[:-4] + '_processed.npz')
        self.assertTrue(val)
        os.remove(test_name[:-4] + '_processed.npz')


    def test_generate_stop_words(self):
        punctuation = string.punctuation
        punctuation += "+-/'\\"
        elements = list(punctuation)
        stopwords = self.indexer._generate_stop_words()
        found_key = False
        for el in elements:
            if el not in stopwords:
                found_key = True
                break
        self.assertEqual(found_key, False)


    def test_number_to_word(self):
        converted = self.indexer._number_to_word(["1930"])
        expected = ['mil', ',', 'novecentos', 'e', 'trinta']
        self.assertEqual(converted, expected)


    def test_get_document_ids(self):
        expected = 16153119
        doc = self.indexer.get_document_ids(['esmalt'])
        self.assertEqual(expected, doc[0])


    def test_preprocess(self):
        doc = 'batata quente amarela'
        text = self.indexer.preprocess(doc)
        expected = ['batat', 'quent', 'amarel']
        self.assertEqual(expected, text)



       


