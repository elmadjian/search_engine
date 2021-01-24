import unittest
import src.indexer as idx
import src.searcher as sch
import os

class TestIndexer(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)
        datapath = os.path.abspath("data/elo7_recruitment_dataset_100.csv")
        indexer = idx.Indexer(datapath)
        self.searcher = sch.Searcher(indexer)


    def test_search_prods_to_show(self):
        result = self.searcher.search('lembrancinha', prods_to_show=5)
        self.assertEqual(5, len(result))
        result = self.searcher.search('lembrancinha', prods_to_show=7)
        self.assertEqual(7, len(result))
        result = self.searcher.search('lembrancinha', prods_to_show=10)
        self.assertLessEqual(10, len(result))


    def test_search_seller_id(self):
        result = self.searcher.search('saia', seller_id=7274093)[0]
        self.assertEqual(result, 7621260)
        result = self.searcher.search('lembrancinha', seller_id=5371868)[0]
        self.assertEqual(result, 6321948)


    def test_search_title(self):
        result = self.searcher.search("difusor", title='Sacolinha com Difusor')[0]
        self.assertEqual(result, 6621912)
        result = self.searcher.search("trocador", title='Trocador portátil')[0]
        self.assertEqual(result, 10267520)


    def test_search_price_min(self):
        result = self.searcher.search("bolsa")
        self.assertGreater(len(result),1)
        result = self.searcher.search("bolsa", price_min=257.65)[0]
        self.assertEqual(result, 7234946)

    
    def test_search_price_max(self):
        result = self.searcher.search("bolsa")
        self.assertGreater(len(result),1)
        result = self.searcher.search("bolsa", price_max=150)[0]
        self.assertEqual(result, 11623598)


    def test_search_weight_min(self):
        result = self.searcher.search("tapete")
        self.assertGreater(len(result),1)
        result = self.searcher.search("tapete", weight_min=1800)[0]
        self.assertEqual(result, 16687430)


    def test_search_weight_max(self):
        result = self.searcher.search("manta")
        self.assertGreater(len(result),1)
        result = self.searcher.search("manta", weight_max=10)[0]
        self.assertEqual(result, 13424151)


    def test_search_express_delivery(self):
        result = self.searcher.search("bolsa")
        self.assertGreater(len(result),2)
        result = self.searcher.search("bolsa", express_delivery=True)
        self.assertEqual(len(result), 2)


    def test_search_min_quantity(self):
        result = self.searcher.search("lembrancinha")
        self.assertGreater(len(result),1)
        result = self.searcher.search("lembrancinha", min_quantity=57)[0]
        self.assertEqual(result, 14863396)


    def test_search_category(self):
        result = self.searcher.search("adesivo", category="Decoração")[0]
        self.assertEqual(result, 15917108)
        result = self.searcher.search("adesivo", category="Papel e Cia")[0]
        self.assertEqual(result, 15534262)