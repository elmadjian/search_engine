import pandas as pd
import src.indexer as idx
import src.searcher as sch
import os
import numpy as np
import matplotlib.pyplot as plt


class Evaluator():

    def __init__(self, dataset_path, engine):
        dataset = pd.read_csv(dataset_path)
        self.products = self._select_fields(dataset)
        self.engine = engine


    def evaluate(self, num_queries=500):
        """
        Performs several random queries on the engine, limited
        to 'num_queries'. The root mean squared value is calculated
        based on the distance between the expected position and
        the actual predicted order. The errors are also returned
        so that a histogram can be visualized.
        """
        print(">>> Performing {} queries. Please wait...".format(num_queries))
        rnd_vec = np.random.randint(len(self.products), size=num_queries)
        errors = []
        for i, rnd_val in enumerate(rnd_vec):
            self._show_progress(i, len(rnd_vec))
            errors.append(self._eval_query(self.products[rnd_val]))
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        std_dev = np.std(errors)
        return errors, rmse, std_dev


    def show_eval(self, rmse, std_dev, errors):
        """
        Visualization of the RMSE, dispersion and histogram of
        the predicted errors
        """
        print("Average error (RMSE):", rmse)
        print("Standard deviation: +-", std_dev)
        plt.hist(errors, bins='auto')
        plt.title('Histogram of mean errors')
        plt.show()  


    def _show_progress(self, i, max_i):
        if i % 25 == 0:
            porc = (i/max_i) * 100
            print("Progress: " + "%2.2f" % porc + "%", 
                   end='\r', flush=True)


    def _eval_query(self, product, limit=1000):
        """
        Returns the error from the difference between
        expected and predicted positions of a product
        in the rank
        """
        query = product['query']
        target_pos = product['position']
        ranking = self.engine.search(query, prods_to_show=limit)
        try:
            position = ranking.index(product['product_id'])
            return np.abs(target_pos-position)
        except:
            return limit


    def _select_fields(self, dataset):
        """
        Selects the fields from the dataset that will
        take in part in evaluation
        """
        dataset = self._transform_position(dataset)
        fields = ['product_id', 'query', 'position']
        dataset = dataset[fields]
        return dataset.to_dict(orient='records')


    def _transform_position(self, dataset):
        """
        Combines 'search_page' and 'position' data
        into a unidimensional scale
        """
        val = (dataset['search_page']-1)*38+1
        dataset['position'] += val
        return dataset



if __name__=="__main__":
    datapath = os.path.abspath("../data/elo7_recruitment_dataset.csv")
    indexer =  idx.Indexer(datapath)
    engine = sch.Searcher(indexer)
    evaluator = Evaluator(datapath, engine)
    errors, rmse, std = evaluator.evaluate()
    evaluator.show_eval(rmse, std, errors)
