import src.indexer as idx
import src.searcher as sch
import src.evaluator as evl
import os


def main():
    dataset, prog_mode = "", ""
    datapath = os.path.abspath("data/elo7_recruitment_dataset.csv")
    welcome_msg = '>>> Welcome to the search engine!'
    print('\n\n'+welcome_msg+'\n{}\n'.format('-'*len(welcome_msg)))
    indexer, searcher, evaluator = "","",""
    while True:
        if dataset == "":
            dataset = input('>>> Please, select a dataset to load [{}]:\n'.format(datapath))
            if dataset == "":
                dataset = datapath
            try:
                indexer = idx.Indexer(dataset)
            except Exception as e:
                print('>>> Error while trying to read the dataset: {}\n'.format(e))
                dataset = ""
        else:
            if prog_mode == "":
                engine = sch.Searcher(indexer)
                evaluator = evl.Evaluator(dataset, engine)
                prog_mode = input('>>> Do you wish run the search engine or the evaluator? [search engine]:\n')
                if prog_mode == '' or prog_mode == 'search engine':
                    prog_mode = 'search_engine'
                elif prog_mode == 'evaluator':
                    prog_mode = 'evaluator'
            elif prog_mode == 'search_engine':
                query = input('>>> Enter your query, type "-o" to show query options, or "-q" to quit:\n')
                if query == '-o':
                    print('>>> Query usage: query [OPTION] \n'
                        +'    --prods_to_show=[int]: limit the number of returned items [default: 10]\n'
                        +'    --seller_id=[int]: filter by seller==#\n'
                        +'    --title=[str]: filter by exact title match\n'
                        +'    --price_min=[float]: show items above a minimum price\n'
                        +'    --price_max=[float]: show items below a maximum price\n'
                        +'    --weight_min=[float]: show items with at least a certain weight\n'
                        +'    --weight_max=[float]: show items with at most a certain weight\n'
                        +'    --express_delivery=[bool]: show only items with express delivery option\n'
                        +'    --min_quantity=[int]: filter by minimum purchase quantity\n'
                        +'    --category=[str]: filter by a valid category\n')
                elif query == '-q':
                    break
                else:
                    query = query.split('--')
                    result = ""
                    if len(query) > 1:
                        kwargs = {}
                        for param in query[1:]:
                            key,value = param.split('=')
                            kwargs[key] = value
                        result = engine.search(query[0], **kwargs)
                    else:
                        result = engine.search(query[0])
                    print("\n>>> Returned products: {}\n".format(result))
            elif prog_mode == 'evaluator':            
                query = input('>>> How many random queries do you wish to evaluate, or type "-q" to quit [500]:\n')
                if query == "":
                    query = 500
                elif query == '-q':
                    break 
                err, rmse, std = evaluator.evaluate(int(query))
                evaluator.show_eval(rmse, std, err)


##############
main()
