import igraph as ig

import pickle
import os
from tqdm import tqdm
from Data_Collection.tool import datasets
import pandas as pd

n_split = 5

directed = False

for dataset_name in tqdm(datasets):

    for i in range(n_split):
        folder_path = '../Data_Collection/Datasets_Complete/{}/Split_{}/'.format(dataset_name, i)
        with open(os.path.join(folder_path, 'instance.pickle'), 'rb') as f:
            pick = pickle.load(f)

        dataset = pick['dataset']

        ll = dataset['training']

        df = pd.DataFrame(dataset['training'], columns=['h', 'r', 't'])

        df_no_rel = df[['h', 't']]

        df_no_rel = df_no_rel.drop_duplicates()

        kg = ig.Graph.DataFrame(df_no_rel, directed=directed)

        # TODO come paper futuro considerare il group by size h,t per definire il peso

        entities = dataset['entity_to_id'].values()

        measures = {'degree': {}, 'pagerank': {}}

        degree = kg.degree(entities)
        pagerank = kg.pagerank(entities)
        harmonic_centrality = kg.harmonic_centrality(entities)
        betweenness = kg.betweenness(entities)

        df_measures = pd.DataFrame({'entity': entities,
                                    'degree': degree,
                                    'pagerank': pagerank,
                                    'harmonic_centrality': harmonic_centrality,
                                    'betweenness': betweenness})

        pick['measures_training'] = {'directed': directed,
                                     'dataframe': df_measures}

        with open(os.path.join(folder_path, 'instance.pickle'), 'wb') as f:
            pickle.dump(pick, f)
