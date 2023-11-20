import igraph as ig

from pykeen.triples import TriplesFactory
from tqdm import tqdm

import random
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
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, 'instance.pickle'), 'rb') as f:
            pick = pickle.load(f)

        dataset = pick['dataset']

        ll = dataset['training']

        df = pd.DataFrame(dataset['training'], columns=['h', 'r', 't'])

        df_no_rel = df[['h', 't']]

        df_no_rel = df_no_rel.drop_duplicates()

        kg = ig.Graph.DataFrame(df_no_rel, directed=directed)

        order = len(kg.vs)
        size = len(kg.es)
        transitivity_undirected = kg.transitivity_undirected()
        connected_components_strong = kg.connected_components(mode='strong')
        connected_components_weak = kg.connected_components(mode='weak')
        diameter = kg.diameter(directed=directed)


        exit()
