import igraph as ig
import pandas as pd
import random
import statistics
import numpy as np

directed = True

import pickle

kg_path = '/home/ramalli/results/TransE/FB15k/FB15k_0_TransE_baseline_0/instance.pickle'

with open(kg_path, 'rb') as f:
    pick = pickle.load(f)

df = pd.DataFrame(pick['training'], columns=['h', 'r', 't'])
df_no_rel = df[['h', 't']]
relationships = df[['r']]

df_no_rel = df_no_rel.drop_duplicates()  # era commentato ma carlo mi aveva detto di computare le metriche droppando i duplicates all'epoca
# kg = ig.Graph.DataFrame(df_no_rel, directed=directed)
kg = ig.Graph.TupleList(df_no_rel.itertuples(index=False), directed=directed)

closeness = kg.closeness()
print(min(closeness), max(closeness), np.nanmean(closeness))

# print(res['dataset']['training'])

# ['degree', 'pagerank', 'harmonic_centrality', 'betweenness']

# dataset = pick['dataset']
# ll = dataset['training']
#
# print(ll)