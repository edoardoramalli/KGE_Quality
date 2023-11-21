import igraph as ig
import pandas as pd
import random
import statistics
#ordine di ore
random.seed(0)
kg = ig.Graph.GRG(50, 0.1)

directed = False

degree = kg.degree()


import time
import os

start_time =time.time()
time.sleep(3)
#### Codice da misurare
elapsed_time = time.time() - start_time


import pickle

kg_path = '../Data_Collection/Datasets_Complete/WN18/Split_0/instance.pickle'


with open(kg_path, 'rb') as f:
    pick = pickle.load(f)

# print(res['dataset']['training'])

# ['degree', 'pagerank', 'harmonic_centrality', 'betweenness']

dataset = pick['dataset']
ll = dataset['training']
df = pd.DataFrame(dataset['training'], columns=['h', 'r', 't'])
df_no_rel = df[['h', 't']]
df_no_rel = df_no_rel.drop_duplicates()
kg = ig.Graph.DataFrame(df_no_rel, directed=directed)


exit()

#solo split 0 tempo e metriche


measures = {
    'order': len(kg.vs),
    'size': kg.ecount(),
    'density': kg.density(),
    'transitivity_undirected': kg.transitivity_undirected(),  # Clustering coefficient
    'max_degree': max(degree),
    'min_degree': min(degree),
    'mean_degree': statistics.mean(degree),
    'connected_components_strong': len(set(kg.connected_components(mode='strong'))),
    'connected_components_weak': len(set(kg.connected_components(mode='weak'))),
    'diameter': kg.diameter(directed=directed),
    'girth': kg.girth(),
    'vertex_connectivity': kg.vertex_connectivity(),  # Non ha senso se è undirected
    'edge_connectivity': kg.edge_connectivity(),
    'clique_number': kg.clique_number(),
    #### CAMILLA ####

}

measures['centralization'] = (measures['size'] / (measures['size'] - 2)) * (
            (measures['max_degree'] / (measures['size'] - 1)) - measures['density'])

print(measures)
