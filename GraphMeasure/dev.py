import igraph as ig
import pandas as pd
import random
import statistics
import numpy as np

directed = True

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
# df_no_rel = df_no_rel.drop_duplicates()
kg = ig.Graph.DataFrame(df_no_rel, directed=directed)

p = {}

p['graph_metrics'] = {}
p['graph_metrics']['order'] = len(kg.vs)
p['graph_metrics']['size'] = kg.ecount()
p['graph_metrics']['density'] = kg.density()
p['graph_metrics']['diameter'] = kg.diameter(directed=directed)
p['graph_metrics']['transitivity_undirected'] = kg.transitivity_undirected()
p['graph_metrics']['connected_components_weak'] = len(kg.connected_components(mode='weak'))
# relative number of nodes in the giant connected component
p['graph_metrics']['connected_components_weak_relative_nodes'] = len(
    kg.connected_components(mode='weak').giant().vs) / len(kg.vs)
# relative number of arcs in the giant connected component
p['graph_metrics']['connected_components_weak_relative_arcs'] = len(
    kg.connected_components(mode='weak').giant().es) / len(kg.es)
p['graph_metrics']['girth'] = kg.girth()
p['graph_metrics']['vertex_connectivity'] = kg.vertex_connectivity()
p['graph_metrics']['edge_connectivity'] = kg.edge_connectivity()
p['graph_metrics']['clique_number'] = kg.clique_number()
p['graph_metrics']['average_degree'] = np.mean(kg.degree())
p['graph_metrics']['degree_deciles'] = list(np.quantile(kg.degree(), np.linspace(0., 1., 11)))
p['graph_metrics']['average_indegree'] = np.mean(kg.indegree())
p['graph_metrics']['indegree_deciles'] = list(np.quantile(kg.indegree(), np.linspace(0., 1., 11)))
p['graph_metrics']['average_outdegree'] = np.mean(kg.outdegree())
p['graph_metrics']['outdegree_deciles'] = list(np.quantile(kg.outdegree(), np.linspace(0., 1., 11)))
p['graph_metrics']['radius'] = kg.radius()
p['graph_metrics']['average_path_length'] = kg.average_path_length()
p['graph_metrics']['assortativity_degree'] = kg.assortativity_degree()
ecc = kg.eccentricity()
p['graph_metrics']['mean_eccentricity'] = np.mean(ecc)
p['graph_metrics']['eccentricity_deciles'] = list(np.quantile(ecc, np.linspace(0., 1., 11)))

# measures['centralization'] = (measures['size'] / (measures['size'] - 2)) * (
#         (measures['max_degree'] / (measures['size'] - 1)) - measures['density'])
#
# print(measures)
