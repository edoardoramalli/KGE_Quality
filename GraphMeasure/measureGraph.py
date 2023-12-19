import pandas as pd
import igraph as ig
import numpy as np


def measure_graph(dataset, directed=True):
    df = pd.DataFrame(dataset, columns=['h', 'r', 't'])
    df_no_rel = df[['h', 't']]
    # df_no_rel = df_no_rel.drop_duplicates()
    kg = ig.Graph.DataFrame(df_no_rel, directed=directed)
    p = {}
    p['order'] = len(kg.vs)
    p['size'] = kg.ecount()
    p['density'] = kg.density()
    p['diameter'] = kg.diameter(directed=directed)
    p['transitivity_undirected'] = kg.transitivity_undirected()
    p['connected_components_weak'] = len(kg.connected_components(mode='weak'))
    # relative number of nodes in the giant connected component
    p['connected_components_weak_relative_nodes'] = len(
        kg.connected_components(mode='weak').giant().vs) / len(kg.vs)
    # relative number of arcs in the giant connected component
    p['connected_components_weak_relative_arcs'] = len(
        kg.connected_components(mode='weak').giant().es) / len(kg.es)
    p['girth'] = kg.girth()
    p['vertex_connectivity'] = kg.vertex_connectivity()
    p['edge_connectivity'] = kg.edge_connectivity()
    p['clique_number'] = kg.clique_number()
    p['average_degree'] = np.mean(kg.degree())
    p['max_degree'] = np.max(kg.degree())
    p['degree_deciles'] = list(np.quantile(kg.degree(), np.linspace(0., 1., 11)))
    p['average_indegree'] = np.mean(kg.indegree())
    p['indegree_deciles'] = list(np.quantile(kg.indegree(), np.linspace(0., 1., 11)))
    p['average_outdegree'] = np.mean(kg.outdegree())
    p['outdegree_deciles'] = list(np.quantile(kg.outdegree(), np.linspace(0., 1., 11)))
    p['radius'] = kg.radius()
    p['average_path_length'] = kg.average_path_length()
    p['assortativity_degree'] = kg.assortativity_degree()
    ecc = kg.eccentricity()
    p['mean_eccentricity'] = np.mean(ecc)
    p['eccentricity_deciles'] = list(np.quantile(ecc, np.linspace(0., 1., 11)))

    p['centralization'] = (p['size'] / (p['size'] - 2)) * (
            (p['max_degree'] / (p['size'] - 1)) - p['density'])

    return p
