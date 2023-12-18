import igraph as ig
import pandas as pd
import random
import statistics
import pickle
import time

#solo split 0 tempo e metriche

# escludere ['degree', 'pagerank', 'harmonic_centrality', 'betweenness']

random.seed(0)
directed = False

kg_path = '../../Data_Collection/Datasets_Complete/WN18/Split_0/instance.pickle'

with open(kg_path, 'rb') as f:
    pick = pickle.load(f)
dataset = pick['dataset']
ll = dataset['training']
df = pd.DataFrame(dataset['training'], columns=['h', 'r', 't'])
df_no_rel = df[['h', 't']]
kg = ig.Graph.DataFrame(df_no_rel, directed=directed)
degree = kg.degree()

measures = {

    #### EDOARDO ####
    'order': kg.vcount(),
    'size': kg.ecount(),
    #'max_degree': max(degree),
    #'min_degree': min(degree),
    #'mean_degree': statistics.mean(degree),
    'diameter': kg.diameter(directed=directed),
    'girth': kg.girth(),

    #### CAMILLA ####

    #Global
    'radius': kg.radius(),
    'average_path_length': kg.average_path_length(),

    #Connectedness
    'all_minimal_st_separators': kg.all_minimal_st_separators(),
    'minimum_size_separators': kg.minimum_size_separators(),

    #Clique & motifs
    'clique_number': kg.clique_number(),
    'maximal_cliques': kg.maximal_cliques(),
    'largest_cliques': kg.largest_cliques(),
    'motifs_randesu_no': kg.motifs_randesu_no(),

    #Optimality
    'farthest_points': kg.farthest_points(),
    'modularity': kg.modularity(),
    'independence_number': kg.independence_number(),
    'maximal_independent_vertex_sets': kg.maximal_independent_vertex_sets(),
    'largest_independent_vertex_sets': kg.largest_independent_vertex_sets(),
    'mincut_value': kg.mincut_value(),
    'feedback_arc_set': kg.feedback_arc_set(),

    #Assortativity
    'assortativity': kg.assortativity(),
    'assortativity_degree': kg.assortativity_degree(),
    'assortativity_nominal': kg.assortativity_nominal(),
    'density': kg.density(),
    'transitivity_undirected': kg.transitivity_undirected(),
    'reciprocity': kg.reciprocity(), # only direct

    #Vertex
    'similarity_dice': kg.similarity_dice(),
    'similarity_jaccard': kg.similarity_jaccard(),
    'similarity_inverse_log_weighted': kg.similarity_inverse_log_weighted(),
    'diversity': kg.diversity(),

    #Structural
    'authority_score': kg.authority_score(),
    'hub_score': kg.hub_score(),
    'closeness': kg.closeness(),
    'bibcoupling': kg.bibcoupling(),
    'constraint': kg.constraint(),
    'cocitation': kg.cocitation(),
    'coreness': kg.coreness(),
    'eccentricity': kg.eccentricity(),
    'strength': kg.strength(),
    'transitivity_local_undirected': kg.transitivity_local_undirected(),

    #Edges
    'laplacian': kg.laplacian(),

    #Flow
    'maxflow_value': kg.maxflow_value(),
    #'edge_connectivity': kg.edge_connectivity(),  # only direct
    #'vertex_connectivity': kg.vertex_connectivity()  # only direct
    
    #Connectivity
    'connected_components_strong': len(set(kg.connected_components(mode='strong'))),
    'connected_components_weak': len(set(kg.connected_components(mode='weak'))),
}

measures['centralization'] = (measures['size'] / (measures['size'] - 2)) * (
(measures['max_degree'] / (measures['size'] - 1)) - measures['density'])

print(measures)
