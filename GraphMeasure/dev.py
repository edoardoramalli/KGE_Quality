import igraph as ig

import random
import statistics

random.seed(0)
kg = ig.Graph.GRG(50, 0.1)

directed = False

degree = kg.degree()



measures = {
    'order': len(kg.vs),
    'size': kg.ecount(),
    'density': kg.density(),
    'transitivity_undirected': kg.transitivity_undirected(),  # Clustering coefficient
    'max_degree': max(degree),
    'mean_degree': statistics.mean(degree),
    'connected_components_strong': len(set(kg.connected_components(mode='strong'))),
    'connected_components_weak': len(set(kg.connected_components(mode='weak'))),
    'diameter': kg.diameter(directed=directed),
    'girth': kg.girth(),
    'vertex_connectivity': kg.vertex_connectivity(),  # Non ha senso se è undirected
    'edge_connectivity': kg.edge_connectivity(),
    'clique_number': kg.clique_number(),
}

measures['centralization'] = (measures['size'] / (measures['size'] - 2)) * ((measures['max_degree'] / (measures['size'] - 1)) - measures['density'])

print(measures)
