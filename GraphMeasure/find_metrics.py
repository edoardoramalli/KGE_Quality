import igraph as ig
import pandas as pd
import random
from statistics import mean
import pickle
import time

#solo split 0 tempo e metriche

# escludere ['degree', 'pagerank', 'harmonic_centrality', 'betweenness']

names = ['WN18', 'FB15k', 'FB15k237', 'YAGO310']

random.seed(0)
directed = False

kg_path = '../Data_Collection/Datasets_Complete/WN18RR/Split_0/instance.pickle'

with open(kg_path, 'rb') as f:
    pick = pickle.load(f)
dataset = pick['dataset']
ll = dataset['training']
df = pd.DataFrame(dataset['training'], columns=['h', 'r', 't'])
df_no_rel = df[['h', 't']]
kg = ig.Graph.DataFrame(df_no_rel, directed=directed)
degree = kg.degree()

if __name__ == '__main__':

    with open("metrics.csv", 'w') as f:

        print("Start...")

        start_global_time = time.time()
    
        start_time = time.time()
        order = kg.vcount()
        elapsed_time = time.time() - start_time
        f.write("order, " + str(order) + ", " + str(elapsed_time)+"\n")
    
        start_time = time.time()
        size = kg.ecount()
        elapsed_time = time.time() - start_time
        f.write("size, " + str(size) + ", " + str(elapsed_time)+"\n")


        start_time = time.time()
        max_degree = max(degree)
        elapsed_time = time.time() - start_time
        f.write("max_degree, " + str(max_degree) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        min_degree = min(degree)
        elapsed_time = time.time() - start_time
        f.write("min_degree, " + str(min_degree) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        mean_degree = mean(degree)
        elapsed_time = time.time() - start_time
        f.write("mean_degree, " + str(mean_degree) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        diameter = kg.diameter(directed=directed)
        elapsed_time = time.time() - start_time
        f.write("diameter, " + str(diameter) + ", " + str(elapsed_time)+"\n")


        start_time = time.time()
        girth = kg.girth()
        elapsed_time = time.time() - start_time
        f.write("girth, " + str(girth) + ", " + str(elapsed_time)+"\n")

    
        #### CAMILLA ####

        print("Global metrics...")
    
        # Global
        start_time = time.time()
        radius = kg.radius()
        elapsed_time = time.time() - start_time
        f.write("radius, " + str(radius) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        average_path_length = kg.average_path_length()
        elapsed_time = time.time() - start_time
        f.write("average_path_length, " + str(average_path_length) + ", " + str(elapsed_time)+"\n")

        print("Clique & motifs metrics...")
    
        # Clique & motifs
        start_time = time.time()
        clique_number = kg.clique_number()
        elapsed_time = time.time() - start_time
        f.write("clique_number, " + str(clique_number) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        maximal_cliques = len(kg.maximal_cliques())
        elapsed_time = time.time() - start_time
        f.write("len_maximal_cliques, " + str(maximal_cliques) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        largest_cliques = len(kg.largest_cliques())
        elapsed_time = time.time() - start_time
        f.write("len_largest_cliques, " + str(largest_cliques) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        motifs_randesu_no = kg.motifs_randesu_no()
        elapsed_time = time.time() - start_time
        f.write("motifs_randesu_no, " + str(motifs_randesu_no) + ", " + str(elapsed_time)+"\n")

        print("Optimality metrics...")

        # Optimality
        start_time = time.time()
        farthest_points = len(kg.farthest_points())
        elapsed_time = time.time() - start_time
        f.write("num_farthest_points, " + str(farthest_points) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        mincut_value = kg.mincut_value()
        elapsed_time = time.time() - start_time
        f.write("mincut_value, " + str(mincut_value) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        feedback_arc_set = len(kg.feedback_arc_set())
        elapsed_time = time.time() - start_time
        f.write("len_feedback_arc_set, " + str(feedback_arc_set) + ", " + str(elapsed_time)+"\n")

        print("Assortativity metrics...")

        # Assortativity
        start_time = time.time()
        assortativity_degree = kg.assortativity_degree()
        elapsed_time = time.time() - start_time
        f.write("assortativity_degree, " + str(assortativity_degree) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        density = kg.density()
        elapsed_time = time.time() - start_time
        f.write("density, " + str(density) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        transitivity_undirected = kg.transitivity_undirected()
        elapsed_time = time.time() - start_time
        f.write("transitivity_undirected, " + str(transitivity_undirected) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        reciprocity = kg.reciprocity()
        elapsed_time = time.time() - start_time # only direct
        f.write("reciprocity, " + str(reciprocity) + ", " + str(elapsed_time)+"\n")

        print("Vertex metrics...")

        # Vertex
        start_time = time.time()
        diversity = kg.diversity()
        elapsed_time = time.time() - start_time
        f.write("min_diversity, " + str(min(diversity)) + ", " + str(elapsed_time)+"\n")
        f.write("max_diversity, " + str(max(diversity)) + ", " + str(elapsed_time) + "\n")
        f.write("mean_diversity, " + str(mean(diversity)) + ", " + str(elapsed_time) + "\n")

        print("Structural metrics...")

        # Structural
        start_time = time.time()
        authority_score = kg.authority_score()
        elapsed_time = time.time() - start_time
        f.write("min_authority_score, " + str(min(authority_score)) + ", " + str(elapsed_time)+"\n")
        f.write("max_authority_score, " + str(max(authority_score)) + ", " + str(elapsed_time) + "\n")
        f.write("mean_authority_score, " + str(mean(authority_score)) + ", " + str(elapsed_time) + "\n")

        start_time = time.time()
        hub_score = kg.hub_score()
        elapsed_time = time.time() - start_time
        f.write("min_hub_score, " + str(min(hub_score)) + ", " + str(elapsed_time)+"\n")
        f.write("max_hub_score, " + str(max(hub_score)) + ", " + str(elapsed_time) + "\n")
        f.write("mean_hub_score, " + str(mean(hub_score)) + ", " + str(elapsed_time) + "\n")

        start_time = time.time()
        closeness = kg.closeness()
        elapsed_time = time.time() - start_time
        f.write("min_closeness, " + str(min(closeness)) + ", " + str(elapsed_time) + "\n")
        f.write("max_closeness, " + str(max(closeness)) + ", " + str(elapsed_time) + "\n")
        f.write("mean_closeness, " + str(mean(closeness)) + ", " + str(elapsed_time) + "\n")

        start_time = time.time()
        bibcoupling = kg.bibcoupling()
        elapsed_time = time.time() - start_time
        f.write("max_bibcoupling, " + str(max(max(bibcoupling))) + ", " + str(elapsed_time)+"\n")
        f.write("min_bibcoupling, " + str(min(min(bibcoupling))) + ", " + str(elapsed_time) + "\n")

        start_time = time.time()
        cocitation = kg.cocitation()
        elapsed_time = time.time() - start_time
        f.write("max_cocitation, " + str(max(max(cocitation))) + ", " + str(elapsed_time)+"\n")
        f.write("min_cocitation, " + str(min(min(cocitation))) + ", " + str(elapsed_time) + "\n")

        start_time = time.time()
        constraint = kg.constraint()
        elapsed_time = time.time() - start_time
        f.write("min_constraint, " + str(min(constraint)) + ", " + str(elapsed_time) + "\n")
        f.write("max_constraint, " + str(max(constraint)) + ", " + str(elapsed_time) + "\n")
        f.write("mean_constraint, " + str(mean(constraint)) + ", " + str(elapsed_time) + "\n")

        start_time = time.time()
        coreness = kg.coreness()
        elapsed_time = time.time() - start_time
        f.write("min_coreness, " + str(min(coreness)) + ", " + str(elapsed_time) + "\n")
        f.write("max_coreness, " + str(max(coreness)) + ", " + str(elapsed_time) + "\n")
        f.write("mean_coreness, " + str(mean(coreness)) + ", " + str(elapsed_time) + "\n")

        start_time = time.time()
        eccentricity = kg.eccentricity()
        elapsed_time = time.time() - start_time
        f.write("min_eccentricity, " + str(min(eccentricity)) + ", " + str(elapsed_time) + "\n")
        f.write("max_eccentricity, " + str(max(eccentricity)) + ", " + str(elapsed_time) + "\n")
        f.write("mean_eccentricity, " + str(mean(eccentricity)) + ", " + str(elapsed_time) + "\n")

        start_time = time.time()
        strength = kg.strength()
        elapsed_time = time.time() - start_time
        f.write("min_strength, " + str(min(strength)) + ", " + str(elapsed_time) + "\n")
        f.write("max_strength, " + str(max(strength)) + ", " + str(elapsed_time) + "\n")
        f.write("mean_strength, " + str(mean(strength)) + ", " + str(elapsed_time) + "\n")

        start_time = time.time()
        transitivity_local_undirected = kg.transitivity_local_undirected()
        elapsed_time = time.time() - start_time
        f.write("min_transitivity, " + str(min(transitivity_local_undirected)) + ", " + str(elapsed_time) + "\n")
        f.write("max_transitivity, " + str(max(transitivity_local_undirected)) + ", " + str(elapsed_time) + "\n")
        f.write("mean_transitivity, " + str(mean(transitivity_local_undirected)) + ", " + str(elapsed_time) + "\n")

        print("Connectivity metrics...")

        # Connectivity
        start_time = time.time()
        connected_components_strong = len(kg.connected_components(mode='strong'))
        elapsed_time = time.time() - start_time
        f.write("connected_components_strong, " + str(connected_components_strong) + ", " + str(elapsed_time)+"\n")

        start_time = time.time()
        connected_components_weak = len(kg.connected_components(mode='weak'))
        elapsed_time = time.time() - start_time
        f.write("connected_components_weak, " + str(connected_components_weak) + ", " + str(elapsed_time)+"\n")

        elapsed_time_global = time.time() - start_global_time
        print(elapsed_time_global)

"""
    # Connectedness
    # all_minimal_st_separators, '+str(kg.all_minimal_st_separators())+",")
    f.write("minimum_size_separators, " + str(minimum_size_separators) + ", " + str(elapsed_time)+"\n")

    # modularity, '+str(kg.modularity(membership=))+",")
    # independence_number, '+str(kg.independence_number())+",")
    # maximal_independent_vertex_sets, '+str(kg.maximal_independent_vertex_sets())+",")
    # largest_independent_vertex_sets, '+str(kg.largest_independent_vertex_sets())+",")

    # Assortativity
    # assortativity, '+str(kg.assortativity())+",")
    # assortativity_nominal, '+str(kg.assortativity_nominal())+",")

    # Flow
    # maxflow_value, '+str(kg.maxflow_value())+",")
    # 'edge_connectivity, '+str(kg.edge_connectivity(),  # only direct
    # 'vertex_connectivity, '+str(kg.vertex_connectivity()  # only direct

    #Vertex
    similarity_dice = max(max(kg.similarity_dice()))
    similarity_jaccard = kg.similarity_jaccard()
    similarity_inverse_log_weighted = kg.similarity_inverse_log_weighted()
    laplacian = kg.laplacian()
"""
