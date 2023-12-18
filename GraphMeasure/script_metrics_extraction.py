import igraph as ig
import pandas as pd
import random
from statistics import mean
import pickle
import time

#NB reciprocity, min_diversity, max_diversity, mean_diversity only if the graph is direct otherwise can be removed
#NB num_farthest_points, mincut_value are always 3 and 0 (I don't know why) maybe could be removed as well

# escludere ['degree', 'pagerank', 'harmonic_centrality', 'betweenness']
names = ['WN18RR', 'WN18', 'FB15k', 'FB15k237', 'YAGO310']

random.seed(0)
directed = False

def extract_metrics():

    with open("metrics_value.csv", 'w') as f1:
        with open("metrics_time.csv", 'w') as f2:

            f1.write("name,order,size,max_degree,mean_degree,diameter,girth,radius,average_path_length,clique_number,len_maximal_cliques,len_largest_cliques,motifs_randesu_no,num_farthest_points,mincut_value,len_feedback_arc_set,assortativity_degree,density,transitivity_undirected,reciprocity,min_diversity,max_diversity,mean_diversity,mean_authority_score,mean_hub_score,min_closeness,min_constraint,max_constraint,mean_constraint,max_coreness,mean_coreness,min_eccentricity,max_eccentricity,mean_eccentricity,max_strength,mean_strength,connected_components_strong,connected_components_weak\n")
            f2.write("name,order,size,max_degree,mean_degree,diameter,girth,radius,average_path_length,clique_number,len_maximal_cliques,len_largest_cliques,motifs_randesu_no,num_farthest_points,mincut_value,len_feedback_arc_set,assortativity_degree,density,transitivity_undirected,reciprocity,min_diversity,max_diversity,mean_diversity,mean_authority_score,mean_hub_score,min_closeness,min_constraint,max_constraint,mean_constraint,max_coreness,mean_coreness,min_eccentricity,max_eccentricity,mean_eccentricity,max_strength,mean_strength,connected_components_strong,connected_components_weak\n")

            for n in names:

                kg_path = '../Data_Collection/Datasets_Complete/' + n + '/Split_0/instance.pickle'

                with open(kg_path, 'rb') as f:
                    pick = pickle.load(f)
                dataset = pick['dataset']
                df = pd.DataFrame(dataset['training'], columns=['h', 'r', 't'])
                df_no_rel = df[['h', 't']]
                df_no_rel = df_no_rel.drop_duplicates()
                kg = ig.Graph.DataFrame(df_no_rel, directed=directed)
                #kg = ig.Graph(50)

                f1.write(n+",")
                f2.write(n+",")

                start_time = time.time()
                order = kg.vcount()
                elapsed_time = time.time() - start_time
                f1.write(str(order) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                size = kg.ecount()
                elapsed_time = time.time() - start_time
                f1.write(str(size) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                degree = kg.degree()
                elapsed_time = time.time() - start_time
                max_degree = max(degree)
                f1.write(str(max_degree) + ",")
                f2.write(str(elapsed_time) + ",")

                mean_degree = mean(degree)
                f1.write(str(mean_degree) + ",")
                f2.write(str(0) + ",")

                start_time = time.time()
                diameter = kg.diameter(directed=directed)
                elapsed_time = time.time() - start_time
                f1.write(str(diameter) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                girth = kg.girth()
                elapsed_time = time.time() - start_time
                f1.write(str(girth) + ",")
                f2.write(str(elapsed_time) + ",")

                #### CAMILLA ####

                print("Global metrics...")

                # Global
                start_time = time.time()
                radius = kg.radius()
                elapsed_time = time.time() - start_time
                f1.write(str(radius) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                average_path_length = kg.average_path_length()
                elapsed_time = time.time() - start_time
                f1.write(str(average_path_length) + ",")
                f2.write(str(elapsed_time) + ",")

                print("Clique & motifs metrics...")

                # Clique & motifs
                start_time = time.time()
                clique_number = kg.clique_number()
                elapsed_time = time.time() - start_time
                f1.write(str(clique_number) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                maximal_cliques = len(kg.maximal_cliques())
                elapsed_time = time.time() - start_time
                f1.write(str(maximal_cliques) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                largest_cliques = len(kg.largest_cliques())
                elapsed_time = time.time() - start_time
                f1.write(str(largest_cliques) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                motifs_randesu_no = kg.motifs_randesu_no()
                elapsed_time = time.time() - start_time
                f1.write(str(motifs_randesu_no) + ",")
                f2.write(str(elapsed_time) + ",")

                print("Optimality metrics...")

                # Optimality
                start_time = time.time()
                farthest_points = len(kg.farthest_points())
                elapsed_time = time.time() - start_time
                f1.write(str(farthest_points) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                mincut_value = kg.mincut_value()
                elapsed_time = time.time() - start_time
                f1.write(str(mincut_value) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                feedback_arc_set = len(kg.feedback_arc_set())
                elapsed_time = time.time() - start_time
                f1.write(str(feedback_arc_set) + ",")
                f2.write(str(elapsed_time) + ",")

                print("Assortativity metrics...")

                # Assortativity
                start_time = time.time()
                assortativity_degree = kg.assortativity_degree()
                elapsed_time = time.time() - start_time
                f1.write(str(assortativity_degree) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                density = kg.density()
                elapsed_time = time.time() - start_time
                f1.write(str(density) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                transitivity_undirected = kg.transitivity_undirected()
                elapsed_time = time.time() - start_time
                f1.write(str(transitivity_undirected) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                reciprocity = kg.reciprocity()
                elapsed_time = time.time() - start_time  # only direct
                f1.write(str(reciprocity) + ",")
                f2.write(str(elapsed_time) + ",")

                print("Vertex metrics...")

                # Vertex
                start_time = time.time()
                diversity = kg.diversity()
                elapsed_time = time.time() - start_time
                f1.write(str(min(diversity)) + ",")
                f2.write(str(elapsed_time) + ",")
                f1.write(str(max(diversity)) + ",")
                f2.write(str(0) + ",")
                f1.write(str(mean(diversity)) + ",")
                f2.write(str(0) + ",")

                print("Structural metrics...")

                # Structural
                start_time = time.time()
                authority_score = kg.authority_score()
                elapsed_time = time.time() - start_time
                f1.write(str(mean(authority_score)) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                hub_score = kg.hub_score()
                elapsed_time = time.time() - start_time
                f1.write(str(mean(hub_score)) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                closeness = kg.closeness()
                elapsed_time = time.time() - start_time
                f1.write(str(min(closeness)) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                constraint = kg.constraint()
                elapsed_time = time.time() - start_time
                f1.write(str(min(constraint)) + ",")
                f2.write(str(elapsed_time) + ",")
                f1.write(str(max(constraint)) + ",")
                f2.write(str(0) + ",")
                f1.write(str(mean(constraint)) + ",")
                f2.write(str(0) + ",")

                start_time = time.time()
                coreness = kg.coreness()
                elapsed_time = time.time() - start_time
                f1.write(str(max(coreness)) + ",")
                f2.write(str(elapsed_time) + ",")
                f1.write(str(mean(coreness)) + ",")
                f2.write(str(0) + ",")

                start_time = time.time()
                eccentricity = kg.eccentricity()
                elapsed_time = time.time() - start_time
                f1.write(str(min(eccentricity)) + ",")
                f2.write(str(elapsed_time) + ",")
                f1.write(str(max(eccentricity)) + ",")
                f2.write(str(0) + ",")
                f1.write(str(mean(eccentricity)) + ",")
                f2.write(str(0) + ",")

                start_time = time.time()
                strength = kg.strength()
                elapsed_time = time.time() - start_time
                f1.write(str(max(strength)) + ",")
                f2.write(str(elapsed_time) + ",")
                f1.write(str(mean(strength)) + ",")
                f2.write(str(0) + ",")

                print("Connectivity metrics...")

                # Connectivity
                start_time = time.time()
                connected_components_strong = len(kg.connected_components(mode='strong'))
                elapsed_time = time.time() - start_time
                f1.write(str(connected_components_strong) + ",")
                f2.write(str(elapsed_time) + ",")

                start_time = time.time()
                connected_components_weak = len(kg.connected_components(mode='weak'))
                elapsed_time = time.time() - start_time
                f1.write(str(connected_components_weak))
                f2.write(str(elapsed_time))

                f1.write("\n")
                f2.write("\n")

            f2.close()
        f1.close()

if __name__ == '__main__':

        print("Start...")
        start_global_time = time.time()

        extract_metrics()

        print("...End")

        elapsed_time_global = time.time() - start_global_time
        print(elapsed_time_global)
