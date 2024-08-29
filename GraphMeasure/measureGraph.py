import pandas as pd
import igraph as ig
import numpy as np
import pickle


def compute_information_loss(_df, entity_lost, reference_property='pagerank'):
    what_left_df = _df[~_df['entity'].isin(entity_lost)]
    return {'information_loss': what_left_df[reference_property].sum(),
            'original_average_pagerank': _df['pagerank'].mean(),
            'original_average_degree': _df['degree'].mean(),
            'original_average_betweenness': _df['betweenness'].mean(),
            'original_average_harmonic_centrality': _df['harmonic_centrality'].mean(),
            }


def missing_entities(_df, current_entity):
    return set(_df['entity']) - set(current_entity)


def get_baseline_pickle(c_dataset, c_split):
    path = '/home/ramalli/KGE_Quality/Data_Collection/Datasets_Complete/{}/Split_{}/instance.pickle'.format(c_dataset,
                                                                                                            c_split)
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['measures_training']['dataframe']


def measure_graph(dataset, directed, dataset_name, c_split):
    from scipy.stats import entropy
    df = pd.DataFrame(dataset, columns=['h', 'r', 't'])
    df_no_rel = df[['h', 't']]
    relationships = df[['r']]
    c_entities = list(set(df['h']) | set(df['t']))
    df_no_rel = df_no_rel.drop_duplicates()  # era commentato ma carlo mi aveva detto di computare le metriche droppando i duplicates all'epoca
    # kg = ig.Graph.DataFrame(df_no_rel, directed=directed)
    kg = ig.Graph.TupleList(df_no_rel.itertuples(index=False), directed=directed)

    baseline_df = get_baseline_pickle(dataset_name, c_split)
    lost_entities = missing_entities(baseline_df, c_entities)
    information_loss = compute_information_loss(baseline_df, lost_entities)

    p = {**information_loss}

    p['average_harmonic_centrality'] = np.mean(kg.harmonic_centrality())
    p['average_betweenness'] = np.mean(kg.betweenness())
    p['average_pagerank'] = np.mean(kg.pagerank())

    p['n_relationships'] = int(relationships.nunique())
    p['order'] = kg.vcount()
    p['size'] = kg.ecount()
    p['density'] = kg.density()
    p['diameter'] = kg.diameter(directed=directed)
    p['transitivity_undirected'] = kg.transitivity_undirected()
    p['connected_components_weak'] = len(kg.connected_components(mode='weak'))
    # relative number of nodes in the giant connected component
    p['connected_components_weak_relative_nodes'] = len(
        kg.connected_components(mode='weak').giant().vs) / p['order']
    # relative number of arcs in the giant connected component
    p['connected_components_weak_relative_arcs'] = len(
        kg.connected_components(mode='weak').giant().es) / p['size']
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
    p['min_eccentricity'] = min(ecc)
    p['max_eccentricity'] = max(ecc)

    p['eccentricity_deciles'] = list(np.quantile(ecc, np.linspace(0., 1., 11)))

    p['centralization'] = (p['size'] / (p['size'] - 2)) * (
            (p['max_degree'] / (p['size'] - 1)) - p['density'])

    # p['len_maximal_cliques'] = len(kg.maximal_cliques()) # TOO MUCH TIME ?
    # p['len_largest_cliques'] = len(kg.largest_cliques())

    p['motifs_randesu_no'] = kg.motifs_randesu_no()

    p['num_farthest_points'] = len(kg.farthest_points())

    p['mincut_value'] = kg.mincut_value()

    p['len_feedback_arc_set'] = len(kg.feedback_arc_set())

    authority_score = kg.authority_score()
    p['mean_authority_score'] = np.nanmean(authority_score)
    p['min_authority_score'] = min(authority_score)
    p['max_authority_score'] = max(authority_score)

    hub_score = kg.hub_score()
    p['mean_hub_score'] = np.nanmean(hub_score)
    p['min_hub_score'] = min(hub_score)
    p['max_hub_score'] = max(hub_score)

    closeness = kg.closeness()
    p['min_closeness'] = min(closeness)
    p['max_closeness'] = max(closeness)
    p['mean_closeness'] = np.nanmean(closeness)

    constraint = kg.constraint()
    p['min_constraint'] = min(constraint)
    p['max_constraint'] = max(constraint)
    p['mean_constraint'] = np.nanmean(constraint)

    coreness = kg.coreness()
    p['max_coreness'] = max(coreness)
    p['min_coreness'] = min(coreness)
    p['mean_coreness'] = np.nanmean(coreness)

    strength = kg.strength()
    p['max_strength'] = max(strength)
    p['min_strength'] = min(strength)
    p['mean_strength'] = np.nanmean(strength)

    p['entropy'] = entropy(relationships.value_counts().values, base=2)

    return p
