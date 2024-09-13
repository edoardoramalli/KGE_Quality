columns = ['dataset', 'model', 'property', 'value',
           'training_gpu_energy', 'training_duration',
           'information_loss', 'original_average_pagerank',
           'original_average_degree', 'original_average_betweenness',
           'original_average_harmonic_centrality', 'average_harmonic_centrality',
           'average_betweenness', 'average_pagerank', 'n_relationships', 'order',
           'size', 'density', 'diameter', 'transitivity_undirected',
           'connected_components_weak', 'connected_components_weak_relative_nodes',
           'connected_components_weak_relative_arcs', 'girth',
           'vertex_connectivity', 'edge_connectivity', 'clique_number',
           'average_degree', 'max_degree', 'average_indegree',
           'average_outdegree', 'radius',
           'average_path_length', 'assortativity_degree', 'mean_eccentricity',
           'min_eccentricity', 'max_eccentricity',
           'centralization', 'motifs_randesu_no', 'num_farthest_points',
           'mincut_value', 'len_feedback_arc_set', 'mean_authority_score',
           'min_authority_score', 'max_authority_score', 'mean_hub_score',
           'min_hub_score', 'max_hub_score', 'min_closeness', 'max_closeness',
           'mean_closeness', 'min_constraint', 'max_constraint', 'mean_constraint',
           'max_coreness', 'min_coreness', 'mean_coreness', 'max_strength',
           'min_strength', 'mean_strength', 'entropy']

def prepare(data):
    
    data = data[columns]
    
    data = data.groupby(['dataset', 'model', 'property', 'value']).mean().reset_index()
    
    data = data[(data["value"] <= 0) & ((data["property"] == "pagerank") | (data["property"] == "harmonic_centrality") | (data["property"] == "baseline"))]
    
    ccc = ['dataset', 'model', 'property', 'value',
           'training_gpu_energy', 'training_duration',
           'information_loss', 'order',
           'size']

    df = data[ccc].copy()
    
    df = df.rename(columns={'size' :'size_'})
    
    df["energy_kwh"] = 0
    df["order_perc"] = 0
    df["size_perc"] = 0
    df["loss_perc"] = 0
    df["time_perc"] = 0

    def get_perc_energy(kwh, kwh_baseline):
      return 1-(kwh/kwh_baseline)
    
    def get_perc(value, baseline):
      return value/baseline
    
    for i in range(0,len(df)):
        df.iloc[i,9] = get_perc_energy(df.iloc[i,4], df[(df["property"] == "baseline") & (df["dataset"] == df.iloc[i,0]) & (df["model"] == df.iloc[i,1])].training_gpu_energy.values[0])
    
    for i in range(0,len(df)):
        df.iloc[i,10] = get_perc(df.iloc[i,7], df[(df["property"] == "baseline") & (df["dataset"] == df.iloc[i,0]) & (df["model"] == df.iloc[i,1])].order.values[0])
    
    for i in range(0,len(df)):
        df.iloc[i,11] = get_perc(df.iloc[i,8], df[(df["property"] == "baseline") & (df["dataset"] == df.iloc[i,0]) & (df["model"] == df.iloc[i,1])].size_.values[0])
    
    for i in range(0,len(df)):
        df.iloc[i,12] = get_perc(df.iloc[i,6], df[(df["property"] == "baseline") & (df["dataset"] == df.iloc[i,0]) & (df["model"] == df.iloc[i,1])].information_loss.values[0])
    
    for i in range(0,len(df)):
        df.iloc[i,13] = get_perc(df.iloc[i,5], df[(df["property"] == "baseline") & (df["dataset"] == df.iloc[i,0]) & (df["model"] == df.iloc[i,1])].training_duration.values[0])

    df['time_perc'] = 1-df['time_perc']
    df = df[df.dataset != 'Kinships']

    return df
