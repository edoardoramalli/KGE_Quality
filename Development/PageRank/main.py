import pickle
import pandas as pd

url = '/Users/edoardo/Documents/Projects/KGE_Quality/Data_Collection/Datasets_Complete/FB15k/Split_0/instance.pickle'







with open(url, 'rb') as f:
    obj = pickle.load(f)


df = pd.DataFrame(obj['dataset']['training'], columns=['h', 'r', 't'])
df_no_rel = df[['h', 't']]

c_entities = set(df['h']) | set(df['t'])

# lost = missing_entities(df, [1, 2, 3])

# a = compute_information_loss(df, lost)

# print(a)
print(obj['measures_training'].keys())
# print(o.sum())
