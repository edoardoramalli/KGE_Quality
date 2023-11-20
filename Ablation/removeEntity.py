import pandas as pd


def remove_entities(df_dataset, entities_to_be_removed):
    to_be_removed = list(set(entities_to_be_removed))
    mask_h = df_dataset['h'].isin(to_be_removed)
    mask_t = df_dataset['t'].isin(to_be_removed)
    return df_dataset[(~mask_h) & (~mask_t)]


# entities = [3]
# #
# df = pd.DataFrame([{'h': 1, 'r': 2, 't': 1}, {'h': 3, 'r': 2, 't': 2},{'h': 2, 'r': 2, 't': 1}, {'h': 1, 'r': 2, 't': 2},])
# #
# #
# #
# # print()
#
# print(remove_entities(df, entities).values.tolist())



