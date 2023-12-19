import pandas as pd


def remove_entities(df_dataset, entities_to_be_removed):
    to_be_removed = list(set(entities_to_be_removed))
    mask_h = df_dataset['h'].isin(to_be_removed)
    mask_t = df_dataset['t'].isin(to_be_removed)
    return df_dataset[(~mask_h) & (~mask_t)]


def compute_test_set_min(testing_df, sorted_desc_df, property_name, max_remove_top, max_remove_bottom,):

    quantile_top = sorted_desc_df.quantile(1 - max_remove_top)[property_name]
    top_entities_to_be_removed = sorted_desc_df[sorted_desc_df[property_name] > quantile_top]['entity'].to_list()

    quantile_bottom = sorted_desc_df.quantile(max_remove_bottom)[property_name]

    bottom_entities_to_be_removed = sorted_desc_df[sorted_desc_df[property_name] < quantile_bottom][
        'entity'].to_list()

    entities_to_be_removed = top_entities_to_be_removed + bottom_entities_to_be_removed
    entities_to_be_removed = set(entities_to_be_removed)

    new_testing_df = remove_entities(df_dataset=testing_df, entities_to_be_removed=entities_to_be_removed)

    return new_testing_df


# entities = [3]
# # #
# df = pd.DataFrame([{'h': 1, 'r': 2, 't': 1}, {'h': 3, 'r': 2, 't': 2},{'h': 2, 'r': 2, 't': 1}, {'h': 1, 'r': 2, 't': 2},])
# # #
# # #
# # #
# # # print()
# #
# print(remove_entities(df, entities).values.tolist())



