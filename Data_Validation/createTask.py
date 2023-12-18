import pickle
import pandas as pd
import time
import string
# import random
import os

# random.seed(int(time.time()))

from Ablation.removeEntity import *

dt = ['FB15k237']# ['WN18', 'WN18RR', 'FB15k', 'FB15k237', 'YAGO310']

models = ['ComplEx']

properties = ['harmonic_centrality']#, 'baseline', 'degree', 'pagerank', 'harmonic_centrality', 'betweenness']

top = [0.01, 0.03]# 0.01, 0.03, 0.05, 0.10]

bottom = [0.05, 0.10]# 0.05, 0.10, 0.15, 0.20]

n_split = [0]

max_remove_top = max(top)
max_remove_bottom = max(bottom)


def create_new_pick(_dataset_name, _c_split, _dataset, _model, _entities_to_be_removed, _training_df, _validation_df,
                    _testing_df,
                    _property_name, _value, _min_testing):
    __entities_to_be_removed = set(_entities_to_be_removed)

    new_training_df = remove_entities(df_dataset=_training_df, entities_to_be_removed=__entities_to_be_removed)
    new_validation_df = remove_entities(df_dataset=_validation_df, entities_to_be_removed=__entities_to_be_removed)
    new_testing_df = remove_entities(df_dataset=_testing_df, entities_to_be_removed=__entities_to_be_removed)

    # ['dataset_name', 'random_state', 'c_random_state', 'splitter', 'n_split', 'c_split', 'ratios',
    #  'entity_to_id', 'relation_to_id', 'training', 'testing', 'validation']

    new_pick = {
        'dataset': _dataset_name,
        'c_split': _c_split,
        'model': _model,
        'entity_to_id': _dataset['entity_to_id'],
        'relation_to_id': _dataset['relation_to_id'],
        'training': new_training_df.values.tolist(),
        'validation': new_validation_df.values.tolist(),
        'testing': new_testing_df.values.tolist(),
        'min_testing': min_testing.values.tolist(),
        'property': _property_name,
        'value': _value,
    }

    return new_pick


def record_task(pick):
    _dataset = pick['dataset']
    _c_split = pick['c_split']
    _model = pick['model']
    _property = pick['property']
    _value = pick['value']

    folder_name = '{}_{}_{}_{}_{}'.format(_dataset, _c_split, _model, _property, _value)

    # print(pick)
    # with open('../Training/register.csv', 'a+') as f:
    #     f.write('{},{},{},{},{},waiting,{}\n'.format(_dataset, _c_split, _model, _property, _value, folder_name))

    project_path = '../Training/TODO/{}'.format(folder_name)

    print(project_path)

    os.makedirs(project_path, exist_ok=True)

    with open(os.path.join(project_path, 'instance.pickle'), 'wb') as f:
        pickle.dump(pick, f)

if __name__ == '__main__':

    for dataset_name in dt:
        for c_split in n_split:
            folder_path = '../Data_Collection/Datasets_Complete/{}/Split_{}/instance.pickle'.format(dataset_name, c_split)
            with open(folder_path, 'rb') as f:
                pick = pickle.load(f)
            for property_name in properties:

                df_measures = pick['measures_training']['dataframe']

                dataset = pick['dataset']

                training_df = pd.DataFrame(dataset['training'], columns=['h', 'r', 't'])
                validation_df = pd.DataFrame(dataset['validation'], columns=['h', 'r', 't'])
                testing_df = pd.DataFrame(dataset['testing'], columns=['h', 'r', 't'])

                for model in models:

                    if property_name == 'baseline':
                        new_pick = {
                            'dataset': dataset_name,
                            'c_split': c_split,
                            'model': model,
                            'entity_to_id': dataset['entity_to_id'],
                            'relation_to_id': dataset['relation_to_id'],
                            'training': training_df.values.tolist(),
                            'validation': validation_df.values.tolist(),
                            'testing': testing_df.values.tolist(),
                            'min_testing': testing_df.values.tolist(),
                            'property': property_name,
                            'value': 0,
                        }
                        # print('qua')
                        # print(new_pick.keys())
                        record_task(new_pick)
                    else:

                        sorted_desc_df = df_measures.sort_values(property_name, ascending=False)

                        min_testing = compute_test_set_min(testing_df, sorted_desc_df, property_name, max_remove_top,
                                                           max_remove_bottom)

                        for t in top:
                            quantile_top = sorted_desc_df.quantile(1 - t)[property_name]
                            top_entities_to_be_removed = sorted_desc_df[sorted_desc_df[property_name] >= quantile_top][
                                'entity'].to_list()

                            new_pick = create_new_pick(dataset_name, c_split, dataset, model, top_entities_to_be_removed,
                                                       training_df,
                                                       validation_df, testing_df, property_name, t, min_testing)

                            record_task(new_pick)

                        for b in bottom:
                            quantile_bottom = sorted_desc_df.quantile(b)[property_name]
                            bottom_entities_to_be_removed = \
                                sorted_desc_df[sorted_desc_df[property_name] <= quantile_bottom]['entity'].to_list()

                            new_pick = create_new_pick(dataset_name, c_split, dataset, model, bottom_entities_to_be_removed,
                                                       training_df,
                                                       validation_df, testing_df, property_name, -b, min_testing)

                            record_task(new_pick)


