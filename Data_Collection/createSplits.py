from pykeen.triples import TriplesFactory
from tqdm import tqdm

import random
import pickle
import os
from tqdm import tqdm
from tool import datasets

splitter = 'CoverageSplitter'
random_state = 1996
n_split = 5
ratios = [0.8, 0.1, 0.1]

for dataset_name in tqdm(datasets):
    random.seed(random_state)
    dataset_path = './Datasets_Complete/{}/{}.csv'.format(dataset_name, dataset_name)
    tf = TriplesFactory.from_path(dataset_path)
    for i in range(n_split):
        c_random_state = random.choice(range(100))
        training, validation, testing = tf.split(ratios=ratios,
                                                 method=splitter,
                                                 random_state=c_random_state)

        relation_to_id = training.relation_to_id
        entity_to_id = training.entity_to_id

        folder_path = './Datasets_Complete/{}/Split_{}/'.format(dataset_name, i)
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, 'instance.pickle'), 'wb') as f:
            pickle.dump({
                'dataset': {
                    'dataset_name': dataset_name,
                    'random_state': random_state,
                    'c_random_state': c_random_state,
                    'splitter': splitter,
                    'n_split': n_split,
                    'c_split': i,
                    'ratios': ratios,
                    'entity_to_id': entity_to_id,
                    'relation_to_id': relation_to_id,
                    'training': training.triples.tolist(),
                    'testing': testing.triples.tolist(),
                    'validation': validation.triples.tolist()
                }}, f)

