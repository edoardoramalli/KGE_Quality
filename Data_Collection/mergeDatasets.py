from pykeen.datasets import *
from tqdm import tqdm
import os
import pandas as pd
from tool import datasets

for dataset_name in tqdm(datasets):

    dataset = eval(dataset_name)()

    training_numpy = dataset.training.triples
    validation_numpy = dataset.validation.triples
    testing_numpy = dataset.testing.triples

    training_pd = pd.DataFrame(training_numpy, columns=['from', 'rel', 'to'])
    validation_pd = pd.DataFrame(validation_numpy, columns=['from', 'rel', 'to'])
    testing_pd = pd.DataFrame(testing_numpy, columns=['from', 'rel', 'to'])

    dataset_complete = pd.concat([training_pd, validation_pd, testing_pd])

    os.makedirs('./Datasets_Complete/{}/'.format(dataset_name), exist_ok=True)

    dataset_complete.to_csv('./Datasets_Complete/{}/{}.csv'.format(dataset_name, dataset_name),
                            sep='\t',
                            header=False,
                            index=False)
