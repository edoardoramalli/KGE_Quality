from pykeen.datasets import *
from tqdm import tqdm

for dataset_name, dataset_class in tqdm(dataset_resolver.lookup_dict.items()):
    try:
        dataset = dataset_class()
    except ValueError:
        print('Error with', dataset_name)

# Run just one time to download on the local machine all the dataset
