from Data_Collection.tool import datasets
from tqdm import tqdm
import pickle
import os

from pykeen.pipeline import pipeline

n_split = 5

for dataset_name in tqdm(datasets):

    for i in range(n_split):
        folder_path = '../Data_Collection/Datasets_Complete/{}/Split_{}/'.format(dataset_name, i)
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, 'instance.pickle'), 'rb') as f:
            pick = pickle.load(f)


        df = pick['measures_training']['dataframe']

        print(df.sort_values('degree'))



        # if not 'measures_training' in pick:
        #     print(dataset_name, i)

        exit()
