from multiprocessing import Pool
import tqdm
import time
import pickle
import os
import tqdm
import glob
from measureGraph import measure_graph
import warnings
warnings.filterwarnings("ignore")


directed = True


def _foo(file):
    with open(file, 'rb') as f:
        pick = pickle.load(f)

    training = pick['training']

    # if 'graph_metrics' in pick:
    #     print('{} already computed'.format(file))
    #     return 0

    props = measure_graph(dataset=training, directed=directed, dataset_name=pick['dataset'], c_split=pick['c_split'])

    pick['graph_metrics'] = {'directed': directed, **props}

    with open(file, 'wb') as f:
        pickle.dump(pick, f)

    return 0


if __name__ == '__main__':
    # job_list = glob.glob('/home/ramalli/results/RotatE/**/**/instance.pickle') # BASTA per un solo modello dopo è tutto uguale
    job_list = glob.glob('/home/ramalli/tmp_results/**/instance.pickle')
    with Pool(25) as p:
        r = list(tqdm.tqdm(p.imap(_foo, job_list), total=len(job_list)))


