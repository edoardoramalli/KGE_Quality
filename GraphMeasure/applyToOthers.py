from tqdm import tqdm
import glob
import pickle


# fnames_transe = glob.glob("../Training/DONE/*TransE*/*.pickle")
fnames_complex = glob.glob("../Training/DONE/*ComplEx*/*.pickle")

res = fnames_complex


for f in tqdm(res):
    with open(f, 'rb') as ff:
        d = pickle.load(ff)

    if 'graph_metrics' not in d:
        with open(f.replace('ComplEx', 'RotatE'), 'rb') as _ff:
            _pick = pickle.load(_ff)
        d['graph_metrics'] = _pick['graph_metrics']

        with open(f, 'wb') as ff:
            pickle.dump(d, ff)
