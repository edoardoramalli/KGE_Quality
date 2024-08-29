from tqdm import tqdm
import glob
import pickle


to_be_replaced = 'TransE'

fnames = glob.glob('/home/ramalli/results/{}/**/**/instance.pickle'.format(to_be_replaced))
# fnames_complex = glob.glob('/home/ramalli/results/ComplEx/**/**/instance.pickle')



res = fnames


for f in tqdm(res):
    with open(f, 'rb') as ff:
        d = pickle.load(ff)

    # if 'graph_metrics' not in d:

    with open(f.replace(to_be_replaced, 'RotatE'), 'rb') as _ff:
        _pick = pickle.load(_ff)
    d['graph_metrics'] = _pick['graph_metrics']

    with open(f, 'wb') as ff:
        pickle.dump(d, ff)
