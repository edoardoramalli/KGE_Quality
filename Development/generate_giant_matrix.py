import glob
import pickle
import pandas as pd
from tqdm import tqdm


def create_df(file_names):
    rr = []

    for f in tqdm(file_names):
        with open(f, 'rb') as ff:
            d = pickle.load(ff)

        tmp = {}

        testing_df = pd.DataFrame(d['testing'], columns=['h', 'r', 't'])
        testing_min_df = pd.DataFrame(d['min_testing'], columns=['h', 'r', 't'])

        tmp['dataset'] = d['dataset']
        tmp['c_split'] = d['c_split']
        tmp['model'] = d['model']
        tmp['property'] = d['property']
        tmp['value'] = d['value']

        tmp['testing_entities'] = len(pd.concat([testing_df.h, testing_df.t], axis=0).unique())
        tmp['testing_min_entities'] = len(pd.concat([testing_min_df.h, testing_min_df.t], axis=0).unique())

        tmp['testing_triples'] = len(testing_df)
        tmp['testing_min_triples'] = len(testing_min_df)

        tmp['results_hits_at_10'] = d['results']['both']['realistic']['hits_at_10']
        tmp['results_hits_at_5'] = d['results']['both']['realistic']['hits_at_5']
        tmp['results_hits_at_3'] = d['results']['both']['realistic']['hits_at_3']
        tmp['results_hits_at_1'] = d['results']['both']['realistic']['hits_at_1']
        tmp['results_inverse_harmonic_mean_rank'] = d['results']['both']['realistic']['inverse_harmonic_mean_rank']
        tmp['results_arithmetic_mean_rank'] = d['results']['both']['realistic']['arithmetic_mean_rank']

        tmp['results_min_hits_at_10'] = d['results_min']['both']['realistic']['hits_at_10']
        tmp['results_min_hits_at_5'] = d['results_min']['both']['realistic']['hits_at_5']
        tmp['results_min_hits_at_3'] = d['results_min']['both']['realistic']['hits_at_3']
        tmp['results_min_hits_at_1'] = d['results_min']['both']['realistic']['hits_at_1']
        tmp['results_min_arithmetic_mean_rank'] = d['results_min']['both']['realistic']['arithmetic_mean_rank']
        tmp['results_min_inverse_harmonic_mean_rank'] = d['results_min']['both']['realistic']['inverse_harmonic_mean_rank']


        # tmp['training_gpu_power'] = d['trackers']['training_tracker']['gpu_power']
        tmp['training_cpu_energy'] = d['trackers']['training_tracker']['cpu_energy']
        tmp['training_gpu_energy'] = d['trackers']['training_tracker']['gpu_energy']
        tmp['training_ram_energy'] = d['trackers']['training_tracker']['ram_energy']
        tmp['training_duration'] = d['trackers']['training_tracker']['duration']

        # tmp['prediction_gpu_power'] = d['trackers']['prediction_tracker']['gpu_power']
        tmp['prediction_cpu_energy'] = d['trackers']['training_tracker']['cpu_energy']
        tmp['prediction_gpu_energy'] = d['trackers']['training_tracker']['gpu_energy']
        tmp['prediction_ram_energy'] = d['trackers']['training_tracker']['ram_energy']
        tmp['prediction_duration'] = d['trackers']['prediction_tracker']['duration']

        # tmp['evaluation_gpu_power'] = d['trackers']['evaluation_tracker']['gpu_power']
        tmp['evaluation_cpu_energy'] = d['trackers']['training_tracker']['cpu_energy']
        tmp['evaluation_gpu_energy'] = d['trackers']['training_tracker']['gpu_energy']
        tmp['evaluation_ram_energy'] = d['trackers']['training_tracker']['ram_energy']
        tmp['evaluation_duration'] = d['trackers']['evaluation_tracker']['duration']

        # tmp['evaluation_min_gpu_power'] = d['trackers']['evaluation_min_tracker']['gpu_power']
        tmp['evaluation_min_cpu_energy'] = d['trackers']['training_tracker']['cpu_energy']
        tmp['evaluation_min_gpu_energy'] = d['trackers']['training_tracker']['gpu_energy']
        tmp['evaluation_min_ram_energy'] = d['trackers']['training_tracker']['ram_energy']
        tmp['evaluation_min_duration'] = d['trackers']['evaluation_min_tracker']['duration']

        tmp['property'] = d['property']
        tmp['value'] = d['value']

        tmp = {**tmp, **d['graph_metrics']}

        rr.append(tmp)

    return pd.DataFrame(rr)


fnames_transe = glob.glob('/home/ramalli/results/TransE/**/**/instance.pickle')
fnames_complex = glob.glob('/home/ramalli/results/ComplEx/**/**/instance.pickle')
fnames_rotate = glob.glob('/home/ramalli/results/RotatE/**/**/instance.pickle')


# names = glob.glob('/Users/edoardo/Desktop/instance.pickle') # For dev
names = glob.glob('/home/ramalli/tmp_results/**/instance.pickle')

# df = create_df(fnames_transe + fnames_complex + fnames_rotate)
df = create_df(names) # For dev

# df = create_df(fnames_transe)

df.to_csv('giant_matrix_2024_04_09.csv', index=False)


# print(df.head())
