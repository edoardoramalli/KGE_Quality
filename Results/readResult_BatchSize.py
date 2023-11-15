import pickle
import os
import torch

dirs = [x for x in os.listdir('./WN18') if os.path.isdir(os.path.join('./WN18', x))]

summary = {}

def walk_dict(diz, indent):
    if not isinstance(diz, dict):
        return '\r', indent
    else:
        keys = diz.keys()
        # print(keys)
        tmp = ''
        for k in keys:
            # tmp += '{}:\n'.format(k) + '  '*(indent + 1) + walk_dict(diz[k], indent + 2)[0]
            tmp += '  ' * (indent) + '{}:\n'.format(k) + walk_dict(diz[k], indent + 2)[0]
        return tmp, indent + 1



for dir in dirs:
    with open('WN18/{}/result.pickle'.format(dir), 'rb') as f:
        results = pickle.load(f)

    checkpoint = torch.load('WN18/{}/model_final.pt'.format(dir), map_location=torch.device('cpu'))

    losses = checkpoint['loss']

    batch_size = results['HP']['batch_size']
    num_epochs = results['HP']['num_epochs']
    evaluation = results['results']

    h_10 = evaluation['both']['realistic']['hits_at_10']
    h_5 = evaluation['both']['realistic']['hits_at_5']
    h_3 = evaluation['both']['realistic']['hits_at_3']
    h_1 = evaluation['both']['realistic']['hits_at_1']
    mr = evaluation['both']['realistic']['arithmetic_mean_rank']
    elapsed_training_time = results['elapsed_training_time']
    epoch_per_sec = num_epochs / elapsed_training_time


    summary[batch_size] = {
        'losses': losses,
        'num_epochs': num_epochs,
        'h_10': h_10,
        'h_5': h_5,
        'h_3': h_3,
        'h_1': h_1,
        'mr': mr,
        'elapsed_training_time': elapsed_training_time,
        'epoch_per_sec': epoch_per_sec,

    }

# print(summary)




# for k in evaluation:
#     print(k)
#     if isinstance(evaluation[k]

# prova = {'head': {'optimistic': {'stdev': 1, 'inverse': 2}}}

print(evaluation)

