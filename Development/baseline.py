from pykeen.pipeline import pipeline
from Training.readConfig import get_hp
import json

# Codice per riprodurre la baseline con i dataset originali

dataset_name = 'FB15k'
model_name = 'ComplEx'

print("BASELINE:", dataset_name, model_name)

HP = get_hp(dataset_name=dataset_name, model_name=model_name)

pipeline_result = pipeline(
    dataset=dataset_name,
    dataset_kwargs=HP['dataset_kwargs'],
    model=model_name,
    random_seed=HP['random_seed'],
    training_loop=HP['training_loop'],
    model_kwargs={
        'embedding_dim': HP['embedding_dim'],
        **HP['model_kwargs']
    },
    loss=HP['loss'],
    loss_kwargs=HP['loss_kwargs'],
    training_kwargs={
        'batch_size': HP['batch_size'],
        'label_smoothing': HP['label_smoothing']
    },
    optimizer=HP['optimizer'],
    optimizer_kwargs={
        **HP['optimizer_kwargs']
    },
    negative_sampler=HP['negative_sampler'],
    negative_sampler_kwargs=HP['negative_sampler_kwargs'],
    epochs=HP['num_epochs'],
    regularizer=HP['regularizer'],
    regularizer_kwargs=HP['regularizer_kwargs'],
)

with open('./Baseline/{}_{}.json'.format(dataset_name, model_name), 'w') as f:
    json.dump(
        {'HP': HP,
         'dataset': dataset_name,
         'model': model_name,
         'results': pipeline_result.metric_results.to_dict()},
        f,
        indent=4
    )

