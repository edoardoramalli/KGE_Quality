# Get a training dataset
import random
import os
import glob
import shutil

path = "./TODO/**/instance.pickle"

tasks = list(glob.glob(path, recursive=True))

if len(tasks) == 0:
    exit()

pickle_file = random.choice(tasks)

print('FILE: {}'.format(pickle_file))

project_path = os.path.dirname(pickle_file)

project_name = os.path.basename(project_path)

shutil.move(project_path, './RUNNING/')

project_path = os.path.join('./RUNNING/', project_name)

pickle_file = pickle_file.replace('TODO', 'RUNNING')

from codecarbon import OfflineEmissionsTracker

tracker = OfflineEmissionsTracker(
    country_iso_code="ITA",
    tracking_mode='process',
    save_to_file=False,
    log_level='critical')

tracker.start_task("import")
from pykeen.models import ComplEx, TransE, RotatE
from torch.optim import Adam, Adagrad
from pykeen.training import SLCWATrainingLoop, LCWATrainingLoop
from pykeen.training.callbacks import TrainingCallback
from pykeen.evaluation import RankBasedEvaluator
from pykeen.predict import predict_triples
from pykeen.triples import TriplesFactory
import torch
import platform
from readConfig import get_hp

import time
import pickle
import numpy as np

import_tracker = tracker.stop_task()
print('Library imported')


class Edo(TrainingCallback):

    def __init__(self, frequency: int, project_path: str, **kwargs):
        super().__init__()
        self.frequency = frequency
        self.project_path = project_path
        if 'triples_factory' in kwargs:
            self.triples_factory = kwargs['triples_factory']

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs):
        if epoch % self.frequency == 0:
            path_file = os.path.join(self.project_path, 'model_*.pt')
            os.system('rm -f {}'.format(path_file))
            self.training_loop._save_state(
                path=os.path.join(self.project_path, 'model_{}.pt'.format(epoch)),
                triples_factory=self.triples_factory)

    def post_train(self, losses, **kwargs) -> None:
        path_file = os.path.join(self.project_path, 'model_*.pt')
        os.system('rm -f {}'.format(path_file))
        self.training_loop._save_state(
            path=os.path.join(self.project_path, 'model_final.pt'),
            triples_factory=self.triples_factory)


tracker.start_task("load_dataset")
with open(pickle_file, 'rb') as f:
    dataset_pickle = pickle.load(f)

print('DATASET: {}'.format(dataset_pickle['dataset']))

training_triples_factory = TriplesFactory(
    mapped_triples=torch.tensor(np.array(dataset_pickle['training']), dtype=torch.long),
    entity_to_id=dataset_pickle['entity_to_id'],
    relation_to_id=dataset_pickle['relation_to_id'],
)

validation_triples_factory = TriplesFactory(
    mapped_triples=torch.tensor(np.array(dataset_pickle['validation']), dtype=torch.long),
    entity_to_id=training_triples_factory.entity_to_id,
    relation_to_id=training_triples_factory.relation_to_id,
)

testing_triples_factory = TriplesFactory(
    mapped_triples=torch.tensor(np.array(dataset_pickle['testing']), dtype=torch.long),
    entity_to_id=training_triples_factory.entity_to_id,
    relation_to_id=training_triples_factory.relation_to_id,
)

min_testing_triples_factory = TriplesFactory(
    mapped_triples=torch.tensor(np.array(dataset_pickle['min_testing']), dtype=torch.long),
    entity_to_id=training_triples_factory.entity_to_id,
    relation_to_id=training_triples_factory.relation_to_id,
)

load_dataset_tracker = tracker.stop_task()

HP = get_hp(dataset_name=dataset_pickle['dataset'], model_name=dataset_pickle['model'])
SETTINGS = {}

# ---- HYPERPARAMETRS ------

# KGEM
# HP['embedding_dim'] = 150
# HP['random_seed'] = 1996
# HP['loss'] = 'MarginRankingLoss'
# HP['loss_kwargs'] = {'margin': 1.0, 'reduction': 'mean'}
# HP['regularizer'] = 'LpRegularizer'
# HP['regularizer_kwargs'] = {'weight': 1.0, 'p': 2.0, 'dim': -1}
# HP['model'] = 'TransE'
#
# # Optimizer
# HP['optimizer_name'] = 'Adam'
# HP['optimizer_kwargs'] = {'lr': 5e-5}
#
# # Training
# HP['num_epochs'] = 250
# HP['batch_size'] = 2 ** 17
# HP['negative_sampler'] = 'basic'
# HP['negative_sampler_kwargs'] = {'corruption_scheme': ('head', 'tail'),
#                                  'num_negs_per_pos': 1}

# ---- END ------


# ---- SETTINGS -----
use_tqdm_batch = True
SETTINGS['frequency_save_model'] = 500

if torch.cuda.is_available():
    SETTINGS['cuda'] = torch.cuda.get_device_name()
else:
    SETTINGS['cuda'] = None

SETTINGS['platform'] = platform.processor()


# Pick a model
model = eval(HP['model'])(
    triples_factory=training_triples_factory,
    embedding_dim=HP['embedding_dim'],
    loss=HP['loss'],
    loss_kwargs=HP['loss_kwargs'],
    random_seed=HP['random_seed'],
    regularizer=HP['regularizer'],
    regularizer_kwargs=HP['regularizer_kwargs'],
    **HP['model_kwargs'])

model.to(torch.device('cuda'))

optimizer = eval(HP['optimizer'])(params=model.get_grad_params(), **HP['optimizer_kwargs'])

training_loop = eval(HP['training_loop'])(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
    negative_sampler=HP['negative_sampler'],
    negative_sampler_kwargs=HP['negative_sampler_kwargs'],
)

tracker.start_task("training")
# Train like Cristiano Ronaldo
training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=HP['num_epochs'],
    batch_size=HP['batch_size'],
    use_tqdm_batch=use_tqdm_batch,
    callbacks=Edo(frequency=SETTINGS['frequency_save_model'],
                  project_path=project_path,
                  **{'triples_factory': training_triples_factory}),
)
training_tracker = tracker.stop_task()

torch.cuda.empty_cache()

evaluator = RankBasedEvaluator()

tracker.start_task("evaluation")
# Evaluate
results = evaluator.evaluate(
    model=model,
    mapped_triples=testing_triples_factory.mapped_triples,
    # batch_size=HP['batch_size'],
    filtered=HP['evaluator_filtered'],
    additional_filter_triples=[
        training_triples_factory.mapped_triples,
        validation_triples_factory.mapped_triples,
    ],
)
evaluation_tracker = tracker.stop_task()

tracker.start_task("evaluation_min")
# MIN TEST SET
results_min = evaluator.evaluate(
    model=model,
    mapped_triples=min_testing_triples_factory.mapped_triples,
    # batch_size=HP['batch_size'],
    filtered=HP['evaluator_filtered'],
    additional_filter_triples=[
        training_triples_factory.mapped_triples,
        validation_triples_factory.mapped_triples,
    ],
)
evaluation_min_tracker = tracker.stop_task()

####

num_predictions = 10
random_triple = testing_triples_factory.mapped_triples[
    random.choices(range(len(testing_triples_factory.mapped_triples)), k=num_predictions)]

tracker.start_task("prediction")
pack = predict_triples(model=model, triples=random_triple, batch_size=HP['batch_size'])
prediction_tracker = tracker.stop_task()

# To be saved to pickle
# SETTINGS, HP, elapsed_training_time, elapsed_evaluation_time, elapsed_prediction_time

with open(os.path.join(project_path, 'instance.pickle'), 'wb') as f:
    pickle.dump({**dataset_pickle, **{
        'HP': HP,
        'SETTINGS': SETTINGS,
        'results': results.to_dict(),
        'results_min': results_min.to_dict(),
        'trackers': {
            'import_tracker': dict(import_tracker.values),
            'load_dataset_tracker': dict(load_dataset_tracker.values),
            'training_tracker': dict(training_tracker.values),
            'evaluation_tracker': dict(evaluation_tracker.values),
            'evaluation_min_tracker': dict(evaluation_min_tracker.values),
            'prediction_tracker': dict(prediction_tracker.values),
        }
    }}, f)

shutil.move(project_path, './DONE/')
