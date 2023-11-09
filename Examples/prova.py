# Get a training dataset
import random

from pykeen.datasets import Nations
from pykeen.models import TransE
from torch.optim import *
from pykeen.training import SLCWATrainingLoop
from pykeen.training.callbacks import TrainingCallback
from pykeen.evaluation import RankBasedEvaluator
from pykeen.predict import predict_triples
from pykeen.triples import TriplesFactory

import time
import pickle
import numpy as np


class Edo(TrainingCallback):

    def __init__(self, frequency: int, **kwargs):
        super().__init__()
        self.frequency = frequency
        if 'triples_factory' in kwargs:
            self.triples_factory = kwargs['triples_factory']

    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs):
        if epoch % self.frequency == 0:
            self.training_loop._save_state(path='../Data_Collection/edo2.pt', triples_factory=self.triples_factory)

    def post_train(self, losses, **kwargs) -> None:
        self.training_loop._save_state(path='../Data_Collection/edo_final.pt', triples_factory=self.triples_factory)


with open('../Data_Collection/Datasets_Complete/WN18/Split_0/instance.pickle', 'rb') as f:
    dataset_pickle = pickle.load(f)

print('DATASET: {} ', dataset_pickle['dataset']['dataset_name'])


training_triples_factory = TriplesFactory.from_labeled_triples(
    triples=np.array(dataset_pickle['dataset']['training']),
    entity_to_id=dataset_pickle['dataset']['entity_to_id'],
    relation_to_id=dataset_pickle['dataset']['relation_to_id'],
)

validation_triples_factory = TriplesFactory.from_labeled_triples(
    triples=np.array(dataset_pickle['dataset']['validation']),
    entity_to_id=training_triples_factory.entity_to_id,
    relation_to_id=training_triples_factory.relation_to_id,
)

testing_triples_factory = TriplesFactory.from_labeled_triples(
    triples=np.array(dataset_pickle['dataset']['testing']),
    entity_to_id=training_triples_factory.entity_to_id,
    relation_to_id=training_triples_factory.relation_to_id,
)


HP = {}
SETTINGS = {}

# ---- HYPERPARAMETRS ------

# KGEM
HP['embedding_dim'] = 256
HP['random_seed'] = 1996
HP['loss'] = 'MarginRankingLoss'
HP['loss_kwargs'] = {'margin': 1.0, 'reduction': 'mean'}
HP['regularizer'] = 'LpRegularizer'
HP['regularizer_kwargs'] = {'weight': 1.0, 'p': 2.0, 'dim': -1}

# Optimizer
HP['optimizer_name'] = 'Adam'
HP['optimizer_kwargs'] = {'lr': 1e-3}

# Training
HP['num_epochs'] = 50
HP['batch_size'] = 256
HP['negative_sampler'] = 'basic'
HP['negative_sampler_kwargs'] = {'corruption_scheme': ('head', 'tail'),
                                 'num_negs_per_pos': 1}

# ---- END ------


# ---- SETTINGS -----
use_tqdm_batch = False
SETTINGS['frequency_save_model'] = 50

# Pick a model
model = TransE(
    triples_factory=training_triples_factory,
    embedding_dim=HP['embedding_dim'],
    loss=HP['loss'],
    loss_kwargs=HP['loss_kwargs'],
    random_seed=HP['random_seed'],
    regularizer=HP['regularizer'],
    regularizer_kwargs=HP['regularizer_kwargs'])

optimizer = eval(HP['optimizer_name'])(params=model.get_grad_params(), **HP['optimizer_kwargs'])

training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
    negative_sampler=HP['negative_sampler'],
    negative_sampler_kwargs=HP['negative_sampler_kwargs'],
    automatic_memory_optimization=False,
)

start_training_time = time.time()
# Train like Cristiano Ronaldo
result = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=HP['num_epochs'],
    batch_size=HP['batch_size'],
    use_tqdm_batch=use_tqdm_batch,
    callbacks=Edo(frequency=SETTINGS['frequency_save_model'], **{'triples_factory': training_triples_factory}),
)
end_training_time = time.time()

training_time = end_training_time - start_training_time

evaluator = RankBasedEvaluator()

start_evaluation_time = time.time()
# Evaluate
results = evaluator.evaluate(
    model=model,
    mapped_triples=testing_triples_factory.mapped_triples,
    batch_size=HP['batch_size'],
    additional_filter_triples=[
        training_triples_factory.mapped_triples,
        validation_triples_factory.mapped_triples,
    ],
)
end_evaluation_time = time.time()

num_predictions = 10
random_triple = testing_triples_factory.mapped_triples[
    random.choices(range(len(testing_triples_factory.mapped_triples)), k=num_predictions)]

start_prediction_time = time.time()
pack = predict_triples(model=model, triples=random_triple, batch_size=HP['batch_size'])
end_prediction_time = time.time()

elapsed_training_time = end_training_time - start_training_time
elapsed_evaluation_time = end_evaluation_time - start_evaluation_time
elapsed_prediction_time = (end_prediction_time - start_prediction_time) / num_predictions

# To be saved to pickle
# SETTINGS, HP, elapsed_training_time, elapsed_evaluation_time, elapsed_prediction_time
