from pykeen.pipeline import pipeline
from pykeen.losses import *
from pykeen.regularizers import *
from pykeen.models import *
import torch



pipeline_result = pipeline(
    # dataset='YAGO310',
    dataset='FB15k',
    model='ComplEx',
    model_kwargs={
        'embedding_dim': 200,
    },
    loss='NSSALoss',  # SoftplusLoss
    loss_kwargs={
        'margin': 1
    },
    training_kwargs={
        'batch_size': 1024 * 4,
    },
    optimizer='Adam',
    optimizer_kwargs={
        "lr": 5e-04
    },
    negative_sampler_kwargs={
        'num_negs_per_pos': 20,
        'corruption_scheme': ('head', 'tail')
    },
    epochs=4000,
    # regularizer='LpRegularizer',
    # regularizer_kwargs={
    #     "weight": 1e-04,
    #     "p": 3.0,
    #     "dim": -1
    # },

)

print(pipeline_result.metric_results.to_dict())

# pipeline_result = pipeline(
#     # dataset='YAGO310',
#     dataset='WN18',
#     model='TransE',
#     model_kwargs={
#         'embedding_dim': 150,
#         'scoring_fct_norm': 1
#     },
#     loss='PairwiseLogisticLoss',
#     training_kwargs={
#         'batch_size': 2 ** 9,
#     },
#     optimizer='Adam',
#     optimizer_kwargs={
#         "lr": 5e-05
#     },
#     negative_sampler_kwargs={
#         'num_negs_per_pos': 10,
#         'corruption_scheme': ('head', 'tail')
#     },
#     epochs=4000,
#     regularizer='LpRegularizer',
#     regularizer_kwargs={
#         "weight": 1e-04,
#         "p": 3.0,
#         "dim": -1
#     },
#
# )
