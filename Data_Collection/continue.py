import torch
from pykeen.constants import PYKEEN_CHECKPOINTS
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import os

print(os.getcwd())


checkpoint = torch.load('./Data_Collection/edo.pt')
# print(checkpoint.keys())

# print(checkpoint['model_state_dict'])

print('FINEE------------------------------')

checkpoint2 = torch.load('./Data_Collection/edo2.pt')
# print(checkpoint2.keys())


print(checkpoint2['random_seed'], checkpoint2['np_random_state'], checkpoint2['torch_random_state'], checkpoint2['torch_cuda_random_state'])





compare_models(checkpoint['model_state_dict'], checkpoint2['model_state_dict'])