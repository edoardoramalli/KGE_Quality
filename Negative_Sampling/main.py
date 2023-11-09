# Get a training dataset
from pykeen.datasets import *
import random
from tqdm import tqdm

random.seed(1)



from multiprocessing import Pool

dataset = WN18RR()

training_triples_factory = dataset.training

entities_set = set([])
relationships = set([])

triples_set = set([])

counter = 0

num_sampling = 1

for triple in tqdm(training_triples_factory.triples):
    entities_set.add(triple[0])
    entities_set.add(triple[2])
    relationships.add(triple[1])
    triples_set.add(tuple(triple))

possibilities = [0, 2]

entities_list = list(entities_set)

# print(entities_set)

# print(triples_set)

for triple in tqdm(training_triples_factory.triples):
    for _ in range(num_sampling):
        choice = random.choice(possibilities)
        corrupted = [triple[0], triple[1], triple[2]]
        tmp_list = list(entities_list)
        # print(corrupted[choice], entities_list)
        tmp_list.remove(corrupted[choice])
        corrupted[choice] = random.choice(tmp_list)
        if tuple(corrupted) in triples_set:
            # print(corrupted, triple)
            counter += 1
            # exit()


print(counter / (len(training_triples_factory.triples) * num_sampling) * 100)

# print(triples_set)
#
#
# print(training_triples_factory.triples)
#
# print()




