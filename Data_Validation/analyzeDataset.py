from Data_Collection.tool import perc_of
import pickle
from pykeen.triples import TriplesFactory
import numpy as np
import pandas as pd

def analyze_kg(kg_triple_factory):
    set_e = set([])
    set_r = set([])
    counter = 0

    for triple in kg_triple_factory.triples:
        counter += 1
        set_e.add(triple[0])
        set_e.add(triple[2])
        set_r.add(triple[1])

    return set_e, set_r, counter

def check_entities(training_triples_factory, testing_triples_factory, validation_triples_factory, num_entities, num_relation):

    # for e in dir(dataset):
    #     print(e, getattr(dataset, e))

    #training_triples_factory = dataset.training
    #testing_triples_factory = dataset.testing
    #validation_triples_factory = dataset.validation

    set_e_tra, set_r_tra, real_num_t_tra = analyze_kg(training_triples_factory)
    set_e_tes, set_r_tes, real_num_t_tes = analyze_kg(testing_triples_factory)
    set_e_val, set_r_val, real_num_t_val = analyze_kg(validation_triples_factory)

    num_t_tra = training_triples_factory.num_triples
    num_t_tes = testing_triples_factory.num_triples
    num_t_val = validation_triples_factory.num_triples

    assert num_t_tra == real_num_t_tra
    assert num_t_tes == real_num_t_tes
    assert num_t_val == real_num_t_val

    print('Triples counting correct.')

    num_e_tra = training_triples_factory.num_entities
    num_e_tes = testing_triples_factory.num_entities
    num_e_val = validation_triples_factory.num_entities

    # if num_e_tra == len(set_e_tra):
    # assert num_e_tes == len(set_e_tes)
    # assert num_e_val == len(set_e_val)

    set_e_tot = set_e_tra | set_e_tes | set_e_val  # union
    real_num_e_tot = len(set_e_tot)
    real_num_e_tra = len(set_e_tra)
    real_num_e_tes = len(set_e_tes)
    real_num_e_val = len(set_e_val)

    set_r_tot = set_r_tra | set_r_tes | set_r_val  # union
    real_num_r_tot = len(set_r_tot)
    real_num_r_tra = len(set_r_tra)
    real_num_r_tes = len(set_r_tes)
    real_num_r_val = len(set_r_val)

    num_r_tra = training_triples_factory.real_num_relations
    num_r_tes = testing_triples_factory.real_num_relations
    num_r_val = validation_triples_factory.real_num_relations

    tot_t = num_t_tra + num_t_tes + num_t_val

    per_t_tra = perc_of(num_t_tra, tot_t)
    per_t_tes = perc_of(num_t_tes, tot_t)
    per_t_val = perc_of(num_t_val, tot_t)

    per_e_tra = perc_of(real_num_e_tra, real_num_e_tot)
    per_e_tes = perc_of(real_num_e_tes, real_num_e_tot)
    per_e_val = perc_of(real_num_e_val, real_num_e_tot)

    per_r_tra = perc_of(real_num_r_tra, real_num_r_tot)
    per_r_tes = perc_of(real_num_r_tes, real_num_r_tot)
    per_r_val = perc_of(real_num_r_val, real_num_r_tot)

    num_e_tra_val = len(set_e_tra - set_e_val)
    num_e_tra_tes = len(set_e_tra - set_e_tes)
    num_e_tes_val = len(set_e_tes - set_e_val)
    num_e_val_tra = len(set_e_val - set_e_tra)
    num_e_tes_tra = len(set_e_tes - set_e_tra)
    num_e_val_tes = len(set_e_val - set_e_tes)

    num_r_tra_val = len(set_r_tra - set_r_val)
    num_r_tra_tes = len(set_r_tra - set_r_tes)
    num_r_tes_val = len(set_r_tes - set_r_val)
    num_r_val_tra = len(set_r_val - set_r_tra)
    num_r_tes_tra = len(set_r_tes - set_r_tra)
    num_r_val_tes = len(set_r_val - set_r_tes)

    #print('Name: {}'.format(dataset.__class__.__name__))
    #print('# Triples: {:,}. \n\t'
    #      'Training D {:,} / R {:,} ({}%) \n\t'
    #      'Validation D {:,} / R {:,} ({}%) \n\t'
    #      'Testing D {:,} / R {:,} ({}%)'.format(
    #        tot_t,
    #        num_t_tra, real_num_t_tra, per_t_tra,
    #        num_t_tes, real_num_t_tes, per_t_tes,
    #        num_t_val, real_num_t_val, per_t_val
    #))

    #print(
    #    'Entities: {:,}. \n\t'
    #    'Training D {:,} / R {:,} ({}%) \n\t'
    #    'Validation D {:,} / R {:,} ({}%) \n\t'
    #    'Testing D {:,} / R {:,} ({}%)\n\t'
    #    'Tra - Val = {:,}, Tra - Tes = {:,}, Tes - Val = {:,}, Val - Tra = {:,}, Tes - Tra = {:,}, Val - Tes = {:,}'.format(
            #dataset.num_entities,
            #num_e_tra, real_num_e_tra, per_e_tra,
            #num_e_tes, real_num_e_tes, per_e_tes,
            #num_e_val, real_num_e_val, per_e_val,
            #num_e_tra_val, num_e_tra_tes, num_e_tes_val,
            #num_e_val_tra, num_e_tes_tra#, num_e_val_tes
        #))

    #print(
    #    '# Relationships: {:,}. \n\t'
    #    'Training D {:,} / R {:,} ({}%) \n\t'
    #    'Validation D {:,} / R {:,} ({}%) \n\t'
    #    'Testing D {:,} / R {:,} ({}%)\n\t'
    #    'Tra - Val = {:,}, Tra - Tes = {:,}, Tes - Val = {:,}, Val - Tra = {:,}, Tes - Tra = {:,}, Val - Tes = {:,}'.format(
    #        dataset.num_relations,
    #        num_r_tra, real_num_r_tra, per_r_tra,
    #        num_r_tes, real_num_r_tes, per_r_tes,
    #        num_r_val, real_num_r_val, per_r_val,
    #        num_r_tra_val, num_r_tra_tes, num_r_tes_val,
    #        num_r_val_tra, num_r_tes_tra, num_r_val_tes
    #    ))

    print('Entities: {:,}. \n\t'.format(num_entities))
    if num_e_val_tra != 0:
        print('Val - Tra = {:,}'.format(
            num_e_val_tra
        ))
    if num_e_tes_tra != 0:
        print('Tes - Tra = {:,}'.format(
            num_e_tes_tra
        ))
    else:
        print("OK!")

    print('Relationships: {:,}. \n\t'.format(num_relation))
    if num_r_val_tra != 0:
        print('Val - Tra = {:,}'.format(
            num_r_val_tra
        ))
    if num_r_tes_tra != 0:
        print('Tes - Tra = {:,}'.format(
            num_r_tes_tra
        ))
    else:
        print("OK!")

if __name__ == '__main__':

    names = ['WN18RR', 'WN18', 'FB15k', 'FB15k237', 'YAGO310']

    for n in names:
        kg_path = '../Data_Collection/Datasets_Complete/' + n + '/Split_1/instance.pickle'

        with open(kg_path, 'rb') as f:
            pick = pickle.load(f)
        dataset = pick['dataset']
        df_training = pd.DataFrame(dataset['training'], columns=['h', 'r', 't'])
        df_training = df_training.drop_duplicates()
        df_testing = pd.DataFrame(dataset['testing'], columns=['h', 'r', 't'])
        df_testing = df_testing.drop_duplicates()
        df_validation = pd.DataFrame(dataset['validation'], columns=['h', 'r', 't'])
        df_validation = df_validation.drop_duplicates()

        training = df_training[['h', 'r', 't']].values
        training = np.char.mod('%d', training)

        testing = df_testing[['h', 'r', 't']].values
        testing = np.char.mod('%d', testing)

        validation = df_validation[['h', 'r', 't']].values
        validation = np.char.mod('%d', validation)

        training_triples_factory = TriplesFactory.from_labeled_triples(training)
        testing_triples_factory = TriplesFactory.from_labeled_triples(testing)
        validation_triples_factory = TriplesFactory.from_labeled_triples(validation)

        num_entities = len(pick["dataset"]["entity_to_id"])
        num_relation = len(pick["dataset"]["relation_to_id"])

        check_entities(training_triples_factory, testing_triples_factory, validation_triples_factory, num_entities, num_relation)


