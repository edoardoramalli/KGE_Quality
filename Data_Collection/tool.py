import torch

datasets = ['WN18', 'WN18RR', 'FB15k', 'FB15k237', 'YAGO310', 'Kinships']


def perc_of(current, total) -> float:
    return round(current / total * 100, 2)


def compare_models(model_1, model_2):
    models_differ = 0
    keys1 = set(model_1.keys())
    keys2 = set(model_2.keys())

    if keys1 != keys2:
        raise ValueError('Modello diverso')

    for k in keys1:
        # print(k, model_1[k])
        if type(model_1[k]) == torch.Tensor:

            if not torch.equal(model_1[k], model_2[k]):
                raise ValueError('Modello diverso')

        elif model_1[k] != model_2[k]:
            raise ValueError('Modello diverso')


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
