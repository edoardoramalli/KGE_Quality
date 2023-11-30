import json

def get_hp(dataset_name, model_name):

    with open('../Training/config.json', 'r') as f:
        configs = json.load(f)

    res = configs.get(dataset_name, {}).get(model_name, {})

    if res:
        return {**res, 'model': model_name}
    else:
        raise ValueError('Missing Config')
