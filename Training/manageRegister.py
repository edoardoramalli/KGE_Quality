import pandas as pd


def check_existence(pick, project_name):
    df = pd.read_csv('./register.csv')

    _dataset = pick['dataset']
    _c_split = pick['c_split']
    _model = pick['model']
    _property = pick['property']
    _value = pick['value']

    index = df.index[(df.dataset == _dataset) & (df.split == _c_split) & (df.model == _model) & (
                df.property == _property) & (df.value == _value) & (df.folder == project_name)].tolist()

    if len(index) == 0:
        raise ValueError('Not present')
    elif len(index) > 1:
        raise ValueError('Too many')
    else:
        return index[0]


def set_running(pick, project_name):
    c_index = check_existence(pick, project_name)
    path = './register.csv'
    df = pd.read_csv(path)
    if df.iloc[c_index].status != 'waiting':
        return False

    df.at[c_index, 'status'] = 'running'
    df.to_csv(path, index=False)

    return True


def set_done(pick, project_name):
    c_index = check_existence(pick, project_name)
    path = './register.csv'
    df = pd.read_csv(path)
    if df.iloc[c_index].status != 'running':
        raise ValueError('Wrong status')

    df.at[c_index, 'status'] = 'done'
    df.to_csv(path, index=False)
