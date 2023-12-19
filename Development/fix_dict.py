import glob
import pickle
path = "../Data_Collection/Datasets_Complete/**/instance.pickle"

for path in glob.glob(path, recursive=True):
    print(path)

    with open(path, 'rb') as f:
        pick = pickle.load(f)

        old_df = pick['measures_training']['dataframe']

        print(old_df.columns)

    if 'entities' in old_df.columns:

        new = old_df.rename(columns={'entities': 'entity'})

        pick['measures_training']['dataframe'] = new

        with open(path, 'wb') as f:
            pickle.dump(pick, f)

            # old_df = pick['measures_training']['dataframe']

