import numpy as np
import pandas as pd

df = pd.read_csv('../giant_matrix_2024_02_11.csv')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score
import matplotlib.pyplot as plt


def encoding_categorical_variables(X):
    def encode(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], dummy_na=True)
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode], axis=1)
        return (res)

    categorical_columns = list(X.select_dtypes(include=['bool', 'object']).columns)

    for col in X.columns:
        if col in categorical_columns:
            X = encode(X, col)
    return X


def train_model(X, y, seed, p):
    model = ExtraTreesRegressor(n_estimators=50, criterion='squared_error', min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_features=1.0, random_state=seed, bootstrap=False,
                                warm_start=False)

    model = model.fit(X, y)

    # cv = ShuffleSplit(n_splits=8, test_size=0.3, random_state=seed)

    cv = KFold(n_splits=10, shuffle=True, random_state=seed)

    evaluation_metrics = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "neg_mean_squared_error",
                          "max_error", "r2"]

    results = {
        "neg_root_mean_squared_error": list(),
        "neg_mean_absolute_error": list(),
        "neg_mean_squared_error": list(),
        "max_error": list(),
        "r2": list()
    }

    for e in evaluation_metrics:
        model2 = ExtraTreesRegressor(n_estimators=50, criterion='squared_error', min_samples_split=2,
                                     min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0,
                                     random_state=seed, bootstrap=False, warm_start=False)
        model_scores = cross_val_score(model2, X, y, cv=cv, scoring=e)
        score_mean = model_scores.mean()
        score_std = model_scores.std()
        results[e] = [score_mean, score_std]

    return results, model


def use_trained_model(X_test, y_test, model):
    y_pred = model.predict(X_test)

    results = {
        "root_mean_squared_error": 0,
        "mean_absolute_error": 0,
        "mean_squared_error": 0,
        "max_error": 0,
        "r2": 0,
    }

    results["root_mean_squared_error"] = mean_squared_error(y_test, y_pred, squared=False)
    results["mean_absolute_error"] = mean_absolute_error(y_test, y_pred)
    results["mean_squared_error"] = mean_squared_error(y_test, y_pred)
    results["max_error"] = max_error(y_test, y_pred)
    results["r2"] = r2_score(y_test, y_pred)

    return results


def training_and_testing(d, df_train, df_test, performance, evaluation_metrics_training, evaluation_metrics, metrics, f,
                         fi, m):
    df_train__ = df_train[metrics]
    df_test__ = df_test[metrics]

    c1 = set(df_train__.columns).union(set(df_test__.columns))

    df_train__ = df_train__.dropna(axis=1)

    df_test__ = df_test__.dropna(axis=1)

    c2 = set(df_train__.columns).intersection(set(df_test__.columns))

    df_train__ = df_train__[list(c2)]
    df_test__ = df_test__[list(c2)]

    print('Ho rimosso: ', c1 - c2, c2 - c1)

    dd = pd.concat([df_train__, df_test__], axis=0)

    dd = encoding_categorical_variables(dd)

    X_train = dd[:df_train__.shape[0]]
    X_test = dd[df_train__.shape[0]:]

    assert X_train.shape[0] == df_train__.shape[0]
    assert X_test.shape[0] == df_test__.shape[0]

    for p in performance:

        ss = StandardScaler()

        print("---- TRAINING PHASE ---- : " + p)
        # training labels
        y = df_train[p]

        X_train = ss.fit_transform(X_train)

        # test labels
        y_test = df_test[p]
        X_test = ss.transform(X_test)

        results_training, model = train_model(X_train, y, 1, p)
        print("Results 4 ExtraTreesRegressor TRAIN" + " -- " + p)
        for e in evaluation_metrics_training:
            print(e + ": " + str(results_training[e]))
        print("\n\n")

        f.write(m + "," + d + ",training," + p + "," + str(
            abs(results_training[evaluation_metrics_training[0]][0])) + "," + str(
            abs(results_training[evaluation_metrics_training[1]][0])) + "," + str(
            abs(results_training[evaluation_metrics_training[2]][0])) + "," + str(
            abs(results_training[evaluation_metrics_training[3]][0])) + "," + str(
            abs(results_training[evaluation_metrics_training[4]][0])) + "\n")
        importances = str(list(model.feature_importances_))
        importances = importances.replace(" ", "")
        importances = importances.replace("[", "")
        importances = importances.replace("]", "")
        fi.write(m + "," + d + "," + p + "," + importances + "\n")

        print("---- TESTING PHASE ---- : " + p)

        results_testing = use_trained_model(X_test, y_test, model)

        print("Results 4 ExtraTreesRegressor TEST" + " -- " + p)
        for e in evaluation_metrics:
            print(e + ": " + str(results_testing[e]))
        print("\n\n")
        f.write(m + "," + d + ",testing," + p + "," + str(results_testing[evaluation_metrics[0]]) + "," + str(
            results_testing[evaluation_metrics[1]]) + "," + str(results_testing[evaluation_metrics[2]]) + "," + str(
            results_testing[evaluation_metrics[3]]) + "," + str(results_testing[evaluation_metrics[4]]) + "\n")


models = ["TransE", "ComplEx", "RotatE"]
performance = ["results_min_hits_at_1", "results_min_hits_at_10", "results_min_inverse_harmonic_mean_rank"]
metrics = ['order', 'size', 'density', 'diameter',
           'transitivity_undirected', 'connected_components_weak',
           'connected_components_weak_relative_nodes',
           'connected_components_weak_relative_arcs', 'girth',
           'vertex_connectivity', 'edge_connectivity', 'clique_number',
           'average_degree', 'max_degree', 'average_indegree',
           'average_outdegree', 'radius',
           'average_path_length', 'assortativity_degree', 'mean_eccentricity',
           'min_eccentricity', 'max_eccentricity',
           'centralization', 'motifs_randesu_no', 'num_farthest_points',
           'mincut_value', 'len_feedback_arc_set', 'mean_authority_score',
           'min_authority_score', 'max_authority_score', 'mean_hub_score',
           'min_hub_score', 'max_hub_score', 'min_closeness', 'max_closeness',
           'mean_closeness', 'min_constraint', 'max_constraint', 'mean_constraint',
           'max_coreness', 'min_coreness', 'mean_coreness', 'max_strength',
           'min_strength', 'mean_strength', 'entropy']
datasets = ['FB15k237', 'FB15k', 'Kinships', 'WN18RR', 'WN18', 'YAGO310']
evaluation_metrics = ["root_mean_squared_error", "mean_absolute_error", "mean_squared_error", "max_error", "r2"]
evaluation_metrics_training = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "neg_mean_squared_error",
                               "max_error", "r2"]

with open("LOGO_v3" + ".csv", 'w') as f:
    with open("LOGO_v3_fi" + ".csv", 'w') as fi:
        f.write("model,dataset,phase,metric," + evaluation_metrics[0] + "," + evaluation_metrics[1] + "," +
                evaluation_metrics[2] + "," + evaluation_metrics[3] + "," + evaluation_metrics[4] + "\n")
        logo = LeaveOneGroupOut()
        for m in models:
            break
            for i, (train_index, test_index) in enumerate(logo.split(df, None, df.dataset)):
                # for d in datasets:
                #     d_test = d
                #
                #     print("---- LEAVE DATASET " + d + " OUT ----")
                #     df_train = df[(df["dataset"] != d_test) & (df["model"] == m)]
                #     df_test = df[(df["dataset"] == d_test) & (df["model"] == m)]

                df_train = df.iloc[train_index]
                df_test = df.iloc[test_index]

                training_and_testing(df_test.dataset.unique()[0], df_train, df_test, performance,
                                     evaluation_metrics_training, evaluation_metrics,
                                     metrics, f, fi, m)

from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)

with open("KFOLD_ALL_DATASETS_v3" + ".csv", 'w') as f:
    with open("KFOLD_ALL_DATASETS_v3_fi" + ".csv", 'w') as fi:
        f.write("model,dataset,phase,metric," + evaluation_metrics[0] + "," + evaluation_metrics[1] + "," +
                evaluation_metrics[2] + "," + evaluation_metrics[3] + "," + evaluation_metrics[4] + "\n")

        for m in models:
            for i, (train_index, test_index) in enumerate(rskf.split(df, df.dataset)):
                print(f"Fold {i}:")
                df_train = df.iloc[train_index]
                df_test = df.iloc[test_index]

                assert df_train.shape[0] + df_test.shape[0] == df.shape[0]

                name = "Split_" + str(i)

                df_train = df_train[df_train["model"] == m]
                df_test = df_test[df_test["model"] == m]

                training_and_testing(name, df_train, df_test, performance, evaluation_metrics_training,
                                     evaluation_metrics, metrics, f, fi, m)
