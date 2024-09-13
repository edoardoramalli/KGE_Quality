import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def encoding_categorical_variables(X):
    def encode(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], dummy_na=False)
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode], axis=1)
        return (res)

    categorical_columns=list(X.select_dtypes(include=['bool','object']).columns)

    for col in X.columns:
        if col in categorical_columns:
            X = encode(X,col)
    return X

def train_tree(X, y):

    X = encoding_categorical_variables(X)

    encoded_columns = X.columns

    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X)

    model = DecisionTreeRegressor()

    model_fit = model.fit(X, y)

    cv = ShuffleSplit(n_splits=8, test_size=0.3, random_state=1)

    evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error","neg_mean_squared_error","max_error","r2"]

    results = {
        "neg_root_mean_squared_error": list(),
        "neg_mean_absolute_error": list(),
        "neg_mean_squared_error": list(),
        "max_error": list(),
        "r2": list()
    }

    for e in evaluation_metrics:
        model_scores = cross_val_score(model_fit, X, y, cv=cv, scoring=e)
        score_mean = model_scores.mean()
        results[e] = score_mean

    return results, model, encoded_columns

def train_model(X, y, model):

    X = encoding_categorical_variables(X)

    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X)

    model_fit = model.fit(X, y)

    cv = ShuffleSplit(n_splits=8, test_size=0.3, random_state=1)

    evaluation_metrics = ["neg_root_mean_squared_error","neg_mean_absolute_error","neg_mean_squared_error","max_error","r2"]

    results = {
        "neg_root_mean_squared_error": list(),
        "neg_mean_absolute_error": list(),
        "neg_mean_squared_error": list(),
        "max_error": list(),
        "r2": list()
    }

    for e in evaluation_metrics:
        model_scores = cross_val_score(model_fit, X, y, cv=cv, scoring=e)
        score_mean = model_scores.mean()
        results[e] = score_mean

    return results, model

def use_trained_model(X_test, y_test, model):

    X_test = encoding_categorical_variables(X_test)
    X_test = StandardScaler().fit_transform(X_test)

    y_pred = model.predict(X_test)

    results = {
        "neg_root_mean_squared_error": 0,
        "neg_mean_absolute_error": 0,
        "neg_mean_squared_error": 0,
        "max_error": 0,
        "r2": 0,
    }

    results["neg_root_mean_squared_error"] = mean_squared_error(y_test, y_pred, squared=True)
    results["neg_mean_absolute_error"] = mean_absolute_error(y_test, y_pred)
    results["neg_mean_squared_error"] = mean_squared_error(y_test, y_pred)
    results["max_error"] = max_error(y_test, y_pred)
    results["r2"] = r2_score(y_test, y_pred)

    return results, y_test, y_pred

def training_and_testing(df_train, df_test, features, target, model):

    #print("---- TRAINING PHASE ----")
    X = df_train[features]
    y = df_train[target]
    results_training, model = train_model(X, y, model)
    #for e in ["neg_root_mean_squared_error","neg_mean_absolute_error","neg_mean_squared_error","max_error","r2"]:
    #    print(e+": "+str(results_training[e]))
    #print("\n\n")

    #print("---- TESTING PHASE ----")
    X = df_test[features]
    y = df_test[target]

    results_testing, y_test, y_pred = use_trained_model(X, y, model)
    #for e in ["neg_root_mean_squared_error","neg_mean_absolute_error","neg_mean_squared_error","max_error","r2"]:
    #    print(e.replace("neg_","")+": "+str(results_testing[e]))
    #print("\n\n")

    return results_training, results_testing, y_test, y_pred


def bar_plot(title, labels, training_means, testing_means):

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x - width/2, training_means, width, label='Training')
    ax.bar(x + width/2, testing_means, width, label='Testing')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    #plt.savefig(your_path + title + "-all.pdf", bbox_inches='tight')
    #plt.ylim(0,55)
    #plt.savefig(your_path + title + "-ylim.pdf", bbox_inches='tight')

    plt.show()
