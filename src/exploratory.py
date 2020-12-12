from sklearn.model_selection import train_test_split
import itertools
from scipy.stats import pearsonr
import pandas as pd

def split(data, standardize=False):
    X = data.drop(["fips", "year", "COPD"], axis=1, inplace=False)
    y = data["COPD"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=325)
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    if standardize == True:
        X_train -= X_train.mean(axis=0)
        X_train /= X_train.std(axis=0)
        X_test -= X_test.mean(axis=0)
        X_test /= X_test.std(axis=0)
    
    return X_train, X_test, y_train, y_test


def get_correlations(data):
    correlations = {}
    columns = data.columns.tolist()

    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[col_a + " vs. " + col_b] = pearsonr(data.loc[:, col_a], data.loc[:, col_b])

    result = pd.DataFrame.from_dict(correlations, orient="index")
    result.columns = ["PCC", "p-value"]
    result = result[(result["PCC"] >= 0.6)]
    result = result.sort_values("PCC", ascending=False)
    
    return result