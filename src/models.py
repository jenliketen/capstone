import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import re

def sub_feature_names(data, feature_names):
    cols = data.columns.tolist()
    feat_map = {"x" + str(num):cat for num, cat in enumerate(cols)}
    feat_string = ",".join(feature_names)
    feat_string
    for key, value in feat_map.items():
        feat_string = re.sub(fr"\b{key}\b", value, feat_string)
    feat_string = feat_string.replace(" ", " : ").split(",")
    return feat_string


def lin_reg(X_train, X_test, y_train, y_test, degree=1):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    p = X_train.shape[1]
    
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)

    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)
    
    adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
    adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p - 1)

    coefs = model.named_steps["linearregression"].coef_.tolist()
    dummy_names = model.named_steps["polynomialfeatures"].get_feature_names()
    feature_names = sub_feature_names(X_train, dummy_names)
    coefficients = {}
    for feature_name, coef in zip(feature_names, coefs):
        coefficients[feature_name] = coef
    coefficients = {key: value for key, value in sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)}

    return model, train_pred, test_pred, mse_train, mse_test, adj_r2_train, adj_r2_test, coefficients


def huber_reg(X_train, X_test, y_train, y_test, degree=1):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    p = X_train.shape[1]
    
    model = make_pipeline(PolynomialFeatures(degree), HuberRegressor())
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)

    coefs = model.named_steps["huberregressor"].coef_.tolist()
    dummy_names = model.named_steps["polynomialfeatures"].get_feature_names()
    feature_names = sub_feature_names(X_train, dummy_names)
    coefficients = {}
    for feature_name, coef in zip(feature_names, coefs):
        coefficients[feature_name] = coef
    coefficients = {key: value for key, value in sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)}

    return model, train_pred, test_pred, mse_train, mse_test, coefficients


def lasso_reg(X_train, X_test, y_train, y_test, degree=1):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    p = X_train.shape[1]

    lasso = Lasso(random_state=0, max_iter=100000, tol=0.1)
    alphas = np.logspace(-3, 3, 50)
    tuned_parameters = [{"alpha": alphas}]
    n_folds = 5
    clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds)
    clf.fit(X_train, y_train)

    lda = clf.best_params_["alpha"]
 
    model = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=lda))
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)
    
    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)
    
    adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
    adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p - 1)
    
    coefs = model.named_steps["lasso"].coef_.tolist()
    dummy_names = model.named_steps["polynomialfeatures"].get_feature_names()
    feature_names = sub_feature_names(X_train, dummy_names)
    coefficients = {}
    for feature_name, coef in zip(feature_names, coefs):
        coefficients[feature_name] = coef
    coefficients = {key: value for key, value in sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)}

    non_important_coefs = []
    for key in coefficients:
        if coefficients[key] == 0:
            non_important_coefs.append(key)
    
    print("Lambda: {}".format(lda))
    return model, train_pred, test_pred, mse_train, mse_test, adj_r2_train, adj_r2_test, coefficients, non_important_coefs


def ridge_reg(X_train, X_test, y_train, y_test, degree=1):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    p = X_train.shape[1]

    ridge = Ridge(random_state=0, max_iter=100000, tol=0.1)
    alphas = np.logspace(-3, 3, 50)
    tuned_parameters = [{"alpha": alphas}]
    n_folds = 5
    clf = GridSearchCV(ridge, tuned_parameters, cv=n_folds)
    clf.fit(X_train, y_train)

    lda = clf.best_params_["alpha"]
 
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=lda))
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)
    
    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)
    
    adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
    adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p - 1)
    
    coefs = model.named_steps["ridge"].coef_.tolist()
    dummy_names = model.named_steps["polynomialfeatures"].get_feature_names()
    feature_names = sub_feature_names(X_train, dummy_names)
    coefficients = {}
    for feature_name, coef in zip(feature_names, coefs):
        coefficients[feature_name] = coef
    coefficients = {key: value for key, value in sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)}
    
    print("Lambda: {}".format(lda))
    return model, train_pred, test_pred, mse_train, mse_test, adj_r2_train, adj_r2_test, coefficients


def enet_reg(X_train, X_test, y_train, y_test, degree=1):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    p = X_train.shape[1]

    enet = ElasticNet(random_state=0, max_iter=100000, tol=0.1)
    alphas = np.logspace(-3, 3, 50)
    tuned_parameters = [{"alpha": alphas}]
    n_folds = 5
    clf = GridSearchCV(enet, tuned_parameters, cv=n_folds)
    clf.fit(X_train, y_train)

    lda = clf.best_params_["alpha"]
 
    model = make_pipeline(PolynomialFeatures(degree), ElasticNet(alpha=lda))
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)
    
    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)
    
    adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
    adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p - 1)
    
    coefs = model.named_steps["elasticnet"].coef_.tolist()
    dummy_names = model.named_steps["polynomialfeatures"].get_feature_names()
    feature_names = sub_feature_names(X_train, dummy_names)
    coefficients = {}
    for feature_name, coef in zip(feature_names, coefs):
        coefficients[feature_name] = coef
    coefficients = {key: value for key, value in sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)}

    non_important_coefs = []
    for key in coefficients:
        if coefficients[key] == 0:
            non_important_coefs.append(key)
    
    print("Lambda: {}".format(lda))
    return model, train_pred, test_pred, mse_train, mse_test, adj_r2_train, adj_r2_test, coefficients, non_important_coefs


def random_forest(X_train, X_test, y_train, y_test):
    Bs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    Rsqs = []
    for B in Bs:
        model = RandomForestRegressor(n_estimators=B, max_depth=20, max_features="sqrt")
        model.fit(X_train, y_train)
        Rsqs.append(model.score(X_train, y_train))

    max_Rsq = max(Rsqs)
    max_index = Rsqs.index(max_Rsq)
    max_index
    B = Bs[max_index]

    model = RandomForestRegressor(n_estimators=B, max_depth=20, max_features="sqrt", oob_score=True)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    oob_score = model.oob_score_

    feature_list = list(X_test.columns)
    importances = list(model.feature_importances_)
    feature_importances_list = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances_list = sorted(feature_importances_list, key=lambda x: x[1], reverse=True)
    feature_importances = {}
    for item in feature_importances_list:
        feature_importances[item[0]] = item[1]

    print("Number of trees: {}".format(B))
    return model, train_pred, test_pred, mse_train, mse_test, train_score, test_score, oob_score, feature_importances


def gbm(X_train, X_test, y_train, y_test):
    Bs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    Rsqs = []
    for B in Bs:
        model = GradientBoostingRegressor(n_estimators=B, max_depth=20, max_features="sqrt")
        model.fit(X_train, y_train)
        Rsqs.append(model.score(X_train, y_train))

    max_Rsq = max(Rsqs)
    max_index = Rsqs.index(max_Rsq)
    max_index
    B = Bs[max_index]

    model = GradientBoostingRegressor(n_estimators=B, max_depth=20, max_features="sqrt")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    feature_list = list(X_test.columns)
    importances = list(model.feature_importances_)
    feature_importances_list = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances_list = sorted(feature_importances_list, key=lambda x: x[1], reverse=True)
    feature_importances = {}
    for item in feature_importances_list:
        feature_importances[item[0]] = item[1]

    print("Number of trees: {}".format(B))
    return model, train_pred, test_pred, mse_train, mse_test, train_score, test_score, feature_importances


def adaboost(X_train, X_test, y_train, y_test):
    Bs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    Rsqs = []
    for B in Bs:
        model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=20, max_features="sqrt"), n_estimators=B)
        model.fit(X_train, y_train)
        Rsqs.append(model.score(X_train, y_train))

    max_Rsq = max(Rsqs)
    max_index = Rsqs.index(max_Rsq)
    max_index
    B = Bs[max_index]

    model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=20, max_features="sqrt"), n_estimators=B)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    feature_list = list(X_test.columns)
    importances = list(model.feature_importances_)
    feature_importances_list = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances_list = sorted(feature_importances_list, key=lambda x: x[1], reverse=True)
    feature_importances = {}
    for item in feature_importances_list:
        feature_importances[item[0]] = item[1]

    print("Number of trees: {}".format(B))
    return model, train_pred, test_pred, mse_train, mse_test, train_score, test_score, feature_importances
