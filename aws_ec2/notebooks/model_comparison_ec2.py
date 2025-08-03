# Import libraries

# Data matipulation
import numpy as np
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt

# Imputation
from sklearn.impute import KNNImputer
from sklearn.impute import MissingIndicator

# Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

# Model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Metrics
from sklearn.metrics import accuracy_score

# Helper functions


# Imputation

def knn_imputer(df):
    #split predictor on numeric and categorical
    numeric_predictors = df.select_dtypes(include=["int64", "float64"])
    categorical_predictors = df.select_dtypes(include="object")

    #get columns
    numeric_columns = numeric_predictors.columns.values
    categorical_columns = categorical_predictors.columns.values

    #imputation by mean / most frequent
    numeric_predictors = KNNImputer(n_neighbors=5).fit_transform(numeric_predictors)

    # predictor numpy.array to pandas.dataframe
    numeric_predictors = pd.DataFrame(numeric_predictors, columns=numeric_columns)
    categorical_predictors = df[categorical_columns]
    df_imputed = pd.concat([numeric_predictors, categorical_predictors], axis=1)
    return df_imputed


def knn_imputer_ind(df):
    #split predictor on numeric and categorical
    numeric_predictors = df.select_dtypes(include=["int64", "float64"])
    categorical_predictors = df.select_dtypes(include="object")


    #get columns
    numeric_columns = numeric_predictors.columns.values
    categorical_columns = categorical_predictors.columns.values
    
    indicator = MissingIndicator(features="missing-only")
    missing_mask = indicator.fit_transform(numeric_predictors)

    numeric_predictors_miss = numeric_predictors.isna().sum()
    numeric_predictors_miss = numeric_predictors_miss[numeric_predictors_miss != 0].index.values

    miss_list = []
    for col in numeric_predictors_miss:
        miss_list.append(f"{col}_was_misssing")


    indicator_df = pd.DataFrame(missing_mask, columns=miss_list)

    #imputation by mean
    numeric_predictors = KNNImputer(n_neighbors=5).fit_transform(numeric_predictors)

    # predictor numpy.array to pandas.dataframe
    numeric_predictors = pd.DataFrame(numeric_predictors, columns=numeric_columns)
    numeric_predictors = pd.concat([numeric_predictors, indicator_df], axis=1)
    categorical_predictors = df[categorical_columns]
    df_imputed = pd.concat([numeric_predictors, categorical_predictors], axis=1)
    return df_imputed


# Convert categorical features into boolean

def get_dummies_fun(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def label_encoder_fun(df):
    le = preprocessing.LabelEncoder()
    for predictor in df.columns:
        if df[predictor].dtype == object:
            df[predictor] = le.fit_transform(df[predictor])
    return df


# Scaling

def standardization(x_train, x_test):
    columns = x_train.columns.values
    index = x_train.index
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train, columns=columns, index=index)

    columns = x_test.columns.values
    index = x_test.index
    scaler = StandardScaler()
    x_test = scaler.fit_transform(x_test)
    x_test = pd.DataFrame(x_test, columns=columns, index=index)
    return x_train, x_test

def normalization(x_train, x_test):
    columns = x_train.columns.values
    index = x_train.index
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train, columns=columns, index=index)

    columns = x_test.columns.values
    index = x_test.index
    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(x_test)
    x_test = pd.DataFrame(x_test, columns=columns, index=index)
    return x_train, x_test

def no_scaling_fun(x_train, x_test):
    return x_train, x_test


# Features selection

random_search__n_iter = 10
def predictors_selector(x_train, y_train):
    model = LogisticRegression(penalty="l1", max_iter=500, solver="liblinear",)
    pipe = Pipeline([
        ("model", model)
    ])
    model_params = {
        "C": np.linspace(0.00001, 0.1)
    }
    rand_search = RandomizedSearchCV(model, model_params, n_iter=random_search__n_iter)
    rand_search.fit(x_train, y_train)
    best_params = rand_search.best_params_
    best_c = best_params["C"]
    log_reg_model = LogisticRegression(penalty="l1", C=best_c, max_iter=10000, solver="liblinear")
    log_reg_model.fit(x_train, y_train)
    coefs = log_reg_model.coef_
    columns = x_train.columns.values
    non_zero_mask = coefs != 0
    selected_predictors = columns[non_zero_mask[0]]
    x_train = x_train[selected_predictors]
    return x_train

def no_feature_selector(x_train, y_train):
    return x_train


# Comparison of all option function
def comparison(imputer, convert_fun, scaler, feature_selector, hyperparameters_tuning, iterator):
    # Import data
    df = pd.read_csv("../data/train.csv")

    # Drop columns
    df = df.drop("Name", axis=1) # has no meaning 
    df = df.drop("Cabin", axis=1) # high percentage of missing data
    df = df.drop("Ticket", axis=1) # lot of unique string data type
    df = df.drop("PassengerId", axis=1) # has no meaning

    # Imputation
    df = imputer(df)
    
    # Drop remaining rows with missing data
    # there is only few missing data
    df = df.dropna()

    df = convert_fun(df)

    # Split data on predictor and responsible feature
    x = df.drop("Survived", axis=1)
    y = df.Survived

    # Split data on training and testing data
    # for choosing best model
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=8)

    # Scaling
    x_train, x_test = scaler(x_train, x_test)

    # Initializing of list for comparing models, no scaling, no features selection, no missing values indicator
    models_list = []

    # Set scorring
    scorring = "accuracy"

    # Features selection
    x_train = feature_selector(x_train, y_train)

    # Models

    # No hyperparameters tuning
    if hyperparameters_tuning == False:

        # K nearest neighbors classifier 
        # defaul setting
        model = KNeighborsClassifier()
        cv = cross_validate(model, x_train, y_train, cv=10, scoring=scorring)
        cv = round(float(cv["test_score"].mean()),5)

        # Add score to summary
        model_params = {
            "id": 1,
            "model": "K Neighbors Classifier",
            "accuracy": cv
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Logistic regression 
        # penalty: None
        model = LogisticRegression(penalty=None, max_iter=500)
        cv = cross_validate(model, x_train, y_train, cv=10, scoring=scorring)
        cv = round(float(cv["test_score"].mean()),5)

        # Add score to summary
        model_params = {
            "id": 2,
            "model": "Logistic Regression, penalty = none",
            "accuracy": cv
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Logistic regression, penalty = l1

        model = LogisticRegression(penalty="l1", max_iter=500, solver="liblinear")
        cv = cross_validate(model, x_train, y_train, cv=10, scoring=scorring)
        cv = round(float(cv["test_score"].mean()),5)

        # Add score to summary
        model_params = {
            "id": 3,
            "model": "Logistic Regression, penalty = l1",
            "accuracy": cv
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Logistic regression, penalty = l2

        model = LogisticRegression(penalty="l2", max_iter=500, solver="liblinear")
        cv = cross_validate(model, x_train, y_train, cv=10, scoring=scorring)
        cv = round(float(cv["test_score"].mean()),5)

        # Add score to summary
        model_params = {
            "id": 4,
            "model": "Logistic Regression, penalty = l2",
            "accuracy": cv
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Decision tree classifier 

        model = DecisionTreeClassifier()
        cv = cross_validate(model, x_train, y_train, cv=10, scoring=scorring)
        cv = round(float(cv["test_score"].mean()),5)

        # Add score to summary
        model_params = {
            "id": 5,
            "model": "Decision Tree Classifier",
            "accuracy": cv
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Bagging 

        estimator = DecisionTreeClassifier()
        model = BaggingClassifier(estimator=estimator, bootstrap=True)
        cv = cross_validate(model, x_train, y_train, cv=10, scoring=scorring)
        cv = round(float(cv["test_score"].mean()),5)

        # Add score to summary
        model_params = {
            "id": 6,
            "model": "Bagging",
            "accuracy": cv
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Random forest Classifier 

        model = RandomForestClassifier()
        cv = cross_validate(model, x_train, y_train, cv=10, scoring=scorring)
        cv = round(float(cv["test_score"].mean()),5)


        # Add score to summary
        model_params = {
            "id": 7,
            "model": "Random Forest Classifier",
            "accuracy": cv
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Gradient boosting classifier 

        model = GradientBoostingClassifier()
        cv = cross_validate(model, x_train, y_train, cv=10, scoring=scorring)
        cv = round(float(cv["test_score"].mean()),5)

        # Add score to summary
        model_params = {
            "id": 8,
            "model": "Gradient Boosting Classifier",
            "accuracy": cv
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Adaboost 1

        model = AdaBoostClassifier()
        cv = cross_validate(model, x_train, y_train, cv=10, scoring=scorring)
        cv = round(float(cv["test_score"].mean()),5)

        # Add score to summary
        model_params = {
            "id": 9,
            "model": "Adaboost Classifier",
            "accuracy": cv
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)



    # With hyperparameters tuning
    else:
        # K nearest neighbors classifier 
        # Tuning:
        # n_neighbors

        # Cell parrameters:
        max_n_neighbors = 30

        # Model
        model = KNeighborsClassifier()

        # Hyperparameter tuning
        pipe = Pipeline([
            ("model", model)
        ])
        param = {
            "model__n_neighbors": range(1,max_n_neighbors)
        }
        grid_search = GridSearchCV(pipe, param,cv=10, scoring=scorring)
        grid_search.fit(x_train, y_train)

        # Best accuracy
        best_accuracy = round(float(grid_search.best_score_),5)

        # Show results
        print(f"Best accuracy: {best_accuracy}")
        print(f"Best parameters: {grid_search.best_params_}")

        # Add score to summary
        model_params = {
            "id": 1,
            "model": "K Neighbors Classifier",
            "accuracy": best_accuracy
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Logistic regression
        # penalty: l1
        # Tuning:
        # c

        # Cell parrameters:
        min_c = 0.00001
        max_c = 0.1

        # Model
        model = LogisticRegression(penalty="l1", max_iter=500, solver="liblinear")

        # Hyperparameter tuning
        pipe = Pipeline([
            ("model", model)
        ])
        param = {
            "model__C": np.linspace(min_c, max_c)
        }
        grid_search = GridSearchCV(pipe, param, cv=10, scoring=scorring)
        grid_search.fit(x_train, y_train)

        # Best accuracy
        best_accuracy = round(float(grid_search.best_score_),5)

        # Show results
        print(f"Best accuracy: {best_accuracy}")
        print(f"Best parameters: {grid_search.best_params_}")

        # Add score to summary
        model_params = {
            "id": 3,
            "model": "Logistic Regression, penalty = l1",
            "accuracy": best_accuracy
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Logistic regression
        # penalty: l2
        # Tuning:
        # c

        # Cell parrameters:
        min_c = 0.00001
        max_c = 0.1

        # Model
        model = LogisticRegression(penalty="l2", max_iter=500, solver="liblinear")

        # Hyperparameter tuning
        pipe = Pipeline([
            ("model", model)
        ])
        param = {
            "model__C": np.linspace(min_c, max_c)
        }
        grid_search = GridSearchCV(pipe, param, cv=10, scoring=scorring)
        grid_search.fit(x_train, y_train)

        # Best accuracy
        best_accuracy = round(float(grid_search.best_score_),5)

        # Show results
        print(f"Best accuracy: {best_accuracy}")
        print(f"Best parameters: {grid_search.best_params_}")

        # Add score to summary
        model_params = {
            "id": 4,
            "model": "Logistic Regression, penalty = l2",
            "accuracy": best_accuracy
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Decision tree classifier

        # Tuning:
        # max depth
        # max leaf nodes
        # ccp

        # Cell parrameters:
        random_search__n_iter = 10

        max_depth_search = 12
        max_leaf_nodes_search = 100
        min_ccp = 0.0001
        max_ccp = 0.001

        # Model
        model = DecisionTreeClassifier()

        # Hyperparameter tuning
        pipe = Pipeline([
            ("model", model)
        ])
        param = {
            "model__max_depth": range(1,max_depth_search),
            "model__max_leaf_nodes": range(2, max_leaf_nodes_search),
            "model__ccp_alpha": np.linspace(min_ccp, max_ccp)
        }
        grid_search = RandomizedSearchCV(pipe, param, cv=10, scoring=scorring, n_iter=random_search__n_iter)
        grid_search.fit(x_train, y_train)

        # Best accuracy
        best_accuracy = round(float(grid_search.best_score_),5)

        # Show results
        print(f"Best accuracy: {best_accuracy}")
        print(f"Best parameters: {grid_search.best_params_}")

        # Add score to summary
        model_params = {
            "id": 5,
            "model": "Decision Tree Classifier",
            "accuracy": best_accuracy
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Bagging

        # Tuning:
        # n estimators
        # max depth
        # max leaf nodes
        # ccp

        # Cell parrameters:
        random_search__n_iter = 10

        max_n_estimators = 202
        max_depth_search = 10
        max_leaf_nodes_search = 100
        min_ccp = 0.0001
        max_ccp = 0.001

        # Model
        estimator = DecisionTreeClassifier()
        model = BaggingClassifier(estimator=estimator, bootstrap=True)

        # Hyperparameter tuning
        pipe = Pipeline([
            ("model", model)
        ])
        param = {
            "model__n_estimators": range(200,max_n_estimators),
            "model__estimator__max_depth": range(3,max_depth_search),
            "model__estimator__max_leaf_nodes": range(20, max_leaf_nodes_search),
            "model__estimator__ccp_alpha": np.linspace(min_ccp, max_ccp)
        }
        grid_search = RandomizedSearchCV(pipe, param, cv=10, scoring=scorring, n_iter=random_search__n_iter)
        grid_search.fit(x_train, y_train)

        # Best accuracy
        best_accuracy = round(float(grid_search.best_score_),5)

        # Show results
        print(f"Best accuracy: {best_accuracy}")
        print(f"Best parameters: {grid_search.best_params_}")

        # Add score to summary
        model_params = {
            "id": 6,
            "model": "Bagging",
            "accuracy": best_accuracy
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Random forest classifier

        # Tuning:
        # n estimatiors
        # max depth
        # max leaf nodes
        # max features

        # Cell parrameters:
        random_search__n_iter = 10
        max_n_estimators = 300
        max_leaf_nodes_search = 100
        max_depth_search = 100
        max_features_search = 6

        # Model
        model = RandomForestClassifier()

        # Hyperparameter tuning
        pipe = Pipeline([
            ("model", model)
        ])
        param = {
            "model__n_estimators": range(100, max_n_estimators),
            "model__max_depth": range(4, max_depth_search),
            "model__max_leaf_nodes": range(20, max_leaf_nodes_search),
            "model__max_features": range(1,max_features_search)
        }
        grid_search = RandomizedSearchCV(pipe, param, cv=10, scoring=scorring, n_iter=random_search__n_iter)
        grid_search.fit(x_train, y_train)

        # Best accuracy
        best_accuracy = round(float(grid_search.best_score_),5)

        # Show results
        print(f"Best accuracy: {best_accuracy}")
        print(f"Best parameters: {grid_search.best_params_}")

        # Add score to summary
        model_params = {
            "id": 7,
            "model": "Random Forest Classifier",
            "accuracy": best_accuracy
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Gradient boosting classifier

        # Tuning:
        # n estimatiors
        # max depth
        # max leaf nodes
        # max features

        # Cell parrameters:
        random_search__n_iter = 10

        max_n_estimators = 200
        min_learning_rate = 0.001
        max_learning_rate = 0.5
        max_depth_search = 10

        # Model
        model = GradientBoostingClassifier()

        # Hyperparameter tuning
        pipe = Pipeline([
            ("model", model)
        ])
        param = {
            "model__n_estimators": range(100, max_n_estimators),
            "model__learning_rate": np.linspace(min_learning_rate, max_learning_rate),
            "model__max_depth": range(1, max_depth_search)
        }
        grid_search = RandomizedSearchCV(pipe, param, cv=10, scoring=scorring, n_iter=random_search__n_iter)
        grid_search.fit(x_train, y_train)

        # Best accuracy
        best_accuracy = round(float(grid_search.best_score_),5)

        # Show results
        print(f"Best accuracy: {best_accuracy}")
        print(f"Best parameters: {grid_search.best_params_}")

        # Add score to summary
        model_params = {
            "id": 8,
            "model": "Gradient Boosting Classifier",
            "accuracy": best_accuracy
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)


        # Adaboost classifier

        # Tuning:
        # n estimatiors
        # max depth
        # max leaf nodes
        # max features

        # Cell parrameters:
        random_search__n_iter = 10

        max_n_estimators = 150
        min_learning_rate = 0.001
        max_learning_rate = 0.5

        # Model
        model = AdaBoostClassifier()

        # Hyperparameter tuning
        pipe = Pipeline([
            ("model", model)
        ])
        param = {
            "model__n_estimators": range(1, max_n_estimators),
            "model__learning_rate": np.linspace(min_learning_rate, max_learning_rate),
        }
        grid_search = RandomizedSearchCV(pipe, param, cv=10, scoring=scorring, n_iter=random_search__n_iter)
        grid_search.fit(x_train, y_train)

        # Best accuracy
        best_accuracy = round(float(grid_search.best_score_),5)

        # Show results
        print(f"Best accuracy: {best_accuracy}")
        print(f"Best parameters: {grid_search.best_params_}")

        # Add score to summary
        model_params = {
            "id": 9,
            "model": "Adaboost Classifier",
            "accuracy": best_accuracy
        }
        model_params = [model_params["id"], model_params["model"], model_params["accuracy"]]
        models_list.append(model_params)



    # Compare models
    columns = ["id", "Model", "Accuracy"]
    data_index = pd.DataFrame(models_list, columns=columns)
    summary = pd.DataFrame(models_list, columns=columns, index=data_index["id"].values).sort_values(by="Accuracy", ascending=False)
    
    # Set setting

    # Setting
    setting = {
        "Imputation": f"{imputer.__name__}",
        "Categorical Features Convert": f"{convert_fun.__name__}",
        "Scaling": f"{scaler.__name__}",
        "Feature selection": f"{feature_selector.__name__}",
        "Hyperparameters tuning": f"{hyperparameters_tuning}"
    }

    df_list = []
    for i in range(summary.shape[0]):
        df_list.append(list(setting.values()))
    df = pd.DataFrame(df_list, columns=setting.keys(), index=data_index["id"].values)
    summary = pd.concat([summary, df], axis=1)
    summary.to_csv(f"../output/models_comparison_{iterator}.csv")
    #iterator += 1



# Compare all options
knn_imputer_list = [knn_imputer, knn_imputer_ind]
convert_fun_list = [get_dummies_fun, label_encoder_fun]
scaler_list = [no_scaling_fun, normalization, standardization]
feature_selector_list = [predictors_selector, no_feature_selector]
hyperparameter_tuning_list = [False, True]

# Set iterator
iterator = 1
for knn_imputer in knn_imputer_list:
    for convert_fun in convert_fun_list:
        for scaler in scaler_list:
            for feature_selector in feature_selector_list:
                for hyperparameter_tuning in hyperparameter_tuning_list:
                    comparison(knn_imputer, convert_fun, scaler, feature_selector, hyperparameter_tuning, iterator)
                    # Update iterator
                    iterator += 1

