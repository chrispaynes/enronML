#!/usr/bin/python

import sys
import pickle
import pprint
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    PowerTransformer,
)
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


import seaborn as sns
from scipy import stats

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from utils import (
    Draw,
    plotData,
    create_features,
    calculate_pct_poi_msgs,
    validate_classifier,
)

from trial_models import trial_models

# setup environment
pp = pprint.PrettyPrinter(indent=4)
sns.set()
sns.set_context("notebook", rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid")

# suppress scientific notation in Pandas
pd.options.display.float_format = "{:.2f}".format
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# set a consistent randomization
RANDOM_STATE = 42
# set a consistent number of folds/splits
FOLDS = 10

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
    "poi",
    "bonus",
    "exercised_stock_options",
    "expenses",
]  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# coerce data features to numeric values or a NaN
df = pd.DataFrame(data_dict).transpose().apply(pd.to_numeric, errors="coerce")

### Task 2: Remove outliers
# remove the TOTAL key from the dict because it's an outlier in the data and represents an aggregate value
# 'THE TRAVEL AGENCY IN THE PARK' doesn't represent a person, but rather an entity
df = df.drop(["TOTAL", "THE TRAVEL AGENCY IN THE PARK"], errors="ignore", axis=0)
df = df.fillna(0)

### Task 3: Create new feature(s)
df["pct_poi_messages"] = df.apply(calculate_pct_poi_msgs, axis=1)

### Store to my_dataset for easy export below.
features, labels, my_dataset = create_features(df, features_list)

# construct a PCA to use as a pipeline step
pca = PCA()

### Task 4: Try a variety of classifiers

# models = trial_models
# these models below were tuned.
# comment out this variable and comment in the `models = trial_models` variable above to run the trial models
models = [
    {
        "title": "DecisionTreeClassifier (RobustScaler + PCA) -- Tuned",
        "pipeline": Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                ("pca", pca),
                ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "param_grid": {
            "scaler__quantile_range": [(25, 75),],
            "pca__n_components": [2],
            "clf__criterion": ["gini"],
            "clf__splitter": ["random"],
            "clf__min_samples_split": [4],
        },
    },
    {
        "title": "AdaBoost (PCA) -- Tuned",
        "pipeline": Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                ("pca", PCA(random_state=RANDOM_STATE)),
                ("clf", AdaBoostClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "param_grid": {
            "scaler__quantile_range": [(25, 75),],
            "pca__n_components": [2],
            "clf__n_estimators": [16],
            "clf__algorithm": ["SAMME"],
            "clf__learning_rate": [1],
        },
    },
    {
        "title": "KNeighbors (PCA) -- Tuned",
        "pipeline": Pipeline(steps=[("pca", PCA()), ("clf", KNeighborsClassifier()),]),
        "param_grid": {
            "pca__n_components": [2],
            "clf__n_neighbors": [2],
            "clf__weights": ["uniform"],
            "clf__algorithm": ["ball_tree"],
            "clf__leaf_size": [2],
            "clf__p": [1],
        },
    },
    {
        "title": "RandomForest (RobustScaler + PCA) -- Tuned",
        "pipeline": Pipeline(
            steps=[
                ("scaler", RobustScaler()),
                ("pca", PCA()),
                ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "param_grid": {
            "pca__n_components": [2],
            "scaler__quantile_range": [(32, 68)],
            "clf__criterion": ["gini"],
            "clf__n_estimators": [20],
            "clf__min_samples_split": [2],
            "clf__class_weight": [None],
        },
    },
]

### Task 4: Try a variety of classifiers
for model in models:
    model_clf = GridSearchCV(
        model.get("pipeline"), model.get("param_grid", {}), cv=3, iid=False,
    )

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    validate_classifier(
        clf_name=model.get("title"),
        clf=model_clf,
        features_list=features_list,
        features=features,
        labels=labels,
        folds=FOLDS,
        reports={
            "classification": False,
            "best_estimator": False,
            "best_params": False,
            "confusion_matrix": False,
        },
        random_state=RANDOM_STATE,
    )

# selected Classifier
clf = Pipeline(
    steps=[
        ("scaler", RobustScaler()),
        ("pca", PCA(n_components=2)),
        (
            "clf",
            DecisionTreeClassifier(
                criterion="gini",
                splitter="random",
                min_samples_split=4,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
test_classifier(clf=clf, dataset=my_dataset, feature_list=features_list, folds=FOLDS)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
