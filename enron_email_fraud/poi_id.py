#!/usr/bin/python

import sys
import pickle
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
    summarizeFeature,
    Draw,
    plotData,
    create_features,
    calculate_poi_msgs,
    printPredictions,
    validate_classifier,
)

# setup environment
pp = pprint.PrettyPrinter(indent=4)
sns.set()
sns.set_context("notebook", rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid")

# suppress scientific notation in Pandas
pd.options.display.float_format = "{:.2f}".format
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

RANDOM_STATE = 0
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

df = pd.DataFrame(data_dict).transpose().apply(pd.to_numeric, errors="coerce")


### Task 2: Remove outliers
# remove the TOTAL key from the dict because it's an outlier in the data
# TOTAL represents an aggregate value
# 'THE TRAVEL AGENCY IN THE PARK' doesn't represent a person
df = df.drop(["TOTAL", "THE TRAVEL AGENCY IN THE PARK"], errors="ignore", axis=0)
df = df.fillna(0)

# print(list(df.index))

# exit()

# summarize selected features (except for POI)
for feature in features_list[1:]:
    summarizeFeature(df, feature)

# commenting out for now because of "38320 segmentation fault"
# plotData(
#     df.to_dict(orient="index"),
#     features=["poi", features_list[1]],
#     xLabel="POI",
#     yLabel=features_list[1],
# )
# plotData(
#     df.to_dict(orient="index"),
#     features=["poi", features_list[2]],
#     xLabel="POI",
#     yLabel=features_list[2],
# )


# ### Task 3: Create new feature(s)
df["poi_messages"] = df.apply(calculate_poi_msgs, axis=1)

### Store to my_dataset for easy export below.
(
    features_train,
    features_test,
    labels_train,
    labels_test,
    features,
    labels,
    my_dataset,
) = create_features(df, features_list)

### Task 4: Try a variety of classifiers

################################################
# DecisionTreeClassifier(StandardScaler + PCA)
################################################
pca = PCA()

# pipe = Pipeline(steps=[("pca", PCA()), ("clf", GaussianNB())])
models = [
    {
        "title": "DecisionTreeClassifier",
        "pipeline": Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("pca", pca),
                ("clf", DecisionTreeClassifier(random_state=1)),
            ]
        ),
        "param_grid": {
            "pca__n_components": range(1, len(features_list) - 1) + [None],
            "clf__criterion": ["gini", "entropy"],
            "clf__splitter": ["best", "random"],
            "clf__min_samples_split": [2, 4, 6, 8, 10, 20, 30, 40],
        },
    }
]

# validate models
for model in models:
    clf = GridSearchCV(model.get("pipeline"), model.get("param_grid"), cv=3, iid=False)
    validate_classifier(
        clf_name=model.get("title"),
        clf=clf,
        features=features,
        labels=labels,
        folds=FOLDS,
        reports={
            "classification": False,
            "best_estimator": False,
            "confusion_matrix": False,
        },
        random_state=RANDOM_STATE,
    )

    test_classifier(
        clf=clf, dataset=my_dataset, feature_list=features_list, folds=FOLDS
    )

    Draw(
        pred.astype(int),
        pca.fit_transform(features_test),
        "poi",
        mark_poi=False,
        name="clusters.pdf",
    )


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
