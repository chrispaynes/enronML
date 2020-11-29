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

# set a consistent randomization
RANDOM_STATE = 42
# set a consistent number of folds/splits
FOLDS = 10

features_list = [
    "poi",
    "bonus",
    "exercised_stock_options",
    "expenses",
]

# construct a PCA to use as a pipeline step
pca = PCA()

trial_models = [
    {
        "title": "GaussianNB (PCA)",
        "pipeline": Pipeline(steps=[("pca", pca), ("clf", GaussianNB()),]),
        "param_grid": {"pca__n_components": [1]},
    },
    {
        "title": "GaussianNB (StandardScaler + PCA)",
        "pipeline": Pipeline(
            steps=[("scaler", StandardScaler()), ("pca", pca), ("clf", GaussianNB())]
        ),
        "param_grid": {"pca__n_components": range(1, len(features_list) - 1) + [None],},
    },
    {
        "title": "GaussianNB (MinMaxScaler + PCA)",
        "pipeline": Pipeline(
            steps=[("scaler", MinMaxScaler()), ("pca", pca), ("clf", GaussianNB())]
        ),
        "param_grid": {"pca__n_components": range(1, len(features_list) - 1) + [None],},
    },
    {
        "title": "DecisionTreeClassifier (PCA)",
        "pipeline": Pipeline(
            steps=[
                ("pca", pca),
                ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "param_grid": {
            "pca__n_components": range(1, len(features_list) - 1) + [None],
            "clf__criterion": ["gini", "entropy"],
            "clf__splitter": ["best", "random"],
            "clf__min_samples_split": [2, 4, 6, 8, 10, 20, 30, 40],
        },
    },
    {
        "title": "DecisionTreeClassifier (StandardScaler + PCA)",
        "pipeline": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", pca),
                ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "param_grid": {
            "pca__n_components": range(1, len(features_list) - 1) + [None],
            "clf__criterion": ["gini", "entropy"],
            "clf__splitter": ["best", "random"],
            "clf__min_samples_split": [2, 4, 6, 8, 10, 20, 30, 40],
        },
    },
    {
        "title": "DecisionTreeClassifier (MinMaxScaler + PCA)",
        "pipeline": Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("pca", pca),
                ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "param_grid": {
            "pca__n_components": range(1, len(features_list) - 1) + [None],
            "clf__criterion": ["gini", "entropy"],
            "clf__splitter": ["best", "random"],
            "clf__min_samples_split": [2, 4, 6, 8, 10, 20, 30, 40],
        },
    },
    {
        "title": "AdaBoost (PCA)",
        "pipeline": Pipeline(
            steps=[
                ("pca", PCA()),
                ("clf", AdaBoostClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "param_grid": {
            "pca__n_components": range(1, len(features_list) - 1) + [None],
            "clf__n_estimators": [25, 50, 75, 100],
            "clf__algorithm": ["SAMME", "SAMME.R"],
        },
    },
    {
        "title": "AdaBoost (StandardScaler + PCA)",
        "pipeline": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("clf", AdaBoostClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "param_grid": {
            "pca__n_components": range(1, len(features_list) - 1) + [None],
            "clf__n_estimators": [25, 50, 75, 100],
            "clf__algorithm": ["SAMME", "SAMME.R"],
        },
    },
    {
        "title": "AdaBoost (MinMaxScaler + PCA)",
        "pipeline": Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("pca", PCA()),
                ("clf", AdaBoostClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "param_grid": {
            "pca__n_components": range(1, len(features_list) - 1) + [None],
            "clf__n_estimators": [25, 50, 75, 100],
            "clf__algorithm": ["SAMME", "SAMME.R"],
        },
    },
    {
        "title": "KNeighbors (PCA)",
        "pipeline": Pipeline(steps=[("pca", PCA()), ("clf", KNeighborsClassifier()),]),
        "param_grid": {
            "pca__n_components": range(1, len(features_list) - 1) + [None],
            "clf__n_neighbors": [2, 4, 6, 8],
            "clf__weights": ["uniform", "distance"],
            "clf__algorithm": ["ball_tree", "kd_tree", "brute",],
            "clf__leaf_size": [5, 10],
            "clf__p": [1, 2],
        },
    },
    {
        "title": "KNeighbors (StandardScaler + PCA)",
        "pipeline": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("clf", KNeighborsClassifier()),
            ]
        ),
        "param_grid": {
            "pca__n_components": range(1, len(features_list) - 1) + [None],
            "clf__n_neighbors": [2, 4, 6, 8],
            "clf__weights": ["uniform", "distance"],
            "clf__algorithm": ["ball_tree", "kd_tree", "brute",],
            "clf__leaf_size": [5, 10],
            "clf__p": [1, 2],
        },
    },
    {
        "title": "KNeighbors (MinMaxScaler + PCA)",
        "pipeline": Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("pca", PCA()),
                ("clf", KNeighborsClassifier()),
            ]
        ),
        "param_grid": {
            "pca__n_components": range(1, len(features_list) - 1) + [None],
            "clf__n_neighbors": [2, 4, 6, 8],
            "clf__weights": ["uniform", "distance"],
            "clf__algorithm": ["ball_tree", "kd_tree", "brute",],
            "clf__leaf_size": [5, 10],
            "clf__p": [1, 2],
        },
    },
    # {
    #     "title": "RandomForest (PCA)",
    #     "pipeline": Pipeline(
    #         steps=[
    #             ("pca", PCA()),
    #             ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    #         ]
    #     ),
    #     "param_grid": {
    #         # "pca__n_components": range(1, len(features_list) - 1) + [None],
    #         "clf__criterion": ["gini", "entropy"],
    #         "clf__n_estimators": [5, 10, 25, 50, 100],
    #         "clf__max_depth": [2, 4, 6, 8, 10, 20],
    #         "clf__min_samples_split": [2, 6, 10, 20, 30],
    #         "clf__class_weight": ["balanced", "balanced_subsample"],
    #     },
    # },
    # {
    #     "title": "RandomForest (StandardScaler + PCA)",
    #     "pipeline": Pipeline(
    #         steps=[
    #             ("scaler", StandardScaler()),
    #             ("pca", PCA()),
    #             ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    #         ]
    #     ),
    #     "param_grid": {
    #         # "pca__n_components": range(1, len(features_list) - 1) + [None],
    #         "clf__criterion": ["gini", "entropy"],
    #         "clf__n_estimators": [5, 10, 25, 50, 100],
    #         "clf__max_depth": [2, 4, 6, 8, 10, 20],
    #         "clf__min_samples_split": [2, 6, 10, 20, 30],
    #         "clf__class_weight": ["balanced", "balanced_subsample"],
    #     },
    # },
    # {
    #     "title": "RandomForest (MinMaxScaler + PCA)",
    #     "pipeline": Pipeline(
    #         steps=[
    #             ("scaler", MinMaxScaler()),
    #             ("pca", PCA()),
    #             ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    #         ]
    #     ),
    #     "param_grid": {
    #         # "pca__n_components": range(1, len(features_list) - 1) + [None],
    #         "clf__criterion": ["gini", "entropy"],
    #         "clf__n_estimators": [5, 10, 25, 50, 100],
    #         "clf__max_depth": [2, 4, 6, 8, 10, 20],
    #         "clf__min_samples_split": [2, 6, 10, 20, 30],
    #         "clf__class_weight": ["balanced", "balanced_subsample"],
    #     },
    # },
]
