import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit

# from sklearn.cross_validation import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    KFold,
    StratifiedShuffleSplit,
)
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


def summarizeFeature(data, feature):
    print "\nSUMMARY FOR: '{}'".format(feature.upper())
    print "=" * 25
    print (data[feature].astype(np.float64).describe())
    print


def Draw(
    pred,
    features,
    poi,
    mark_poi=False,
    name="image.png",
    f1_name="feature 1",
    f2_name="feature 2",
):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


def plotData(data_dict, features, xLabel, yLabel, markOutlier=True):
    data = featureFormat(data_dict, features=features)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(
        "POIs with {} and {}:".format(yLabel.upper(), xLabel.upper()), loc="center"
    )
    feature_x, feature_y = features

    zScores = abs(stats.zscore(data, axis=0))

    #  plot data points
    for index, (x, y) in enumerate(data):
        xZ, yZ = zScores[index]
        plt.scatter(x, y)

    if features[0] == "poi":
        plt.xticks([0, 1])

    # annotate the data point if X or Y is greater than 2 Stdevs.
    if (yZ > 2) and markOutlier:
        # find the person associated with the data point
        poi_name = next(
            (
                key
                for (key, value) in data_dict.items()
                if value.get(feature_x) == x or value.get(feature_y) == y
            ),
            None,
        )

        ax.annotate("%s" % (poi_name), xy=(x, y), textcoords="data")

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid(True)
    plt.figure()
    plt.show()


def create_features(df, features_list):
    my_dataset = df[features_list].transpose().to_dict()

    data = featureFormat(my_dataset, features_list, sort_keys=True, remove_NaN=True)
    labels, features = targetFeatureSplit(data)
    labels, features = np.array(labels), np.array(features)

    # save 30% of data for testing
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.3, random_state=1
    )

    return (
        features_train,
        features_test,
        labels_train,
        labels_test,
        features,
        labels,
        my_dataset,
    )


def calculate_poi_msgs(x):
    total_msgs = x["from_messages"] + x["to_messages"]
    x["poi_messages"] = 0

    if total_msgs > 0:
        x["poi_messages"] = (
            (x["from_poi_to_this_person"] + x["from_this_person_to_poi"])
            / float(total_msgs)
        ) * 100

    return x["poi_messages"]


def printPredictions(predictions, labels_test):
    print
    print (
        "{} predictions / {} test data points".format(
            len(predictions), len(labels_test)
        )
    )
    print "PREDICTIONS: ", pd.DataFrame(predictions, dtype=int).transpose().values
    print "TEST:\t", pd.DataFrame(labels_test, dtype=int).transpose().values
    print "{} POIs detected".format(len([n for n in predictions if n == 1]))
    print


def validate_classifier(
    clf_name,
    clf,
    features,
    labels,
    folds=10,
    reports={
        "classification": False,
        "best_estimator": False,
        "confusion_matrix": False,
    },
    random_state=None,
):
    print "\nValidating {}".format(clf_name)
    print "Mean {}-Fold Cross Validation Test Score: {:.2f}%\n".format(
        folds,
        cross_validate(clf, features, labels, cv=folds, scoring="accuracy")[
            "test_score"
        ].mean()
        * 100,
    )

    sss = StratifiedShuffleSplit(n_splits=folds, random_state=random_state)

    cf_matricies = []
    TP, TN, FP, FN = 0, 0, 0, 0

    for train_index, test_index in sss.split(features, labels):
        # print ("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        clf.fit(X_train, y_train)
        if reports.get("best_estimator"):
            print ("GridSearch Best Estimator", clf.best_estimator_)

        y_pred = clf.predict(X_test)

        if reports.get("classification"):
            # Build a text report showing the main classification metrics.
            print classification_report(
                y_true=y_test, y_pred=y_pred, target_names=["NOT POI", "POI"]
            )

        # Compute Confusion Matrix to evaluate the accuracy of a classification.
        cfm = confusion_matrix(y_true=y_test, y_pred=y_pred)

        # classifier accurately predicted person was a POI
        TP += float(cfm[1, 1])
        # classifier accurately predicted person not a POI
        TN += float(cfm[0, 0])
        # classifier falsely predicted person was a POI
        FP += float(cfm[0, 1])
        # classifier falsely predicted person not a POI
        FN += float(cfm[1, 0])

        # store the Confusion Matrix for this split so we can aggregate results
        cf_matricies.append(cfm)

        if reports.get("confusion_matrix"):
            print "Confusion Matrix"
            print cfm
            print

    # aggregate summary information
    population = np.sum(cf_matricies)
    acc = 1 * (TP + TN) / population
    precision = 1 * TP / (TP + FP)
    recall = 1 * TP / (TP + FN)
    f1 = (2.0 * TP) / ((2 * TP) + FP + FN)

    print "Aggregate Model Classification Performance"
    print "=" * 50
    print "Population:\t", np.sum(cf_matricies)
    print "Accuracy:\t", round(acc, 2)
    print "Precision:\t", round(precision, 2)
    print "Recall:\t\t", round(recall, 2)
    print "F1 Score:\t", round(f1, 2)
    print
    print "Aggregate Confusion Matrix"
    print "=" * 50
    print np.sum(cf_matricies, axis=0)
