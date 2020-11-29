import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    KFold,
    StratifiedShuffleSplit,
)
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
from scipy import stats


def Draw(
    pred, features, poi, mark_poi=False, f1_name="Feature 1", f2_name="Feature 2",
):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        # print (ii, pp)
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")

    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.grid(True)
    # plt.show()


def plotData(data_dict, features, xLabel, yLabel, scaler=None, markOutlier=True):
    data = featureFormat(data_dict, features=features)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("{} and {}:".format(yLabel.upper(), xLabel.upper()), loc="center")
    feature_x, feature_y = features

    # print ("DATA", data)
    # exit()
    if scaler:
        # print ("featurex", data[:, 0])
        data = scaler.fit_transform(data)

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


def create_features(df, features_list):
    """creates the feature/target split, using the supplied DataFrame and the desired Feature List

    Args:
        df (pandas.core.frame.DataFrame): DataFrame containing the feature set
        features_list (list): str, list of DataFrame features to include

    Returns:
        list: (
            features (list): feature_list features
            labels (list): POIs data feature
            my_dataset (dict): original dataset with included features
        )
    """

    my_dataset = df[features_list].transpose().to_dict()

    data = featureFormat(my_dataset, features_list, sort_keys=True, remove_NaN=True)
    labels, features = targetFeatureSplit(data)

    return (
        np.array(features),
        np.array(labels),
        my_dataset,
    )


def calculate_pct_poi_msgs(series):
    """calculates the percentage of a potential POI's
       total messages that are to or from a POI

    Args:
        series (pandas.core.series.Series): data series associated with a single POI

    Returns:
        [pandas.core.series.Series]: data series with the added POI message calculate
    """
    total_msgs = series["from_messages"] + series["to_messages"]
    series["pct_poi_messages"] = 0

    if total_msgs > 0:
        series["pct_poi_messages"] = (
            (series["from_poi_to_this_person"] + series["from_this_person_to_poi"])
            / float(total_msgs)
        ) * 100

    return series["pct_poi_messages"]


def validate_classifier(
    clf_name,
    clf,
    features_list,
    features,
    labels,
    folds=10,
    reports={
        "classification": False,
        "best_estimator": False,
        "best_params": False,
        "confusion_matrix": False,
    },
    random_state=None,
):
    """fits a classifier to a test and train dataset and performs
       validation summaries and reports using classifier predictions

    Args:
        clf_name (string): display name for the classifier
        clf ([type]): [description]
        features ([type]): [description]
        labels ([type]): [description]
        folds (int, optional): [description]. Defaults to 10.
        reports (dict, optional): [description]. Defaults to { "classification": False, "best_estimator": False, "confusion_matrix": False, }.
        random_state ([type], optional): [description]. Defaults to None.
    """

    print "\nValidating {}: random_state = {}".format(clf_name, random_state)
    print "Mean {}-Fold Cross Validation Test Score: {:.2f}%\n".format(
        folds,
        cross_validate(clf, features, labels, cv=folds, scoring="accuracy")[
            "test_score"
        ].mean()
        * 100,
    )

    # print ("features", features)
    # exit()

    sss = StratifiedShuffleSplit(n_splits=folds, random_state=random_state)

    cf_matricies = []
    TP, TN, FP, FN = 0, 0, 0, 0
    y_pred = None
    X_train, X_test, y_train, y_test = None, None, None, None

    for train_index, test_index in sss.split(features, labels):
        # print ("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        clf.fit(X_train, y_train)
        if reports.get("best_estimator"):
            print ("GridSearch Best Estimator", clf.best_estimator_)

        if reports.get("best_params"):
            print ("GridSearch Best Params", clf.best_params_)

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

    Draw(
        y_pred.astype(int),
        PCA(n_components=2).fit_transform(MinMaxScaler().fit_transform(X_test)),
        poi=y_test,
        mark_poi=True,
        f1_name=features_list[1],
        f2_name=features_list[2],
    )

    # aggregate summary information
    population, acc, precision, recall, f1 = 0, 0, 0, 0, 0
    population = np.sum(cf_matricies)

    try:
        acc = 1 * (TP + TN) / population
    except ZeroDivisionError:
        pass

    try:
        precision = 1 * TP / (TP + FP)
    except ZeroDivisionError:
        pass

    try:
        recall = 1 * TP / (TP + FN)
    except ZeroDivisionError:
        pass

    try:
        f1 = (2.0 * TP) / ((2 * TP) + FP + FN)
    except ZeroDivisionError:
        pass

    mc_title = "{} Aggregate Model Classification Performance".format(clf_name)
    print mc_title
    print "=" * len(mc_title)
    print "Population:\t", np.sum(cf_matricies)
    print "Accuracy:\t", round(acc, 2)
    print "Precision:\t", round(precision, 2)
    print "Recall:\t\t", round(recall, 2)
    print "F1 Score:\t", round(f1, 2)
    print
    fm_title = "{} Aggregate Confusion Matrix".format(clf_name)
    print fm_title
    print "=" * len(fm_title)
    print np.sum(cf_matricies, axis=0)
