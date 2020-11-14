#!/usr/bin/python

import sys
import pickle
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.style.use("classic")
import seaborn as sns
from scipy import stats

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

pp = pprint.PrettyPrinter(indent=4)
sns.set()
sns.set_context("notebook", rc={"lines.linewidth": 2.5})
sns.set_style("whitegrid")

# suppress scientific notation in Pandas
pd.options.display.float_format = "{:.2f}".format

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
    "poi",
    "salary",
    "bonus",
    "long_term_incentive",
    "exercised_stock_options",
    "total_payments",
]  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# remove the TOTAL key from the dict because it's an outlier in the data
# TOTAL represents an aggregate value
data_dict.pop("TOTAL", 0)

### Task 2: Remove outliers
for data in data_dict.items():
    # pp.pprint(data[1])
    pass
# pp.pprint(data_dict)

# salary  = filter(lambda person: person[1], [for data in data_dict.items()])
# remove NaN salaries
salary = np.array(
    filter(
        lambda person: person[1].get("salary") != "NaN",
        [data for data in data_dict.items()],
    )
)


def filterNaNs(feature):
    # return np.array(
    #     filter(
    #         lambda person: person[1].get(feature) != "NaN",
    #         [data for data in data_dict.items()],
    #     )
    # )
    return {
        key: value for (key, value) in data_dict.items() if value.get(feature) != "NaN"
    }
    # return filter(
    #     lambda person: person[1].get(feature) != "NaN",
    #     [data for data in data_dict.items()],
    # )


salary = filterNaNs("salary")
# print "salary:", salary

bonus = np.array(
    filter(
        lambda person: person[1].get("bonus") != "NaN",
        [data for data in data_dict.items()],
    )
)

# for key, person in salary.items():
#     pp.pprint(person.get("salary"))


# pp.pprint(salary)

salaries = np.array([s[1].get("salary") for s in salary.items()])
# print "SALARIES:", salaries

max_salary = filter(
    lambda person: person[1].get("salary") == np.max(salaries), salary.items()
)
# SKILLING JEFFREY K'
print "person with max salary: ", max_salary

mean = np.mean(salaries)
standard_deviation = np.std(salaries)
print "mean:", mean
print "standard dev:", standard_deviation
# pp.pprint(salary)
# print ("SALARIES")
# reverse sort
# salaries = -np.sort(-salaries)
# pp.pprint(salaries)


data = featureFormat(
    data_dict,
    features=[
        "salary",
        "bonus",
        "long_term_incentive",
        "exercised_stock_options",
        "total_payments",
    ],
)


def summarizeFeature(feature, title):
    print "\nSUMMARY FOR {}".format(title.upper())
    print "=" * 25
    pp.pprint(pd.DataFrame(feature).describe())
    print "\n"


for feature in features_list[1:]:
    f = filter(None, [d.get(feature) for d in filterNaNs(feature).values()])
    summarizeFeature(f, feature)


def plotData(data_dict, features, xLabel, yLabel):
    data = featureFormat(data_dict, features=features)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # mean = np.mean(data)
    print "type data:", type(data)

    print "mean: {}, {}".format(xLabel, yLabel), np.mean(data, axis=0)

    # xMean, yMean = np.mean(data, axis=0)
    x75pct, y75thpct = np.percentile(data, 75, axis=0)
    # print "Zscore:", abs(stats.zscore(data, axis=0))
    zScores = abs(stats.zscore(data, axis=0))

    # sns.scatterplot(data=data, hue=tip_rate)
    for x, y in data:
        plt.scatter(x, y)

    for index, (x, y) in enumerate(data):

        # pass
        xZ, yZ = zScores[index]
        if xZ > 2 or yZ > 2:
            print "{} outlier x: {}, y: {}".format(features, x, y)
            # person = dict(
            #     lambda person: person[1].get(features[0]) == x, data_dict.items()
            # )

            person = {
                key: value
                for (key, value) in data_dict.items()
                if value.get(features[0]) == x or value.get(features[1]) == y
            }

            print "outlier person:", person
            person_name = person.keys()[0]

            ax.annotate("%s" % (person_name), xy=(x, y), textcoords="data")

        # print "index: {}, x: {}, y: {}".format(index, x, y)
        # ax.annotate("(%s)" % xy, xy=xy, textcoords="data")

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.grid()
    plt.figure()


plotData(data_dict, features=["salary", "bonus"], xLabel="Salary", yLabel="Bonus")
# plotData(
#     data_dict,
#     features=["salary", "long_term_incentive"],
#     xLabel="Salary",
#     yLabel="long_term_incentive",
# )
# plotData(
#     data_dict,
#     features=["salary", "exercised_stock_options"],
#     xLabel="Salary",
#     yLabel="exercised_stock_options",
# )

# plotData(
#     data_dict,
#     features=["salary", "total_payments"],
#     xLabel="Salary",
#     yLabel="total_payments",
# )

plt.show()


# count      94.00
# mean   284087.54
# std    177131.12
# min       477.00
# 25%    211802.00
# 50%    258741.00
# 75%    308606.50
# max   1111258.00
salary_df = pd.DataFrame(salaries)
# pp.pprint(pd.DataFrame(salaries).describe())
mean = salary_df.mean()
standard_deviation = salary_df.std()
print "mean:", mean
print "standard dev:", standard_deviation
print "len salaries:", len(salaries)


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


print (reject_outliers(salaries, m=2))


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
