# What will I learn?

Deal with an imperfect, real-world dataset
Validate a machine learning result using test data
Evaluate a machine learning result using quantitative metrics
Create, select and transform features
Compare the performance of machine learning algorithms
Tune machine learning algorithms for maximum performance
Communicate your machine learning algorithm results clearly

# Table of Contents
- [What will I learn?](#what-will-i-learn)
- [Table of Contents](#table-of-contents)
- [Requirements](#requirements)
- [File Archicture](#file-archicture)
- [Project Summary](#project-summary)
- [Data Exploration](#data-exploration)
      - [Statistical Summary](#statistical-summary)
      - [Data Features](#data-features)
      - [Sample Records](#sample-records)
- [Outlier Removal](#outlier-removal)
- [Feature Selection](#feature-selection)
    - [Step 1: Missing Value Ratio](#step-1-missing-value-ratio)
    - [Step 2: Low Variance](#step-2-low-variance)
    - [Step 3: High Feature Correlation](#step-3-high-feature-correlation)
      - [General Feature Correlation](#general-feature-correlation)
      - [Features With High Correlation to 'POI'](#features-with-high-correlation-to-poi)
      - [Features With High Correlation](#features-with-high-correlation)
      - [Features With Low Correlation](#features-with-low-correlation)
      - [Features With High Mean Correlation To Other Features](#features-with-high-mean-correlation-to-other-features)
      - [Features With Low Mean Correlation To Other Features](#features-with-low-mean-correlation-to-other-features)
      - [Recursive Feature Reduction (RFE) with Logistic Regression Classifier](#recursive-feature-reduction-rfe-with-logistic-regression-classifier)
      - [Recursive Feature Reduction (RFE) with Random Forest Classifier](#recursive-feature-reduction-rfe-with-random-forest-classifier)
    - [Selected Features](#selected-features)
- [Feature Engineering](#feature-engineering)
- [Dimensionality Reduction / Feature Scaling](#dimensionality-reduction--feature-scaling)
- [Model Parameter Tuning](#model-parameter-tuning)
- [Model Validation](#model-validation)
- [Model Evaluation Metrics](#model-evaluation-metrics)
    - [Confusion Matrix](#confusion-matrix)
    - [Classification Report](#classification-report)
- [Classifier Validation Results](#classifier-validation-results)
    - [Gaussian Naive Bayes](#gaussian-naive-bayes)
        - [Validating GaussianNB (PCA): random_state = 0](#validating-gaussiannb-pca-random_state--0)
        - [Validating GaussianNB (StandardScaler + PCA): random_state = 0](#validating-gaussiannb-standardscaler--pca-random_state--0)
        - [Validating GaussianNB (MinMaxScaler + PCA): random_state = 0](#validating-gaussiannb-minmaxscaler--pca-random_state--0)
    - [Decision Tree](#decision-tree)
        - [Validating DecisionTreeClassifier (PCA): random_state = 0](#validating-decisiontreeclassifier-pca-random_state--0)
        - [Validating DecisionTreeClassifier (StandardScaler + PCA): random_state = 0](#validating-decisiontreeclassifier-standardscaler--pca-random_state--0)
        - [Validating DecisionTreeClassifier (MinMaxScaler + PCA): random_state = 0](#validating-decisiontreeclassifier-minmaxscaler--pca-random_state--0)
    - [AdaBoost](#adaboost)
        - [Validating AdaBoost (PCA): random_state = 0](#validating-adaboost-pca-random_state--0)
        - [Validating AdaBoost (StandardScaler + PCA): random_state = 0](#validating-adaboost-standardscaler--pca-random_state--0)
        - [Validating AdaBoost (MinMaxScaler + PCA): random_state = 0](#validating-adaboost-minmaxscaler--pca-random_state--0)
    - [KNN](#knn)
        - [Validating KNeighbors (PCA): random_state = 0](#validating-kneighbors-pca-random_state--0)
        - [Validating KNeighbors (StandardScaler + PCA): random_state = 0](#validating-kneighbors-standardscaler--pca-random_state--0)
        - [Validating KNeighbors (MinMaxScaler + PCA): random_state = 0](#validating-kneighbors-minmaxscaler--pca-random_state--0)
    - [Random Forest](#random-forest)
        - [Validating RandomForest (PCA): random_state = 0](#validating-randomforest-pca-random_state--0)
        - [Validating RandomForest (StandardScaler + PCA): random_state = 0](#validating-randomforest-standardscaler--pca-random_state--0)
      - [Validating RandomForest (MinMaxScaler + PCA): random_state = 0](#validating-randomforest-minmaxscaler--pca-random_state--0)


# Requirements
- Python 2.7
- [Starter Code + Enron dataset]("https://github.com/udacity/ud120-projects.git)
-


# File Archicture

| Filename                  | Description                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| enron_email.ipynb         | Jupyter Notebook contained detailed data exploration and charts                                                                                                                                                                                                                                                                                                                                                           |
| poi_id.py                 | Starter code for the POI identifier, you will write your analysis here. You will also submit a version of this file for your evaluator to verify your algorithm and results.                                                                                                                                                                                                                                              |
| utils.py                  |                                                                                                                                                                                                                                                                                                                                                                                                                           |
| final_project_dataset.pkl | The dataset for the project, more details below.                                                                                                                                                                                                                                                                                                                                                                          |
| tester.py                 | When you turn in your analysis for evaluation by Udacity, you will submit the algorithm, dataset and list of features that you use (these are created automatically in poi_id.py). The evaluator will then use this code to test your result, to make sure we see performance that’s similar to what you report. You don’t need to do anything with this code, but we provide it for transparency and for your reference. |
| emails_by_address         | this directory contains many text files, each of which contains all the messages to or from a particular email address. It is for your reference, if you want to create more advanced features based on the details of the emails dataset. You do not need to process the e-mail corpus in order to complete the project.                                                                                                 |

# Project Summary
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

Steps to Success
We will provide you with starter code that reads in the data, takes your features of choice, then puts them into a NumPy array, which is the input form that most sklearn functions assume. Your job is to engineer the features, pick and tune an algorithm, and to test and evaluate your identifier.

As preprocessing to this project, we've combined the Enron email and financial data into a dictionary, where each key-value pair in the dictionary corresponds to one person. The dictionary key is the person's name, and the value is another dictionary, which contains the names of all the features and their values for that person.

You are encouraged to make, transform or rescale new features from the starter features. If you do this, you should store the new feature to my_dataset, and if you use the new feature in the final algorithm, you should also add the feature name to my_feature_list, so your evaluator can access it during testing.

# Data Exploration
- 20 features
- 146 records

#### Statistical Summary
| Statistic    | bonus       | deferral_payments | deferred_income | director_fees | email_address | exercised_stock_options | expenses   | from_messages | from_poi_to_this_person | from_this_person_to_poi | loan_advances | long_term_incentive | other       | restricted_stock | restricted_stock_deferred | salary      | shared_receipt_with_poi | to_messages | total_payments | total_stock_value |
| ------------ | ----------- | ----------------- | --------------- | ------------- | ------------- | ----------------------- | ---------- | ------------- | ----------------------- | ----------------------- | ------------- | ------------------- | ----------- | ---------------- | ------------------------- | ----------- | ----------------------- | ----------- | -------------- | ----------------- |
| <b>count</b> | 82.00       | 39.00             | 49.00           | 17.00         | 0.00          | 102.00                  | 95.00      | 86.00         | 86.00                   | 86.00                   | 4.00          | 66.00               | 93.00       | 110.00           | 18.00                     | 95.00       | 86.00                   | 86.00       | 125.00         | 126.00            |
| <b>mean</b>  | 2374234.61  | 1642674.15        | -1140475.14     | 166804.88     | nan           | 5987053.77              | 108728.92  | 608.79        | 64.90                   | 41.23                   | 41962500.00   | 1470361.45          | 919064.97   | 2321741.14       | 166410.56                 | 562194.29   | 1176.47                 | 2073.86     | 5081526.49     | 6773957.45        |
| <b>std</b>   | 10713327.97 | 5161929.97        | 4025406.38      | 319891.41     | nan           | 31062006.57             | 533534.81  | 1841.03       | 86.98                   | 100.07                  | 47083208.70   | 5942759.32          | 4589252.91  | 12518278.18      | 4201494.31                | 2716369.15  | 1178.32                 | 2582.70     | 29061716.40    | 38957772.73       |
| <b>min</b>   | 70000.00    | -102500.00        | -27992891.00    | 3285.00       | nan           | 3285.00                 | 148.00     | 12.00         | 0.00                    | 0.00                    | 400000.00     | 69223.00            | 2.00        | -2604490.00      | -7576788.00               | 477.00      | 2.00                    | 57.00       | 148.00         | -44093.00         |
| <b>25</b>%   | 431250.00   | 81573.00          | -694862.00      | 98784.00      | nan           | 527886.25               | 22614.00   | 22.75         | 10.00                   | 1.00                    | 1600000.00    | 281250.00           | 1215.00     | 254018.00        | -389621.75                | 211816.00   | 249.75                  | 541.25      | 394475.00      | 494510.25         |
| <b>50</b>%   | 769375.00   | 227449.00         | -159792.00      | 108579.00     | nan           | 1310813.50              | 46950.00   | 41.00         | 35.00                   | 8.00                    | 41762500.00   | 442035.00           | 52382.00    | 451740.00        | -146975.00                | 259996.00   | 740.50                  | 1211.00     | 1101393.00     | 1102872.50        |
| <b>75</b>%   | 1200000.00  | 1002671.50        | -38346.00       | 113784.00     | nan           | 2547724.00              | 79952.50   | 145.50        | 72.25                   | 24.75                   | 82125000.00   | 938672.00           | 362096.00   | 1002369.75       | -75009.75                 | 312117.00   | 1888.25                 | 2634.75     | 2093263.00     | 2949846.75        |
| <b>max</b>   | 97343619.00 | 32083396.00       | -833.00         | 1398517.00    | nan           | 311764000.00            | 5235198.00 | 14368.00      | 528.00                  | 609.00                  | 83925000.00   | 48521928.00         | 42667589.00 | 130322299.00     | 15456290.00               | 26704229.00 | 5521.00                 | 15149.00    | 309886585.00   | 434509511.00      |

#### Data Features
The features in the data fall into three major types, namely financial features, email features and POI labels.

| Feature                   | feature type | notes                                                                                           |
| ------------------------- | ------------ | ----------------------------------------------------------------------------------------------- |
| bonus                     | financial    |                                                                                                 |
| deferral_payments         | financial    | can be a positive or negative value                                                             |
| deferred_income           | financial    | can be a positive or negative value                                                             |
| director_fees             | financial    |                                                                                                 |
| email_address             | email        |                                                                                                 |
| exercised_stock_options   | financial    |                                                                                                 |
| expenses                  | financial    |                                                                                                 |
| from_messages             | email        |                                                                                                 |
| from_poi_to_this_person   | email        |                                                                                                 |
| from_this_person_to_poi   | email        |                                                                                                 |
| loan_advances             | financial    |                                                                                                 |
| long_term_incentive       | financial    |                                                                                                 |
| other                     | financial    |                                                                                                 |
| poi                       | label        | boolean, represented as integer, indicating whether an individual is a Person of Interest (POI) |
| restricted_stock          | financial    | can be a positive or negative value                                                             |
| restricted_stock_deferred | financial    | can be a positive or negative value                                                             |
| salary                    | financial    |                                                                                                 |
| shared_receipt_with_poi   | email        |                                                                                                 |
| to_messages               | email        |                                                                                                 |
| total_payments            | financial    |                                                                                                 |
| total_stock_value         | financial    | can be a positive or negative value                                                             |


#### Sample Records
| Person             | bonus      | deferral_payments | deferred_income | director_fees | email_address | exercised_stock_options | expenses  | from_messages | from_poi_to_this_person | from_this_person_to_poi | loan_advances | long_term_incentive | other      | poi   | restricted_stock | restricted_stock_deferred | salary    | shared_receipt_with_poi | to_messages | total_payments | total_stock_value |
| ------------------ | ---------- | ----------------- | --------------- | ------------- | ------------- | ----------------------- | --------- | ------------- | ----------------------- | ----------------------- | ------------- | ------------------- | ---------- | ----- | ---------------- | ------------------------- | --------- | ----------------------- | ----------- | -------------- | ----------------- |
| ALLEN PHILLIP K    | 4175000.00 | 2869717.00        | -3081055.00     | nan           | nan           | 1729541.00              | 13868.00  | 2195.00       | 47.00                   | 65.00                   | nan           | 304805.00           | 152.00     | False | 126027.00        | -126027.00                | 201955.00 | 1407.00                 | 2902.00     | 4484442.00     | 1729541.00        |
| BADUM JAMES P      | nan        | 178980.00         | nan             | nan           | nan           | 257817.00               | 3486.00   | nan           | nan                     | nan                     | nan           | nan                 | nan        | False | nan              | nan                       | nan       | nan                     | nan         | 182466.00      | 257817.00         |
| BANNANTINE JAMES M | nan        | nan               | -5104.00        | nan           | nan           | 4046157.00              | 56301.00  | 29.00         | 39.00                   | 0.00                    | nan           | nan                 | 864523.00  | False | 1757552.00       | -560222.00                | 477.00    | 465.00                  | 566.00      | 916197.00      | 5243487.00        |
| BAXTER JOHN C      | 1200000.00 | 1295738.00        | -1386055.00     | nan           | nan           | 6680544.00              | 11200.00  | nan           | nan                     | nan                     | nan           | 1586055.00          | 2660303.00 | False | 3942714.00       | nan                       | 267102.00 | nan                     | nan         | 5634343.00     | 10623258.00       |
| BAY FRANKLIN R     | 400000.00  | 260455.00         | -201641.00      | nan           | nan           | nan                     | 129142.00 | nan           | nan                     | nan                     | nan           | nan                 | 69.00      | False | 145796.00        | -82782.00                 | 239671.00 | nan                     | nan         | 827696.00      | 63014.00          |

# Outlier Removal

# Feature Selection
Feature Selection was handled in multiple steps to methodically reduce 20+ features to 3:

---

### Step 1: Missing Value Ratio

| Feature                        | % Missing Values |
| ------------------------------ | ---------------- |
| poi                            | 0.00%            |
| <b>totl_stock_value</b>        | <b>13.70%</b>    |
| <b>total_payments</b>          | <b>14.38%</b>    |
| <b>restricted_stock</b>        | <b>24.66%</b>    |
| <b>exercised_stock_options</b> | <b>30.14%</b>    |
| <b>salary</b>                  | <b>34.93%</b>    |
| <b>expenses</b>                | <b>34.93%</b>    |
| other                          | 36.30%           |
| <b>to_messages</b>             | <b>41.10%</b>    |
| <b>shared_receipt_with_poi</b> | <b>41.10%</b>    |
| <b>from_messages</b>           | <b>41.10%</b>    |
| <b>from_poi_to_this_person</b> | <b>41.10%</b>    |
| <b>from_this_person_to_poi</b> | <b>41.10%</b>    |
| <b>bonus</b>                   | <b>43.84%</b>    |
| long_term_incentive            | 54.79%           |
| deferred_income                | 66.44%           |
| deferral_payments              | 73.29%           |
| restricted_stock_deferred      | 87.67%           |
| director_fees                  | 88.36%           |
| loan_advances                  | 97.26%           |
| email_address                  | 100.00%          |

- Any features that were missing from more than 50% individuals were dropped
- Each individual had an Email Address however, each datapoint was coerced to a numeric value which resulted in a NaN value for Email Address strings. Also, in this instance, the email_address feature was also irrelevant for statistical modeling.
- Other was dropped because it was unclear what that feature encompasses

---

### Step 2: Low Variance
features with low statistical variability were candidates for removal

| Feature                        | Variance                 |
| ------------------------------ | ------------------------ |
| poi                            | 0.11                     |
| from_poi_to_this_person        | 7565.39                  |
| from_this_person_to_poi        | 10014.63                 |
| shared_receipt_with_poi        | 1388432.46               |
| from_messages                  | 3389406.00               |
| to_messages                    | 6670344.36               |
| <b>expenses</b>                | <b>2125982471.43    </b> |
| <b>salary</b>                  | <b>31375432034.87   </b> |
| <b>bonus</b>                   | <b>2078439602903.44 </b> |
| <b>restricted_stock</b>        | <b>5061466658800.40 </b> |
| <b>exercised_stock_options</b> | <b>30243945891878.36</b> |
| <b>total_stock_value</b>       | <b>42678561561688.52</b> |
| <b>total_payments</b>          | <b>90024146625871.03</b> |

---

### Step 3: High Feature Correlation

#### General Feature Correlation
 | <b>FEATURES</b>                | expenses | salary | bonus | restricted_stock | total_payments | exercised_stock_options | total_stock_value | poi  |
 | ------------------------------ | -------- | ------ | ----- | ---------------- | -------------- | ----------------------- | ----------------- | ---- |
 | <b>expenses</b>                | 1.00     | 0.15   | 0.03  | 0.04             | 0.02           | 0.03                    | 0.11              | 0.06 |
 | <b>salary</b>                  | 0.15     | 1.00   | 0.52  | 0.55             | 0.61           | 0.61                    | 0.58              | 0.26 |
 | <b>bonus</b>                   | 0.03     | 0.52   | 1.00  | 0.38             | 0.51           | 0.51                    | 0.57              | 0.30 |
 | <b>restricted_stock</b>        | 0.04     | 0.55   | 0.38  | 1.00             | 0.69           | 0.86                    | 0.60              | 0.22 |
 | <b>exercised_stock_options</b> | 0.02     | 0.61   | 0.51  | 0.69             | 1.00           | 0.96                    | 0.59              | 0.50 |
 | <b>total_stock_value</b>       | 0.03     | 0.61   | 0.51  | 0.86             | 0.96           | 1.00                    | 0.67              | 0.37 |
 | <b>total_payments</b>          | 0.11     | 0.58   | 0.57  | 0.60             | 0.59           | 0.67                    | 1.00              | 0.23 |
 | <b>poi</b>                     | 0.06     | 0.26   | 0.30  | 0.22             | 0.50           | 0.37                    | 0.23              | 1.00 |

#### Features With High Correlation to 'POI'
| Feature                        | Correlation |
| ------------------------------ | ----------- |
| poi                            | 1.00        |
| <b>exercised_stock_options</b> | <b>0.50</b> |
| total_stock_value              | 0.37        |
| bonus                          | 0.30        |
| salary                         | 0.26        |
| total_payments                 | 0.23        |
| restricted_stock               | 0.22        |
| expenses                       | 0.06        |


#### Features With High Correlation
 | <b>FEATURES</b>                | expenses | salary | bonus | restricted_stock | total_payments | exercised_stock_options | total_stock_value | poi  |
 | ------------------------------ | -------- | ------ | ----- | ---------------- | -------------- | ----------------------- | ----------------- | ---- |
 | <b>salary</b>                  | -        | -      | 0.52  | 0.55             | 0.61           | 0.61                    | 0.58              | -    |
 | <b>bonus</b>                   | -        | 0.52   | -     | -                | 0.51           | 0.51                    | 0.57              | -    |
 | <b>restricted_stock</b>        | -        | 0.55   | -     | -                | 0.69           | 0.86                    | 0.60              | -    |
 | <b>exercised_stock_options</b> | -        | 0.61   | 0.51  | 0.69             | -              | 0.96                    | 0.59              | 0.50 |
 | <b>total_stock_value</b>       | -        | 0.61   | 0.51  | 0.86             | 0.96           | -                       | 0.67              | -    |
 | <b>total_payments</b>          | -        | 0.58   | 0.57  | 0.60             | 0.59           | 0.67                    | -                 | -    |
 | <b>poi</b>                     | -        | -      | -     | -                | 0.50           | -                       | -                 | -    |

#### Features With Low Correlation
 | <b>FEATURES</b>                | expenses | salary | bonus | restricted_stock | total_payments | exercised_stock_options | total_stock_value | poi  |
 | ------------------------------ | -------- | ------ | ----- | ---------------- | -------------- | ----------------------- | ----------------- | ---- |
 | <b>expenses</b>                | -        | 0.15   | 0.03  | 0.04             | 0.02           | 0.03                    | 0.11              | 0.06 |
 | <b>salary</b>                  | 0.15     | -      | -     | -                | -              | -                       | -                 | 0.26 |
 | <b>bonus</b>                   | 0.03     | -      | -     | 0.38             | -              | -                       | -                 | 0.30 |
 | <b>restricted_stock</b>        | 0.04     | -      | 0.38  | -                | -              | -                       | -                 | 0.22 |
 | <b>exercised_stock_options</b> | 0.02     | -      | -     | -                | -              | -                       | -                 | -    |
 | <b>total_stock_value</b>       | 0.03     | -      | -     | -                | -              | -                       | -                 | 0.37 |
 | <b>total_payments</b>          | 0.11     | -      | -     | -                | -              | -                       | -                 | 0.23 |
 | <b>poi</b>                     | 0.06     | 0.26   | 0.30  | 0.22             | -              | 0.37                    | 0.23              | -    |


#### Features With High Mean Correlation To Other Features
| <b>FEATURES</b>                | # Highly Correlated Features | Mean Correlation |
| ------------------------------ | ---------------------------- | ---------------- |
| <b>exercised_stock_options</b> | <b>6</b>                     | <b>0.64</b>      |
| salary                         | 5                            | 0.58             |
| total_stock_value              | 5                            | 0.72             |
| total_payments                 | 5                            | 0.60             |
| bonus                          | 4                            | 0.53             |
| restricted_stock               | 4                            | 0.67             |
| poi                            | 1                            | 0.50             |
| expenses                       | 0                            | nan              |

Choosing the `exercised_stock_options` feature because it has the highest correlation to the `POI` feature, and it can serve as a stand in for the 5 other features it shares a high correlation with.

#### Features With Low Mean Correlation To Other Features
| <b>FEATURES</b>         | # Lowly Correlated Features | Mean Correlation |
| ----------------------- | --------------------------- | ---------------- |
| <b>expenses</b>         | <b>7.00</b>                 | <b>0.06</b>      |
| poi                     | 6.00                        | 0.24             |
| <b>bonus</b>            | <b>3.00</b>                 | <b>0.24</b>      |
| restricted_stock        | 3.00                        | 0.21             |
| salary                  | 2.00                        | 0.21             |
| total_stock_value       | 2.00                        | 0.20             |
| total_payments          | 2.00                        | 0.17             |
| exercised_stock_options | 1.00                        | 0.02             |

Since the `exercised_stock_options` is highly correlated with other features, we'll choose 2 additional features with low Mean Correlation values so that we better represent the larger feature set. Of the remaining features, `expenses` feature has the lowest mean correlation between features, so that seems like a good choice for a highly independent variable.  `bonus` has a moderately high correlation with `POI` so that feels like a good choice to supplement the other feature that's highly-correlated with POI.

#### Recursive Feature Reduction (RFE) with Logistic Regression Classifier
| Feature                        | Feature Importance Rank (lower is better) |
| ------------------------------ | ----------------------------------------- |
| total_stock_value              | 1                                         |
| restricted_stock               | 1                                         |
| to_messages                    | 1                                         |
| shared_receipt_with_poi        | 1                                         |
| from_this_person_to_poi        | 1                                         |
| <b>exercised_stock_options</b> | 2                                         |
| from_poi_to_this_person        | 3                                         |
| poi                            | 4                                         |
| from_messages                  | 5                                         |
| <b>expenses</b>                | 6                                         |
| salary                         | 7                                         |
| <b>bonus</b>                   | 8                                         |
| total_payments                 | 9                                         |

#### Recursive Feature Reduction (RFE) with Random Forest Classifier
| Feature                        | Feature Importance Rank (lower is better) |
| ------------------------------ | ----------------------------------------- |
| poi                            | 1                                         |
| total_payments                 | 1                                         |
| restricted_stock               | 1                                         |
| <b>expenses</b>                | 1                                         |
| shared_receipt_with_poi        | 1                                         |
| <b>bonus</b>                   | 2                                         |
| <b>exercised_stock_options</b> | 3                                         |
| from_messages                  | 4                                         |
| from_this_person_to_poi        | 5                                         |
| to_messages                    | 6                                         |
| from_poi_to_this_person        | 7                                         |
| total_stock_value              | 8                                         |
| salary                         | 9                                         |

### Selected Features
1. exercised_stock_options
1. bonus
1. expenses

# Feature Engineering

# Dimensionality Reduction / Feature Scaling
- Principle Component Analysis
- StandardScaler
- MinMaxScaler

# Model Parameter Tuning
- GridCV

# Model Validation
- cross_validate
- StratifiedShuffleSplit

# Model Evaluation Metrics
- confusion_matrix
- precision
- recall
- accuracy
- F1 Score
-

### Confusion Matrix
| n = x      | Predicted POI | Actual POI |
| ---------- | ------------- | ---------- |
| Actual POI |               |            |
| Actual POI |               |            |

### Classification Report


# Classifier Validation Results
Below are individual validation results for various classifier types and parameters that I experimented with.

Each permutation used a Scikit Learn Pipeline with PCA as well as Standard or MinMax feature scaling.

Classifier parameters were tuned using GridSearchCV cross validation.

Classifiers were fit to a randomized training set generated by StratifiedShuffleSplit and were validated against a testing set generated during the same StratifiedShuffleSplit.

Calculations were aggregrated across each of the 10 StratifiedShuffleSplit splits.

Note: the results below were generated using the `validate_classifier()` function and will differ slightly from results from the `test_classifier()` function, due to the different `random_state` values supplied to the `StratifiedShuffleSplit()` function.

### Gaussian Naive Bayes
##### Validating GaussianNB (PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 85.92%

GaussianNB (PCA) Aggregate Model Classification Performance
===========================================================
Population:     140
Accuracy:       0.87
Precision:      0.6
Recall:         0.3
F1 Score:       0.4

GaussianNB (PCA) Aggregate Confusion Matrix
===========================================
[[116   4]
 [ 14   6]]
GaussianNB (StandardScaler + PCA)
```

##### Validating GaussianNB (StandardScaler + PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 85.76%

GaussianNB (StandardScaler + PCA) Aggregate Model Classification Performance
============================================================================
Population:     140
Accuracy:       0.87
Precision:      0.63
Recall:         0.25
F1 Score:       0.36

GaussianNB (StandardScaler + PCA) Aggregate Confusion Matrix
============================================================
[[117   3]
 [ 15   5]]
```

##### Validating GaussianNB (MinMaxScaler + PCA): random_state = 0

```
Mean 10-Fold Cross Validation Test Score: 84.93%

GaussianNB (MinMaxScaler + PCA) Aggregate Model Classification Performance
==========================================================================
Population:     140
Accuracy:       0.86
Precision:      0.67
Recall:         0.1
F1 Score:       0.17

GaussianNB (MinMaxScaler + PCA) Aggregate Confusion Matrix
==========================================================
[[119   1]
 [ 18   2]]
DecisionTreeClassifier (PCA)
```

### Decision Tree
##### Validating DecisionTreeClassifier (PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 82.02%

DecisionTreeClassifier (PCA) Aggregate Model Classification Performance
=======================================================================
Population:     140
Accuracy:       0.85
Precision:      0.46
Recall:         0.3
F1 Score:       0.36

DecisionTreeClassifier (PCA) Aggregate Confusion Matrix
=======================================================
[[113   7]
 [ 14   6]]

```

##### Validating DecisionTreeClassifier (StandardScaler + PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 84.21%

DecisionTreeClassifier (StandardScaler + PCA) Aggregate Model Classification Performance
========================================================================================
Population:     140
Accuracy:       0.87
Precision:      0.6
Recall:         0.3
F1 Score:       0.4

DecisionTreeClassifier (StandardScaler + PCA) Aggregate Confusion Matrix
========================================================================
[[116   4]
 [ 14   6]]
```

##### Validating DecisionTreeClassifier (MinMaxScaler + PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 82.02%

DecisionTreeClassifier (MinMaxScaler + PCA) Aggregate Model Classification Performance
======================================================================================
Population:     140
Accuracy:       0.85
Precision:      0.46
Recall:         0.3
F1 Score:       0.36

DecisionTreeClassifier (MinMaxScaler + PCA) Aggregate Confusion Matrix
======================================================================
[[113   7]
 [ 14   6]]
```

### AdaBoost
##### Validating AdaBoost (PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 86.36%

AdaBoost (PCA) Aggregate Model Classification Performance
=========================================================
Population:     140
Accuracy:       0.86
Precision:      0.55
Recall:         0.3
F1 Score:       0.39

AdaBoost (PCA) Aggregate Confusion Matrix
=========================================
[[115   5]
 [ 14   6]]
```

##### Validating AdaBoost (StandardScaler + PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 85.51%

AdaBoost (StandardScaler + PCA) Aggregate Model Classification Performance
==========================================================================
Population:     140
Accuracy:       0.86
Precision:      0.5
Recall:         0.15
F1 Score:       0.23

AdaBoost (StandardScaler + PCA) Aggregate Confusion Matrix
==========================================================
[[117   3]
 [ 17   3]]
```

##### Validating AdaBoost (MinMaxScaler + PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 84.27%

AdaBoost (MinMaxScaler + PCA) Aggregate Model Classification Performance
========================================================================
Population:     140
Accuracy:       0.84
Precision:      0.38
Recall:         0.25
F1 Score:       0.3

AdaBoost (MinMaxScaler + PCA) Aggregate Confusion Matrix
========================================================
[[112   8]
 [ 15   5]]
```

### KNN
##### Validating KNeighbors (PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 84.93%

KNeighbors (PCA) Aggregate Model Classification Performance
===========================================================
Population:     140
Accuracy:       0.87
Precision:      0.67
Recall:         0.2
F1 Score:       0.31

KNeighbors (PCA) Aggregate Confusion Matrix
===========================================
[[118   2]
 [ 16   4]]
```


##### Validating KNeighbors (StandardScaler + PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 82.07%

KNeighbors (StandardScaler + PCA) Aggregate Model Classification Performance
============================================================================
Population:     140
Accuracy:       0.86
Precision:      0.5
Recall:         0.05
F1 Score:       0.09

KNeighbors (StandardScaler + PCA) Aggregate Confusion Matrix
============================================================
[[119   1]
 [ 19   1]]
```

##### Validating KNeighbors (MinMaxScaler + PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 82.78%

KNeighbors (MinMaxScaler + PCA) Aggregate Model Classification Performance
==========================================================================
Population:     140
Accuracy:       0.85
Precision:      0.4
Recall:         0.1
F1 Score:       0.16

KNeighbors (MinMaxScaler + PCA) Aggregate Confusion Matrix
==========================================================
[[117   3]
 [ 18   2]]
```


### Random Forest
#####  Validating RandomForest (PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 81.13%

RandomForest (PCA) Aggregate Model Classification Performance
=============================================================
Population:     140
Accuracy:       0.87
Precision:      0.67
Recall:         0.2
F1 Score:       0.31

RandomForest (PCA) Aggregate Confusion Matrix
=============================================
[[118   2]
 [ 16   4]]
```

#####  Validating RandomForest (StandardScaler + PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 82.44%

RandomForest (StandardScaler + PCA) Aggregate Model Classification Performance
==============================================================================
Population:     140
Accuracy:       0.89
Precision:      0.75
Recall:         0.3
F1 Score:       0.43

RandomForest (StandardScaler + PCA) Aggregate Confusion Matrix
==============================================================
[[118   2]
 [ 14   6]]
```

#### Validating RandomForest (MinMaxScaler + PCA): random_state = 0
```
Mean 10-Fold Cross Validation Test Score: 83.32%

RandomForest (MinMaxScaler + PCA) Aggregate Model Classification Performance
============================================================================
Population:     140
Accuracy:       0.86
Precision:      0.56
Recall:         0.25
F1 Score:       0.34

RandomForest (MinMaxScaler + PCA) Aggregate Confusion Matrix
============================================================
[[116   4]
 [ 15   5]]
```