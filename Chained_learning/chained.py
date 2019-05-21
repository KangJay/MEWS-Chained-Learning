from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#pip install -q -U tensorflow==1.7.0

import itertools
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Import the slearn utility to compare algorithms
from sklearn import model_selection

# Import all the algorithms we want to test
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

import seaborn as sns

def read_csv():
    #pima-indians-diabetes.csv
    DATA_PATH = "pima-indians-diabetes.csv"
    dataset = pd.read_csv(DATA_PATH, header=None)
    dataset.columns = [
        "NumTimesPrg", "PlGlcConc", "BloodP",
        "SkinThick", "TwoHourSerIns", "BMI",
        "DiPedFunc", "Age", "HasDiabetes"]
    return dataset


def show_plot(): plt.show()

def get_median_data(dataset):
    #Don't transform 0 values for things that make sense
    # number of times pregnant = 0 makes sense. 0 bmi does not.

    #Getting the median bmi from the dataet
    median_bmi = dataset['BMI'].median()
    #replace 0 values inthe dataset with the median value
    dataset['BMI'] = dataset['BMI'].replace(to_replace=0, value=median_bmi)

    # Calculate the median value for BloodP
    median_bloodp = dataset['BloodP'].median()
    # Substitute it in the BloodP column of the
    # dataset where values are 0
    dataset['BloodP'] = dataset['BloodP'].replace(to_replace=0, value=median_bloodp)

    # Calculate the median value for PlGlcConc
    median_plglcconc = dataset['PlGlcConc'].median()
    # Substitute it in the PlGlcConc column of the
    # dataset where values are 0
    dataset['PlGlcConc'] = dataset['PlGlcConc'].replace(
        to_replace=0, value=median_plglcconc)

    # Calculate the median value for SkinThick
    median_skinthick = dataset['SkinThick'].median()
    # Substitute it in the SkinThick column of the
    # dataset where values are 0
    dataset['SkinThick'] = dataset['SkinThick'].replace(
        to_replace=0, value=median_skinthick)

    # Calculate the median value for TwoHourSerIns
    median_twohourserins = dataset['TwoHourSerIns'].median()
    # Substitute it in the TwoHourSerIns column of the
    # dataset where values are 0
    dataset['TwoHourSerIns'] = dataset['TwoHourSerIns'].replace(
        to_replace=0, value=median_twohourserins)

    return dataset

def train_model(dataset):
    #Splitting the dataset into testing and training.
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

    #Separate the labels from the rest of the dataset. We're testing for diabetes
    train_set_labels = train_set["HasDiabetes"].copy()
    train_set = train_set.drop("HasDiabetes", axis=1)

    test_set_label = test_set["HasDiabetes"].copy()
    test_set = test_set.drop("HasDiabetes", axis=1)
    return train_set_labels, train_set, test_set_label, test_set

def scale_and_train(train_set_labels, train_set, test_set_label, test_set):
    '''
    One of the most important data transformations we need to apply is the features
    scaling. Basically most of the machine learning algorithms don't work very well
    if the features have a different set of values. In our case for example the Age
    ranges from 20 to 80 years old, while the number of times a patient has been
    pregnant ranges from 0 to 17. For this reason we need to apply a proper
    transformation.
    '''

    scaler = Scaler()
    scaler.fit(train_set)
    train_set_scaled = scaler.transform(train_set)
    test_set_scaled = scaler.transform(test_set)

    df = pd.DataFrame(data=train_set_scaled)

    # Prepare an array with all the algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVC', SVC()))
    models.append(('LSVC', LinearSVC()))
    models.append(('RFC', RandomForestClassifier()))
    models.append(('DTR', DecisionTreeRegressor()))

    seed = 7
    results, names = [], []
    X = train_set_scaled
    Y = train_set_labels

    for name, model in models:
        kfold = model_selection.KFold(
            n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(
            model, X, Y, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (
            name, cv_results.mean(), cv_results.std())
        print(msg)

    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    #plt.show()


def main():
    dataset = read_csv()
    #Get how storngly correlated the data is between different characteristics \
    #-1 to 1. 1 = Strong, -1 = Weak
    corr = dataset.corr()
    sns.heatmap(corr, annot=True)
    dataset.hist(bins=50, figsize=(20, 15))
    dataset = get_median_data(dataset)
    train_set_labels, train_set, test_set_label, test_set = train_model(dataset)
    scale_and_train(train_set_labels, train_set, test_set_label, test_set)


if __name__ == "__main__":
    main()
