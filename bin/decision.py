#!/usr/bin/env python

# decision.py - given a matrix of scores and a matrix of labels, create a model and classify a vector


# require
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from subprocess import call
import pandas as pd

# initialize the scores and labels
scores = pd.read_csv( './scores.csv', index_col='id' )
labels = pd.read_csv( './labels.csv', index_col='id' )

# randomly create a set of training and testing data
scores_train, scores_test, labels_train, labels_test = train_test_split( scores, labels, random_state = 1 )

# model the data
model = tree.DecisionTreeClassifier()
model.fit( scores_train, labels_train )

# calculate and output accuracy, approximately 75%
labels_predict = model.predict( scores_test )
print accuracy_score( labels_test, labels_predict )

# here, given a paragraph of text, return a harms/principles/actors/rights vector

# given a vector, predict a label
print model.predict( [ [ 12,34,8,26 ] ] )