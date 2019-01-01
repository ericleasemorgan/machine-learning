#!/usr/bin/env python


# configure
FILE = './ideas.tsv'

# require
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd

# create a data frame from the given file
df = pd.read_csv( FILE, sep='\t' )

# based on the file name, create an author column (labels)
df[ 'author' ] = df[ 'file' ].str.extract( '^.*/(.*?)-', expand=False )

# extract a feature matrix (X) and the list of associated labels (Y)
X = df.drop( [ 'file', 'author' ], axis=1 )
y = df[ 'author' ]

# create sets for training and testing
Xtrain, Xtest, ytrain, ytest = train_test_split( X, y, random_state=1 )

# initialize a (simple) model, and do the work
model = GaussianNB()
model.fit( Xtrain, ytrain )

# create a set of predictions, output accuracy, and done
predictions = model.predict( Xtest )
print( accuracy_score( ytest, predictions ) )
exit()

