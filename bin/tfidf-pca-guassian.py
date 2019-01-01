#!/usr/bin/env python

# configure
AUTHOR    = 0
BOOK      = 1

# require
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import seaborn as sns; sns.set()
import sys

# sanity check
if len( sys.argv ) != 2 :
	sys.stderr.write( 'Usage: ' + sys.argv[ 0 ] + " <directory>\n" )
	quit()

# get input
directory = sys.argv[ 1 ]

# given a directory, create a corpus and set of labels
corpus = []
labels = []
for file in os.listdir( directory ) :
	
	# we only want plain text files
	if file.endswith( '.txt' ) :
		
		# update the corpus
		corpus.append( open( directory + '/' + file ).read() )
		
		# based on BOOK or AUTHOR, update the list of labels; tricky
		labels.append( re.split( '-', file )[ AUTHOR ] )

# initialize and fill a TFIDF model (feature extraction), and then convert it to a data frame (X)
vectorizer = TfidfVectorizer( stop_words='english' )
corpus     = vectorizer.fit_transform( corpus )
X          = pd.DataFrame( corpus.toarray(), columns=vectorizer.get_feature_names() )

# create and fill a PCA model (dimension reduction); update the data frame with the results
pca        = PCA( n_components=2 )
components = pca.fit_transform( X )

# create and fill a Gaussian model (cluster); update the data frame with the results
gaussian = GaussianMixture( n_components=3, covariance_type='full' )
gaussian.fit( X )
clusters = gaussian.predict( X )

# update the data frame with the additional observations
X[ 'label' ]   = labels
X[ 'PCA #1' ]  = components[ :, 0 ]
X[ 'PCA #2' ]  = components[ :, 1 ]
X[ 'cluster' ] = clusters

# plot the results; are their visual clusters?
sns.lmplot( 'PCA #1', 'PCA #2', data=X, col='cluster', hue='label', fit_reg=False )
plt.show()
exit()
