#!/usr/bin/env python

# tsv2gaussian.py - given a TSV file of a particular shape and an integer, plot Gaussian clusters after doing PCA

# configure
PCA_COMPONENTS = 3

# require
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
import sys

# sanity check
if len( sys.argv ) != 3 :
	sys.stderr.write( 'Usage: ' + sys.argv[ 0 ] + " <TSV> <integer>\n" )
	quit()

# get input
file           = sys.argv[ 1 ]
gmm_components = int( sys.argv[ 2 ] )

# create a data frame from the given file
df = pd.read_csv( file, sep='\t' )

# based on the file name, create an author column (labels)
df[ 'author' ] = df[ 'file' ].str.extract( '^.*/(.*?)-', expand=False )
df[ 'book' ]   = df[ 'file' ].str.extract('^.*/\w+-(.*?)-', expand=False)

# create a feature set; "just give me the data"
X = df.drop( [ 'file', 'author', 'book' ], axis=1 )

# instantiate PCA and identify the components
model      = PCA( n_components=PCA_COMPONENTS )
components = model.fit_transform( X )

# for future reference, insert the components back into the data frame
df['component #1'] = components[ :, 0 ]
df['component #2'] = components[ :, 1 ]

# instantiate a Gaussian model, model, and generate a set of clusters
model = GaussianMixture ( n_components=gmm_components, covariance_type='full' )
model.fit( X )
clusters = model.predict( X )

# insert the clusters into the data frame, plot, and done
df['cluster'] = clusters
sns.lmplot( "component #1", "component #2", data=df, col='cluster', hue='author', fit_reg=False )
plt.show()
exit()
