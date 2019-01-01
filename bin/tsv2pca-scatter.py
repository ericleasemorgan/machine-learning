#!/usr/bin/env python

# tsv2pca-scatter.py - given a TSV file of a particular shape, plot the two principle components


# configure
COMPONENTS = 2

# require
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
import sys

# sanity check
if len( sys.argv ) != 2 :
	sys.stderr.write( 'Usage: ' + sys.argv[ 0 ] + " <file>\n" )
	exit()

# get input
file       = sys.argv[ 1 ]

# create a data frame from the given file
df = pd.read_csv( file, sep='\t' )

# based on the file name, create an author column (labels)
df[ 'author' ] = df[ 'file' ].str.extract( '^.*/(.*?)-', expand=False )
df[ 'book' ]   = df[ 'file' ].str.extract('^.*/\w+-(.*?)-', expand=False)

# extract the feature set
X  = df.drop( [ 'file', 'author', 'book' ], axis=1 )

# instantiate a model and then model them in to components
model      = PCA( n_components=COMPONENTS )
components = model.fit_transform( X )

# insert the results back into the data frame, plot, and done
df['component #1'] = components[ :, 0 ]
df['component #2'] = components[ :, 1 ]
sns.lmplot( "component #1", "component #2", data=df, hue='author', fit_reg=False )
plt.show()
exit()