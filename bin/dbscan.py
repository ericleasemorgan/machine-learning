#!/usr/bin/env python

# require
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import sys

# sanity check
if len( sys.argv ) != 2 :
	sys.stderr.write( 'Usage: ' + sys.argv[ 0 ] + " <file>\n" )
	exit()

# get input
file = sys.argv[ 1 ]

# create a dataframe, and extract the values
df = pd.read_csv( file, sep='\t' )
X  = df.drop( ['file'], axis=1 ).values

# initialize a scalar and scale
scalar = StandardScaler()
X      = scalar.fit_transform( X )

# initialize the scanner and scan
dbscan   = DBSCAN()
clusters = dbscan.fit_predict( X )

# plot, output, and done
plt.scatter( X[:,0], X[:,1], c=clusters )
plt.show()
exit()
