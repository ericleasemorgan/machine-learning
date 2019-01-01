#!/usr/bin/env python

# tsv2kmeans.py - given a TSV file of a particular shape, use KMeans to plot possible clusters

# require
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import sys

# sanity check
if len( sys.argv ) != 3 :
	sys.stderr.write( 'Usage: ' + sys.argv[ 0 ] + " <file> <integer>\n" )
	exit()

# get input
file     = sys.argv[ 1 ]
clusters = int( sys.argv[ 2 ] )

# create a dataframe, and extract the values
df = pd.read_csv( file, sep='\t' )
X  = df.drop(['file'], axis=1).values

# initialize a scalar and scale
scalar = StandardScaler()
X      = scalar.fit_transform( X )

# initialize K-means analysis, and do the work
kmeans = KMeans( n_clusters=clusters )  
kmeans.fit( X ) 

# plot the result
plt.scatter( X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow' )  
plt.show()
