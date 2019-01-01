#!/usr/bin/env python


# require
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import os
import pandas as pd
import sys

# sanity check
if len( sys.argv ) != 4 :
	sys.stderr.write( 'Usage: ' + sys.argv[ 0 ] + " <directory> <integer> <integer>\n" )
	exit()

# get input
directory  = sys.argv[ 1 ]
topics     = int( sys.argv[ 2 ] )
dimensions = int( sys.argv[ 3 ] )

# initialize
data = []
keys = []

# process each file in the given directory
for file in os.listdir( directory ) :
	
	# only process .txt files
	if file.endswith( '.txt' ) : 
		keys.append( file.split( '.' )[0] )
		data.append( open( directory + '/' + file ).read() )

# initialize and instantiate a vectorizer
vectorizer = CountVectorizer( stop_words='english', min_df=4 )
X          = vectorizer.fit_transform( data )

# initialize and instantiate a model
model = KMeans( n_clusters=topics )
model.fit( X )

# get the list of features and centers
features = vectorizer.get_feature_names()
centers  = model.cluster_centers_.argsort()[ :, : :-1 ] 

# from the centers & features, create a list of topic phrases (labels)
labels = []
for i in range( topics ) :
	
	# re-initialize and build a topic from the index
	topic = str( i )
	for j in centers[ i, :dimensions ] : topic = topic + ' ' + features[ j ]
	print( 'Topic #' + str( i ) + ': ' + topic )
	labels.append( topic )

# get the scores and combine them with the labels into a data frame
scores = model.transform( X )
df     = pd.DataFrame( scores, columns=labels )

# update the data frame with the keys and clusters
df[ 'keys' ]     = keys
df[ 'clusters' ] = model.labels_.tolist()
df.set_index( 'keys', inplace=True )
df.sort_index( inplace=True )

# done, so far
#print( df )
print( df['clusters'].value_counts() )
exit()

