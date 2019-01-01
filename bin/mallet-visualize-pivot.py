#!/usr/bin/env python

# mallet-visualize-pivot.py - given a (MALLET) CSV file and quite a few parameters, visualize topics based on a pivot table

# require
import pandas as pd
import matplotlib.pyplot as plt
import sys

# sanity check
if len( sys.argv ) < 5 :
	sys.stderr.write( 'Usage: ' + sys.argv[ 0 ] + " <CSV> <bar|barh|line> <index> <topic> <another topic> [<another topic> ...]\n" )
	quit()

# get input
file  = sys.argv[ 1 ]
type  = sys.argv[ 2 ]
index = sys.argv[ 3 ]

# get the directories to process
topics = []
for i in range( 4, len( sys.argv ) ) : topics.append( sys.argv[ i ] )

df = pd.read_csv( file, index_col='filename')
df = df.drop(['docId', 'id'], axis=1)
table = df.pivot_table( topics, index=index )
table.plot( kind=type )
plt.show()
exit()