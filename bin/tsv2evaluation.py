#!/usr/bin/env python

# evaluate-lexicon.py - given a TSV file of a particular shape, output ordered lists of significant items


# require
import pandas as pd
import sys

# sanity check
if len( sys.argv ) != 2 :
	sys.stderr.write( 'Usage: ' + sys.argv[ 0 ] + " <TSV>\n" )
	quit()

# get input
file = sys.argv[ 1 ]

# import the given file; create a dataframe
df = pd.read_csv( file, sep='\t' )

# sum each column to create our overall coefficient
df['coefficient'] = df.sum( axis=1 )

# parse the filename to create author and book columns
df['author'] = df['file'].str.extract( '^.*/(.*?)-', expand=False )
df['book']   = df['file'].str.extract('^.*/\w+-(.*?)-', expand=False)

# output the coefficient for each file (chapter)
print( "Most significant files:" )
print( df[ [ 'coefficient', 'file' ] ].sort_values( by=[ 'coefficient' ], ascending=False ) )
print()

# output the coefficient for each book
print( "Most significant books:" )
print( df.groupby( 'book' )[ 'file', 'coefficient' ].sum().sort_values( by='coefficient', ascending=False ) )
print()

# output the coefficient for each book; consider changing 'book' to 'author'
print( "Most significant authors:" )
print( df.groupby( 'author' )[ 'file', 'coefficient' ].sum().sort_values( by='coefficient', ascending=False ) )
print()

# done
exit()