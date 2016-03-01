import json
from glob import glob
from pprint import *
import csv
import operator
import re
from math import sqrt
from math import log
import operator
from itertools import islice
from operator import sub
import random
import sys
import numpy as np
from scipy import spatial
import bz2,json,contextlib
import numpy as np
from random import sample
from nltk import PorterStemmer
# 211 plain



if len(sys.argv)!=5:
    print "Use:"
    print "hash2GloVe.py [log|wei|plain] vector_leng window_size file"
    print "example: hash2Glove.py wei 211 5 text8/text8.txt"
    quit()

arg_model = sys.argv[1]
arg_vec_len = sys.argv[2]
arg_k = sys.argv[3]
arg_file = sys.argv[4]


# Hash2Vec Experimentation (next try with a log transformation)
print "Hash2Vec v1.0"

# Context window length
k = int(arg_k)
# Text file
text_filename=arg_file

# Vector lengths
vec_len = int(arg_vec_len)

# Create a vector for the previously seen k words
context = [""]*k
# Count of total words
total_words = 0
# The word embeddings vector
word_vectors = {}


# Dot product
def dot(a,b):
	return np.dot(a,b)

# Normalize a vector
def norm(V):
    L = np.linalg.norm(V)
    if L>0: return V/L
    return V

# Cosine distance
def distance(a,b):
	return scipy.spatial.distance.cosine(a,b) # ya incluye el 1-cos(ab)


for line in file(text_filename):
    words = line.split()
    word_position = 0
    for w in words:
        word_position+=1
		total_words+=1 
		if w not in word_vectors:
			word_vectors[w] = {}
			word_vectors[w]['frq']=0
			word_vectors[w]['vec'] = [float(0)] * vec_len
			word_vectors[w]['h']= hash(PorterStemmer().stem_word(w))%vec_len

		word_vectors[w]['frq']+=1
		if total_words % 1000000 == 0:
			print "Processed:"+str(total_words)+" words"
		
		# Do the context operations
		for j in range(1,min([word_position-1,k])+1):
			context_word = context[k-j]
			#weight = float(1)/j
			weight = float(k-j+1)/k
			# For w hash context word and add it
			position = hash(context_word)%vec_len
			
			#don't calculate it unless it's the model
			sign = 1
			if arg_model == "wei":
				h2 = hash(context_word[::-1]) %2
				if h2 == 1:
					sign = -1
			
			word_vectors[w]['vec'][position]+=(sign*weight)
			
			# For context word hash w and add it
			position = hash(w)%vec_len
			sign = 1
			if arg_model == "wei":
				h2 = h2 = hash(w[::-1]) %2
				if h2 == 1:
					sign = -1
			word_vectors[context_word]['vec'][position]+=(sign*weight)

            # Shift the context
            context.pop(0)
            context.append(w)

print "Processed a total of: "+str(total_words)+" words in text found: "+str(len(word_vectors))+" different words"

if arg_model=='log':
    print "Log normalizing..."
    for w in word_vectors:
        for i in range(len(word_vectors[w]['vec'])):
            if word_vectors[w]['vec'][i]>0:
                word_vectors[w]['vec'][i]=log(word_vectors[w]['vec'][i])
    print "Done!"

# Store the word embeddings
modelname = text_filename+"_stem_"+str(k)+"_"+str(vec_len)+"_"+str(arg_model)+".json.bz2"
print "Storing model in "+modelname
with contextlib.closing(bz2.BZ2File(modelname, 'wb')) as f:
  json.dump(word_vectors, f)

print "Done!"