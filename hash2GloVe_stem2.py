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
from nltk.stem.lancaster import LancasterStemmer
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


# Hash2Vec Experimentation (next try with a log tranformation)
print "Hash2Vec v1.0"

# Context window length
k = int(arg_k)
# Text file
#text_filename="text8/text8.txt"
#text_filename="billion/train_v2.txt"
text_filename=arg_file

# Vector lengths
vec_len = int(arg_vec_len)

words_to_find = ['computer','king','queen','physics','north','italy','wounded','car','church','wednesday','two','man','woman','anglican']

# Create a vector for the previously seen k words
context = [""]*k
# Count of total words
total_words = 0
# The word embeddings vector
word_vectors = {}
# A very short list of stop words
stoplist = set('for a of the and to in s was is that'.split())

lancaster = LancasterStemmer()

# Dot product
def dot(a,b):
    return sum( [a[i]*b[i] for i in range(len(b))] )

# Normalize a vector
def norm(V):
    L = sqrt( sum( [x**2 for x in V] ) )
    if L>0:
        return [ x/L for x in V ]
    else:
        return V

# Cosine distance
def distance(a,b):
    return  1-dot(norm(a),norm(b))

negative_samples=[]

previous_word = ''
for line in file(text_filename):
    #words = re.findall(r"[\w']+", line.lower())
    words = line.split()
    word_position = 0
    for w in words:
        
        word_position+=1
        #print w
        #print "["+str(word_position)+"]= "+w
        if w not in stoplist:
            total_words+=1
            #if total_words>20:
            #    quit()
            if w not in word_vectors:
                word_vectors[w] = {}
                word_vectors[w]['frq']=0
                word_vectors[w]['vec'] = [float(0)] * vec_len
                word_vectors[w]['h']= hash(lancaster.stem(w))%vec_len

            word_vectors[w]['frq']+=1
            if total_words % 1000000 == 0:
                print "Processed:"+str(total_words)+" words"
            # Do the context operations
            # ADD NEGATIVE SAMPLING
            #if total_words>5:
             #  negative_samples = random.sample(word_vectors.keys(), 5)
            for j in range(1,min([word_position-1,k])+1):
                context_word = context[k-j]
                #print "context contains: " + context[1] +" "+ context[2] +" "+ context[3] +" "+ context[4] 

                #weight = float(1)/j
                weight = float(k-j+1)/k
                # For w hash context word and add it
                position = hash(lancaster.stem(context_word))%vec_len
                #if w=='electrodynamics':
                    #print "word:"+w+" checking...."+context_word+" weight:"+str(weight)+" in position:"+str(position)
                h2 = hash(context_word[::-1])
                if h2%2 == 0:
                    sign = +1
                else:
                    sign = -1
                if arg_model!="wei":
                        sign=1
                word_vectors[w]['vec'][position]+=(sign*weight)

                

                # NEGATIVE SAMPLING
                #for wn in negative_samples:
                 #   word_vectors[wn]['vec'][position]-=(sign*weight)

                # For context word hash w and add it
                if context_word in word_vectors:
                    position = hash(lancaster.stem(w))%vec_len
                    #if context_word=='electrodynamics':
                      #  print "word:"+context_word+" checking...."+w+" weight:"+str(weight)+" in position:"+str(position)
                    h2 = hash(w[::-1])
                    if h2%2 == 0:
                        sign = +1
                    else:
                        sign = -1
                    if arg_model!="wei":
                        sign=1
                    word_vectors[context_word]['vec'][position]+=(sign*weight)


            # Shift the context
            context.pop(0)
            
            context.append(w)

#sorted_x = sorted(word_vectors.items(), key=lambda x: x[1], reverse=True)
#pprint(sorted_x,width=200)
#print "-------------------------------"
# Sort by word freq


print "Processed a total of: "+str(total_words)+" words in text found: "+str(len(word_vectors))+" different words"

if arg_model=='log':
    print "Log normalizing..."
    for w in word_vectors:
        for i in range(len(word_vectors[w]['vec'])):
            if word_vectors[w]['vec'][i]>0:
                word_vectors[w]['vec'][i]=log(word_vectors[w]['vec'][i])
    print "Done!"

# SMOOTHING
#print "Smoothing!"
#for w in word_vectors:
#    for i in range(len(word_vectors[w]['vec'])):
#        s=cmp(word_vectors[w]['vec'][i],0)
#        word_vectors[w]['vec'][i]=(abs(word_vectors[w]['vec'][i])**0.7)*s

# Store the word embeddings
modelname = text_filename+"_stem2_"+str(k)+"_"+str(vec_len)+"_"+str(arg_model)+".json.bz2"
print "Storing model in "+modelname
with contextlib.closing(bz2.BZ2File(modelname, 'wb')) as f:
  json.dump(word_vectors, f)
#pickle.dump( word_vectors, open( "word_vectors.p", "wb" ) )
print "Done!"
