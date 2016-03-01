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
import numpy as np
from scipy import spatial
import pickle
import bz2,json,contextlib
import sys
import scipy.stats as sci

if len(sys.argv)!=5:
    print "Use:"
    print "hash2GloVe_test.py [log|wei|plain] vector_len window_size file"
    print "example: hash2Glove_test.py wei 211 5 text8/text8.txt"
    quit()

arg_model = sys.argv[1]
arg_vec_len = sys.argv[2]
arg_k = sys.argv[3]
arg_file = sys.argv[4]

modelname =arg_file+"_"+str(arg_k)+"_"+str(arg_vec_len)+"_"+str(arg_model)+".json.bz2"
print "Model: "+modelname

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
	#return sum(pow(a[i]-b[i],2) for i in range(len(b))) #euclidean norm
	#pearson correlation in negative so lower is better
	#return 1- dot(norm(a),norm(b))
	#tanimoto distance
	#return 1 - dot(a,b)/(dot(a,a) + dot(b,b) - dot(a,b))
	#return sci.kendalltau(a,b) #kendall tau



# Load the benchmark

benchmarks = ['benchmarks/wordsim353.csv','benchmarks/Mtruk.csv']

print "Loading Model..."
with contextlib.closing(bz2.BZ2File(modelname, 'rb')) as f:
    word_vectors = json.load(f)
print "Done!"



for bench in benchmarks:
    test_words = []
    with open(bench, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            test_words.append([row[0],row[1],float(row[2])])
    list_score1 = []
    list_score2 = []
    for test in test_words:
        word1 = test[0].lower()
        word2 = test[1].lower()
        if word1 in word_vectors and word2 in word_vectors:
            list_score1.append(test[2])
            vec1 = word_vectors[word1]['vec']
            vec2 = word_vectors[word2]['vec']
            score2 = 1-distance(vec1,vec2)
            list_score2.append(score2)
            #print "Word1:"+word1+" Word2:"+word2+" Human Score:"+str(test[2])+" Hash2Vec:"+str(score2)
    rho = sci.stats.spearmanr(list_score1, list_score2)
    print bench+" Rho:"+str(rho[0])




##### FIRST TEST: FIND TOP 10 SIMILAR WORDS TO THE ONES LISTED
res = {}
words_to_find = ['computer','king','queen','physics','north','italy','wounded','car','church','wednesday','man','woman','anglican']
for sw in words_to_find:
    vec = word_vectors[sw]['vec']
    print sw
    res.clear()
    for w in word_vectors:
        #dist = distance(map(sub,dict['man'],dict[w]),vec)
        dist = distance(word_vectors[w]['vec'],vec)
        if dist<0.3:
            res[w]=dist
    sorted_x = sorted(res.items(), key=operator.itemgetter(1))

    pprint(sorted_x[0:10])
    print "------------------------"


##### THIRD TEST: FIND KING - QUEEN + WOMAN or FRANCE - PARIS + ITALY
eps = 0.0000000000000000000001
print "king is to queen like man is to....x"
a = word_vectors['king']['vec']
a_star = word_vectors['queen']['vec']
b = word_vectors['man']['vec']
#vec = vec1-vec2+vec3
#vec = vec.tolist()
res.clear()
for w in word_vectors:
    b_star = word_vectors[w]['vec']
    d1 = distance(b_star,a_star)
    d2 = distance(b_star,b)
    d3 = distance(b_star,a)
    dist = (d1*d2)/(d3+eps)
    res[w]=dist
sorted_x = sorted(res.items(), key=operator.itemgetter(1),reverse=False)
pprint(sorted_x[0:10])
print "------------------------"

print "paris is to france like moscow is to....x"
a =word_vectors['paris']['vec']
a_star = word_vectors['france']['vec']
b = word_vectors['moscow']['vec']
#vec = vec1-vec2+vec3
#vec = vec.tolist()
res.clear()
for w in word_vectors:
    b_star = word_vectors[w]['vec']
    d1 = distance(b_star,a_star)
    d2 = distance(b_star,b)
    d3 = distance(b_star,a)
    dist = (d1*d2)/(d3+eps)
    res[w]=dist
sorted_x = sorted(res.items(), key=operator.itemgetter(1),reverse=False)
pprint(sorted_x[0:10])
print "------------------------"


