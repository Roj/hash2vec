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

# This is the Wikipedia Reader
k = 10
vec_len = 211

# First task is to parse all wikipedia just count the time it takes and output something every now and then
# second task is to count the number of lines in total
# third task is to do the hash2vec procedure but only for words in a given file (get all words from benchmarks)

# We are only interested in the vectors for these words
interesting_words = []

# Store benchmark to perform test quickly
test_words = {}

benchmarks = ['benchmarks/wordsim353.csv','benchmarks/Mtruk.csv']
for bench in benchmarks:
    if not bench in test_words:
        test_words[bench]=[]
    with open(bench, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            test_words[bench].append([row[0],row[1],float(row[2])])
            if row[0] not in interesting_words:
                interesting_words.append(row[0])
            if row[1] not in interesting_words:
                interesting_words.append(row[1])

print "Total # of interesting words: " + str(len(interesting_words))

word_vectors = {}

file = open("wikipedia/wikipedia.corpus.nodups.clean")
total_lines=0
sections = 0
while 1:
    lines = file.readlines(100000)
    if not lines:
        break
    total_lines+=100000
    sections += 1
    perc = round(float(sections)/81200*100)
    if total_lines % 20000000==0:
        print "Processed:"+str(perc)+"% of Wikipedia"
    for line in lines:
        word_position = 0
        words = line.split()
        for w in words:
            word_position+=1
            if w in interesting_words and not w in word_vectors:
                word_vectors[w] = {}
                word_vectors[w]['frq']=0
                word_vectors[w]['vec'] = [float(0)] * vec_len
                word_vectors[w]['h']= hash(w)%vec_len
            if w in interesting_words:
                word_vectors[w]['frq']+=1