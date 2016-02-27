from gensim.models import word2vec
import logging
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
import scipy.stats as sci
import bz2,json,contextlib
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#sentences = word2vec.Text8Corpus('text8/text8.txt')

#model = word2vec.Word2Vec(sentences, size=200)
#model.save('text8.model')
#model.save_word2vec_format('text.model.bin', binary=True)
model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)


#pprint(model.most_similar(['man']))
#pprint(model.most_similar(['frog']))

#pprint(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))

#pprint(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=2))


#pprint(model.most_similar(['girl', 'father'], ['boy'], topn=3))


benchmarks = ['benchmarks/wordsim353.csv','benchmarks/Mtruk.csv']


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
        if word1 in model and word2 in model:
            list_score1.append(test[2])
            #score2 = 1-distance(vec1,vec2)
            score2 = model.similarity(word1, word2)
            list_score2.append(score2)
            print "Word1:"+word1+" Word2:"+word2+" Human Score:"+str(test[2])+" Hash2Vec:"+str(score2)
    rho = sci.stats.spearmanr(list_score1, list_score2)
    print bench+" Rho:"+str(rho[0])

words_to_find = ['computer','king','queen','physics','north','italy','wounded','car','church','wednesday','two','man','woman']

for w in words_to_find:
    print w
    pprint(model.most_similar(w,topn=10))