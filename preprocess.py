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

# HACER TEXT CLEANUP
# PARSING IN SENTENCES
# REMOVE INFREQUENT WORDS
# WRITE RESULTING TXT ONE SENTENCE PER LINE

if len(sys.argv)!=2:
    print "Use: preprocess.py textfile"
    print "example: preprocess.py text8/text8.txt"

arg_file = sys.argv[1]

total_lines = 0
total_words = 0
dif_words = 0
word_count = {}

# Open aux file where we will write one line per sentence with tokens parsed
# but not filtered yet by frequency!

for line in file(arg_file):
    words =line.split()
    for w in words:
        total_words+=1
        if w not in word_count:
            word_count[w]=1
            dif_words+=1
        else:
            word_count[w]+=1


print "Processed a total of "+str(total_lines)+" lines of text"
print "found a total of "+str(total_words)+ " with "+str(dif_words)+" different words"

stoplist = set('for a of the and to in s was is that'.split())

protected_words =['american','world','war','city','goverment','century','university','life','british']
#sorted_x = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
#pprint(sorted_x)
#quit()

total_words=0
final_words = {}
# Repeat the process filtering words with freq below threshold
f = open(arg_file+"clean.txt", 'w')
for line in file(arg_file):
    words =line.split()
    for w in words:
        if word_count[w]>=5 and (word_count[w]<10000 or w in protected_words):
            f.write(w+" ")
            total_words+=1
            if w not in final_words:
                final_words[w]=1

print "After filtering we kept "+str(total_words)+ " words with "+str(len(final_words.keys()))+" different words"
f.close()
