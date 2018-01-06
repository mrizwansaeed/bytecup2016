from sim_measures2 import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import pandas as pd
from multiprocessing import Process, Pool
import time
import math
import shutil
import scipy.spatial.distance
import multiprocessing.managers
from sklearn.metrics.pairwise import cosine_similarity

origdatadir = './original_data/'
moddatadir = './modified_data/'

print('starting')

user_info = pd.read_csv(origdatadir + 'user_info.txt',delimiter = '\t', header=None, usecols=[0,1])#, skiprows = 10)
userids = user_info[:][0]

#uniquetags = usertags.str.split('/').apply(pd.Series, 1).stack().unique()
#uniquetags = uniquetags[uniquetags != '']
#uniquetags = uniquetags[uniquetags != '/']
#maxtag = max(uniquetags.astype(int))
#print maxtag
numRows = long(len(userids))

corpus = user_info[:][1]
#corpus = corpus.replace({'/':' '}, regex=True)
tt = corpus.as_matrix()

vect = TfidfVectorizer(min_df=1, tokenizer = tokens)
X = vect.fit_transform(tt)

op_path = 'uni_u2u_tag_sim_temp'
matrixsim = np.memmap(op_path, dtype='float32', mode='w+', shape=(numRows,numRows))
#matrixsim = cosine_similarity(tt)
#matrixsim[matrixsim < 0] = 0
matrixsim = (X*X.T).A
print np.trace(matrixsim)
print np.sum(matrixsim[0,:])

#the following loop will introduce asymmetry in the similarity matrix
#e.g. if u1 has tags 2/3/4 and u2 has tags 3/4.. Then based on tags only
#u1 will answer all questions u2 will answers but not vice versa

for i in range(0,numRows):
   for j in range(i+1, numRows):
       tag1 = set(tt[i].split('/'))
       tag2 = set(tt[j].split('/'))
       if tag2.issubset(tag1):
          matrixsim[i,j] = 1.0
       elif tag1.issubset(tag2):
          matrixsim[j,i] = 1.0
   if (i % 1000 == 0):
      print 'Processing record # ' + str(i)


op_path = 'uni_u2u_tag_sim_final'
print('read from file')
matrixsim2 = np.memmap(op_path, dtype='float32', mode='w+', shape=(numRows,numRows))
matrixsim2[:,:] = matrixsim[:,:]
print np.trace(matrixsim2)
print np.sum(matrixsim2[0,:])

del matrixsim
del matrixsim2

print('read from file')
matrixsim = np.memmap(op_path, dtype='float32', mode='r', shape=(numRows,numRows))
print np.trace(matrixsim)
print np.sum(matrixsim[0,:])
del matrixsim

