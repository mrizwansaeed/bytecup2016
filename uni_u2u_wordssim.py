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

origdatadir = './original_data/'
moddatadir = './modified_data/'

print('starting')

user_info = pd.read_csv(origdatadir + 'user_info.txt',delimiter = '\t', header=None)#, usecols=[0,1,2,3])#, skiprows = 10)
userids = user_info[:][0]

#uniquetags = usertags.str.split('/').apply(pd.Series, 1).stack().unique()
#uniquetags = uniquetags[uniquetags != '']
#uniquetags = uniquetags[uniquetags != '/']
#maxtag = max(uniquetags.astype(int))
#print maxtag
numRows = long(len(userids))

#words
corpus = user_info[:][1]
tt = corpus.as_matrix()
vect = TfidfVectorizer(min_df=1, tokenizer = tokens)
X = vect.fit_transform(tt)
op_path = 'uni_u2u_word_sim_temp'
matrixsim = np.memmap(op_path, dtype='float32', mode='w+', shape=(numRows,numRows))
#matrixsim = cosine_similarity(tt)
#matrixsim[matrixsim < 0] = 0
matrixsim = (X*X.T).A
print np.trace(matrixsim)
print np.sum(matrixsim[0,:])

op_path = 'uni_u2u_word_sim_final'
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


#char sim
corpus = user_info[:][2]
tt = corpus.as_matrix()
vect = TfidfVectorizer(min_df=1, tokenizer = tokens)
X = vect.fit_transform(tt)
op_path = 'uni_u2u_char_sim_temp'
matrixsim = np.memmap(op_path, dtype='float32', mode='w+', shape=(numRows,numRows))
#matrixsim = cosine_similarity(tt)
#matrixsim[matrixsim < 0] = 0
matrixsim = (X*X.T).A
print np.trace(matrixsim)
print np.sum(matrixsim[0,:])

op_path = 'uni_u2u_char_sim_final'
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

