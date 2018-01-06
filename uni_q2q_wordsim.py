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

ques_info = pd.read_csv(origdatadir + 'question_info.txt',delimiter = '\t', header=None, usecols=[0,2,3])#,nrows=6)#, skiprows = 10)
quesids = ques_info[:][0]

#uniquetags = questags.str.split('/').apply(pd.Series, 1).stack().unique()
#uniquetags = uniquetags[uniquetags != '']
#uniquetags = uniquetags[uniquetags != '/']
#maxtag = max(uniquetags.astype(int))
#print maxtag
numRows = long(len(quesids))

corpus = ques_info[:][2]
tt = corpus.as_matrix()
vect = TfidfVectorizer(min_df=1, tokenizer = tokens)
X = vect.fit_transform(tt)
op_path = 'uni_q2q_word_sim_temp'
matrixsim = np.memmap(op_path, dtype='float32', mode='w+', shape=(numRows,numRows))
matrixsim = (X*X.T).A
print np.trace(matrixsim)
print np.sum(matrixsim[0,:])
op_path = 'uni_q2q_word_sim_final'
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


#CHAR SIM
corpus = ques_info[:][3]
tt = corpus.as_matrix()
vect = TfidfVectorizer(min_df=1, tokenizer = tokens)
X = vect.fit_transform(tt)
op_path = 'uni_q2q_char_sim_temp'
matrixsim = np.memmap(op_path, dtype='float32', mode='w+', shape=(numRows,numRows))
matrixsim = (X*X.T).A
print np.trace(matrixsim)
print np.sum(matrixsim[0,:])
op_path = 'uni_q2q_char_sim_final'
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


