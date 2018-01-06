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

ques_info = pd.read_csv(origdatadir + 'question_info.txt',delimiter = '\t', header=None, usecols=[1,4,5,6])#,nrows=5)#, skiprows = 10)

ques_info[1] = (ques_info[1] - ques_info[1].mean())/ques_info[1].std()
ques_info[4] = (ques_info[4] - ques_info[4].mean())/ques_info[4].std()
ques_info[5] = (ques_info[5] - ques_info[5].mean())/ques_info[5].std()
ques_info[6] = (ques_info[6] - ques_info[6].mean())/ques_info[6].std()
ques_info.columns = ['qtags','upvotes','numAnswers','numQAns']

numRows = long(len(ques_info))

tt = ques_info.as_matrix()
#X = cosine_similarity(tt)

op_path = 'uni_q2q_features_sim_temp'
matrixsim = np.memmap(op_path, dtype='float32', mode='w+', shape=(numRows,numRows))
matrixsim = cosine_similarity(tt)
matrixsim[matrixsim < 0] = 0
print np.trace(matrixsim)
print np.sum(matrixsim[0,:])

op_path = 'uni_q2q_features_sim_final'
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

