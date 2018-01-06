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

#user_info = pd.read_csv(origdatadir + 'user_info.txt',delimiter = '\t', header=None, usecols=[0])#, skiprows = 10)
#userids = user_info[:][0]
#numRows = long(len(userids))

ques_info = pd.read_csv(origdatadir + 'question_info.txt',delimiter = '\t', header=None, usecols=[0,2,3])#,nrows=6)#, skiprows = 10)
quesids = ques_info[:][0]
numRows = long(len(quesids))

print ('Reading user-question history')
#0 if no data, 1 if user answered, -1 if ignored
uxq_mat = np.load('UxQAmatrix_minus_CV.npy')
uxq_mat_t = uxq_mat.transpose()
#tt = ques_info.as_matrix()
#X = cosine_similarity(tt)

op_path = 'uni_q2q_qhistory_sim_CV_temp'
matrixsim = np.memmap(op_path, dtype='float32', mode='w+', shape=(numRows,numRows))
matrixsim = cosine_similarity(uxq_mat_t)
matrixsim[matrixsim < 0] = 0
print np.trace(matrixsim)
print np.sum(matrixsim[0,:])

op_path = 'uni_q2q_qhistory_sim_CV_final'
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

