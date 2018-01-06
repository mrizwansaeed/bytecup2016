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

ques_info = pd.read_csv(origdatadir + 'question_info.txt',delimiter = '\t', header=None, usecols=[0,1,2])#,nrows=5)#, skiprows = 10)

#ques_info[1] = (ques_info[1] - ques_info[1].mean())/ques_info[1].std()
#ques_info[4] = (ques_info[4] - ques_info[4].mean())/ques_info[4].std()
#ques_info[5] = (ques_info[5] - ques_info[5].mean())/ques_info[5].std()
#ques_info[6] = (ques_info[6] - ques_info[6].mean())/ques_info[6].std()
temp = ques_info[:][1]#'upvotes','numAnswers','numQAns']

numQues = long(len(ques_info))

tt = temp.as_matrix()
#X = cosine_similarity(tt)

#op_path = 'uni_q2q_tag_sim_temp2'
matrixsim = np.zeros((numQues,numQues),dtype='int8')

#matrixsim = np.memmap(op_path, dtype='float32', mode='w+', shape=(numRows,numRows))

for i in range(0,numQues):
   matrixsim[i,:] = (tt == temp[i])*1


print np.trace(matrixsim)
print np.sum(matrixsim[0,:])

np.save('uni_q2q_tag_sim_final',matrixsim)

