from sim_measures2 import *
from score_computations import *
import numpy as np
import os
import pandas as pd
import multiprocessing.managers
from multiprocessing import Process, Pool
import time
import math
import pickle
from sklearn import linear_model
from rank_metrics import *
from sklearn.externals import joblib
import datetime
from sklearn.svm import SVC

#probabilities of previously answered questions based on tags
def compute_model(user_info, results, results2, invited_info):

   for i in range(0,len(user_info)):
      user_id = user_info.get_value(i,1)
      dataslice = invited_info[invited_info['user_id'] == user_id]
      dataslice_1 = invited_info.loc[(invited_info['user_id'] == user_id) & (invited_info['answered'] == 1)]
      if i%500 == 0:
         print 'Evaluating user: ' + str(i)

      if len(dataslice) > 0 and len(dataslice_1) > 0:
         
         alltags = dataslice['q_tag'].as_matrix().tolist()
         alltags_1 = dataslice_1['q_tag'].as_matrix().tolist()

         unique_qtags = np.unique(alltags)

         for elem in unique_qtags:
            temp = alltags.count(elem) * 1.0   
            temp1 = alltags_1.count(elem) * 1.0    
            results[i,elem] = temp1/temp
            results2[i,elem] = temp1*temp1/temp
   return 0

#probabilities of previously answered questions not based on tags
def compute_model2(user_info, results, results2, invited_info):

   for i in range(0,len(user_info)):
      user_id = user_info.get_value(i,1)
      dataslice = invited_info[invited_info['user_id'] == user_id]
      dataslice_1 = invited_info.loc[(invited_info['user_id'] == user_id) & (invited_info['answered'] == 1)]
      if i%500 == 0:
         print 'Evaluating user: ' + str(i)

      #i is user_indx
      results[i] = 0.0
      results2[i] = 0.0

      if len(dataslice) > 0 and len(dataslice_1) > 0:
         total_posed = 1.0 * len(dataslice)
         total_answered = 1.0 * len(dataslice_1)
         results[i] = total_answered/total_posed;
         results2[i] = total_answered*total_answered/total_posed;
   return 0

#ankit question version
def compute_model3(q_info, results, results2, invited_info):

   for i in range(0,len(q_info)):
      q_id = q_info.get_value(i,0)
      dataslice = invited_info[invited_info['q_id'] == q_id]
      dataslice_1 = invited_info.loc[(invited_info['q_id'] == q_id) & (invited_info['answered'] == 1)]
      if i%500 == 0:
         print 'Evaluating question: ' + str(i)

      #i is user_indx
      results[i] = 0.0
      results2[i] = 0.0

      if len(dataslice) > 0 and len(dataslice_1) > 0:
         total_posed = 1.0 * len(dataslice)
         total_answered = 1.0 * len(dataslice_1)
         results[i] = total_answered/total_posed;
         results2[i] = total_answered*total_answered/total_posed;
   return 0


origdatadir = './original_data/'
moddatadir = './modified_data/'

#===============================================================================================
#These predictions are based on if similar users have answered a question or ignored a question
#===============================================================================================

user_info = pd.read_csv(moddatadir + 'userIDtoIndex.csv',delimiter = ',', header=None)#, nrows = 10)
numUsers = len(user_info)

uniqueusertags = user_info[2].str.split('/').apply(pd.Series, 1).stack().unique()
maxusertag = max(uniqueusertags.astype(int))

userQtagProbmat_minus = np.zeros((numUsers,maxusertag+1),dtype='float')
userQtagEmat_minus = np.zeros((numUsers,maxusertag+1),dtype='float')    
userLikely2answer_minus = np.zeros((numUsers,1),dtype='float')
userLikely2answersq_minus = np.zeros((numUsers,1),dtype='float')
qLikely2banswered_minus = np.zeros((numUsers,1),dtype='float')
qLikely2banswered_sq_minus = np.zeros((numUsers,1),dtype='float')

q_info = pd.read_csv(origdatadir + 'question_info.txt',delimiter = '\t', header=None, usecols=[0])#,nrows=5)#, skiprows = 10)
invited_info = pd.read_csv(moddatadir + 'merged_invited_info_full.csv',delimiter = ',')#,nrows=10)

compute_model3(q_info, qLikely2banswered_minus, qLikely2banswered_sq_minus, invited_info)
compute_model(user_info, userQtagProbmat_minus, userQtagEmat_minus, invited_info)
compute_model2(user_info, userLikely2answer_minus, userLikely2answersq_minus, invited_info)

#print userQtagProbmat_minus
#print np.sum(userQtagProbmat_minus[4532,:])
np.save('userQtagProbmat_minus',userQtagProbmat_minus)
np.save('userQtagProbRank_minus',userQtagEmat_minus)
np.save('userLikely2answer_minus',userLikely2answer_minus)
np.save('userLikely2answersq_minus',userLikely2answersq_minus)
np.save('qLikely2banswered_minus', qLikely2banswered_minus)
np.save('qLikely2banswered_sq_minus', qLikely2banswered_sq_minus)

