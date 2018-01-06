from sim_measures2 import *
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
import time
import math
import pickle
from sklearn import linear_model
from rank_metrics import *
from sklearn.externals import joblib
import datetime

#Compute user to user similarity
def compute_u2u_simscore(usersim, uxq_mat, pandafile, scorecol, purpose):

   numRecords = len(pandafile) #records in training data
   for i in range(0,numRecords):
      user_indx = pandafile.get_value(i,'user_indx') #current user
      q_indx = pandafile.get_value(i,'q_indx') #current question
      
      if (i%1000 == 0):
         print ('Calculation ' + purpose + ' for record #' + str(i))

      #read similarity score of this users with all others
      sim_users = usersim[user_indx,:]

      #get list of all users who answered the current question
      q_history = np.copy(uxq_mat[:,q_indx])
   
      #to ensure during training that if the question has already been answered by current
      #user, then that doesn't affect the score
      q_history[user_indx] = 0

      #normalize based on number of non-zeros in the question history (normalized based on
      #available question answered or ignored)
      normby = np.count_nonzero(q_history)
      #print normby
      score1 = 0
      if (normby != 0):
         score1 = np.dot(sim_users,q_history)#/normby #this gives between -1 to 1
         #score1 = (score1+1)/2 #this gives between 0 to 1
         #print score1
      pandafile.set_value(i, scorecol, score1)

#Compute question to question similarity
def compute_q2q_simscore(quessim, uxq_mat, pandafile, scorecol, purpose):
   numRecords = len(pandafile)
   for i in range(0,numRecords):
      user_indx = pandafile.get_value(i,'user_indx')
      q_indx = pandafile.get_value(i,'q_indx')

      if (i%1000 == 0):
      #print user_indx
      #print q_indx
         print ('Calculation ' + purpose + ' for record #' + str(i))
      
      #get questions similar to this question
      simquestions = quessim[:,q_indx].astype(float)
 
      #check if same user answered similar questions in the past 
      user_history =  np.copy(uxq_mat[user_indx,:])

      #to ensure during training that if the current question is answered by user, it doesn't affect
      #the score
      user_history[q_indx] = 0

      normby = np.count_nonzero(user_history)
      #print normby
      score1 = 0
      if (normby != 0):
         score1 = np.dot(simquestions,user_history)#/normby #this gives between -1 to 1
         #score1 = (score1+1)/2 #this gives between 0 to 1
         #print score1
      pandafile.set_value(i, scorecol, score1)

