from sim_measures2 import *
from score_computations import *
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

origdatadir = './original_data/'
moddatadir = './modified_data/'

#===============================================================================================
#These predictions are based on if similar users have answered a question or ignored a question
#===============================================================================================

user_info = pd.read_csv(origdatadir + 'user_info.txt',delimiter = '\t', header=None, usecols=[0,1])#, skiprows = 10)
userids = user_info[:][0]
numUsers = long(len(userids))

no_label_info = pd.read_csv(moddatadir + 'merged_invited_info_full.csv',delimiter = ',') #, nrows=2)

numScores = 23

for i in np.arange(0,numScores):
   scorecol = 'score' + str(i+1)
   no_label_info[scorecol] = 0.0


#no_label_info = no_label_info[['q_id', 'user_id', 'q_indx', 'user_indx', 'usertags', 'q_tag','score1', 'score2', 'score3', 'score4', 'score5','score6','score7','score8','answered']]

numRecords = len(no_label_info)

print ('Reading user-question history')
uxq_mat = np.load('UxQAmatrix_minus.npy')

#=======================================================================================================
#computing user tag similarity based scores
#user-user tag similarity score
print ('Reading user similarity matrices')
op_path = 'uni_u2u_tag_sim_final'
usersim = np.memmap(op_path, dtype='float32', mode='r', shape=(numUsers,numUsers))
print np.trace(usersim)
compute_u2u_simscore(usersim, uxq_mat, no_label_info, 'score1', 'u2u tag sim')
del usersim
#no_label_info.to_csv('score1',index=False, columns = ['q_indx','user_indx','score1'])

#=======================================================================================================
#=======================================================================================================
#computing user 2 user word similarity based scores
#uxq_mat = np.load('UxQAmatrix_minus_CV.npy')
print ('Reading user 2 user word similarity')
op_path = 'uni_u2u_word_sim_final'
usersim = np.memmap(op_path, dtype='float32', mode='r', shape=(numUsers,numUsers))
print np.trace(usersim)
compute_u2u_simscore(usersim, uxq_mat, no_label_info, 'score2', 'u2u word sim')
del usersim
#no_label_info.to_csv('score2',index=False, columns = ['q_indx','user_indx','score2'])
#=======================================================================================================
#=======================================================================================================
#computing user 2 user char similarity based scores
#uxq_mat = np.load('UxQAmatrix_minus_CV.npy')
print ('Reading user 2 user char similarity')
op_path = 'uni_u2u_char_sim_final'
usersim = np.memmap(op_path, dtype='float32', mode='r', shape=(numUsers,numUsers))
print np.trace(usersim)
compute_u2u_simscore(usersim, uxq_mat, no_label_info, 'score3', 'u2u char sim')
del usersim
#no_label_info.to_csv('score3',index=False, columns = ['q_indx','user_indx','score3'])
#=======================================================================================================
#computing user 2 user question history similarity based scores
#uxq_mat = np.load('UxQAmatrix_minus_CV.npy')
print ('Reading user 2 user qhistory similarity')
op_path = 'uni_u2u_qhistory_sim_final'
usersim = np.memmap(op_path, dtype='float32', mode='r', shape=(numUsers,numUsers))
print np.trace(usersim)
compute_u2u_simscore(usersim, uxq_mat, no_label_info, 'score7', 'u2u q_history sim')
del usersim
#no_label_info.to_csv('score7',index=False, columns = ['q_indx','user_indx','score7'])
#=======================================================================================================
ques_info = pd.read_csv(origdatadir + 'question_info.txt',delimiter = '\t', header=None, usecols=[1])#,nrows=5)#, skiprows = 10)
numQues = long(len(ques_info))
#=======================================================================================================
#computing score based on feature similarity of questions
#uxq_mat = np.load('UxQAmatrix_minus_CV.npy')
print ('Reading question features matrix')
op_path = 'uni_q2q_features_sim_final'
quessim = np.memmap(op_path, dtype='float32', mode='r', shape=(numQues,numQues))
print np.trace(quessim)
compute_q2q_simscore(quessim, uxq_mat, no_label_info, 'score4', 'q2q features sim')
del quessim
#no_label_info.to_csv('score4',index=False, columns = ['q_indx','user_indx','score4'])
#=======================================================================================================
#computing score based on word similarity of questions
#uxq_mat = np.load('UxQAmatrix_minus_CV.npy')
print ('Reading question words matrix')
op_path = 'uni_q2q_word_sim_final'
quessim = np.memmap(op_path, dtype='float32', mode='r', shape=(numQues,numQues))
print np.trace(quessim)
compute_q2q_simscore(quessim, uxq_mat, no_label_info, 'score5', 'q2q word sim')
del quessim
#no_label_info.to_csv('score5',index=False, columns = ['q_indx','user_indx','score5'])
#=======================================================================================================
#computing score based on char similarity of questions
#uxq_mat = np.load('UxQAmatrix_minus_CV.npy')
print ('Reading question char matrix')
op_path = 'uni_q2q_char_sim_final'
quessim = np.memmap(op_path, dtype='float32', mode='r', shape=(numQues,numQues))
print np.trace(quessim)
compute_q2q_simscore(quessim, uxq_mat, no_label_info, 'score6', 'q2q char sim')
#no_label_info.to_csv('score6',index=False, columns = ['q_indx','user_indx','score6'])
del quessim

#=======================================================================================================


#=======================================================================================================
#computing score based on user similarity of questions
#uxq_mat = np.load('UxQAmatrix_minus_CV.npy')
print ('Reading question u history matrix')
op_path = 'uni_q2q_qhistory_sim_final'
quessim = np.memmap(op_path, dtype='float32', mode='r', shape=(numQues,numQues))
print np.trace(quessim)
compute_q2q_simscore(quessim, uxq_mat, no_label_info, 'score11', 'q2q qhistory sim')
del quessim
#no_label_info.to_csv('score5',index=False, columns = ['q_indx','user_indx','score5'])
#=======================================================================================================
#=======================================================================================================
#computing score based on feature similarity of questions
#uxq_mat = np.load('UxQAmatrix_minus_CV.npy')
print ('Reading question features 2 matrix')
op_path = 'uni_q2q_features_sim_final2'
quessim = np.memmap(op_path, dtype='float32', mode='r', shape=(numQues,numQues))
print np.trace(quessim)
compute_q2q_simscore(quessim, uxq_mat, no_label_info, 'score23', 'q2q features sim2')
del quessim
#no_label_info.to_csv('score4',index=False, columns = ['q_indx','user_indx','score4'])
#=======================================================================================================
#computing score based on tag similarity of questions
#uxq_mat = np.load('UxQAmatrix_minus_CV.npy')
print ('Reading question char matrix')
#op_path = 'uni_q2q_char_sim_final'
quessim = np.load('uni_q2q_tag_sim_final.npy')
#quessim = np.memmap(op_path, dtype='float32', mode='r', shape=(numQues,numQues))
print np.trace(quessim)
compute_q2q_simscore(quessim, uxq_mat, no_label_info, 'score22', 'q2q tag sim')
#no_label_info.to_csv('score6',index=False, columns = ['q_indx','user_indx','score6'])
#del quessim

#=======================================================================================================

userQtagProbmat_minus  = np.load('userQtagProbmat_minus.npy')      
usertagsQtagsmat_minus = np.load('usertagsQtagsmat_minus.npy')
userQtagProbRank_minus = np.load('userQtagProbRank_minus.npy')     
userLikely2answer_minus = np.load('userLikely2answer_minus.npy')
userLikely2answersq_minus = np.load('userLikely2answersq_minus.npy')
qLikely2banswered_minus = np.load('qLikely2banswered_minus.npy')
qLikely2banswered_sq_minus = np.load('qLikely2banswered_sq_minus.npy')

numRecords = len(no_label_info)

for i in range(0,numRecords):
   user_indx = no_label_info.get_value(i,'user_indx') #current user
   q_indx = no_label_info.get_value(i,'q_indx') #current question
   q_tag = no_label_info.get_value(i,'q_tag') #current question   

   if (i%1000 == 0):
      print ('Calculation for record #' + str(i))

   no_label_info.set_value(i, 'score8' , userQtagProbmat_minus[user_indx, q_tag])
   no_label_info.set_value(i, 'score9' , userQtagProbRank_minus[user_indx, q_tag])
   no_label_info.set_value(i, 'score10', 1.0*usertagsQtagsmat_minus[user_indx, q_indx])
   no_label_info.set_value(i, 'score12', no_label_info.get_value(i,'upvotes'))
   numAnswers = 1.0 * no_label_info.get_value(i,'numAnswers')
   numQualAns = 1.0 * no_label_info.get_value(i,'numQualAns')
   temp = 0.0
   if numAnswers != 0:
      temp = numQualAns/numAnswers
   no_label_info.set_value(i, 'score13', numAnswers)
   no_label_info.set_value(i, 'score14', numQualAns)
   no_label_info.set_value(i, 'score15', temp)
   no_label_info.set_value(i, 'score16', userLikely2answer_minus[user_indx])
   no_label_info.set_value(i, 'score17', userLikely2answersq_minus[user_indx])
   if usertagsQtagsmat_minus[user_indx, q_indx] < 1:
      no_label_info.set_value(i, 'score18', userLikely2answer_minus[user_indx])
      no_label_info.set_value(i, 'score19', userLikely2answersq_minus[user_indx])
   else:   
      no_label_info.set_value(i, 'score18', userQtagProbmat_minus[user_indx, q_tag])
      no_label_info.set_value(i, 'score19', userQtagProbRank_minus[user_indx, q_tag])
   no_label_info.set_value(i, 'score20', qLikely2banswered_minus[q_indx])
   no_label_info.set_value(i, 'score21', qLikely2banswered_sq_minus[q_indx])


n = datetime.datetime.now()
n = str(n).replace('-','_').replace(':','_').replace(' ','_').replace('.','_')
n = '2'
filename = 'computed_score_all_training' + n + '.csv'
no_label_info.to_csv(filename,index=False)#, columns = ['q_id', 'user_id', 'q_indx','user_indx','score1', 'score2', 'score3', 'score4', 'score5', 'score6','score7','answered'])


