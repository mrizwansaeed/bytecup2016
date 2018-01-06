from sim_measures2 import *
import numpy as np
import os
import pandas as pd
import time
import math

origdatadir = './original_data/'
moddatadir = './modified_data/'

#=======================User=info====================================

invited_info = pd.read_csv(moddatadir + 'merged_invited_info_full.csv',delimiter = ',')#,nrows=10)
#invited_info = invited_info.drop_duplicates()

#invited_info = invited_info[invited_info.answered != 0]
#invited_info.to_csv(moddatadir + 'reduced_invited_info2.csv',index=False, header=True)
#invited_info = pd.read_csv(moddatadir + 'reduced_invited_info2.csv',delimiter = ',')

user_info = pd.read_csv(moddatadir + 'userIDtoIndex.csv',delimiter = ',', header=None)
question_info = pd.read_csv(moddatadir + 'questionIDtoIndex.csv',delimiter = ',', header=None)

numQues = len(question_info)
numUsers = len(user_info)
#numRows = long(numRows)

#userQmat = np.memmap('userQmat', dtype='uint8', mode='w+', shape=(numRows,numQues))    

#only set answered questions to 1
#userQmat = np.zeros((numUsers,numQues),dtype='uint8')

#set ignored questions to -1
userQmat_minus = np.zeros((numUsers,numQues),dtype='int8')
usertagsQtagsmat_minus = np.zeros((numUsers,numQues),dtype='int8')    

numRecords = len(invited_info)

for i in range(0,numRecords):
   answered = invited_info.get_value(i,'answered')
   user_indx = invited_info.get_value(i,'user_indx')
   q_indx = invited_info.get_value(i,'q_indx')
   temp = userQmat_minus[user_indx, q_indx] 
   if (answered == 1):
      userQmat_minus[user_indx, q_indx] = 1
   else:
      if (temp != 1):
         userQmat_minus[user_indx, q_indx] = -1

   usertags = str(invited_info.get_value(i,'usertags'))
   q_tags = str(invited_info.get_value(i,'q_tag'))
   tag1 = set(usertags.split('/'))
   tag2 = set(q_tags.split('/'))
   if tag2.issubset(tag1):
      usertagsQtagsmat_minus[user_indx, q_indx] = 1

#np.save(moddatadir + 'UxQmatrix',userQmat)
np.save('UxQAmatrix_minus',userQmat_minus)
np.save('usertagsQtagsmat_minus',usertagsQtagsmat_minus)

