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

class MyManager(multiprocessing.managers.BaseManager):
    pass 
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)

def chunks(l, n):
   for i in range(0, l, n):
      t = 0
      if ((i+n) > l):
         t = l
      else:
         t = i + n  
      yield np.array(np.arange(i,t))


def compute_model(modeldir, user_info, training, idx):

   startcol = min(idx)
   endcol = max(idx) + 1  #+1 so endcol is inclusive in the loop
   #print len(no_label_info)
   print('Worker started with range: ' + str(startcol) + ' to ' + str(endcol))
   k = 1
   
   for i in range(startcol,endcol):

   #reading from training 
      user_id = user_info.get_value(i,1)
      dataslice = training[training['user_id'] == user_id]
   
      #print 'Evaluating user: ' + str(i)

      if len(dataslice) > 0:   

         Xtrain = np.array(dataslice[['score1','score2','score3','score4','score5','score6','score7']].as_matrix())
         Ytrain = np.array(dataslice[['answered']].as_matrix())
      
         if len(dataslice) < 10:
            reg = linear_model.RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], normalize = True)#, cv = 3)
         else:
            reg = linear_model.RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], normalize = True, cv = 2)
         reg.fit(Xtrain, Ytrain) 

         filename = modeldir + 'user_' + user_id + '_logistic.pkl'
         joblib.dump(reg, filename)


      if (k%500==0):
         print('Worker with range: ' + str(startcol) + ' to ' + str(endcol) + ' is ' + str(100*k/len(idx)) + '% done')
      k = k + 1

   print('Worker finished with range: ' + str(startcol) + ' to ' + str(endcol))
   return 0

if __name__ == '__main__':  
   origdatadir = './original_data/'
   moddatadir = './modified_data/'
   modeldir = './savemodel/all/'



#===============================================================================================
#These predictions are based on if similar users have answered a question or ignored a question
#===============================================================================================

   user_info = pd.read_csv(moddatadir + 'userIDtoIndex.csv',delimiter = ',', header=None, usecols=[0,1])#, skiprows = 10)
   numRecords = long(len(user_info))
   #numUsers = 10

   filename = 'computed_score_all_training.csv'
   training = pd.read_csv(filename,delimiter = ',') #, nrows=2)

   m = MyManager()
   m.start()

   #how many processes (not threads) to start
   divisions = 12
   pool = Pool(processes=divisions)

   workdiv = math.ceil(float(numRecords)/divisions)
   workdiv = int(workdiv)

   mygenerator = chunks(numRecords,workdiv)
   #results = m.np_zeros((numRecords,1),float)

   k = 0
   for i in mygenerator:
      pool.apply_async(compute_model, (modeldir, user_info, training, i))
      print k 
      k = k + 1
   pool.close()
   pool.join()
   #print np.sum(results)



