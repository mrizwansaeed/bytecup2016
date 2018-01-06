import pandas as pd
from rank_metrics import *

#===============================================================================================
#These predictions are based on if similar users have answered a question or ignored a question
#===============================================================================================

filename = './scores_based_on_training_CV_logistic/computed_score_validation_2016_11_15_01_18_04_809571_final.csv'
output = pd.read_csv(filename,delimiter = ',') #, usecols=[0,1])#, skiprows = 10)
print evaluate_ndcg(output,'score')


filename = './scores_based_on_training_CV_SVM/computed_score_CV_validation_2016_11_15_01_18_04_809571_final.csv'
output = pd.read_csv(filename,delimiter = ',') #, usecols=[0,1])#, skiprows = 10)
print evaluate_ndcg(output,'score')

filename = './scores_based_on_training_CV_SVM/computed_score_CV_validation_2016_11_15_01_18_04_809571_final2.csv'
output = pd.read_csv(filename,delimiter = ',') #, usecols=[0,1])#, skiprows = 10)
print evaluate_ndcg(output,'score')





















