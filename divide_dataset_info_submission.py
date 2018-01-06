import numpy as np
import pandas as pd

#This files create a csv with following columns
#['q_id', 'user_id', 'q_indx', 'user_indx', 'q_tag', 'upvotes', 'numAns', 'numQAns', 'answered']
#Basically it takes the invited_info text file and merges some information from
#user and question info txt files. Also it coverts long IDs to indices.
#For each user,question ID in inverted_info.txt files, it adds the integer index of the IDs from their respective files.

origdatadir = './original_data/'
moddatadir = './modified_data/'

#In this file, we will do random shuffle of the records and divide it into 70 and 30% sets

#assign each user a numeric index from 0 to n-1 where n is the number of total users
user_info = pd.read_csv(origdatadir + 'user_info.txt',delimiter = '\t', header=None, usecols=[0,1])#,nrows=5900)
user_info.to_csv(moddatadir + 'userIDtoIndex.csv',header=False)#, columns = [0])

#assign each question a numeric index from 0 to m-1 where m is the number of total ques
question_info = pd.read_csv(origdatadir + 'question_info.txt',delimiter = '\t', header=None)#, usecols=[0,1,4,5,6])#,nrows=100)
question_info.to_csv(moddatadir + 'questionIDtoIndex.csv',header=False)

#read the above created files with integer indices assigned to alphanumeric IDs
user_info = pd.read_csv(moddatadir + 'userIDtoIndex.csv',delimiter = ',', header=None)#, usecols=[0,1])
question_info = pd.read_csv(moddatadir + 'questionIDtoIndex.csv',delimiter = ',', header=None)# usecols=[0,1,2])


#===========================Merged invited info========================================

invited_info = pd.read_csv(origdatadir + 'invited_info_train.txt',delimiter = '\t', header=None)#,nrows=10)
#invited_info = invited_info.drop_duplicates()

#sorted_invited_info = invited_info.sort(2, ascending = False)
#no_dup = sorted_invited_info.drop_duplicates(subset=[0,1], keep='first', inplace = False)

sorted_list  = invited_info.sort_values(by = [0,1,2], ascending=[1, 1, 0])
no_dup = sorted_list.drop_duplicates(subset=[0,1], keep='first', inplace = False)
invited_info = no_dup

numRecords = len(invited_info)
question_info.drop(question_info.columns[[3,4]], axis=1, inplace=True) #drop question words and tags

#===========================Merged invited info Training========================================
user_info = pd.read_csv(moddatadir + 'userIDtoIndex.csv',delimiter = ',', header=None)#, usecols=[0,1])
#question_info = pd.read_csv(moddatadir + 'questionIDtoIndex.csv',delimiter = ',', header=None)# usecols=[0,1,2])


#===========================Merged invited info Total========================================
#user_info = pd.read_csv(moddatadir + 'userIDtoIndex.csv',delimiter = ',', header=None)#, usecols=[0,1])
#question_info = testing_questions
#question_info.drop(question_info.columns[[3,4]], axis=1, inplace=True) #drop question words and tags
#invited_info = pd.read_csv(origdatadir + 'invited_info_train.txt',delimiter = '\t', header=None)#,nrows=10)

temp = pd.merge(question_info, invited_info, left_on = [1], right_on=[0], sort=False)
temp2 = pd.merge(temp, user_info, left_on = '1_y', right_on=[1], sort=False, how = 'inner')
temp2.columns = np.arange(temp2.shape[1])
temp2.drop(temp2.columns[[2,7,11]], axis=1, inplace=True)
temp2.columns = ['q_id', 'q_indx', 'q_tag', 'upvotes', 'numAnswers', 'numQualAns', 'user_id', 'answered', 'user_indx','usertags']
temp2 = temp2[['q_id', 'user_id', 'q_indx', 'user_indx', 'usertags', 'q_tag', 'upvotes', 'numAnswers', 'numQualAns', 'answered']]
temp2.to_csv(moddatadir + 'merged_invited_info_full.csv',index=False, header=True)


#==============no_label info
nolabel_info = pd.read_csv(origdatadir + 'validate_nolabel.txt',delimiter = ',')#, header=True)#,nrows=10)
numRecords = len(nolabel_info)
temp = pd.merge(question_info, nolabel_info, left_on = [1], right_on=['qid'], sort=False)
temp2 = pd.merge(temp, user_info, left_on = 'uid', right_on=[1], sort=False, how = 'inner')
temp2.columns = np.arange(temp2.shape[1])
temp2.drop(temp2.columns[[6,10]], axis=1, inplace=True)
temp2.columns = ['q_indx', 'q_id', 'q_tag', 'upvotes', 'numAnswers', 'numQualAns', 'user_id', 'label', 'user_indx','usertags']
temp2 = temp2[['q_id', 'user_id', 'q_indx', 'user_indx', 'usertags', 'q_tag', 'upvotes', 'numAnswers', 'numQualAns', 'label']]
temp2.to_csv(moddatadir + 'merged_invited_info_submission.csv',index=False, header=True)

#==============test_label info
nolabel_info = pd.read_csv(origdatadir + 'test_nolabel.txt',delimiter = ',')#, header=True)#,nrows=10)
numRecords = len(nolabel_info)
temp = pd.merge(question_info, nolabel_info, left_on = [1], right_on=['qid'], sort=False)
temp2 = pd.merge(temp, user_info, left_on = 'uid', right_on=[1], sort=False, how = 'inner')
temp2.columns = np.arange(temp2.shape[1])
temp2.drop(temp2.columns[[6,10]], axis=1, inplace=True)
temp2.columns = ['q_indx', 'q_id', 'q_tag', 'upvotes', 'numAnswers', 'numQualAns', 'user_id', 'label', 'user_indx','usertags']
temp2 = temp2[['q_id', 'user_id', 'q_indx', 'user_indx', 'usertags', 'q_tag', 'upvotes', 'numAnswers', 'numQualAns', 'label']]
temp2.to_csv(moddatadir + 'merged_invited_info_testing.csv',index=False, header=True)
