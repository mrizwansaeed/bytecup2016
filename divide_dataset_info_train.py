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
#shuffle training data... probabaly i need to make sure that you get sufficient number of 1's in both
#training and testing sets. Let's keep a ratio of 11%.

divide = 0.8

ones_in_data = np.sum(invited_info[2])
ones_in_testing = 0
ones_in_training = 0

ones_test_limit = (1-divide-0.05)*ones_in_data  #i.e., 20% of total set, should have at least 15% ones
ones_training_limit = (divide-0.05)*ones_in_data  #i.e., 80% of total set, should have at least 75% ones

train_size = divide * numRecords
train_size = int(train_size)
test_size = numRecords - train_size

#this is to make sure that there are enough ones in both data sets
while (ones_in_testing < ones_test_limit and ones_in_training < ones_training_limit):
   
  #random shuffle data
   shuffled_info = invited_info.reindex(np.random.permutation(invited_info.index))
   new_index = np.arange(numRecords)
   #re=sort indices
   shuffled_info = shuffled_info.set_index(new_index)

   #select training
   training_info = shuffled_info[:train_size]

   #select testing, re-organize indices
   testing_info = shuffled_info[-test_size:]
   new_index = np.arange(testing_info.shape[0])
   testing_info = testing_info.set_index(new_index)

   ones_in_testing = np.sum(testing_info[2])
   ones_in_training = np.sum(training_info[2])

#training_info.to_csv(moddatadir + 'training_questions_info.csv',index=False, header=False)#, columns = [0])
#testing_info.to_csv(moddatadir + 'testing_questions_info.csv',index=False, header=False)#, columns = [0])

question_info.drop(question_info.columns[[3,4]], axis=1, inplace=True) #drop question words and tags

#===========================Merged invited info Training========================================
user_info = pd.read_csv(moddatadir + 'userIDtoIndex.csv',delimiter = ',', header=None)#, usecols=[0,1])
#question_info = pd.read_csv(moddatadir + 'questionIDtoIndex.csv',delimiter = ',', header=None)# usecols=[0,1,2])

#invited_info = pd.read_csv(origdatadir + 'invited_info_train.txt',delimiter = '\t', header=None)#,nrows=10)

temp = pd.merge(question_info, training_info, left_on = [1], right_on=[0], sort=False)
temp2 = pd.merge(temp, user_info, left_on = '1_y', right_on=[1], sort=False, how = 'inner')
temp2.columns = np.arange(temp2.shape[1])
temp2.drop(temp2.columns[[2,7,11]], axis=1, inplace=True)
temp2.columns = ['q_id', 'q_indx', 'q_tag', 'upvotes', 'numAnswers', 'numQualAns', 'user_id', 'answered', 'user_indx','usertags']
temp2 = temp2[['q_id', 'user_id', 'q_indx', 'user_indx', 'usertags', 'q_tag', 'answered', 'upvotes', 'numAnswers', 'numQualAns']]
temp2.to_csv(moddatadir + 'merged_invited_info_training.csv',index=False, header=True)

#===========================Merged invited info Testing========================================
#user_info = pd.read_csv(moddatadir + 'userIDtoIndex.csv',delimiter = ',', header=None)#, usecols=[0,1])
#question_info = testing_questions
#question_info.drop(question_info.columns[[3,4]], axis=1, inplace=True) #drop question words and tags
#invited_info = pd.read_csv(origdatadir + 'invited_info_train.txt',delimiter = '\t', header=None)#,nrows=10)

temp = pd.merge(question_info, testing_info, left_on = [1], right_on=[0], sort=False)
temp2 = pd.merge(temp, user_info, left_on = '1_y', right_on=[1], sort=False, how = 'inner')
temp2.columns = np.arange(temp2.shape[1])
temp2.drop(temp2.columns[[2,7,11]], axis=1, inplace=True)
temp2.columns = ['q_id', 'q_indx', 'q_tag', 'upvotes', 'numAnswers', 'numQualAns', 'user_id', 'answered', 'user_indx','usertags']
#temp2 = temp2[['q_id', 'user_id', 'q_indx', 'user_indx', 'usertags', 'q_tag', 'answered', 'upvotes', 'numAnswers', 'numQualAns']]
#temp2.to_csv(moddatadir + 'merged_invited_info_validation2.csv',index=False, header=True)
temp2 = temp2[['q_id', 'user_id', 'q_indx', 'user_indx', 'usertags', 'q_tag', 'answered', 'upvotes', 'numAnswers', 'numQualAns']]
temp2.to_csv(moddatadir + 'merged_invited_info_validation.csv',index=False, header=True)


