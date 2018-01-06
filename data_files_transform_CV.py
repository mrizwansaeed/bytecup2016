import numpy as np
import pandas as pd

def convert_letor(textfile, training, features):
   text_file = open(textfile, "w")
   #features = np.arange(1,11) #use all features
   numRecords = len(training)
   #numRecords = 300

   for i in range(0,numRecords):
      line = ''
      answered = str(training.get_value(i, 'answered'))
      q_indx  = str(training.get_value(i, 'q_indx'))
      user_indx  = str(training.get_value(i, 'user_indx'))
      line = answered + ' qid:' + q_indx + ' ' 
      k = 1
      for j in features:
         scorecol = 'score' + str(j)
         score = str(training.get_value(i, scorecol))
         line = line + str(k) + ':' + score + ' '
         k = k + 1
      line = line + '#userID = ' + user_indx + '\n'
      text_file.write(line)
   text_file.close()

#for online validation and testing files
def convert_letor2(textfile, training, features):
   text_file = open(textfile, "w")
   #features = np.arange(1,11) #use all features
   numRecords = len(training)
   #numRecords = 300

   for i in range(0,numRecords):
      line = ''
      #answered = str(training.get_value(i, 'label')) simply insert 0 as first element of each line
      q_indx  = str(training.get_value(i, 'q_indx'))
      user_indx  = str(training.get_value(i, 'user_indx'))
      line = '0 qid:' + q_indx + ' ' 
      k = 1
      for j in features:
         scorecol = 'score' + str(j)
         score = str(training.get_value(i, scorecol))
         line = line + str(k) + ':' + score + ' '
         k = k + 1
      line = line + '#userID = ' + user_indx + '\n'
      text_file.write(line)
   text_file.close()

def normalizeColumns(training, numFeatures):

    for j in range(0,numFeatures):
       scorecol = 'score' + str(j+1)
       training[scorecol] = training[scorecol] - training[scorecol].min()
       if training[scorecol].max() != 0:
          training[scorecol] = training[scorecol]/training[scorecol].max()
    return training

filename1 = 'computed_score_CV_training2.csv'
filename2 = 'computed_score_CV_validation2.csv'
filename3 = 'computed_score_CV_submission2.csv'
filename4 = 'computed_score_CV_testing2.csv'

training   = pd.read_csv(filename1,delimiter = ',')
validation = pd.read_csv(filename2,delimiter = ',')
submission = pd.read_csv(filename3,delimiter = ',')
testing    = pd.read_csv(filename4,delimiter = ',')

#training.drop(training.columns[[4,5]], axis=1, inplace=True)
#validation.drop(validation.columns[[4,5]], axis=1, inplace=True)
#submission.drop(submission.columns[[4,5]], axis=1, inplace=True)
#testing.drop(testing.columns[[4,5]], axis=1, inplace=True)

training.sort_values(by = ['q_indx','user_indx'], ascending=[1, 1], inplace = True)
new_index = np.arange(len(training))
training = training.set_index(new_index)
training.fillna(0.0, axis = 1, inplace = True)


validation.sort_values(by = ['q_indx','user_indx'], ascending=[1, 1], inplace = True)
new_index = np.arange(len(validation))
validation = validation.set_index(new_index)
validation.fillna(0.0, axis = 1, inplace = True)


submission.sort_values(by = ['q_indx','user_indx'], ascending=[1, 1], inplace = True)
new_index = np.arange(len(submission))
submission = submission.set_index(new_index)
#existing value of label will be ignored, filled with zero in convert_letor2
submission.fillna(0.0, axis = 1, inplace = True)
#submission.label = submission.label.astype(int)


testing.sort_values(by = ['q_indx','user_indx'], ascending=[1, 1], inplace = True)
new_index = np.arange(len(testing))
testing = testing.set_index(new_index)
#existing value of label will be ignored
testing.fillna(0.0, axis = 1, inplace = True)
#testing.label = testing.label.astype(int)

#score1: similar user answered same question based on tag similarity
#score2: similar user answered same question based on word similarity
#score3: similar user answered same question based on char similarity
#score4: same user answered similar question based on features similarity
#score5: same user answered similar question based on words similarity
#score6: same user answered similar question based on char similarity
#score7: similar used answered same question based on question history similarity
#score8: probability of answering question based on this tag for this user
#score9: weighted probability of answering question based on this tag for this user
#score10: qtag in user tag
#score11: question similarity based on users answered
#score12: upvotes
#score13: numAns
#score14: quality answers
#score15: score14/score13
#score16: probability of answering any question
#score17: weighted probability of answering any question
#score18,19: composite of score 8 and 16, 9 and 17
#score20: prob of this question getting answerd
#score21: prob of this question getting answerd_ sq
#score22: q2q tag sim
#score23: q2q features (minus tag) sim
numScores = 23
training = normalizeColumns(training, numScores)
submission = normalizeColumns(submission, numScores)
testing = normalizeColumns(testing, numScores)
validation = normalizeColumns(validation, numScores)
#features = np.arange(1,11) #0.478304451155421 
#features = [4,5,7,8,9] #0.481747337503593 
features = [2,4,5,7,9,11] #0.485681991898277  (highest based on all data 0.4912)
#features = [4,5,7,9] 
features = [2,4,5,7,8,9,11,12,13,14]  #best
#features = [1,2,4,5,7,8,9,11,12,13,14,16,17]
#features = [2,4,5,7,8,9,11,12,13,14,16,17] #submitted saturday
#features = [4,5,7,8,9,11,12,13,14,16,17]
#sunday attempts
#features = [2,4,5,7,8,9,11,12,13,14,16,17,23] #submitted saturday
#features = [2,4,5,7,8,9,11,12,13,14,16,17,20,21,23] #2
#features = [2,4,5,7,8,9,11,12,13,14,16,17,20,21,22,23] #3
#features = [4,5,7,8,9,11,12,13,14,16,17,20,21,22,23] #4
#features = [4,5,7,8,9,11,12,13,14,16,17,20,22,23] #5
#features = [4,5,7,8,9,11,12,13,14,16,17,18,19,20,22,23] #6
#features = [4,5,7,8,9,11,12,13,14,16,17,18,19,22,23] #7
#features = [4,5,7,8,9,11,12,13,14,16,17,18,19,23] #8
#features = [4,5,7,8,9,11,12,13,14,16,17,20,22,23] #9
#features = [2,4,5,7,8,9,11,12,13,14,17,20,23] #10

#features = [1,2,4,5,7,8,9,11,12,13,14]
#features = np.arange(1,numScores+1)
#features = [1,2,4,5,7,8,9,11,12,13,14]

print 'Writing training'
convert_letor("trainCV.txt", training, features)
print 'Writing validation'
convert_letor("valiCV.txt", validation, features)
print 'Writing testing'
convert_letor2("testCV.txt", testing, features)
print 'Writing submission'
convert_letor2("submitCV.txt", submission, features)

filename1 = 'reduced_computed_score_CV_training.csv'
filename2 = 'reduced_computed_score_CV_validation.csv'
filename3 = 'reduced_computed_score_CV_submission.csv'
filename4 = 'reduced_computed_score_CV_testing.csv'

training.to_csv(filename1,index=False, header=True)
validation.to_csv(filename2,index=False, header=True)
submission.to_csv(filename3,index=False, header=True)
testing.to_csv(filename4,index=False, header=True)


temp = submission[['q_id','user_id']]
temp.columns = [['qid','uid']]
temp.to_csv('temp.csv',index=False, header=True)

temp = testing[['q_id','user_id']]
temp.columns = [['qid','uid']]
temp.to_csv('final.csv',index=False, header=True)
