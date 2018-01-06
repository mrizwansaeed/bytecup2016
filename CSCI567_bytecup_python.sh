#!/bin/bash  
echo "Create modified data sets CSVs"
python divide_dataset_info_train.py
python divide_dataset_info_submission.py

echo "Create User-Question history matrix based on 80% training set"
echo "Create naive classifier which give output 1 if question tag is in user tags, 0 otherwise"
python createUxQmatrices_CV.py
#python createUxQmatrices.py

echo "Compute probability of a user answering a question based on tag"
#python createUxTagsProbability_minus.py
python createUxTagsProbability_minus_CV.py


echo "Run user to user similarity matrices"
echo "Based on tag similarity"
python uni_u2u_tagsim.py
echo "Based on profile similarity"
python uni_u2u_wordssim.py
echo "Based on answering history similarity"
python uni_u2u_qhistory_sim_CV.py
#python uni_u2u_qhistory_sim.py

echo "Run question to question similarity matrices"
echo "Based on tag similarity"
python uni_q2q_tagsim.py
echo "Based on features similarity"
python uni_q2q_featuressim.py
python uni_q2q_featuressim2.py
echo "Based on question words similarity"
python uni_q2q_wordsim.py
echo "Based on user who answered the question"
python uni_q2q_qhistory_sim_CV.py

echo "Compute all scores"
python compute_score_CV_submission.py
python compute_score_CV_testing.py
python compute_score_CV_training.py
python compute_score_CV_validation.py

echo "Feature selection and output format as per LETOR algorithms"
python data_files_transform_CV.py

