List of Python Programs to run  

#Create data set for all  
#Creates joined full invited_info, validate_nolabel and test_nolabel  
divide_dataset_info_train.py  

#Create data set for validation  
#Divides invited_info in 80-20% sets  
divide_dataset_info_submission.py  


#Create UxQ Matrices based on validation/all files  
#Creates 2 matrices  
#a) Questions answered by users  
#b) if a given question tag is in user tags  
createUxQmatrices_CV.py  
createUxQmatrices.py  

#What is the probability of a user answered a question based on tag  
#for each user and each tag computes P(Answered = Yes, tag)/P(Ans = yes, tag) + P(Ans = no, tag)  
createUxTagsProbability_minus.py  
createUxTagsProbability_minus_CV.py  


#Run following to create different similarity measures between u2u or q2q  
#These are independent of training/all data  
uni_u2u_tagsim.py  
uni_q2q_tagsim.py  
uni_q2q_featuressim.py  
uni_q2q_featuressim2.py  
uni_q2q_wordsim.py  
uni_u2u_tagsim.py  
uni_u2u_wordssim.py  

#except these  
uni_u2u_qhistory_sim_CV.py  
uni_u2u_qhistory_sim.py  


#helper files  
rank_metrics.py  
score_computations.py  
sim_measures2.py  
testing_output_rank.py  

#score computations  
python compute_score_all_submission.py  
python compute_score_all_testing.py  
python compute_score_all_training.py  


python compute_score_CV_submission.py  
python compute_score_CV_testing.py  
python compute_score_CV_training.py  
python compute_score_CV_validation.py  


