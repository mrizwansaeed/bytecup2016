#!/bin/bash  
python uni_q2q_qhistory_sim_CV.py

echo "Compute all scores"
python compute_score_CV_submission.py
python compute_score_CV_testing.py
python compute_score_CV_training.py
python compute_score_CV_validation.py

echo "Feature selection and output format as per LETOR algorithms"
python data_files_transform_CV.py

