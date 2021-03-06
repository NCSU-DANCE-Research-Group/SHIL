# SHIL: Self-Supervised Hybrid Learning for Security Attack Detection in Containerized Applications

This contains the code and data for our paper "SHIL: Self-Supervised Hybrid Learning for Security Attack Detection in Containerized Applications" accepted by ACSOS 2022. 

## Approach
SHIL is a hybrid machine learning solution, which intelligently combines unsupervised and supervised learning method to achieve high attack detection accuracy without requiring any manual data labelling.

## Data

We evaluated 41 CVEs. The traces are in `shaped-transformed` folder.

## Environment

This system was evaluated using Python 3.6.9, 3.7.0, and 3.7.3. The packages used and the corresponding versions are listed below, which can be installed using `pip3 install -r requirements.txt`:

```
joblib==1.1.0
Keras==2.2.4
numpy==1.19.5
pandas==0.24.0
scikit-learn==0.21.3
scipy==1.1.0
tensorboard==1.14.0
tensorflow==1.13.1
xlrd==1.2.0
XlsxWriter==0.9.6
```

## One button script to reproduce the result

The partial result of the unsupervised model and supervised models are saved in the `data` and `result` folder.
Please keep all contents in the `data`, `result` and `shaped-transformed` folders to run the `verify.sh` with `sh` or `bash`. After it stops running, you can copy the whole content from `./result/SHIL/boundary-2.0/testing-res-formatted.csv` and paste into cell `L3` of the sheet `./result/result.xlsx` to compare with the paper result of SHIL using 200% boundary case. 

## Unsupervised Model

`classification&save_file.py` prepares the data for training the unsupervised model.
`train_all-classified.sh` trains the unsupervised model.
`classification&testing.py` tests the unsupervised model.

## Outlier detection

`outlier_detection_IsolationForest_nonoutlier_normal.py` labels the outliers and non-outliers and saves to CSV files.

## Self-supervised Models

### Self-supervised random forest (used in SHIL)

`supervised_binary_randomforest_training.py` is the training code for self-supervised random forest.
`supervised_binary_randomforest_testing.py` is the testing code for self-supervised random forest. The first argument is the minimum confident. 

### Self-supervised CNN (alternative model)

`supervised_CNN_training.py` is the training code for self-supervised CNN.
`supervised_CNN_testing.py` is the testing code for self-supervised CNN. The first argument is the minimum confident. 

## SHIL

`SHIL_analysis_experiment.py` contains the code for final cross validaton. Please change the paths in this code so that it compares the decisions made by the unsupervised model and the correct supervised model.
