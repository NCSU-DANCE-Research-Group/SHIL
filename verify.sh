#!/bin/sh

# Unsupervised model training
# python3 classification\&save_file.py > classification.log
# mv -i ./classified/ ./data/classified-randomforest-200/ # move to the correct path
# ./train_all-classified.sh

# Unsupervised model testing
# python3 classification\&testing.py 2>&1 | tee testing.log
# python3 format_res_csv.py
# dir=./result/CDL/AE95-41CVE
# mkdir -p $dir
# mv ./detected.csv ./predicted.csv ./testing-res.csv ./thresholds.csv ./recon_errors.csv ./testing-res-formatted.csv $dir 

# Outlier detection
python3 outlier_detection_IsolationForest_nonoutlier_normal.py
dir=./data/label_using_outlier
mkdir -p $dir
for container in {1..4}
do
    mv ./outlier_$container.csv ./nonoutlier_$container.csv $dir
done

# Self-supervised random forest model (used in SHIL)
python3 supervised_binary_randomforest_training.py
for confidence in 0.6
do
    python3 supervised_binary_randomforest_testing.py $confidence
    python3 format_res_csv.py

    dir=./result/supervisedRF/RF-$confidence
    mkdir -p $dir
    mv ./detected.csv ./predicted.csv ./testing-res.csv ./testing-res-formatted.csv ./probabiltity.csv $dir
done

# Self-supervised CNN model (alternative model, not used)
# python3 supervised_CNN_training.py
# for confidence in 0.6
# do
#     python3 supervised_CNN_testing.py $confidence
#     python3 format_res_csv.py

#     dir=./result/supervisedCNN/CNN-$confidence
#     mkdir -p $dir
#     mv ./detected.csv ./predicted.csv ./testing-res.csv ./testing-res-formatted.csv ./probabiltity.csv $dir
# done

# SHIL
# Test two boundary 120% and 200% used in the paper
for boundary in 1.2 2.0
do
    python3 SHIL_analysis_experiment.py $boundary
    python3 format_res_csv.py
    python3 get_final_stats.py

    dir=./result/SHIL/boundary-$boundary
    mkdir -p $dir
    mv ./detected.csv ./predicted.csv ./testing-res.csv ./testing-res-formatted.csv ./thresholds.csv ./recon_errors.csv ./final-stats.txt $dir
done