# python3 classification\&save_file.py > classification.log
# mv -i classified/ ./data/classified-randomforest-200/ # move to the correct path
# ./train_all-classified.sh

python3 classification\&testing.py 2>&1 | tee testing.log

# python3 format_res_csv.py
# dir=AE
# mkdir $dir
# mv -t $dir detected.csv predicted.csv testing-res.csv thresholds.csv recon_errors.csv testing-res-formatted.csv testing.log

# outlier detection
# python3 outlier_detection_IsolationForest_nonoutlier.py
# dir=data/label_using_outlier
# mkdir $dir
# for container in {1..4}
# do
#     mv -t $dir outlier_$container.csv nonoutlier_$container.csv
# done

# self-supervised model training
# python3 supervised_binary_randomforest_training.py 
# python3 supervised_CNN_training.py

# self-supervised RF model testing 
# for confidence in 0.4 0.5 0.6 0.7 0.8 0.9
# do
#     python3 supervised_binary_randomforest_testing.py $confidence
#     python3 format_res_csv.py

#     dir=RF-$confidence
#     mkdir $dir
#     mv -t $dir detected.csv predicted.csv testing-res.csv testing-res-formatted.csv 
# done

# self-supervised CNN model testing 
# for confidence in 0.4 0.5 0.6 0.7 0.8 0.9
# do
#     python3 supervised_CNN_testing.py $confidence
#     python3 format_res_csv.py

#     dir=CNN-$confidence
#     mkdir $dir
#     mv -t $dir detected.csv predicted.csv testing-res.csv testing-res-formatted.csv 
# done

# HML testing
# for boundary in 1.2 1.4 1.6 1.8 2.0
# do
#     python3 HML_analysis.py $boundary
#     python3 format_res_csv.py

#     dir=HML-RF-$boundary
#     mkdir $dir
#     mv -t $dir detected.csv predicted.csv testing-res.csv testing-res-formatted.csv thresholds.csv recon_errors.csv
# done

# alternative combined method
# python3 badsupervised_binary_randomforest_training.py
# for confidence in 0.4 0.5 0.6 0.7 0.8 0.9
# do
#     python3 badsupervised_binary_randomforest_testing.py $confidence
#     python3 format_res_csv.py

#     dir=badRF-$confidence
#     mkdir $dir
#     mv -t $dir detected.csv predicted.csv testing-res.csv testing-res-formatted.csv 
# done

# for boundary in 1.2 1.4 1.6 1.8 2.0
# do
#     python3 HML_analysis.py $boundary
#     python3 format_res_csv.py

#     dir=HML-badRF-$boundary
#     mkdir $dir
#     mv -t $dir detected.csv predicted.csv testing-res.csv testing-res-formatted.csv thresholds.csv recon_errors.csv
# done

python3 ../testing/send_email.py

