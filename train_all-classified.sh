#!/bin/bash
input="data/apps-classified.txt" # originally: data/apps-classified.txt
while IFS= read -r app
do
  trash ./models/$app
  mkdir -p ./models/$app
  
  folder=data/classified-randomforest-200
  
  python3 autoEncoderTrain.py $folder/$app.csv
  python3 autoEncoderGetThreshold.py $folder/$app.csv
  mv $app.pkl $app.txt ./model
  mv ./model ./$app
  mv ./$app ./models

done < "$input"
