#!/bin/bash
ls ./data/attack_category -1 | sed -r 's/.csv//' > "data/attack_type.txt"
input="./data/attack_type.txt" # originally: data/apps-classified.txt
while IFS= read -r app
do
  trash ./models/$app
  mkdir -p ./models/$app
  
  folder=data/attack_category
  
  python3 autoEncoderTrain.py $folder/$app.csv
  python3 autoEncoderGetThreshold.py $folder/$app.csv
  mv $app.pkl $app.txt ./model
  mv ./model ./$app
  mv ./$app ./models

done < "$input"
