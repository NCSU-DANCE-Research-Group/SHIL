#!/bin/bash
input="./apps.txt"
while IFS= read -r app
do
  trash ./models/$app
  mkdir -p ./models/$app
  
  folder=shaped-transformed/$app
  
  for dir in 1 2 3 4; do
      python3 autoEncoderTrain.py $folder/$app-$dir\_freqvector.csv
      python3 autoEncoderGetThreshold.py $folder/$app-$dir\_freqvector.csv
      mv $app-$dir.pkl $app-$dir.txt ./model
      mv ./model ./models/$app/$dir
  done

  rm -rf ./log/*
done < "$input"
