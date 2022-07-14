input="apps.txt"
while IFS= read -r app
do

echo $app

raw_dir=raw-input/$app
shaped_dir=shaped-input/$app

mkdir -p $shaped_dir

python3 unifyTraceShape.py $raw_dir/$app-1_freqvector_full.csv,$raw_dir/$app-2_freqvector_full.csv,$raw_dir/$app-3_freqvector_full.csv,$raw_dir/$app-4_freqvector_full.csv $shaped_dir/$app-1_freqvector_full.csv,$shaped_dir/$app-2_freqvector_full.csv,$shaped_dir/$app-3_freqvector_full.csv,$shaped_dir/$app-4_freqvector_full.csv

done < $input
