app=""
if [ "$1" != "" ]; then
    app=$1
else
    echo "Error: please supply an application name as the first argument"
	exit 1
fi

# copy the needed Java class to the folder
cd $app
cp ../code/FrequencyProcessing.class .

# get raw data
for count in {1..4}
do
	[ -f $app-$count\_freqvector_full.csv ] || java FrequencyProcessing $app-$count.txt $app-$count\_freqvector_full.csv
	#rm $app-$count.txt
done

rm FrequencyProcessing.class
#zip $app-raw.zip *.txt
zip $app.zip *.csv time.txt
#rm *.txt

