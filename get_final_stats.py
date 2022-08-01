import os

NUM_CONTAINER = 4
NUM_CVE = 41
RAW_FILE_NAME = "testing-res.csv"
OUTPUT_FILE_NAME = "final-stats.txt"
PATH = "./"

fpr_list = []
leadtime_list = []
detection_list = []

data = []

# Read in the raw data file.
with open(os.path.join(PATH, RAW_FILE_NAME)) as fin:
    for line in fin:
        line = line.strip()
        if "," not in line:  # found a new container
            data.append([])
        else:
            raw_values = line.split(",")[1:]
            fpr_list.append(float(raw_values[0]))
            leadtime_list.append(float(raw_values[6]))
            detection_list.append(float(raw_values[7]))
            data[-1].append(",".join(raw_values))


# Check if the 
assert(len(data) == NUM_CONTAINER)
for i in range(len(data)):
    try:
        assert(len(data[i]) == NUM_CVE)
    except AssertionError:
        print(f"Error: container {i + 1} has only {len(data[i])} rows, which is less than expected {NUM_CVE} rows")

print("Final stats for the experiment:")
# Print the final stats of FPR, lead time and detection 
with open(os.path.join(PATH, OUTPUT_FILE_NAME), "w") as fout:
    stats_str = f"FPR: {sum(fpr_list) / len(fpr_list):.2f}%\nDetection rate: {sum(detection_list) / len(detection_list):.2f}%\nLead time: {sum(leadtime_list) / len(leadtime_list):.2f}s\n"
    print(stats_str)
    fout.write(stats_str)