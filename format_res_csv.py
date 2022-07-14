import os


NUM_CONTAINER = 4
NUM_CVE = 41
RAW_FILE_NAME = "testing-res.csv"
EXPORT_FILE_NAME = "testing-res-formatted.csv"
PATH = "./"
print(os.path.join(PATH, RAW_FILE_NAME))

data = []

# read in testing-res.csv
with open(os.path.join(PATH, RAW_FILE_NAME)) as fin:
    for line in fin:
        line = line.strip()
        if "," not in line:  # found a new container
            data.append([])
        else:
            data[-1].append(",".join(line.split(",")[1:]))

# self checking
assert(len(data) == NUM_CONTAINER)
for i in range(len(data)):
    try:
        assert(len(data[i]) == NUM_CVE)
    except AssertionError:
        print(f"Error: container {i + 1} has less than expected rows")

curr_cve = 0
# save to a new file testing-res-formatted.csv
with open(os.path.join(PATH, EXPORT_FILE_NAME), "w") as fout:
    while curr_cve < NUM_CVE:
        line = []
        for i in range(NUM_CONTAINER):
            line.append(data[i][curr_cve])
        line.append("\n")
        fout.write(",".join(line))
        curr_cve += 1
