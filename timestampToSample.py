from getSamples import calculateSamples

# read from apps-all.txt
application_list = []
with open("apps.txt") as fin_apps:
    for line in fin_apps:
        application_list.append(line.strip())

print("There are {} applications. ".format(len(application_list)))
print(application_list)
NUM_CONTAINER = 4

for app in application_list:
    print(f"app: {app}")
    # get time.txt to get all time stamps
    query_times = []
    with open(f"raw-input/{app}/time.txt") as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue # empty lines
            if line[0] == 'c':
                query_times.append([])
            elif len(query_times):
                parts = line.split(":")
                parts[-1] = float(parts[-1])
                parts[-1] = "{:.3f}".format(parts[-1])
                query_times[-1].append(":".join(parts))
    for i in range(1, NUM_CONTAINER + 1):
        time = 0
        # get the first epoch time stamp
        with open(f"raw-input/{app}/{app}-{i}_freqvector_full.csv") as fin:
            next(fin) # skip the first line
            time = int(fin.readline().strip().split(",")[0])
        print(f"Container {i}") 
        calculateSamples(time, query_times[i - 1][1:-1])
