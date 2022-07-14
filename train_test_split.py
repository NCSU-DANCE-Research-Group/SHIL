# read from apps-all.txt
application_list = []
with open("apps.txt") as fin_apps:
    for line in fin_apps:
        application_list.append(line.strip())

start = 1
end = 4 
num_train_split = 3

for app in application_list:
    print(f"app: {app}")
    folder = f"shaped-transformed/{app}"
    for i in range(start, end + 1):
        with open(f"{folder}/{app}-{i}_freqvector_full.csv", "r") as fin:
            header = fin.readline()
            first_line = fin.readline()
            first_time = int(first_line.split(",")[0])
            last_line = first_line
            for train_index in range(1, num_train_split + 1):
                train_file_num = train_index + (i - 1) * num_train_split
                print(f"train_file_num: {train_file_num}")
                with open(f"{folder}/{app}-{train_file_num}_freqvector.csv", "w") as fout:
                    fout.write(header)
                    while True:
                        time = int(last_line.split(",")[0])
                        # 1 min = 60 * 1000 ms
                        if time < first_time + 60 * 1000 * train_index:
                            fout.write(last_line)
                        else:
                            break
                        last_line = fin.readline()
            
            # write the rest to the test file
            with open(f"{folder}/{app}-{i}_freqvector_test.csv", "w") as fout:
                fout.write(header)
                fout.write(last_line)
                for line in fin:
                    fout.write(line)
