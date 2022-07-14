from DataConversion import transform_data

# read from apps-all.txt
application_list = []
with open("apps.txt") as fin_apps:
    for line in fin_apps:
        application_list.append(line.strip())

start = 1
end = 4 

file_list = [f'{i}_freqvector_full' for i in range(start, end + 1)]
transform_data(application_list, file_list)
