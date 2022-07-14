import numpy as np
import os
import pandas as pd
import csv

def get_system_calls():
    """
    Loads list of all system calls from a text file.
    """
    system_calls_set = set()
    with open("./data/combine_system_calls.txt") as file:
        for line in file:
            system_calls_set.add(line.strip())
    return system_calls_set

def get_stats(application_list, system_calls_set, file_num=1):
    """
    Add any new system call to the system call set.
    """
    is_changed = False
    for index in range(len(application_list)):
        app_name = application_list[index]
        file_name = 'shaped-input/{}/{}-{}_freqvector.csv'.format(app_name, app_name, file_num)
        df = pd.read_csv(file_name)
        header = list(df)
        for item in header:
            if "[" not in item:
                continue
            system_call = item.split("[")[0]
            if system_call not in system_calls_set:
                print("{} has not been seen ".format(system_call))
                is_changed = True
    return is_changed           

def combination(file_num_list, size):
    if (len(file_num_list) < size):
        return
    res = []
    dfs(file_num_list, size, 0, res, [])
    return res

def dfs(file_num_list, size, start, res, temp):
    if len(temp) == size:
        res.append(temp.copy())
        return
    for i in range(start, len(file_num_list)):
        temp.append(file_num_list[i])
        dfs(file_num_list, size, i + 1, res, temp)
        temp.pop()

def get_pairs(file_num_list):
    res = []
    last = ""
    for num in file_num_list:
        last += str(num)
        if len(last) < 2:
            continue
        res.append(last)
    return res
    
def transform_all_data(application_list, file_num_list, app_added=None):
    """
    
    """
    # generate a list of file names
    # for itself and test
    file_list = []
    for num in file_num_list: 
        file_list.append("{}_freqvector".format(num))
        file_list.append("{}_freqvector_test".format(num))
    # for combined 
    res = get_pairs(file_num_list)
    if len(res) > 0:
        for item in res:
            file_list.append("{}_freqvector".format(item))
    system_calls_set = get_system_calls()
    if app_added is None:
        get_stats(application_list, system_calls_set)
        transform_data(application_list, file_list, system_calls_set)
    else:
        is_changed = get_stats(app_added, system_calls_set)
        transform_data(app_added, file_list, system_calls_set)
    

def transform_data(application_list, file_list, system_calls_set=None):       
    """
    """
    if system_calls_set is None:
        system_calls_set = get_system_calls()
    for file_name in file_list:
        for index in range(len(application_list)):
            app_name = application_list[index]
            input_file = 'shaped-input/{}/{}-{}.csv'.format(app_name, app_name, file_name)
            df = pd.read_csv(input_file)
            header = list(df)
            current = 0
            sc_name_list = list(system_calls_set)
            sc_name_list.sort()
            sc_index_list = []
            for i, item in enumerate(header):
                if i == 0:
                    continue # ignore the first item because it doesn't contain any system call
                system_call = item.split("[")[0]
                while current < len(sc_name_list) and system_call != sc_name_list[current]:
                    current += 1
                    sc_index_list.append(-1)
                if current < len(sc_name_list) and system_call == sc_name_list[current]:
                    sc_index_list.append(i)
                    current += 1
            rows, columns = df.shape
            output_file = 'shaped-transformed/{}/{}-{}.csv'.format(app_name, app_name, file_name)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            print(output_file)
            if os.path.isfile(output_file):
                continue
            with open(output_file, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',
                                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
                header = ["timestamp"]
                for index, item in enumerate(sc_name_list):
                    header.append("{}[node1]:{}".format(item, index+1))
                csvwriter.writerow(header)
                for row in range(rows):
                    row_data = [0] * len(sc_name_list)
                    for j, item in enumerate(sc_index_list):
                        if item != -1:
                            row_data[j] = df.iat[row, item]
                    row_data.insert(0, df.iat[row, 0]) # add timestamp information from the original file
                    csvwriter.writerow(row_data)

