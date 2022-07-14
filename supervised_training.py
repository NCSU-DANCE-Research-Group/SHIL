from collections import defaultdict
from training_util import is_empty_line, get_key
from exportCSV import exportCSV
from training_util import NUM_CVE, NUM_CONTAINER, USE_10MS, read_record
from sklearn.metrics.pairwise import manhattan_distances
import pandas as pd
import numpy as np

data_folder = 'shaped-transformed'

def is_similar(normal_df, sample, similar_threshold):
    sample = np.array(sample)
    sample = np.expand_dims(sample, axis=0)
    # get the L1 distance
    distances = manhattan_distances(normal_df, sample)
    min_distance = distances.min()
    # find the most similar line from the dataframe
    return min_distance <= similar_threshold

def get_attack_period_not_similar(app, file_num, picked, label, similar_threshold):
    curr_pt = 0
    data = []
    labels = []
    # read in the normal period as a df
    df_train = pd.DataFrame()
    for offset in range(1, 4):
        train_file_num = (file_num - 1) * 3 + offset
        train_file = f"{data_folder}/{app}/{app}-{train_file_num}_freqvector.csv"
        print(train_file)
        df = pd.read_csv(train_file)
        df = df.iloc[:, 1:]
        df_train = pd.concat([df_train, df], axis=0)
        print(df_train.shape)
    with open(f"{data_folder}/{app}/{app}-{file_num}_freqvector_test.csv") as fin:
        fin.readline()  # skip the header
        counter = 0
        for line in fin:
            counter += 1
            if curr_pt < len(picked) and counter == picked[curr_pt]:
                # ignore the first column
                sample = list(map(int, line.strip().split(",")))[1:]
                data.append(sample)  
                if label == 1:
                    if is_similar(df_train, sample, similar_threshold): # find any line that is similar to this vector
                        labels.append(0)
                    else:
                        labels.append(1)  # non zero vector
                else:
                    labels.append(0)
                curr_pt += 1
                if curr_pt >= len(picked):
                    break
    return data, labels

def get_attack_period_negative_zero_vector(app, file_num, picked, label):
    curr_pt = 0
    data = []
    labels = []
    with open(f"{data_folder}/{app}/{app}-{file_num}_freqvector_test.csv") as fin:
        fin.readline()  # skip the header
        counter = 0
        for line in fin:
            counter += 1
            if curr_pt < len(picked) and counter == picked[curr_pt]:
                sample = list(map(int, line.strip().split(",")))[1:]
                is_empty_line_current = is_empty_line(sample)
                # if is_empty_line_current:
                #     empty_as_outlier += 1
                data.append(sample)  # ignore the first column
                # print(sample)
                # if label == 1:
                #     exportCSV(sample, f"raw_outlier_{file_num}.csv")
                if label == 1:
                    if is_empty_line_current:
                        # force the zero vectors to have a label of 0
                        labels.append(0)
                    else:
                        labels.append(1)  # non zero vector
                else:
                    labels.append(0)
                curr_pt += 1
                if curr_pt >= len(picked):
                    break
    return data, labels

def get_attack_period(app, file_num, picked, label):
    curr_pt = 0
    data = []
    labels = []
    # empty_as_outlier = 0
    with open(f"{data_folder}/{app}/{app}-{file_num}_freqvector_test.csv") as fin:
        fin.readline()  # skip the header
        counter = 0
        for line in fin:
            counter += 1
            if curr_pt < len(picked) and counter == picked[curr_pt]:
                sample = list(map(int, line.strip().split(",")))[1:]
                # if is_empty_line(sample):
                #     empty_as_outlier += 1
                data.append(sample)  # ignore the first column
                # print(sample)
                # if label == 1:
                #     exportCSV(sample, f"raw_outlier_{file_num}.csv")
                labels.append(label)
                curr_pt += 1
                if curr_pt >= len(picked):
                    break
    return data, labels

def get_data_by_app(app, normal_file_list, attack_file_list, label_mode, use_nonoutlier=True, similar_threshold=None):
    data = defaultdict(list)
    labels = defaultdict(list)
    # memory = defaultdict(set)  # keep consistent labels
    # relabelled = defaultdict(int)
    # empty_outliers = defaultdict(list)
    for file_num in normal_file_list:
        with open(f"data/label_using_outlier/outlier_{file_num}.csv") as fin_outlier:
            with open(f"data/label_using_outlier/nonoutlier_{file_num}.csv") as fin_nonoutlier:
                not_found = True
                while not_found:
                    parts = fin_outlier.readline().strip().split(",")
                    parts2 = fin_nonoutlier.readline().strip().split(",")
                    if parts is None:
                        break # we have searched the whole file
                    if parts[0] != app:
                        continue
                    not_found = False
                if file_num in attack_file_list: # default: [1]
                    # we want to use these containers' attack periods only
                    outliers = [int(parts[i]) for i in range(1, len(parts))]
                    nonoutliers = [int(parts2[i])
                                for i in range(1, len(parts2))]
                    if label_mode == 'zero':
                        data_outlier, label_outlier = get_attack_period_negative_zero_vector(app, file_num, outliers, 1)
                    elif label_mode == 'similar':
                        data_outlier, label_outlier = get_attack_period_not_similar(app, file_num, outliers, 1, similar_threshold)
                    else: # 'normal'
                        data_outlier, label_outlier = get_attack_period(app, file_num, outliers, 1)
                    data[app].extend(data_outlier)
                    # empty_outliers[app].append(
                    #     f"{empty_as_outlier}")
                    # for item in data_outlier:
                    #     memory[app].add(get_key(item))
                    labels[app].extend(label_outlier)
                    if use_nonoutlier:
                        data_nonoutlier, label_nonoutlier = get_attack_period(
                            app, file_num, nonoutliers, 0)
                        data[app].extend(data_nonoutlier)
                        labels[app].extend(label_nonoutlier)
    for file_num in normal_file_list:
        for file_num_offset in range(1, 4):
            train_file_num = 3 * (file_num - 1) + file_num_offset
            file_name = f'{data_folder}/{app}/{app}-{train_file_num}_freqvector.csv'
            with open(file_name) as fin:
                fin.readline()
                for line in fin:
                    item = list(
                        map(float, line.strip().split(",")))[1:]
                    # ignore the first column
                    data[app].append(item)
                    # if is_empty_line(item) and get_key(item) in memory[app] and False:
                    #     # print(f"{app} relabel")
                    #     relabelled[app] += 1
                    #     labels[app].append(1)
                    # else:
                    labels[app].append(0)
    # print(f"relabelled CVE:")
    # for key in relabelled:
    #     print(key)
    # for app in empty_outliers:
    #     line = f"{app},"
    #     line += ",".join(empty_outliers[app])
    #     print(line)
    return data, labels


def get_data(application_list, normal_file_list, attack_file_list, label_zero_vector_negative=False, use_nonoutlier=True):
    data = defaultdict(list)
    labels = defaultdict(list)
    # memory = defaultdict(set)  # keep consistent labels
    # relabelled = defaultdict(int)
    # empty_outliers = defaultdict(list)
    for file_num in normal_file_list:
        with open(f"data/label_using_outlier/outlier_{file_num}.csv") as fin_outlier:
            with open(f"data/label_using_outlier/nonoutlier_{file_num}.csv") as fin_nonoutlier:
                for app in application_list:
                    parts = fin_outlier.readline().strip().split(",")
                    parts2 = fin_nonoutlier.readline().strip().split(",")
                    if parts[0] != app:
                        print(parts[0], app)
                        continue
                    if file_num in attack_file_list: # default: [1]
                        # we want to use these containers' attack periods only
                        outliers = [int(parts[i]) for i in range(1, len(parts))]
                        nonoutliers = [int(parts2[i])
                                    for i in range(1, len(parts2))]
                        if label_zero_vector_negative:
                            data_outlier, label_outlier = get_attack_period_negative_zero_vector(
                                app, file_num, outliers, 1)
                        else:
                            data_outlier, label_outlier = get_attack_period(
                                app, file_num, outliers, 1)
                        data[app].extend(data_outlier)
                        # empty_outliers[app].append(
                        #     f"{empty_as_outlier}")
                        # for item in data_outlier:
                        #     memory[app].add(get_key(item))
                        labels[app].extend(label_outlier)
                        if use_nonoutlier:
                            data_nonoutlier, label_nonoutlier = get_attack_period(
                                app, file_num, nonoutliers, 0)
                            data[app].extend(data_nonoutlier)
                            labels[app].extend(label_nonoutlier)
    for app in application_list:
        for file_num in normal_file_list:
            for file_num_offset in range(1, 4):
                train_file_num = 3 * (file_num - 1) + file_num_offset
                file_name = f'{data_folder}/{app}/{app}-{train_file_num}_freqvector.csv'
                with open(file_name) as fin:
                    fin.readline()
                    for line in fin:
                        item = list(
                            map(float, line.strip().split(",")))[1:]
                        # ignore the first column
                        data[app].append(item)
                        # if is_empty_line(item) and get_key(item) in memory[app] and False:
                        #     # print(f"{app} relabel")
                        #     relabelled[app] += 1
                        #     labels[app].append(1)
                        # else:
                        labels[app].append(0)
    # print(f"relabelled CVE:")
    # for key in relabelled:
    #     print(key)
    # for app in empty_outliers:
    #     line = f"{app},"
    #     line += ",".join(empty_outliers[app])
    #     print(line)
    return data, labels


def get_data_nonoutlier_positive(application_list, file_num_list, attack_file_num_list):
    data = defaultdict(list)
    labels = defaultdict(list)
    # memory = defaultdict(set)  # keep consistent labels
    # empty_outliers = defaultdict(list)
    for file_num in file_num_list:
        with open(f"data/label_using_outlier/nonoutlier_{file_num}.csv") as fin_nonoutlier:
            for app in application_list:
                parts2 = fin_nonoutlier.readline().strip().split(",")
                if parts2[0] != app:
                    print(parts2[0], app)
                    continue
                if file_num in attack_file_num_list: # default: [1]
                    # we want to use these containers' attack periods only
                    nonoutliers = [int(parts2[i])
                                for i in range(1, len(parts2))]
                    data_nonoutlier, label_nonoutlier = get_attack_period(
                        app, file_num, nonoutliers, 1) # use non outliers as positive 
                    data[app].extend(data_nonoutlier)
                    labels[app].extend(label_nonoutlier)
    for app in application_list:
        for file_num in file_num_list:
            for file_num_offset in range(1, 4):
                train_file_num = 3 * (file_num - 1) + file_num_offset
                file_name = f'{data_folder}/{app}/{app}-{train_file_num}_freqvector.csv'
                with open(file_name) as fin:
                    fin.readline()
                    for line in fin:
                        item = list(
                            map(float, line.strip().split(",")))[1:]
                        # ignore the first column
                        data[app].append(item)
                        labels[app].append(0)
    # for app in empty_outliers:
    #     line = f"{app},"
    #     line += ",".join(empty_outliers[app])
    #     print(line)
    return data, labels
