from collections import defaultdict

NUM_CVE = 41
NUM_CONTAINER = 4
USE_10MS = False

def get_key(num_list):
    int_num_list = [int(val) for val in num_list]
    return ",".join([f"{val}" for val in int_num_list])


def is_empty_line(item):
    for val in item:
        if val != 0:
            return False
    return True

def get_application_dict():
    application_dict = defaultdict(int)
    index = 0
    with open("data/apps-all.txt") as fin_apps:
        for line in fin_apps:
            app = line.strip()
            application_dict[app] = index
            index += 1
    print(f"There are {len(application_dict)} applications: \n{application_dict}")
    return application_dict

def read_record(path, is_int, keep_float, file_name):
    data = []
    rows = []
    print(f"{path}/{file_name}")
    with open(f"{path}/{file_name}") as fin:
        for line in fin:
            line = line.strip()
            if "," not in line:
                # new container list
                if len(rows):
                    data.append(rows)
                rows = []
            else:
                rows.append([float(i) for i in line.split(",")])
    data.append(rows)
    assert(len(data) == NUM_CONTAINER)
    for i in range(len(data)):
        assert(len(data[i]) == NUM_CVE)
    return data