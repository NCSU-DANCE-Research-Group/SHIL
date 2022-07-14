import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from exportCSV import exportCSV
import pickle
from sklearn.metrics import classification_report

class ApplicationClassifier:
    def __init__(self, interval, n_estimators=200, num_container=4, step=10, algorithm='randomforest', moving_window=True):
        self.interval = interval # 300 means 30 seconds
        self.n_estimators = n_estimators # default value is 200
        self.num_container = num_container
        self.has_printed = False
        # leave the first one empty on purpose
        self.data = []
        self.labels = []
        self.step = step
        self.algorithm = algorithm
        classifier_folder = 'data/classifier'
        os.makedirs(classifier_folder, exist_ok=True)
        classifier_data_folder = "data/classifier_data"
        os.makedirs(classifier_data_folder, exist_ok=True)
        self.clf_file = f'{classifier_folder}/{self.algorithm}-{self.n_estimators}.pkl'
        self.transform_raw = {'jboss':{'CVE-2015-8103', 'CVE-2017-12149'}, 'jetty':{'CVE-2021-28164', 'CVE-2021-28169', 'CVE-2021-34429'}, 'ghostscript':{'CVE-2018-16509', 'CVE-2018-19475', 'CVE-2019-6116b'}}
        self.transform_dict = dict()
        for key in self.transform_raw:
            for item in self.transform_raw[key]:
                self.transform_dict[item] = key
        self.moving_window = moving_window
    
    def get_data(self, app_name, label, file_name):
        X = []
        Y = []
        raw_data = []
        
        if self.moving_window:
            df = pd.read_csv(file_name)
            rows, columns = df.shape
            sc_index_list = [i for i in range(1, columns)]

            last_data = None
            for i in range(0, rows, self.step):
                count = 0
                end = min(i + self.interval, rows)
                if last_data is not None:
                    row_data = last_data[:]
                    start = end - self.step
                    count = self.interval - self.step # we have some old data
                    # remove the last step rows from the sum
                    for row in range(i - self.step, i):
                        for j, item in enumerate(sc_index_list):
                            row_data[j] -= df.iat[row, item]
                else:
                    row_data = [0] * (columns - 1)
                    start = i
                #print("{} to {}".format(start, end))
                temp = []
                for row in range(start, end): # one line lasts 0.1 second
                    temp.append(list(df.iloc[row]))
                    for j, item in enumerate(sc_index_list):
                        row_data[j] += df.iat[row, item]
                raw_data.append(temp)
                count += end - start
                last_data = row_data
                row_data = [t / count for t in row_data]
                X.append(row_data)
                Y.append(label)
                if end == rows:
                    break
        else: # simply get the original data line by line
            with open(file_name, "r") as fin:
                header = fin.readline() # ingore the header
                for line in fin:
                    # ignore the timestamp column
                    val_row = [int(val) for val in line.strip().split(",")[1:]]
                    X.append(val_row)
                    Y.append(label)
                    raw_data.append([val_row])
        return X, Y, raw_data

    def test(self, app_name, file_name, model, save_raw_classified, save_classifier_testdata):
        print("Testing {}".format(file_name))
        testX, testY, raw_data = self.get_data(app_name, app_name, file_name)
        predictY = model.predict(testX)
        # print(len(raw_data), len(testY), len(predictY))
        for i in range(len(testY)):
            if save_raw_classified:
                self.append_file(raw_data[i], f'classified/{predictY[i]}.csv', predictY[i])
            if save_classifier_testdata:
                self.append_classifier_data(testX[i], f'data/classifier_data/data.csv')
                self.append_classifier_data([testY[i]], f'data/classifier_data/label.csv')
        for i in range(len(predictY)):
            if predictY[i] in self.transform_raw and testY[i] in self.transform_raw[predictY[i]]:
                predictY[i] = testY[i]
            if predictY[i] != testY[i]:
                print("Application {} was wrongly classified to be {}.".format(testY[i], predictY[i]))
        return testY, predictY
    
    def append_file(self, data, file_name, app=None):
        if not os.path.isfile(file_name) and app:
            self.init_file(app)
        #print(len(data))
        for item in data:
            exportCSV(item, file_name)
    
    def append_classifier_data(self, data, file_name):
        exportCSV(data, file_name)
    
    def init_file(self, app):
        if app:
            folder = 'classified'
            os.makedirs(folder, exist_ok=True)
            file_name = '{}/{}.csv'.format(folder, app)
            with open(file_name, "w") as fout:
                template = app
                if app in self.transform_raw:
                    template = list(self.transform_raw[app])[0]
                with open(f"shaped-transformed/{template}/{template}-1_freqvector.csv") as fin:
                    fout.write(fin.readline().strip() + "\n")
    
    def train(self, application_list, new_app=None, save_classifier_traindata=False):
        # prepare training data for training classifier
        for app_name in application_list:
            if new_app is not None and app_name != new_app:
                continue
            for file_num in range(1, self.num_container * 3 + 1):
                if file_num % 3 == 1:
                    file_name = 'shaped-transformed/{}/{}-{}_freqvector.csv'.format(app_name, app_name, file_num)
                    X, Y, _ = self.get_data(app_name, app_name, file_name)
                    self.data.extend(X)
                    self.labels.extend(Y)
        if self.algorithm == 'randomforest':
            clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0, n_jobs=-1)
        elif self.algorithm == 'extratrees':
            clf = ExtraTreesClassifier(n_estimators=self.n_estimators, random_state=0, n_jobs=-1)
        elif self.algorithm == 'SVC':
            clf = SVC(random_state=0)
        if save_classifier_traindata:
            for item in self.data:
                self.append_classifier_data(item, f'data/classifier_data/train_data.csv')
            for item in self.labels:
                self.append_classifier_data([item], f'data/classifier_data/train_label.csv')
        # transform the raw data using the transform_dict:
        if len(self.transform_dict):
            for i in range(len(self.labels)):
                if self.labels[i] in self.transform_dict:
                    self.labels[i] = self.transform_dict[self.labels[i]]
        clf = clf.fit(self.data, self.labels)
        pickle.dump(clf, open(self.clf_file, 'wb'))
        return clf
        
    def test_all(self, application_list, new_app=None, save_raw_classified=False, save_classifier_traindata=False, save_classifier_testdata=False, retrain=False):
        if save_raw_classified:
            for item in set(self.labels):
                self.init_file(item)
        
        if not retrain and os.path.isfile(self.clf_file):
            with open(self.clf_file, "rb") as input_file:
                clf = pickle.load(input_file)
            print("Restored classifer")
        else:
            clf = self.train(application_list, save_classifier_traindata=save_classifier_traindata)
        
        if not self.has_printed:
            print(clf)
            self.has_printed = True
        true_labels = []
        predicted = []
        for app_name in application_list:
            if new_app is not None and app_name != new_app:
                continue
            # training data for the autoencoder
            for file_num in range(1, self.num_container * 3 + 1):
                if file_num % 3 != 1:
                    file_name = 'shaped-transformed/{}/{}-{}_freqvector.csv'.format(app_name, app_name, file_num)
                    testY, predictY = self.test(app_name, file_name, clf, save_raw_classified, save_classifier_testdata)
                    true_labels.extend(testY)
                    predicted.extend(predictY)
            # testing data for the autoencoder
            for file_num in range(1, self.num_container + 1):
                file_name = 'shaped-transformed/{}/{}-{}_freqvector_test.csv'.format(app_name, app_name, file_num)
                testY, predictY = self.test(app_name, file_name, clf, False, save_classifier_testdata)
                true_labels.extend(testY)
                predicted.extend(predictY)
        print(classification_report(true_labels, predicted))
        return true_labels, predicted
