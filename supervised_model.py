import numpy as np

# load the testing data after triggering the attack, and before the attack success time 
def get_training_data(app_time, file_num_list, application_list):
    data = []
    labels = []
    for app in application_list:
        for file_num in file_num_list:
            with open(f"shaped-transformed/{app}/{app}-{file_num}_freqvector_test.csv") as fin:
                times = app_time[app][file_num - 1]
                anomalous_sample_start = int(times[0])
                anomalous_sample_shell = int(times[1])
                fin.readline() # skip the header
                counter = 0
                for line in fin:
                    counter += 1
                    if counter <= 1200:
                        data.append(list(map(float, line.strip().split(",")))[1:]) # ignore the first column
                        labels.append(app)
    return np.array(data), np.array(labels)

def evaluate(clf, X_train, y_train, X_test, y_test):
    # training
    clf = clf.fit(X_train, y_train)

    # testing
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print(f"TPR: {TPR}")


    print(f"FPR: {FPR}")
    print(f"mean TPR: {np.mean(TPR) * 100:.2f}%")
    print(f"mean FPR: {np.mean(FPR) * 100:.2f}%")
    
    return clf
