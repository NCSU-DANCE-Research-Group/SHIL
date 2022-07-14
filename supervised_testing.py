from sklearn.metrics import confusion_matrix
from training_util import USE_10MS
import numpy as np
from exportCSV import exportCSV

def evaluate(app_time, predicted, predicted_probs, app, file_num):
    """
    Get and save summary results
    """
    times = app_time[app][int(file_num) - 1]
    anomalous_sample_start = int(times[0])
    anomalous_sample_shell = int(times[1])
    # anomalous_sample_stop = int(times[2])
    predicted = list(map(int, predicted))
    print(len(predicted))
    print(times)
    if USE_10MS:
        sample_rate = 0.01
    else:
        sample_rate = 0.1  # seconds
    truth_labels = np.zeros(len(predicted))
    truth_labels[anomalous_sample_start:].fill(1)

    lead_sample = anomalous_sample_shell + 1
    detected_samples = []
    for i in range(anomalous_sample_start, min(anomalous_sample_shell + 1, len(predicted))):
        if predicted[i] == 1:
            lead_sample = i
            print(f"first anomalous sample: {lead_sample}")
            break
    lead_time = sample_rate * (anomalous_sample_shell - lead_sample)
    is_detected = 1
    if lead_time < 0:
        is_detected = 0
        lead_time = 0
    else:
        for i in range(lead_sample, min(anomalous_sample_shell + 1, len(predicted))):
            if predicted[i] == 1:
                detected_samples.append(i)
    tn, fp, fn, tp = confusion_matrix(
        truth_labels, predicted, labels=[0, 1]).ravel()
    print(tn, fp, fn, tp)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    data = [app, fpr * 100, tpr * 100, fp, tp,
            fn, tn, lead_time, is_detected * 100]
    # save results to a file
    exportCSV(data, "testing-res.csv")
    exportCSV([app], "detected.csv")
    exportCSV(predicted, "predicted.csv")
    exportCSV(detected_samples, "detected.csv")
    exportCSV(predicted_probs, "probabiltity.csv")