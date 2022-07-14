"""
Author: Fogo Tunde-Onadele

- getSamples.py
gets sample numbers of requested timestamps, given an initial timestamp in epoch format


// command line args
sys.argv[0]: first epoch timestamp
sys.argv[1]: timestamp of needed samples (hh:mm:ss.mm) (comma separated)
e.g: `python getSamples.py 1565621400825 16:54:00.112,16:54:00.158`

NOTE:
currently assumes the step between timestamps is even == 100ms
"""

from __future__ import division, print_function, absolute_import
from datetime import datetime
import time
import sys

def calculateSamples(first_epoch_time, query_times):
    print(first_epoch_time)
    print(query_times)
    print()

    # parameters
    time_step = 100

    # grab date / convert to datetime
    first_epoch_obj = datetime.utcfromtimestamp(first_epoch_time/1000.0)
    date_string = first_epoch_obj.strftime("%Y-%m-%d")


    # output
    for t in query_times:
        # convert to epoch
        t_obj = datetime.strptime(date_string+" "+t, "%Y-%m-%d %H:%M:%S.%f")

        # calculate sample
        diff = t_obj - first_epoch_obj
        sample = 1 + diff.total_seconds()*1000/time_step

        # print samples
        #print("%s  => %d" % (t, sample))
        print("%d" % sample)

if __name__ == "__main__":
    # input args
    first_epoch_time = int(sys.argv[1].strip())
    query_times = sys.argv[2].split(",")
    calculateSamples(first_epoch_time, query_times)
