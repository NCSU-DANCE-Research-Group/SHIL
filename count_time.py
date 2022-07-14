from timeit import default_timer as timer
from exportCSV import exportCSV

def count_time(last_time, message):
    diff = timer() - last_time
    time_elapsed_str = f"{message}: {diff}"
    print(time_elapsed_str)
    exportCSV([diff], "time_log.txt")
    last_time += diff
    return last_time