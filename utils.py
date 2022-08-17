import os

import numpy as np


def number_manager(number):
    if number > 1e3:
        val = f"{round(number/1000,1)} KCounts/s"
    if number > 1e6:
        val = f"{round(number/1e6,1)} MCounts/s"
    if number > 1e9:
        val = f"{round(number/1e9,1)} GCounts/s"
    return val


def sleeper(t, iter, tbla=0):
    # time.sleep(t)
    for i in range(1000):
        q = np.sin(np.linspace(0, 5, 1000000))
    print("sleeping for: ", t)
    print("tbla is: ", tbla)
    return t


def get_file_list(path):
    ls = os.listdir(path)
    files = []
    for item in ls:
        if os.path.isfile(os.path.join(path, item)):
            files.append(item)
    return files
