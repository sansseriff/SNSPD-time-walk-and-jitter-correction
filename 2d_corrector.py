from data_obj import DataObj
import numpy as np
import matplotlib.pyplot as plt
from phd import viz


"""
Used for removing outliers from 2d calibration arrays

"""


colors, swatches = viz.phd_style()

name = ".//peacoq_results//Wire_1//41mV_15.2uA//2d//calibration_results_2d_02.09.2022_16.23.27.json"

data_2d = DataObj(name)

data_2d.medians = np.array(data_2d.medians)

print(np.shape(data_2d.medians))
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
data_2d.medians[4, 21] = 0
data_2d.medians[4, 28:] = 0
data_2d.medians[5, 28:] = 0

ax.imshow(data_2d.medians)

print(name[:-5] + "_corrected.json")
data_2d.export(name[:-5] + "_corrected.json")
