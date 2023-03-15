from data_obj import DataObj
import numpy as np
import matplotlib.pyplot as plt
from phd import viz
import scipy
import matplotlib.patches as patches


"""
Used for removing outliers from 2d calibration arrays

"""


colors, swatches = viz.phd_style()

name = ".//peacoq_results//Wire_1//41mV_15.2uA//2d//calibration_results_2d_02.09.2022_16.23.27.json"

data_2d = DataObj(name)

data_2d.medians = np.array(data_2d.medians)

# print(np.shape(data_2d.medians))
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# data_2d.medians[4, 21] = 0
# data_2d.medians[4, 28:] = 0
# data_2d.medians[5, 28:] = 0

# ax.imshow(data_2d.medians)



mask = np.roll(np.roll(np.eye(len(data_2d.medians)),3,axis=1),-3,axis=0).astype('bool')
mask2 = np.roll(np.roll(np.eye(len(data_2d.medians)),2,axis=1),-3,axis=0).astype('bool')
mask3 = np.roll(np.roll(np.eye(len(data_2d.medians)),2,axis=1),-2,axis=0).astype('bool')
medians = np.array(data_2d.medians)
medians[mask] = 0
medians[mask2] = 0
medians[mask3] = 0
# medians[0:7,27:] = 0
# medians[0:7,:14] = 0
medians[0:6, 0:] = 0

def create_triangle_mask(size, top_cutoff, diagonal_cutoff):
    b = np.zeros((size,size)).astype('bool')
    b[np.triu_indices(size,diagonal_cutoff)] = True
    b[:top_cutoff,:] = False
    # special_mask = np.triu_indices(150,30) | np.ones((150,150))[15:,:]
    # full[b] = .2
    return b

b = create_triangle_mask(150, 10, 10)


fig,ax = plt.subplots(1,1,figsize=(6,6),dpi=100)
medians_show = medians.copy()
medians_show[b] = 0.2
plt.imshow(medians_show)

medians_slice = medians[b]

edge = medians[:,-1]
# edge_t = print(np.shape(medians[:,-5:]))
edge = np.average(medians[:,-5:],axis=1)



down = np.multiply.outer(edge, np.ones(150))
fig,ax = plt.subplots(1,1,figsize=(6,6),dpi=100)
plt.imshow(down)
plt.title("down")
down_slice = down[b]

side = np.multiply.outer(np.ones(150), edge)
fig,ax = plt.subplots(1,1,figsize=(6,6),dpi=100)
plt.imshow(side)
plt.title("side")
side_slice = side[b]
# print(len(side_slice))
# print(len(side.flatten()))

A = np.array([side_slice,down_slice]).T
B = medians_slice


# find the scalars inside the x vector (a, b) so that the 2d grid made from a*side + b*down
# results in the best approximation of the original data 2d grid.
print(len(A))
print(len(B))
x, res, rnk, s = scipy.linalg.lstsq(A,B)
print(x)





fig, ax = plt.subplots(1,3, figsize=(26,6), dpi=200)

medians[medians==0] = np.nan
medians_ = ax[0].imshow(medians,vmin=-.02,vmax=0.12)
ax[0].set_ylabel("t' (ns)")
ax[0].set_xlabel("t'' (ns)")
ax[0].set_title(f"collected data", fontsize=14)
rect = patches.Rectangle((144, 0), 5, 150, linewidth=1, edgecolor='r', facecolor='none')
# Add the patch to the Axes
ax[0].add_patch(rect)

outside = create_triangle_mask(150,6,7)
output = np.zeros((150,150))
output[outside] = (side*x[0] + down*x[1])[outside]
# output[b] = 0
output[output==0] = np.nan
outputs_ = ax[1].imshow(output,vmin=-.02,vmax=0.12)
ax_small = ax[1].inset_axes([0.1, 0.1, 0.15, 0.15])
ax_small.imshow(down)
ax[1].text(42, 126, "+", color='black', fontsize=19)
ax_small = ax[1].inset_axes([0.44, 0.1, 0.15, 0.15])
ax_small.imshow(side)
ax[1].set_title(f"axese sum from red box, amplitudes: [t', t''] = [{round(x[0],2)},{round(x[1],2)}]", fontsize=14)

difference = (medians-output)
difference[difference==0] = np.nan
diffs_ = ax[2].imshow(difference, cmap='plasma',vmin=-.01,vmax=0.05)
ax[2].set_title(f"difference", fontsize=14)
plt.colorbar(outputs_, label="offset (ns)")
plt.colorbar(medians_, label="offset (ns)")
cbar_diffs = plt.colorbar(diffs_, label='offset (ns)')




# Now instead of just using the t' axis data for the t'' axis, let's
# subtract the t' from the 2d grid, and use the result to construct a t'' axis.
# After the 2d subtraction, this requires an averaging step to bring the
# grid back down to a vector. (project the grid down/up/vertically).
side_2d_subtracted = medians - down
side_mean = np.nanmean(side_2d_subtracted, axis=0)

fig, ax = plt.subplots(1,2, figsize=(17,6), dpi=100)
side_2d_subtracted_ = ax[0].imshow(side_2d_subtracted) #,vmin=-.02,vmax=0.07)
difference_2d_subtracted = side_2d_subtracted - np.multiply.outer(np.ones(150), side_mean)
total_diffs = ax[1].imshow(side_2d_subtracted - np.multiply.outer(np.ones(150), side_mean), cmap='plasma', vmin=-.01,vmax=0.05)
plt.colorbar(total_diffs, label="offset (ns)")
plt.colorbar(side_2d_subtracted_, label="offset (ns)")


data_2d.medians = output

print(name[:-5] + "_corrected_axese_sum_simple_linear.json")
data_2d.export(name[:-5] + "_corrected_axese_sum_simple_linear.json")

# data_2d.medians = output
#
# print(name[:-5] + "_corrected_axese_sum_simple_linear.json")
# data_2d.export(name[:-5] + "_corrected_axese_sum_simple_linear.json")
