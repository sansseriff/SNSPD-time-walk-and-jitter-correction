import numpy as np
from scipy.signal import find_peaks
from matplotlib import cm
import matplotlib.pyplot as plt
from tqdm import tqdm
'''
MASK GENERATORS
the diffs histogram needs to be split up into many time windows, each of which has one guassian of counts
for each t' value. For slower laser rep rate when the laser period is much larger than the maximum jitter observed
from the detector, this is straightforward. generate_masks_from_period is used for this. It just splits up the 
histograms by timing. 

When the laser rep rate is higher, individual guassians may still be visible, but they are much closer together
and they can be shifted by the jitterate effect in such a way to setting the bounds for cutting up the original large
histogram is more challenging. In this case the generate_mask_from_peaks method looks for peaks in the large histogram
(actually a lower res version of it, to minimize unwanted peaks) and defines bounds around each peak. 

For the 4GHz peacoq analysis, the peaks method was used, for the 1 GHz peacoq data, the from_period method should be
sufficient, and likely more reliable. 
'''


class MaskGenerator:
    def __init__(self, diffs, max, inter_pulse_time, figures = False):
        self.diffs = diffs
        self.mask_type = "nan"
        self.max = max  # in picoseconds
        self.inter_pulse_time = inter_pulse_time
        self.bins = np.linspace(0, 200, max + 1)
        self.hist, self.bins = np.histogram(self.diffs, self.bins, density=True)
        self.figures = figures
        self.pulses = np.array([i * self.inter_pulse_time for i in range(1, 400)])
        if self.figures:
            plt.figure()
            print("Mask Generator: length of bins", len(self.bins))
            print("Mask Generator: length of hist: ", len(self.hist))
            plt.plot(self.bins[:-1], self.hist, color="black")

            print("pulses: ", self.pulses[:40])

    def apply_mask_from_period(self, adjustment_prop = 0.00023, adjustment_mult = 2.9):
        self.mask_type = "period"
        adjustment = [(i ** adjustment_mult) * adjustment_prop for i in range(20)]
        adjustment.reverse()
        st = 1
        t_start = np.zeros(len(self.pulses))
        t_end = np.zeros(len(self.pulses))
        for i in range(len(self.pulses)):
            if i < st:
                continue
            time_start = self.pulses[i] - self.inter_pulse_time / 2.1
            time_end = self.pulses[i] + self.inter_pulse_time / 2.1
            if (i >= st) and i < (st + len(adjustment)):
                time_start = time_start + adjustment[i - st]
                time_end = time_end + adjustment[i - st]
            if self.figures:
                map = cm.get_cmap("viridis")
                plt.axvspan(time_start, time_end, alpha=0.3, color=map(i / 120))
                plt.vlines(self.pulses, 0.01, 1, color="orange", alpha=0.8)
            t_start[i] = time_start
            t_end[i] = time_end

    def apply_mask_from_peaks(self, down_sample):
        self.mask_type = "peaks"
        self.down_sample = down_sample
        bins_peaks = np.linspace(0, 200, self.max // down_sample + 1)  # lower res for peak finding
        hist_peaks, bins_peaks = np.histogram(self.diffs, self.bins_peaks, density=True)
        # then you gotta go out to far field time and find agreement bewteen laser timing and pulse timing

        peaks, props = find_peaks(hist_peaks, height=0.01)
        peaks = np.sort(peaks)
        print(self.bins[peaks * 10][:10])
        peaks_rh = self.bins[peaks * down_sample]
        pulses = self.pulses[self.pulses < 1000]
        peaks_rh = peaks_rh[peaks_rh < 1000]
        if self.figures:
            plt.plot(bins_peaks[:-1], hist_peaks, color="blue")
            plt.vlines(peaks_rh, 0.01, 1, color="purple", alpha=0.8)

        pulses = np.sort(pulses)
        peaks_rh = np.sort(peaks_rh)
        peaks_rh = peaks_rh.tolist()
        pulses = pulses.tolist()

        while len(peaks_rh) != len(pulses):
            pulses.pop(0)

        offsets = []
        pulses_x = []
        for i in tqdm(range(len(peaks_rh) - 2, 0, -1)):
            # print(i)
            j = j + 1
            bin_center = peaks_rh[i]
            bin_left = bin_center - (peaks_rh[i] - peaks_rh[i - 1]) / 2
            bin_right = bin_center + (peaks_rh[i + 1] - peaks_rh[i]) / 2

            bin_left_choked = bin_center - (peaks_rh[i] - peaks_rh[i - 1]) / 2.1
            bin_right_choked = bin_center + (peaks_rh[i + 1] - peaks_rh[i]) / 2.1
            mask = (self.diffs > bin_left) & (self.diffs < bin_right)
            mask_choked = (self.diffs > bin_left_choked) & (self.diffs < bin_right_choked)
            bin_tags = self.diffs[mask]
            bin_tags_choked = self.diffs[mask_choked]

            mini_bins = np.linspace(bin_left, bin_right, 50)
            mini_hist, mini_bins = np.histogram(bin_tags_choked, mini_bins)

            offset = np.median(bin_tags)
            if self.figures:
                plt.plot(mini_bins[:-1], mini_hist)
                plt.axvspan(bin_left_choked, bin_right_choked, alpha=0.3, color=map(i / len(peaks_rh)))
                plt.axvline(offset, color = 'red')
            offset = offset - pulses[i]

            offset_choked = np.median(bin_tags_choked) - pulses[i]
            offsets.append(offset_choked)
            pulses_x.append(pulses[i])

        pulses_x = np.array(pulses_x)
        offsets = np.array(offsets)

        sorter = np.argsort(pulses_x)
        self.pulses_x = pulses_x[sorter]
        offsets = offsets[sorter]

        zero_offset = np.mean(offsets[-40:])
        self.offsets = offsets - zero_offset

    def plot_tprime_offset(self):
        plt.figure()
        plt.plot(self.pulses_x, self.offsets)
        plt.xlabel("time (ns)")
        plt.ylabel("offsets (ps)")
        plt.plot(self.pulses_x[-40:], self.offsets[-40:], lw=2.4, color="red")
        plt.grid()
