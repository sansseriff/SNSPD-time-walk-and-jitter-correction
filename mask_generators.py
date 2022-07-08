import numpy as np
from scipy.signal import find_peaks
from matplotlib import cm
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from scipy import signal
import math
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
    def __init__(self, diffs: np.ndarray, max: int, inter_pulse_time: float, figures: bool = False):
        """
        :param diffs: SNSPD tag minus previous laser-based time
        :param max: maximum time on t' vs d curve in ps
        :param inter_pulse_time: time bewteen laser pulses
        :param figures: turn off or on figures
        """
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

    def apply_mask_from_period(self, adjustment_prop: float, adjustment_mult: float, bootstrap_errorbars: bool = False):
        """
        :param adjustment_prop: used to distort the spacing of bins for short t' values. Useful when the delays
        start to approach the period of the laser
        :param adjustment_mult: used to distort the spacing of bins for short t' values. Useful when the delays
        start to approach the period of the laser
        :param bootstrap_errorbars: use bootstrap method to generate error bars for the median (\tilde{d}) and width
        of the distributions
        """
        self.mask_type = "period"
        adjustment = [(i ** adjustment_mult) * adjustment_prop for i in range(20)]
        adjustment.reverse()
        st = 1

        # make t_start and t_end arrays.
        # Used for settings the bounds used to chop up the original giant histogram
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

        if self.figures:
            plt.xlim(0, 120)
            plt.yscale('log')
            plt.grid()

        ##############################
        ##############################
        # Set up for fitting histograms
        # avgOffset = np.zeros(len(self.pulses))
        # stdOffset = np.zeros(len(self.pulses))
        # background = np.zeros(len(self.pulses))
        counts = np.zeros(len(self.pulses))
        t_prime = np.zeros(len(self.pulses))
        ranges = np.zeros(len(self.pulses))
        r_widths = np.zeros(len(self.pulses))
        l_widths = np.zeros(len(self.pulses))
        median = np.zeros(len(self.pulses))
        fwhm_ranges = np.zeros(len(self.pulses))

        kde = KernelDensity(bandwidth=.025, kernel='gaussian')  # bw has units of nanoseconds
        for jj, pulse in enumerate(self.pulses):

            if jj >= 160:  # avoiding some error I don't want to deal with
                break
            section = self.diffs[(self.diffs > t_start[jj]) & (self.diffs < t_end[jj])]
            section_org = section
            section = section[:50000]  # don't need more stats than this.
            if len(section) > 10:  # if more than 5 counts, try to fit the counts to a guassian
                # q = time.time()

                bins_idx_left = np.searchsorted(self.bins, t_start[jj], side="left") - 1
                bins_idx_right = np.searchsorted(self.bins, t_end[jj], side="left")
                section_bins = self.bins[bins_idx_left:bins_idx_right]

                single_distribution_hist, section_bins = np.histogram(section, section_bins, density=True)
                # print("size of section: ", len(section))
                # print("length of bins: ", len(self.bins))

                if bootstrap_errorbars:
                    fwhm_l_ranges, fwhm_r_ranges = \
                        self.bootstrap_kde(section, kde, self.bins, bins_idx_left, bins_idx_right, 18)
                    fwhm_ranges[jj] = math.sqrt(fwhm_l_ranges ** 2 + fwhm_r_ranges ** 2)

                kde.fit(section[:8000, None])
                logprob = kde.score_samples(section_bins[:, None])
                peaks, _ = signal.find_peaks(np.exp(logprob))
                max = peaks[np.argmax(logprob[peaks])]
                width, w_height, lw, rw = signal.peak_widths(np.exp(logprob), np.array([max]), rel_height=0.5)
                r_widths[jj] = np.interp(
                    rw, np.arange(0, bins_idx_right - bins_idx_left),
                    self.bins[bins_idx_left:bins_idx_right])[0]
                l_widths[jj] = np.interp(
                    lw, np.arange(0, bins_idx_right - bins_idx_left),
                    self.bins[bins_idx_left:bins_idx_right])[0]

                if bootstrap_errorbars:
                    ranges[jj] = self.bootstrap_median(section, 500) * 4  # 4x sigma for 95% confidence interval

                print("this is rwidth: ", r_widths[jj])
                print("this is lwidth: ", l_widths[jj])

                t_prime[jj] = pulse
                counts[jj] = len(section_org)
                median[jj] = np.median(section) - pulse

                if self.figures:
                    plt.plot(section_bins[1:], single_distribution_hist)
                    plt.plot(section_bins, np.exp(logprob), alpha=1, color='red')
                    plt.axvspan(r_widths[jj], l_widths[jj], color='green', alpha=0.3)
                    # plt.axvline(x=Mu2, color = 'red')
                    # plt.axvline(x=np.median(section), color = 'green')
                    plt.axvline(median[jj], color='blue', alpha=0.4)
                    # plt.axvline(x[jj], color='red', alpha=0.3)


    def bootstrap_kde(self, section, estimator, bins, bins_idx_left, bins_idx_right, number=50):
        lims_l = np.zeros(number)
        lims_r = np.zeros(number)
        for i in range(number):
            sect = np.random.choice(section[:8000], size=len(section))
            estimator.fit(sect[:, None])
            logprob = estimator.score_samples(bins[bins_idx_left:bins_idx_right, None])
            peaks, _ = signal.find_peaks(np.exp(logprob))
            max = peaks[np.argmax(logprob[peaks])]
            _, _, lw, rw = signal.peak_widths(np.exp(logprob), np.array([max]), rel_height=0.5)

            lims_r[i] = np.interp(rw, np.arange(0, bins_idx_right - bins_idx_left), bins[bins_idx_left:bins_idx_right])[
                0]
            lims_l[i] = np.interp(lw, np.arange(0, bins_idx_right - bins_idx_left), bins[bins_idx_left:bins_idx_right])[
                0]

        return np.std(lims_l) * 4, np.std(lims_r) * 4

    def bootstrap_median(self, section, number=100):
        """
        Used for making error bars for the \tilde{d} delay values in the t' vs \tilde{d} curve.
        :param section: delay measurments for a particular t'
        :param number: number of times sections is sampled (bootstrap iterations)
        :return: error estimate
        """
        meds = np.zeros(number)
        for i in range(number):
            meds[i] = np.median(np.random.choice(section, size=len(section)))

        return np.std(meds)

    def apply_mask_from_peaks(self, down_sample):
        """
        :param down_sample: Peaks are found from a histogram that is lower resolution than the native 1ps. These are
        scaled down by the factor down_sample. 
        """
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
        # loop over the peaks and use them to define window bounds that sets of measurements fall into. 
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
        if self.figures:
            plt.figure()
            plt.plot(pulses_x, self.offsets)
            plt.xlabel("time (ns)")
            plt.ylabel("offsets (ps)")
            plt.plot(pulses_x[-40:], self.offsets[-40:], lw=2.4, color="red")
            plt.grid()

    def plot_tprime_offset(self):
        plt.figure()
        plt.plot(self.pulses_x, self.offsets)
        plt.xlabel("time (ns)")
        plt.ylabel("offsets (ps)")
        plt.plot(self.pulses_x[-40:], self.offsets[-40:], lw=2.4, color="red")
        plt.grid()


