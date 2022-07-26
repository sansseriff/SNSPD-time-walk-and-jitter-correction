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
    def __init__(self, diffs: np.ndarray,
                 max: int,
                 inter_pulse_time: float,
                 figures: bool = False,
                 main_hist_downsample = 1):
        """
        :param diffs: SNSPD tag minus previous laser-based time
        :param max: maximum time on t' vs d curve in picoseconds
        :param inter_pulse_time: time bewteen laser pulses in nanoseconds
        :param figures: turn off or on figures
        :param main_hist_downsample: larger hists are slow to generate and zoom. Overall downsample makes them faster
        """
        print("inter pulse time is: ", inter_pulse_time)
        self.diffs = diffs
        self.mask_type = "nan"
        self.max = max  # in picoseconds
        self.inter_pulse_time = inter_pulse_time
        self.bins = np.linspace(0, max/1000, max + 1)
        self.d = main_hist_downsample
        # bins is in nanoseconds, with 1000 bins per nanosecond (discretized by 1 ps)
        # print("bins: ", self.bins[:100])
        self.hist, self.bins = np.histogram(self.diffs, self.bins)
        self.total_hist_counts = np.sum(self.hist)
        self.hist = self.hist/self.total_hist_counts
        self.figures = figures

        pulse_numer = int((max/1000)/inter_pulse_time)
        self.pulses = np.array([i * self.inter_pulse_time for i in range(1, pulse_numer)])

        if self.figures:
            plt.figure()
            self.color_map = cm.get_cmap("viridis")
            print("Mask Generator: length of bins", len(self.bins))
            print("Mask Generator: length of hist: ", len(self.hist))
            plt.plot(self.bins[:-1:self.d], self.hist[::self.d], color="black")
            # print("pulses: ", self.pulses[:40])

    def apply_mask_from_period(self, adjustment_prop: float = 0,
                               adjustment_mult: float = 0,
                               bootstrap_errorbars: bool = False,
                               kde_bandwidth: float = 0.025,
                               low_cutoff = 0):
        """
        when the period of the laser pulse train is significantly larger than the maximimum jitter of the detector,
        it is straightforward to divide up the measurments into sections the length of the laser period. This way for
        each t' a unique distribution of events can be analyzed. Measurements of delta-t are separated out by what bin
        they fall into where each bin is indexed by the number of laser periods since delta_t = 0.

        :param adjustment_prop: used to distort the spacing of bins for short t' values. Useful when the delays
        start to approach the period of the laser. Set to zero for no adjustment
        :param adjustment_mult: used to distort the spacing of bins for short t' values. Useful when the delays
        start to approach the period of the laser. Set to zero for no adjustment
        :param bootstrap_errorbars: use bootstrap method to generate error bars for the median (\tilde{d}) and width
        of the distributions.
        :param kde_bandwidth: smoothing factor for kernel density estimation (KDE). KDE is used to measure the FWHM
        of distributions along the t' vs delay curve.
        :param low_cutoff: number of skipped pulses at the beginning (small t'). For very short time scales there
        can be nonsensical distributions of counts that should ignored
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
                plt.axvspan(time_start, time_end, alpha=0.3, color=self.color_map(i / 120))
                for x_location in self.pulses:
                    plt.axvline(x_location, color="black",ls=':', alpha=0.8)
            t_start[i] = time_start
            t_end[i] = time_end

        if self.figures:
            plt.xlim(0, self.bins[-1])
            # plt.yscale('log')
            plt.grid()

        ##############################
        ##############################
        counts = np.zeros(len(self.pulses))
        t_prime = np.zeros(len(self.pulses))
        ranges = np.zeros(len(self.pulses))
        r_widths = np.zeros(len(self.pulses))
        l_widths = np.zeros(len(self.pulses))
        offsets = np.zeros(len(self.pulses))
        fwhm_ranges = np.zeros(len(self.pulses))

        # kernel density estimation is used for measuring width (sigma) of histograms
        kde = KernelDensity(bandwidth=kde_bandwidth, kernel='gaussian')
        # bw has units of nanoseconds
        # so bandwidth=0.25 results in a kind of smoothing with spread of 25 picoseconds

        for jj, pulse in enumerate(tqdm(self.pulses)):
            if jj < low_cutoff:
                continue
            section = self.diffs[(self.diffs > t_start[jj]) & (self.diffs < t_end[jj])]
            section_org = section
            section = section[:50000]  # don't need more stats than this for accurate mean and sigma.
            if len(section) > 10:  # if more than 5 counts, try to fit the counts to a guassian
                bins_idx_left = np.searchsorted(self.bins, t_start[jj], side="left") - 1
                bins_idx_right = np.searchsorted(self.bins, t_end[jj], side="left")
                section_bins = self.bins[bins_idx_left:bins_idx_right]
                single_distribution_hist, section_bins = np.histogram(section, section_bins, density=True)
                if bootstrap_errorbars:
                    fwhm_l_ranges, fwhm_r_ranges = \
                        self.bootstrap_kde(section, kde, self.bins, bins_idx_left, bins_idx_right, 18)
                    fwhm_ranges[jj] = math.sqrt(fwhm_l_ranges ** 2 + fwhm_r_ranges ** 2)

                kde.fit(section[:8000, None])
                logprob = kde.score_samples(section_bins[:, None])
                peaks, _ = signal.find_peaks(np.exp(logprob)) # returns location of the peaks in picoseconds
                # (because section_bins is in units of picoseconds)

                # There might be more than the main peak due to the noise floor. Want statistics on largest peak
                # max: the index of the largest amplitude peak. Found using:
                # (logprob[peaks] is a list of the heights of the peaks)
                # (np.argmax(logprob[peaks]) is the index of the largest peak in this list of peaks)
                # (peaks[np.argmax(logprob[peaks])] is the location in picoseconds of largest peak
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
                t_prime[jj] = pulse
                counts[jj] = len(section_org)
                offsets[jj] = np.median(section) - pulse

                if self.figures:
                    scale_factor = (len(section)/self.total_hist_counts)/1000
                    normed_single_distribution_hist = single_distribution_hist*scale_factor
                    normed_logprob_hist = np.exp(logprob)*scale_factor
                    plt.plot(section_bins[1::self.d], normed_single_distribution_hist[::self.d], color='purple')
                    plt.plot(section_bins[::self.d], normed_logprob_hist[::self.d], alpha=1, color='red')
                    plt.axvspan(r_widths[jj], l_widths[jj], color='green', alpha=0.3)
                    # plt.axvline(x=Mu2, color = 'red')
                    # plt.axvline(x=np.median(section), color = 'green')
                    # plt.axvline(offsets[jj], color='blue', alpha=0.4)
                    # plt.axvline(x[jj], color='red', alpha=0.3)

        # adjust the offsets graph so that the last fifth is centered around zero
        fifth = int(len(offsets)/5)
        offsets = offsets - np.average(offsets[-fifth:])
        widths = r_widths - l_widths

        if self.figures:
            plt.figure()
            plt.grid()
            plt.plot(t_prime, offsets)
            plt.plot(t_prime[-fifth:], offsets[-fifth:], lw=2, color='red')
            plt.title("delay vs t_prime curve")

            plt.figure()
            plt.grid()
            plt.plot(t_prime, (r_widths - l_widths)*1000)
            plt.ylabel("FWHM (ps)")
            plt.xlabel("t_prime")
            plt.title("FWHM vs t_prime curve")


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

        When the period of the laser approaches the maximum jitter of the detector or the maximum uncorrected delays,
        then seperating out the measurments by t' is less straightforward than with the apply_mask_from_period method.
        However, the separation may still be possible if the max jitter is similar magnitude to the laser period.
        Here it's not assumed that the counts corresponding to t' = x*laser_period are found in a time bin starting at
        delta_t = x*laser_period and ending at (x+1)*laser_period. At least not for small t'.

        For large t' (say 200 to 1000 ns for generic WSi SNSDPS), this can be assumed.

        This method finds a list of peak locations in a histogram of counts vs delta_t. Then it takes a peak in the far
        field (t' ~ 100 ns or larger) for which it can deduce the corresponding t' value. Then it works backward
        toward shorter t' matching peaks with t' values and defines bins for separating out the counts by
        corresponding t'.


        :param down_sample: Peaks are found from a histogram that is lower resolution than the native 1ps. These are
        scaled down by the factor down_sample. 
        """
        self.mask_type = "peaks"
        self.down_sample = down_sample
        self.bins_peaks = np.linspace(0, 200, self.max // down_sample + 1)  # lower res for peak finding
        hist_peaks, self.bins_peaks = np.histogram(self.diffs, self.bins_peaks, density=True)
        # then you gotta go out to far field time and find agreement bewteen laser timing and pulse timing

        peaks, props = find_peaks(hist_peaks, height=0.01)
        peaks = np.sort(peaks)
        print(self.bins[peaks * 10][:10])
        peaks_rh = self.bins[peaks * down_sample]
        pulses = self.pulses[self.pulses < 1000]
        peaks_rh = peaks_rh[peaks_rh < 1000]
        if self.figures:
            plt.plot(self.bins_peaks[:-1], hist_peaks, color="blue")
            plt.vlines(peaks_rh, 0.01, 1, color="purple", alpha=0.8)

        pulses = np.sort(pulses)
        peaks_rh = np.sort(peaks_rh)
        peaks_rh = peaks_rh.tolist()
        pulses = pulses.tolist()

        while len(peaks_rh) != len(pulses):
            pulses.pop(0)

        offsets = []
        pulses_x = []
        # work backward through the peaks list to define bins
        for i in tqdm(range(len(peaks_rh) - 2, 0, -1)):
            # print(i)
            # j = j + 1
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
                plt.plot(mini_bins[:-1], mini_hist/(down_sample/4), color='purple')
                plt.axvspan(bin_left_choked, bin_right_choked, alpha=0.3, color=self.color_map(i / len(peaks_rh)))
                # plt.axvline(offset, color = 'red')
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
            plt.plot(self.pulses_x, self.offsets)
            plt.xlabel("time (ns)")
            plt.ylabel("offsets (ps)")
            plt.plot(self.pulses_x[-40:], self.offsets[-40:], lw=2.4, color="red")
            plt.grid()

    def plot_tprime_offset(self):
        plt.figure()
        plt.plot(self.pulses_x, self.offsets)
        plt.xlabel("time (ns)")
        plt.ylabel("offsets (ps)")
        plt.plot(self.pulses_x[-40:], self.offsets[-40:], lw=2.4, color="red")
        plt.grid()


