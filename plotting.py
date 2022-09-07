import os
import time

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import special
from scipy.interpolate import CubicSpline
from scipy.stats import rv_continuous

from data_obj import DataObj
from utils import number_manager


def checkLocking(Clocks, RecoveredClocks):
    basis = np.linspace(Clocks[0], Clocks[-1], len(Clocks))
    diffs = Clocks - basis
    diffsRecovered = RecoveredClocks - basis
    x = np.arange(0, len(diffs))
    # fig1 = plt.figure()
    plt.figure()
    plt.plot(x, diffs)
    plt.plot(x, diffsRecovered)
    plt.title("Raw Clock and PLL Clock")
    # plt.plot(x,diffsRecovered)
    # plt.ylim(-1000, 1000)


def make_histogram(data_tags, nearest_pulse_times, delay, stats, figures):
    diffsorg = data_tags[1:-1] - nearest_pulse_times[1:-1]
    guassDiffs = diffsorg + delay

    guassEdges = np.linspace(
        int(-stats["inter_pulse_time"] * 1000 * 0.5),
        int(stats["inter_pulse_time"] * 1000 * 0.5),
        4001,
    )  # 1 period width
    guassHist, guassBins = np.histogram(guassDiffs, guassEdges, density=True)
    gaussianBG = gaussian_bg(
        a=guassDiffs.min() / 1000, b=guassDiffs.max() / 1000, name="gaussianBG"
    )
    start = time.time()
    scalefactor = 1000
    guassStd2, guassAvg2, back, flock, fscale = gaussianBG.fit(
        guassDiffs[-30000:] / scalefactor, floc=0, fscale=1
    )
    guassStd = np.std(guassDiffs[-30000:])
    end = time.time()
    print("time of fit: ", end - start)
    guassStd2 = guassStd2 * scalefactor
    guassAvg2 = guassAvg2 * scalefactor

    if figures:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(guassBins[1:], guassHist)
        ax[1].plot(guassBins[1:], guassHist)
        ax[1].set_yscale("log")
        ax[0].set_title("histogram of counts wrt clock")


def plot_and_analyze_histogram(
    r, data, corrected_diffs, uncorrected_diffs, uncorrupted_diffs, corr_params, edge
):
    # resolution or smoothing of CubicSpline is determined by the resolution
    # of these _interp arrays
    r.hist_bins_interp = np.linspace(
        r.hist_bins[0],
        r.hist_bins[-1],
        corr_params["spline_interpolation_resolution"],
    )

    # lower res histograms to be used with CubicSpline
    r.hist_corrected_interp, r.hist_bins_interp = np.histogram(
        corrected_diffs, r.hist_bins_interp, density=True
    )
    r.hist_uncorrected_interp, r.hist_bins_interp = np.histogram(
        uncorrected_diffs, r.hist_bins_interp, density=True
    )

    r.corrected_mean = np.mean(corrected_diffs)
    r.corrected_median = np.median(corrected_diffs)

    r.uncorrected_mean = np.mean(uncorrected_diffs)
    r.uncorrected_media = np.median(uncorrected_diffs)

    r.uncorrupted_median = np.median(uncorrupted_diffs)
    r.uncorrupted_mean = np.mean(uncorrupted_diffs)
    r.uncorrupted_number = len(uncorrupted_diffs)
    r.uncorrupted_std = np.std(uncorrupted_diffs)

    r.hist_bins_interp = (
        r.hist_bins_interp[1:] - (r.hist_bins_interp[1] - r.hist_bins_interp[0]) / 2
    )

    spline_corrected = CubicSpline(
        r.hist_bins_interp,
        r.hist_corrected_interp,
    )
    spline_uncorrected = CubicSpline(
        r.hist_bins_interp,
        r.hist_uncorrected_interp,
    )

    r.fwhm_corrected = LineObj(
        r.hist_bins_interp,
        r.hist_corrected_interp,
        0.5,
        500,
        "#eb4034",
        "-",
        label="FWHM corrected",
    )
    r.fwhm_uncorrected = LineObj(
        r.hist_bins_interp,
        r.hist_uncorrected_interp,
        0.5,
        500,
        "#eb4034",
        "--",
        label="FWHM uncorrected",
    )

    r.fwtm_corrected = LineObj(
        r.hist_bins_interp,
        r.hist_corrected_interp,
        1 / 10,
        500,
        "#c92eb2",
        "-",
        label="FW(1/10)M corrected",
    )
    r.fwtm_uncorrected = LineObj(
        r.hist_bins_interp,
        r.hist_uncorrected_interp,
        1 / 10,
        500,
        "#c92eb2",
        "--",
        label="FW(1/10)M uncorrected",
    )

    r.fwhum_corrected = LineObj(
        r.hist_bins_interp,
        r.hist_corrected_interp,
        1 / 100,
        500,
        "#3d2ec9",
        "-",
        label="FW(1/100)M corrected",
    )
    r.fwhum_uncorrected = LineObj(
        r.hist_bins_interp,
        r.hist_uncorrected_interp,
        1 / 100,
        500,
        "#3d2ec9",
        "--",
        label="FW(1/100)M uncorrected",
    )

    print("FWHM corrected: ", r.fwhm_corrected.roots(-1, 0))
    print("FWTM corrected: ", r.fwtm_corrected.roots(-1, 0))
    print("FW100M corrected: ", r.fwhum_corrected.roots(-1, 0))
    print()
    print("FWHM uncorrected: ", r.fwhm_uncorrected.roots(-1, 0))
    print("FWTM uncorrected: ", r.fwtm_uncorrected.roots(-1, 0))
    print("FW100M uncorrected: ", r.fwhum_uncorrected.roots(-1, 0))

    # if corr_params["view"]["show_figures"]:
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    ax.plot(
        r.hist_bins,
        r.hist_uncorrected,
        label="uncorrected raw data",
        alpha=0.3,
        color="orange",
    )
    ax.plot(
        r.hist_bins,
        r.hist_corrected,
        label="corrected raw data",
        alpha=0.3,
    )
    ax.plot(
        r.hist_bins,
        spline_corrected(r.hist_bins),
        "k",
        alpha=1,
        label="cubic spline corrected",
    )
    ax.plot(
        r.hist_bins,
        spline_uncorrected(r.hist_bins),
        "k",
        alpha=0.3,
        label="cubic spline uncorrected",
        ls="--",
    )
    ax.plot(
        r.hist_bins,
        r.hist_uncorrupted,
        color="green",
        label="uncorrupted tags",
        alpha=0.2,
    )

    ax.axvline(x=r.uncorrupted_mean, color="green")
    ax.axvline(x=r.uncorrupted_median, color="green", ls="--")

    # data does not exit in local scope. And the global object is shared...
    print("raw count rate: ", data.stats["count_rate"])

    print(f"count rate: {number_manager(data.stats['count_rate'])}")
    title = f"count rate: {number_manager(data.stats['count_rate'])}"
    ax.set_title(title)

    r.hist_spline_corrected = spline_corrected(r.hist_bins)
    r.hist_spline_uncorrected = spline_uncorrected(r.hist_bins)

    line_objs = [
        r.fwhm_uncorrected,
        r.fwtm_uncorrected,
        r.fwhum_uncorrected,
        r.fwhm_corrected,
        r.fwtm_corrected,
        r.fwhum_corrected,
    ]
    for line_obj in line_objs:
        label = f"{line_obj.label} {round(line_obj.roots(-1, 0), 1)} ps"
        ax.hlines(
            line_obj.level,
            line_obj.root_list[-1],
            line_obj.root_list[0],
            label=label,
            color=line_obj.color,
            ls=line_obj.line_style,
        )
    ax.grid()
    ax.set_xlim(-edge, edge)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("normalized counts")
    plt.legend(fancybox=True, frameon=False, loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 0.1)

    ax_lin = inset_axes(ax, width="30%", height=1, loc=1)
    ax_lin.plot(r.hist_bins, spline_corrected(r.hist_bins), "k", alpha=1)
    ax_lin.plot(
        r.hist_bins,
        spline_uncorrected(r.hist_bins),
        "k",
        alpha=0.3,
        ls="--",
    )
    ax_lin.set_xlim(-edge, edge)
    if corr_params["output"]["save_fig"]:
        rg = corr_params["output"]["data_file_snip"]
        save_name = (
            f"{data.params['data_file'][rg[0] : rg[-1]]}_{corr_params['type']}.png"
        )

        save_name = os.path.join(corr_params["output"]["save_location"], save_name)
        print("saving figure to : ", save_name)
        plt.savefig(save_name)

    if not corr_params["view"]["show_figures"]:
        plt.close(fig)

    return r


def guassian_background(x, sigma, mu, back, l, r):
    "d was found by symbolically integrating in mathematica"
    n = back + (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * (((x - mu) / sigma) ** 2)
    )
    d = 0.5 * (
        2 * back * (-l + r)
        + special.erf((-l + mu) / (np.sqrt(2) * sigma))
        - special.erf((mu - r) / (np.sqrt(2) * sigma))
    )
    return n / d


class LineObj(DataObj):
    def __init__(self, x, y, level, analysis_range, color, line_style, label=""):
        self.level = level
        self.analysis_range = analysis_range
        self.color = color
        self.line_style = line_style
        self.root_list = validate_roots(
            CubicSpline(x, y - max(y) * level).roots(),
            -analysis_range,
            analysis_range,
        )
        self.level = y.max() * self.level
        self.label = label

    def roots(self, index_1, index_2):
        try:
            return self.root_list[index_1] - self.root_list[index_2]
        except IndexError:
            return 0
        except TypeError:
            return 0


class gaussian_bg(rv_continuous):
    "Gaussian distributionwithj Background parameter 'back'"

    def _pdf(self, x, sigma, mu, back):
        return guassian_background(x, sigma, mu, back, self.a, self.b)


def validate_roots(roots, right_lim, left_lim):
    valid_roots = []
    for root in roots:
        if root < right_lim or root > left_lim:
            continue
        else:
            valid_roots.append(root)
    if len(valid_roots) >= 2:
        return valid_roots
    else:
        return [0, 0]
