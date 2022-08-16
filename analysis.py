from TimeTagger import createTimeTagger, FileWriter, FileReader

from time import sleep
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
import os
import numpy as np
import yaml
import time
import json
import timeit
from bokeh.plotting import figure, output_file, show
from scipy.stats import norm
from numba import njit
from scipy.interpolate import interp1d
import math
from scipy.stats import rv_continuous
from scipy import special

from clock_tools import clockLock
from clock_tools import RepeatedPll
from datetime import date
import datetime
import phd.viz
import concurrent.futures
import glob
from matplotlib import cm
from scipy.signal import find_peaks
from os.path import exists
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from mask_generators import MaskGenerator
from data_obj import DataObj

from tqdm import tqdm


Colors, palette = phd.viz.phd_style(text=1)


def delayCorrect(_dataTags):
    delays = np.load("Delays_1.npy")  # in picoseconds
    delayTimes = np.load("Delay_Times.npy")  # in nanoseconds
    f2 = interp1d(delayTimes, delays, kind="cubic")
    print(max(delayTimes))
    print(min(delayTimes))
    xnew = np.linspace(min(delayTimes), max(delayTimes), num=200, endpoint=True)
    random = [5, 15, 22, 17, 100]
    Outs = []
    for item in random:
        if item <= min(delayTimes):
            out = delays[0]
        elif item >= max(delayTimes):
            out = delays[-1]
        else:
            out = f2(item)
        Outs.append(out)

    plt.figure()
    plt.plot(xnew, f2(xnew))
    plt.plot(random, Outs)

    newTags = np.zeros(len(_dataTags))
    unprocessed = np.zeros(len(_dataTags))
    prevTag = 0
    print("start")
    for i, tag in enumerate(_dataTags):
        deltaTime = (tag - prevTag) / 1000  # now in ns
        if deltaTime <= min(delayTimes) + 0.01:
            out = delays[0]
        elif deltaTime >= max(delayTimes) - 0.01:
            out = delays[-1]
        else:
            out = f2(deltaTime)
            # unprocessed[i] = tag

        newTags[i] = tag - out

        prevTag = tag
    print("end")
    return newTags  # , unprocessed


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


class gaussian_bg(rv_continuous):
    "Gaussian distributionwithj Background parameter 'back'"

    def _pdf(self, x, sigma, mu, back):
        return guassian_background(x, sigma, mu, back, self.a, self.b)


def getCountRate(path_, file_):
    full_path = os.path.join(path_, file_)
    file_reader = FileReader(full_path)
    n_events = 100000000  # Number of events to read at once
    data = file_reader.getData(n_events)
    channels = data.getChannels()  # these are numpy arrays
    timetags = data.getTimestamps()
    SNSPD_tags = timetags[channels == -3]
    count_rate = 1e12 * (len(SNSPD_tags) / (SNSPD_tags[-1] - SNSPD_tags[0]))
    print("Count rate is: ", count_rate)

    return count_rate


def load_snspd_and_clock_tags(
    full_path, snspd_ch: int, clock_ch: int, read_events=1e9, debug=False
):
    file_reader = FileReader(full_path)
    data = file_reader.getData(int(read_events))
    if debug:
        print(
            "load_snspd_and_clock_tags: Size of the returned data chunk: {:d} events\n".format(
                data.size
            )
        )
    channels = data.getChannels()
    timetags = data.getTimestamps()
    print("length of timetags: ", len(timetags))

    SNSPD_tags = timetags[channels == snspd_ch]
    CLOCK_tags = timetags[channels == clock_ch]
    return SNSPD_tags, CLOCK_tags, channels, timetags


def data_statistics(modu_params, snspd_tags, clock_tags, debug=True):
    pulses_per_clock = modu_params["pulses_per_clock"]
    count_rate = 1e12 * (len(snspd_tags) / (snspd_tags[-1] - snspd_tags[0]))
    clock_rate = 1e12 * (len(clock_tags) / (clock_tags[-1] - clock_tags[0]))
    pulse_rate = (clock_rate * pulses_per_clock) / 1e9
    inter_pulse_time = 1 / pulse_rate  # time between pulses in nanoseconds
    time_elapsed = 1e-12 * (snspd_tags[-1] - snspd_tags[0])

    if debug:
        print("SNSPD TAGS:   ", len(snspd_tags))
        print("Count rate is: ", count_rate)
        print("Clock rate is: ", clock_rate)
        print("pulse rate: ", pulse_rate)
        print("inter_pulse_time: ", inter_pulse_time)
        print("time elapsed: ", time_elapsed)

    return {
        "count_rate": count_rate,
        "clock_rate": clock_rate,
        "pulse_rate": pulse_rate,
        "inter_pulse_time": inter_pulse_time,
        "time_elapsed": time_elapsed,
    }


def parse_count_rate(number):
    if number > 1e3:
        val = f"{round(number/1000,1)} KCounts/s"
    if number > 1e6:
        val = f"{round(number/1e6,1)} MCounts/s"
    if number > 1e9:
        val = f"{round(number/1e9,1)} GCounts/s"
    return val


def delay_analysis(
    channels, timetags, clock_channel, snspd_channel, stats, delay, deriv, prop
):
    dataNumbers = []
    delayRange = np.array([i - 500 for i in range(1000)])
    Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(
        channels[:100000],
        timetags[:100000],
        clock_channel,
        snspd_channel,
        stats["pulses_per_clock"],
        delay,
        window=0.01,
        deriv=deriv,
        prop=prop,
    )
    checkLocking(Clocks, RecoveredClocks)
    for i, delay in enumerate(delayRange):
        Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(
            channels[:100000],
            timetags[:100000],
            clock_channel,
            snspd_channel,
            stats["pulses_per_clock"],
            delay,
            window=0.01,
            deriv=deriv,
            prop=prop,
        )
        deltaTimes = dataTags[1:-1] - np.roll(dataTags, 1)[1:-1]
        dataNumbers.append(len(deltaTimes))
    dataNumbers = np.array(dataNumbers)
    # delay = delayRange[np.argmax(dataNumbers)]
    plt.figure()
    plt.plot(delayRange, dataNumbers)
    delay = delayRange[np.argmax(dataNumbers)]
    print("Max counts found at delay: ", delayRange[np.argmax(dataNumbers)])
    print(
        "You can update the analysis_params.yaml file with this delay and turn off delay_scan"
    )
    plt.title("peak value is phase (ps) bewteen clock and SNSPD tags")
    print("Offset time: ", delay)
    return delay  # after


def make_histogram(dataTags, nearestPulseTimes, delay, stats, Figures):
    diffsorg = dataTags[1:-1] - nearestPulseTimes[1:-1]
    guassDiffs = diffsorg + delay

    guassEdges = np.linspace(
        int(-stats["inter_pulse_time"] * 1000 * 0.5),
        int(stats["inter_pulse_time"] * 1000 * 0.5),
        4001,
    )  # 1 period width
    print("length of guassDiffs: ", len(guassDiffs))
    guassHist, guassBins = np.histogram(guassDiffs, guassEdges, density=True)
    gaussianBG = gaussian_bg(
        a=guassDiffs.min() / 1000, b=guassDiffs.max() / 1000, name="gaussianBG"
    )
    start = time.time()
    print("starting fit")
    scalefactor = 1000
    guassStd2, guassAvg2, back, flock, fscale = gaussianBG.fit(
        guassDiffs[-30000:] / scalefactor, floc=0, fscale=1
    )
    guassStd = np.std(guassDiffs[-30000:])
    end = time.time()
    print("time of fit: ", end - start)
    guassStd2 = guassStd2 * scalefactor
    guassAvg2 = guassAvg2 * scalefactor

    if Figures:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(guassBins[1:], guassHist)
        ax[1].plot(guassBins[1:], guassHist)
        ax[1].set_yscale("log")
        ax[0].set_title("histogram of counts wrt clock")


def calculate_diffs(data_tags, nearest_pulse_times, delay):
    """
    Calculate time between a given snspd tag and the clock-time of the snspd tag that preceded it.
    :param data_tags: snspd tags
    :param nearest_pulse_times: clock-time, analogouse to the time the photon hit the nanowire
    :param delay: constant offset or phase
    :return: diffs
    """
    # this function subtracts the previous laser-based timing from the timing of snspd tags
    # output is in nanoseconds

    nearest_pulse_times = np.roll(nearest_pulse_times, 1)
    diffs = data_tags[1:-1] - nearest_pulse_times[1:-1]
    diffs = diffs + delay
    diffs = diffs / 1000  # Now in nanoseconds
    return diffs


def calculate_2d_diffs(data_tags, nearest_pulse_times, delay):
    delays = data_tags[2:-2] + delay - nearest_pulse_times[2:-2]

    # this makes more sense to me than using np.diff
    prime_1 = nearest_pulse_times[2:-2] - np.roll(nearest_pulse_times, 1)[2:-2]
    prime_2 = (
        np.roll(nearest_pulse_times, 1)[2:-2] - np.roll(nearest_pulse_times, 2)[2:-2]
    )

    return delays / 1000, prime_1 / 1000, prime_2 / 1000


# def get_full_widths(x, y, level, analysis_range):
#     """
#     For finding FWHM, FW(1/10)M, etc. Returns dataObj with roots and level
#     :param x: x-axis, could be in ps
#     :param y: y-axis of response function
#     :param level: use (1/2) for FWHM, (1/10) for FW(1/10)M, (1/100) for FW(1/100)M
#     :param analysis_range: range inside the domain of x to search for roots, if less than x
#     :return: a list of roots
#     """
#
#     obj = DataObj()
#     obj.roots = validate_roots(
#         CubicSpline(x, y - max(y) * level).roots(), -analysis_range, analysis_range
#     )
#     obj.level = y.max() * level
#     return obj


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
        return self.root_list[index_1] - self.root_list[index_2]


def validate_roots(roots, right_lim, left_lim):
    valid_roots = []
    for root in roots:
        if root < right_lim or root > left_lim:
            continue
        else:
            valid_roots.append(root)
    return valid_roots


def number_manager(number):
    if number > 1e3:
        val = f"{round(number/1000,1)} KCounts/s"
    if number > 1e6:
        val = f"{round(number/1e6,1)} MCounts/s"
    if number > 1e9:
        val = f"{round(number/1e9,1)} GCounts/s"
    return val


def do_correction(corr_params, calibration_obj, data):
    if corr_params["type"] == "1d":
        return do_1d_correction(corr_params, calibration_obj, data)
    if corr_params["type"] == "2d":
        return do_2d_correction(corr_params, calibration_obj, data)


def do_2d_correction(corr_params, calibration_obj, data):
    r = DataObj()  # results object

    diff_1 = np.roll(np.diff(data.data_tags), 1)
    diff_2 = np.roll(diff_1, 1)

    diff_1 = diff_1[2:-2]
    diff_2 = diff_2[2:-2]

    uncorrected_diffs = data.data_tags - data.nearest_pulse_times
    uncorrected_diffs = uncorrected_diffs[2:-3]

    medians = calibration_obj.medians

    # make sure edges has the correct scaling from the calibration file
    # put edges in units of picoseconds
    edges = (np.arange(len(medians)) / calibration_obj.stats["pulse_rate"]) * 1000

    # interpolator = interp2d(edges, edges, medians, "linear")
    func = RegularGridInterpolator(
        (edges, edges), medians, bounds_error=False, fill_value=0
    )

    print("len of diff_1 and diff_2:", len(diff_1), len(diff_2))
    diff_12 = list(zip(diff_1, diff_2))
    shifts = func(diff_12)
    # shifts = interpolator(diff_1, diff_2)  # now in picoseconds

    print("length of shifts: ", len(shifts))
    shifts = shifts * 1000

    corrected = data.data_tags[2:-3] - shifts
    corrected_diffs = corrected - data.nearest_pulse_times[2:-3]

    # mostly included for compatibility with 1d correction
    data_tags = data.data_tags[2:-3]
    nearest_pulse_times = data.nearest_pulse_times[2:-3]
    uncorrupted_mask = diff_1 / 1000 > 200  # nanoseconds
    uncorrupted_tags = data_tags[uncorrupted_mask]
    uncorrupted_diffs = uncorrupted_tags - nearest_pulse_times[uncorrupted_mask]

    edge = int(data.stats["inter_pulse_time"] * 1000 / 2)
    const_offset = uncorrected_diffs.min()
    uncorrected_diffs = uncorrected_diffs - const_offset - edge  # cancel offset
    corrected_diffs = corrected_diffs - const_offset - edge
    uncorrupted_diffs = uncorrupted_diffs - const_offset - edge

    r.hist_bins = np.arange(-edge, edge, 1)
    r.hist_uncorrected, r.hist_bins = np.histogram(
        uncorrected_diffs, r.hist_bins, density=True
    )
    r.hist_corrected, r.hist_bins = np.histogram(
        corrected_diffs, r.hist_bins, density=True
    )

    r.hist_uncorrupted, r.hist_bins = np.histogram(
        uncorrupted_diffs, r.hist_bins, density=True
    )

    hist, bins = np.histogram(shifts, bins=1000)
    plt.figure()
    plt.plot(bins[1:], hist)
    plt.title("shifts")

    r.hist_bins = r.hist_bins[1:] - (r.hist_bins[1] - r.hist_bins[0]) / 2

    # plt.figure()
    # plt.plot(r.hist_bins, r.hist_uncorrected, label="uncorrected")
    # plt.plot(r.hist_bins, r.hist_corrected, ls="--", label="corrected")
    # plt.title(
    #     f"count rate: {parse_count_rate(data.stats['count_rate'])}, "
    #     f"data_file: {data.params['data_file']}"
    # )
    # plt.legend()

    r = plot_and_analyze_histogram(
        r,
        corrected_diffs,
        uncorrected_diffs,
        uncorrupted_diffs,
        corr_params,
        edge,
    )

    r.corr_params = corr_params
    r.data_stats = data.stats
    r.data_params = data.params

    if corr_params["output"]["save_correction_result"]:
        rg = corr_params["output"]["data_file_snip"]
        file_name = (
            corr_params["output"]["save_name"]
            + "2d_"
            + data.params["data_file"][rg[0] : rg[-1]]
        )

        r.export(
            os.path.join(
                corr_params["output"]["save_location"],
                file_name,
            ),
            print_info=True,
            include_time_inside=True,
        )
        return r


def plot_and_analyze_histogram(
    r, corrected_diffs, uncorrected_diffs, uncorrupted_diffs, corr_params, edge
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
        plt.savefig(save_name)

    if not corr_params["view"]["show_figures"]:
        plt.close(fig)

    return r


def do_1d_correction(corr_params, calibration_obj, data):
    r = DataObj()  # results object
    delta_ts = data.data_tags - np.roll(data.data_tags, 1)
    delta_ts = delta_ts[1:-1] / 1000  # now in nanoseconds
    data_tags = data.data_tags[1:-1]
    nearest_pulse_times = data.nearest_pulse_times[1:-1]

    plt.figure()
    hist, bins = np.histogram(delta_ts, bins=500)
    plt.plot(bins[:-1], hist)
    plt.title("dist of deltaTs")

    # GENERATE SHIFTS
    shifts = 1000 * np.interp(
        delta_ts, calibration_obj.t_prime, calibration_obj.offsets
    )  # in picoseconds. Remove the 1st couple data

    uncorrupted_tags = data_tags[delta_ts > 200]  # nanoseconds
    uncorrupted_diffs = uncorrupted_tags - nearest_pulse_times[delta_ts > 200]
    # print("length of uncorrupted tags: ", len(uncorrupted_tags))

    plt.figure()
    hist, bins = np.histogram(shifts, bins=500)
    plt.plot(bins[:-1], hist)
    plt.title("dist of shifts")
    # points because they are not generally valid.

    corrected_tags = data_tags - shifts

    uncorrected_diffs = data_tags - nearest_pulse_times
    corrected_diffs = corrected_tags - nearest_pulse_times

    edge = int(data.stats["inter_pulse_time"] * 1000 / 2)
    const_offset = uncorrected_diffs.min()
    uncorrected_diffs = uncorrected_diffs - const_offset - edge  # cancel offset
    corrected_diffs = corrected_diffs - const_offset - edge
    uncorrupted_diffs = uncorrupted_diffs - const_offset - edge

    #############################
    r.hist_bins = np.arange(-edge, edge, 1)
    r.hist_uncorrected, r.hist_bins = np.histogram(
        uncorrected_diffs, r.hist_bins, density=True
    )
    r.hist_corrected, r.hist_bins = np.histogram(
        corrected_diffs, r.hist_bins, density=True
    )

    r.hist_uncorrupted, r.hist_bins = np.histogram(
        uncorrupted_diffs, r.hist_bins, density=True
    )

    r.hist_bins = r.hist_bins[1:] - (r.hist_bins[1] - r.hist_bins[0]) / 2

    #############################
    # if corr_params["view"]["show_figures"]:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=275)
    ax.plot(
        r.hist_bins,
        r.hist_uncorrected,
        "--",
        color="black",
        alpha=0.8,
        label="uncorrected data",
    )
    ax.plot(
        r.hist_bins,
        r.hist_corrected,
        color="black",
        label="corrected data",
    )

    ax.grid(which="both")
    plt.legend()
    plt.title(
        f"count rate: {parse_count_rate(data.stats['count_rate'])}, "
        f"data_file: {data.params['data_file']}"
    )
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("counts (ps)")
    if not corr_params["view"]["show_figures"]:
        plt.close(fig)

    r = plot_and_analyze_histogram(
        r, corrected_diffs, uncorrected_diffs, uncorrupted_diffs, corr_params, edge
    )

    r.corr_params = corr_params
    r.data_stats = data.stats
    r.data_params = data.params

    if corr_params["output"]["save_correction_result"]:
        rg = corr_params["output"]["data_file_snip"]
        file_name = (
            corr_params["output"]["save_name"]
            + data.params["data_file"][rg[0] : rg[-1]]
        )

        r.export(
            os.path.join(
                corr_params["output"]["save_location"],
                file_name,
            ),
            print_info=True,
            include_time_inside=True,
        )
        return r


def prepare_data(data_params, path_dic=None):

    if path_dic is None:
        full_path = os.path.join(data_params["data_path"], data_params["data_file"])
    else:
        full_path = os.path.join(path_dic["path"], path_dic["file"])

    data = DataObj()

    Figures = data_params["view"]["show_figures"]
    pulses_per_clock = data_params["modulation_params"]["pulses_per_clock"]
    # get data set up
    snspd_channel = data_params["snspd_channel"]
    clock_channel = data_params["clock_channel"]
    snspd_tags, clock_tags, channels, timetags = load_snspd_and_clock_tags(
        full_path,
        data_params["snspd_channel"],
        data_params["clock_channel"],
        data_params["data_limit"],
    )
    data.stats = data_statistics(
        data_params["modulation_params"], snspd_tags, clock_tags, debug=False
    )

    # optional delay analysis
    if data_params[
        "delay_scan"
    ]:  # to be done with a low detection rate file (high attentuation)
        delay = delay_analysis(
            channels,
            timetags,
            clock_channel,
            snspd_channel,
            data.stats,
            data_params["delay"],
            data_params["phase_locked_loop"]["deriv"],
            data_params["phase_locked_loop"]["prop"],
        )
    (
        data.clocks,
        data.recovered_clocks,
        data.data_tags,
        data.nearest_pulse_times,
        cycles,
    ) = clockLock(
        channels,
        timetags,
        data_params["clock_channel"],
        data_params["snspd_channel"],
        data_params["modulation_params"]["pulses_per_clock"],
        data_params["delay"],
        window=data_params["phase_locked_loop"]["window"],
        deriv=data_params["phase_locked_loop"]["deriv"],
        prop=data_params["phase_locked_loop"]["prop"],
        guardPeriod=data_params["phase_locked_loop"]["guard_period"],
    )
    if Figures:
        checkLocking(data.clocks, data.recovered_clocks)

    make_histogram(
        data.data_tags[: data_params["view"]["histogram_max_tags"]],
        data.nearest_pulse_times[: data_params["view"]["histogram_max_tags"]],
        data_params["delay"],
        data.stats,
        Figures,
    )
    data.params = data_params
    if path_dic is not None:
        # update data.params to point to the passed file
        data.params["data_path"] = path_dic["path"]
        data.params["data_file"] = path_dic["file"]
    return data


# @njit
def do_2d_scan(prime_1_masks, prime_2_masks, delays, prime_steps):

    medians = np.zeros((prime_steps, prime_steps))
    means = np.zeros((prime_steps, prime_steps))
    std = np.zeros((prime_steps, prime_steps))
    counts = np.zeros((prime_steps, prime_steps))
    for i in tqdm(range(len(prime_1_masks))):
        # print(i / prime_steps)
        for j in range(len(prime_2_masks)):
            # inside is prime_2
            sub_delays = delays[prime_1_masks[i] & prime_2_masks[j]]
            counts[i, j] = len(sub_delays)

            if len(sub_delays > 10):
                medians[i, j] = np.median(sub_delays)
                means[i, j] = np.mean(sub_delays)
                std[i, j] = np.std(sub_delays)

    valid = counts > 10  # boolean mask
    adjustment = np.mean(medians[-30:, -30:])
    medians[valid] = medians[valid] - adjustment
    means[valid] = means[valid] - adjustment

    return medians, means, std, counts


def do_calibration(cal_params, data):
    if cal_params["type"] == "1d":
        return do_1d_calibration(cal_params, data)
    if cal_params["type"] == "2d":
        return do_2d_calibration(cal_params, data)
    else:
        print("calibration type unknown: [use '1d' or '2d']")
        quit()


def do_2d_calibration(cal_params, data):
    cal_results_obj = DataObj()  # for storing results of calibration
    cal_results_obj.stats = data.stats
    cal_results_obj.data_params = data.params
    cal_results_obj.cal_params = cal_params

    delays, prime_1, prime_2 = calculate_2d_diffs(
        data.data_tags, data.nearest_pulse_times, data.params["delay"]
    )  # returns diffs in nanoseconds

    prime_steps = cal_params["prime_steps"]
    prime_1 = (
        prime_1 / data.stats["pulse_rate"]
    )  # prime_1 must be in units of ns before division
    prime_2 = prime_2 / data.stats["pulse_rate"]
    prime_1 = prime_1.astype("int")
    prime_2 = prime_2.astype("int")

    prime_1_masks = []
    prime_2_masks = []

    for i in tqdm(range(prime_steps)):
        prime_1_masks.append(prime_1 == i)
        prime_2_masks.append(prime_2 == i)

    print("starting 2d analysis")
    medians, means, std, counts = do_2d_scan(
        prime_1_masks, prime_2_masks, delays, prime_steps
    )

    x = np.arange(0, prime_steps)
    y = np.arange(0, prime_steps)
    x, y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_ylabel("prime 2")
    ax.set_xlabel("prime_1")
    ax.plot_surface(x, y, medians)

    cal_results_obj.medians = medians
    cal_results_obj.means = means
    cal_results_obj.std = std
    cal_results_obj.counts = counts

    if cal_params["output"]["save_analysis_result"]:
        cal_results_obj.export(
            cal_params["output"]["save_name"] + "2d_",
            include_time=True,
            print_info=True,
        )

    return cal_results_obj


def do_1d_calibration(cal_params, data):
    cal_results_obj = DataObj()  # for storing results of calibration
    cal_results_obj.stats = data.stats
    cal_results_obj.cal_params = cal_params

    diffs = calculate_diffs(
        data.data_tags, data.nearest_pulse_times, data.params["delay"]
    )  # returns diffs in nanoseconds

    mask_manager = MaskGenerator(
        diffs,
        cal_params["analysis_range"],
        data.stats["inter_pulse_time"],
        figures=cal_params["view"]["show_figures"],
        main_hist_downsample=1,
    )  # used to separate out the data into discrete distributions for each t'

    if cal_params["mask_method"] == "from_period":
        cal_results_obj = mask_manager.apply_mask_from_period(
            cal_params["mask_from_period"], cal_results_obj
        )

    elif cal_params["mask_method"] == "from_peaks":
        cal_results_obj = mask_manager.apply_mask_from_peaks(
            cal_params["mask_from_peaks"], cal_results_obj
        )
    else:
        print("Unknown decoding method")
        return 1

    cal_results_obj.data_params = data.params

    if cal_params["output"]["save_analysis_result"]:
        cal_results_obj.export(
            cal_params["output"]["save_name"], include_time=True, print_info=True
        )

    return cal_results_obj


# def do_correction():
#     if params["correction"]["load_pre-generated_calibration"]:
#         calibration_obj = DataObj(
#             os.path.join(
#                 params["correction"]["pre-generated_calibration"]["path"],
#                 params["correction"]["pre-generated_calibration"]["file"],
#             )
#         )
#     else:
#         # else would I do this?
#         # you could initialize the object no matter what, and give it some label inside if it is used
#         try:
#             cal_results_obj  # check if it exists
#         except UnboundLocalError:
#             print(
#                 "Error: if calibration is not loaded externally, it must be enabled. \n"
#                 "Set do_calibration to True, or set load_pre-generated_calibration to True"
#             )
#             return 1
#
#     # should I loop over correction files here? No, you need to load multiple files...
#     corr_results_obj = DataObj()
#     corr_results_obj = do_correction(
#         calibration_obj,
#         corr_results_obj,
#         data_tags,
#         nearest_pulse_times,
#         stats,
#         params,
#     )
#
#     if params["correction"]["output"]["save_correction_result"]:
#         corr_results_obj.export()
#
#     # rep rate of the calibration set may be different than that of the corrected set.


def sleeper(t, iter, tbla=0):
    # time.sleep(t)
    for i in range(1000):
        q = np.sin(np.linspace(0, 5, 1000000))
    print("sleeping for: ", t)
    print("tbla is: ", tbla)
    return t


# class RunAnalysisCopier(object):
#     def __init__(self, path, modu_params, DERIV, PROP, delayScan, delay, Figures):
#         self.Path = path
#         self.Deriv = DERIV
#         self.Prop = PROP
#         self.DelayScan = delayScan
#         self.Delay = delay
#         self.Figures = Figures
#         self.modu_params = modu_params
#
#     def __call__(self, file_iterator):
#
#         return run_analysis(
#             self.Path,
#             file_iterator,
#             self.modu_params,
#             DERIV=self.Deriv,
#             PROP=self.Prop,
#             delayScan=self.DelayScan,
#             delay=self.Delay,
#             Figures=self.Figures,
#         )


def get_file_list(path):
    ls = os.listdir(path)
    files = []
    for item in ls:
        if os.path.isfile(os.path.join(path, item)):
            files.append(item)
    return files


class MultiprocessLoaderCorrector:
    def __init__(self, _params, _calibration_obj):
        # self.params = _params
        self.data_params = _params["data"]
        self.correction_params = _params["correction"]
        self.calibration_object = _calibration_obj
        self.path = _params["correction"]["multiple_files_path"]

    def __call__(self, file):
        return caller(
            self.data_params,
            self.path,
            file,
            self.correction_params,
            self.calibration_object,
        )
        # print("you are being called")
        # data = prepare_data(self._params["data"], full_path=data_path)
        # # need some way of signaling it should use explicit file
        # print("data exists")
        # return do_correction(self._params["correction"],
        # self.calibration_object, data)


def caller(data_params, path, file, correction_params, calibration_object):
    path_dic = {"path": path, "file": file}
    data = prepare_data(data_params, path_dic=path_dic)
    return do_correction(correction_params, calibration_object, data)


if __name__ == "__main__":

    with open("analysis_params.yaml", "r") as f:
        params = yaml.safe_load(f)["params"]

    # just calibrate and save
    if params["do_calibration"] and not params["do_correction"]:
        data = prepare_data(params["data"])

        calibration_obj = do_calibration(params["calibration"], data)

    if params["do_calibration"] and params["do_correction"]:
        print(
            "calibrating and correcting data_file: ", params["calibration"]["data_file"]
        )
        if params["correction"]["load_pre-generated_calibration"]:
            print(
                "WARNING: the pre-generated calibration will not be used unless "
                "do_calibration is turned off"
            )
        if params["correction"]["correct_multiple_files"]:
            print(
                "WARNING: multi-file correction requires "
                "'load_pre-generated_calibration = True' and \n"
                "'do_calibration = False'"
            )

        data = prepare_data(params["data"])
        calibration_obj = do_calibration(params["calibration"], data)
        results_obj = do_correction(params["correction"], calibration_obj, data)

    if not params["do_calibration"] and params["do_correction"]:
        if params["correction"]["load_pre-generated_calibration"]:
            calibration_obj = DataObj(
                os.path.join(
                    params["correction"]["pre-generated_calibration"]["path"],
                    params["correction"]["pre-generated_calibration"]["file"],
                )
            )
            if params["correction"]["correct_multiple_files"]:
                # override show_figures

                params["view"]["show_figures"] = False
                params["data"]["view"]["show_figures"] = False
                params["correction"]["view"]["show_figures"] = True

                file_list = get_file_list(params["correction"]["multiple_files_path"])

                # for some reason the multiprocessing does not work in pycharm
                # python console!
                with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
                    ls = executor.map(
                        MultiprocessLoaderCorrector(params, calibration_obj), file_list
                    )

            else:
                # load and do single file correction
                data = prepare_data(params["data"])
                results_obj = do_correction(params["correction"], calibration_obj, data)

        else:
            print(
                "Error: Cannot do a correction without either calibrating loaded data\n"
                "calibration file. Set 'do_calibration' to True, or set "
                "'load_pre-generated_calibration' to True."
            )
