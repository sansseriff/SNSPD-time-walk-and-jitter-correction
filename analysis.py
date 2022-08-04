from TimeTagger import createTimeTagger, FileWriter, FileReader

from time import sleep
import matplotlib.pyplot as plt
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
from scipy.interpolate import CubicSpline
from mask_generators import MaskGenerator
from data_obj import DataObj


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
    file: str, path: str, snspd_ch: int, clock_ch: int, read_events=1e9, debug=False
):
    full_path = os.path.join(path, file)
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


def get_full_widths(x, y, level, analysis_range):
    """
    For finding FWHM, FW(1/10)M, etc.
    :param x: x-axis, could be in ps
    :param y: y-axis of response function
    :param level: use (1/2) for FWHM, (1/10) for FW(1/10)M, (1/100) for FW(1/100)M
    :param analysis_range: range inside the domain of x to search for roots, if less than x
    :return: a list of roots
    """
    return validate_roots(
        CubicSpline(x, y - max(y) * level).roots(), -analysis_range, analysis_range
    )


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


def do_correction(d, r, data_tags, nearest_pulse_times, stats, params):
    delta_ts = data_tags - np.roll(data_tags, 1)
    delta_ts = delta_ts[1:-1] / 1000  # now in nanoseconds
    data_tags = data_tags[1:-1]
    nearest_pulse_times = nearest_pulse_times[1:-1]

    plt.figure()
    hist, bins = np.histogram(delta_ts, bins=500)
    plt.plot(bins[:-1], hist)
    plt.title("dist of deltaTs")

    # GENERATE SHIFTS
    shifts = 1000 * np.interp(
        delta_ts, d.t_prime, d.offsets
    )  # in picoseconds. Remove the 1st couple data

    plt.figure()
    hist, bins = np.histogram(shifts, bins=500)
    plt.plot(bins[:-1], hist)
    plt.title("dist of shifts")
    # points because they are not generally valid.

    corrected_tags = data_tags - shifts

    uncorrected_diffs = data_tags - nearest_pulse_times
    corrected_diffs = corrected_tags - nearest_pulse_times

    edge = int(stats["inter_pulse_time"] * 1000 / 2)
    const_offset = uncorrected_diffs.min()
    uncorrected_diffs = uncorrected_diffs - const_offset - edge  # cancel offset
    corrected_diffs = corrected_diffs - const_offset - edge

    r.hist_bins = np.arange(-edge, edge, 1)
    r.hist_uncorrected, r.hist_bins = np.histogram(
        uncorrected_diffs, r.hist_bins, density=True
    )
    r.hist_corrected, r.hist_bins = np.histogram(
        corrected_diffs, r.hist_bins, density=True
    )

    plt.yscale("log")
    plt.ylim(1e-6, 0.1)
    plt.legend()

    #############################
    if params["show_figures"]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=275)
        ax.plot(
            r.hist_bins[:-1],
            r.hist_uncorrected,
            "--",
            color="black",
            alpha=0.8,
            label="uncorrected data",
        )
        ax.plot(
            r.hist_bins[:-1],
            r.hist_corrected,
            color="black",
            label="corrected data",
        )
        # ax.set_yscale('log')
        ax.grid(which="both")
        plt.legend()
        plt.title(
            f"count rate: {parse_count_rate(stats['count_rate'])}, data_file: {params['data_file']}"
        )
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("counts (ps)")

    # resolution or smoothing of CubicSpline is determined by the resolution of these _interp arrays
    r.hist_bins_interp = np.linspace(
        r.hist_bins[1],
        r.hist_bins[-1],
        params["spline_interpolation_resolution"],
    )
    r.hist_corrected_interp = np.interp(
        r.hist_bins_interp, r.hist_bins[1:], r.hist_corrected
    )
    r.hist_uncorrected_interp = np.interp(
        r.hist_bins_interp, r.hist_bins[1:], r.hist_uncorrected
    )

    spline_corrected = CubicSpline(r.hist_bins_interp, r.hist_corrected_interp)
    spline_uncorrected = CubicSpline(r.hist_bins_interp, r.hist_uncorrected_interp)

    r.fwhm_corrected = get_full_widths(
        r.hist_bins_interp, r.hist_corrected_interp, 0.5, 500
    )
    r.fwhm_uncorrected = get_full_widths(
        r.hist_bins_interp, r.hist_uncorrected_interp, 0.5, 500
    )

    r.fwtm_corrected = get_full_widths(
        r.hist_bins_interp, r.hist_corrected_interp, 1 / 10, 500
    )
    r.fwtm_uncorrected = get_full_widths(
        r.hist_bins_interp, r.hist_uncorrected_interp, 1 / 10, 500
    )

    r.fwhum_corrected = get_full_widths(
        r.hist_bins_interp, r.hist_corrected_interp, 1 / 100, 500
    )
    r.fwhum_uncorrected = get_full_widths(
        r.hist_bins_interp, r.hist_uncorrected_interp, 1 / 100, 500
    )

    print("FWHM corrected: ", r.fwhm_corrected[-1] - r.fwhm_corrected[-2])
    print("FWTM corrected: ", r.fwtm_corrected[-1] - r.fwtm_corrected[-2])
    print("FW100M corrected: ", r.fwhum_corrected[-1] - r.fwhum_corrected[-2])

    print("FWHM uncorrected: ", r.fwhm_uncorrected[-1] - r.fwhm_uncorrected[-2])
    print("FWTM uncorrected: ", r.fwtm_uncorrected[-1] - r.fwtm_uncorrected[-2])
    print("FW100M uncorrected: ", r.fwhum_uncorrected[-2] - r.fwhum_uncorrected[0])

    if params["show_figures"]:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

        half_c = "#eb4034"
        tenth_c = "#c92eb2"
        hundreth_c = "#3d2ec9"

        ax.plot(
            r.hist_bins[1:],
            r.hist_uncorrected,
            label="uncorrected raw data",
            alpha=0.3,
        )
        ax.plot(
            r.hist_bins[1:],
            r.hist_corrected,
            label="corrected raw data",
            alpha=0.3,
        )

        title = f"count rate: {number_manager(stats['count_rate'])}"
        ax.set_title(title)
        ax.plot(
            r.hist_bins[1:],
            spline_corrected(r.hist_bins[1:]),
            "k",
            alpha=1,
            label="cubic spline corrected",
        )
        ax.plot(
            r.hist_bins[1:],
            spline_uncorrected(r.hist_bins[1:]),
            "k",
            alpha=0.3,
            label="cubic spline uncorrected",
            ls="--",
        )

        data.array_cs_corrected = spline_corrected(data.guassBins[1:])
        data.array_cs_uncorrected = spline_uncorrected(data.guassBins[1:])

        label = f"FWHM corrected: {round(data.roots_fw_half_c[-1] - data.roots_fw_half_c[-2], 1)} ps"
        ax.hlines(
            max(data.shorty_corrected) / 2,
            data.roots_fw_half_c[-1],
            data.roots_fw_half_c[-2],
            label=label,
            color=half_c,
        )
        label = f"FWHM uncorrected: {round(data.roots_fw_half_u[-1] - data.roots_fw_half_u[-2], 1)} ps"
        ax.hlines(
            max(data.shorty_uncorrected) / 2,
            data.roots_fw_half_u[-1],
            data.roots_fw_half_u[-2],
            label=label,
            color=half_c,
            ls="--",
        )

        label = f"FW(1/10)M corrected: {round(data.roots_fw_tenth_c[-1] - data.roots_fw_tenth_c[-2], 1)} ps"
        ax.hlines(
            max(data.shorty_corrected) / 10,
            data.roots_fw_tenth_c[-1],
            data.roots_fw_tenth_c[-2],
            label=label,
            color=tenth_c,
        )
        label = f"FW(1/10)M uncorrected: {round(data.roots_fw_tenth_u[-1] - data.roots_fw_tenth_u[-2], 1)} ps"
        ax.hlines(
            max(data.shorty_uncorrected) / 10,
            data.roots_fw_tenth_u[-1],
            data.roots_fw_tenth_u[-2],
            label=label,
            color=tenth_c,
            ls="--",
        )

        label = f"FW(1/100)M corrected: {round(data.roots_fw_hundreth_c[-1] - data.roots_fw_hundreth_c[-2], 1)} ps"
        ax.hlines(
            max(data.shorty_corrected) / 100,
            data.roots_fw_hundreth_c[-1],
            data.roots_fw_hundreth_c[-2],
            label=label,
            color=hundreth_c,
        )
        print(data.roots_fw_hundreth_u)

        ################
        label = f"FW(1/100)M uncorrected: {round(data.roots_fw_hundreth_u[-2] - data.roots_fw_hundreth_u[0], 1)} ps"
        ax.hlines(
            max(data.shorty_uncorrected) / 100,
            data.roots_fw_hundreth_u[-2],
            data.roots_fw_hundreth_u[0],
            label=label,
            color=hundreth_c,
            ls="--",
        )
        ################

        ax.grid()
        ax.set_xlim(-460, 200)
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("normalized counts")
        plt.legend(fancybox=True, frameon=False)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 0.1)
        ax_lin = inset_axes(ax, width="30%", height=1, loc=1)
        ax_lin.plot(
            data.guassBins[1:], spline_corrected(data.guassBins[1:]), "k", alpha=1
        )
        ax_lin.plot(
            data.guassBins[1:],
            spline_uncorrected(data.guassBins[1:]),
            "k",
            alpha=0.3,
            ls="--",
        )

        ax_lin.hlines(
            max(data.shorty_corrected) / 2,
            data.roots_fw_half_c[-1],
            data.roots_fw_half_c[-2],
            color=half_c,
        )
        ax_lin.hlines(
            max(data.shorty_uncorrected) / 2,
            data.roots_fw_half_u[-1],
            data.roots_fw_half_u[-2],
            color=half_c,
            ls="--",
        )
        ax_lin.hlines(
            max(data.shorty_corrected) / 10,
            data.roots_fw_tenth_c[-1],
            data.roots_fw_tenth_c[-2],
            color=tenth_c,
        )
        ax_lin.hlines(
            max(data.shorty_uncorrected) / 10,
            data.roots_fw_tenth_u[-1],
            data.roots_fw_tenth_u[-2],
            color=tenth_c,
            ls="--",
        )
        ax_lin.hlines(
            max(data.shorty_corrected) / 100,
            data.roots_fw_hundreth_c[-1],
            data.roots_fw_hundreth_c[-2],
            color=hundreth_c,
        )
        ax_lin.hlines(
            max(data.shorty_uncorrected) / 100,
            data.roots_fw_hundreth_u[-1],
            data.roots_fw_hundreth_u[-2],
            color=hundreth_c,
            ls="--",
        )

        ax_lin.set_xlim(-150, 150)
        ax_lin.grid()


def run_analysis(params):
    Figures = params["show_figures"]
    pulses_per_clock = params["modulation_params"]["pulses_per_clock"]
    # get data set up
    snspd_channel = params["snspd_channel"]
    clock_channel = params["clock_channel"]
    snspd_tags, clock_tags, channels, timetags = load_snspd_and_clock_tags(
        params["data_file"],
        params["data_path"],
        params["snspd_channel"],
        params["clock_channel"],
        params["data_limit"],
    )
    stats = data_statistics(
        params["modulation_params"], snspd_tags, clock_tags, debug=False
    )

    delay = params["delay"]
    # optional delay analysis
    if params[
        "delay_scan"
    ]:  # to be done with a low detection rate file (high attentuation)
        delay = delay_analysis(
            channels,
            timetags,
            clock_channel,
            snspd_channel,
            stats,
            delay,
            params["phase_locked_loop"]["deriv"],
            params["phase_locked_loop"]["prop"],
        )
    clocks, recovered_clocks, data_tags, nearest_pulse_times, cycles = clockLock(
        channels,
        timetags,
        params["clock_channel"],
        params["snspd_channel"],
        params["modulation_params"]["pulses_per_clock"],
        delay,
        window=params["phase_locked_loop"]["window"],
        deriv=params["phase_locked_loop"]["deriv"],
        prop=params["phase_locked_loop"]["prop"],
        guardPeriod=params["phase_locked_loop"]["guard_period"],
    )
    if Figures:
        checkLocking(clocks, recovered_clocks)

    make_histogram(
        data_tags[: params["histogram_max_tags"]],
        nearest_pulse_times[: params["histogram_max_tags"]],
        delay,
        stats,
        Figures,
    )

    # do_analysis = True
    # do_correction = True

    if params["do_calibration"]:
        cal_results_obj = DataObj()  # for storing results of calibration
        cal_results_obj.stats = stats
        cal_results_obj.params = params

        diffs = calculate_diffs(
            data_tags, nearest_pulse_times, delay
        )  # returns diffs in nanoseconds

        mask_manager = MaskGenerator(
            diffs,
            analysis_params["analysis_range"],
            stats["inter_pulse_time"],
            figures=params["show_figures"],
            main_hist_downsample=1,
        )

        if params["mask_method"] == "from_period":
            cal_results_obj = mask_manager.apply_mask_from_period(
                params["mask_from_period"], cal_results_obj
            )

        elif params["mask_method"] == "from_peaks":
            cal_results_obj = mask_manager.apply_mask_from_peaks(
                params["mask_from_peaks"], cal_results_obj
            )
        else:
            print("Unknown decoding method")
            return 1

        if params["output"]["save_analysis_result"]:
            cal_results_obj.export(
                params["output"]["save_name"], include_time=True, print_info=True
            )

    # use data_tags and nearest_pulse_times for making the histograms.
    if params["correction"]["do_correction"]:
        if params["correction"]["load_pre-generated_calibration"]:
            calibration_obj = DataObj(
                os.path.join(
                    params["correction"]["pre-generated_calibration"]["path"],
                    params["correction"]["pre-generated_calibration"]["file"],
                )
            )
        else:
            # else would I do this?
            # you could initialize the object no matter what, and give it some label inside if it is used
            try:
                cal_results_obj  # check if it exists
            except UnboundLocalError:
                print(
                    "Error: if calibration is not loaded externally, it must be enabled. \n"
                    "Set do_calibration to True, or set load_pre-generated_calibration to True"
                )
                return 1

        corr_results_obj = DataObj()
        do_correction(
            calibration_obj,
            corr_results_obj,
            data_tags,
            nearest_pulse_times,
            stats,
            params,
        )

        # rep rate of the calibration set may be different than that of the corrected set.


def sleeper(t, iter, tbla=0):
    # time.sleep(t)
    for i in range(1000):
        q = np.sin(np.linspace(0, 5, 1000000))
    print("sleeping for: ", t)
    print("tbla is: ", tbla)
    return t


class RunAnalysisCopier(object):
    def __init__(self, path, modu_params, DERIV, PROP, delayScan, delay, Figures):
        self.Path = path
        self.Deriv = DERIV
        self.Prop = PROP
        self.DelayScan = delayScan
        self.Delay = delay
        self.Figures = Figures
        self.modu_params = modu_params

    def __call__(self, file_iterator):

        return run_analysis(
            self.Path,
            file_iterator,
            self.modu_params,
            DERIV=self.Deriv,
            PROP=self.Prop,
            delayScan=self.DelayScan,
            delay=self.Delay,
            Figures=self.Figures,
        )


if __name__ == "__main__":

    with open("analysis_params.yaml", "r") as f:
        analysis_params = yaml.safe_load(f)["params"]

    if analysis_params["analyze_set"]:
        LS = []
        # ** this is very specific to the dataset collected on April 5
        dBlist = [i * 2 + 26 for i in range(21)]
        path = "..//data//537.5MHz_0.1s//"
        file_list = [
            "jitterate_537.5MHz_-.025V_" + str(dB) + ".0.1.ttbin" for dB in dBlist
        ]
        params_file = "..//modu//537.5MHz//0_21.08.27.13.24_params.yml"
        with open(params_file, "r") as f:
            modu_params = yaml.full_load(f)
        sleep_list = [(1.05**i) for i in range(10)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            dictList = executor.map(
                R(path, modu_params, 200000, 1e-14, False, -230, False),
                file_list,
            )
            # takes 10 - 20 minutes on 16 threads
        LS = []
        for dB, item in zip(dBlist, dictList):
            print(item)
            item["dB"] = dB
            LS.append(item)
        today_now = datetime.now().strftime("%y.%m.%d.%H.%M")
        with open(
            "jitterate_swabianHighRes_537.5MHz_" + today_now + ".json", "w"
        ) as outfile:
            json.dump(LS, outfile, indent=4)

    else:
        run_analysis(analysis_params)
