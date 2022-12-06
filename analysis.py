from TimeTagger import FileReader

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import numpy as np
import yaml
from scipy.interpolate import interp1d

from clock_tools import clockLock
import phd.viz
import concurrent.futures
from scipy.interpolate import RegularGridInterpolator
import scipy.interpolate as interpolate
from mask_generators import MaskGenerator
from data_obj import DataObj
from tqdm import tqdm
from numba import njit

from plotting import (
    checkLocking,
    make_histogram,
    plot_and_analyze_histogram,
)
from utils import number_manager, get_file_list

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

    # square
    # prime_2 = np.roll(nearest_pulse_times,1)[2:-2] - np.roll(nearest_pulse_times, 2)[2:-2]

    # triangle
    prime_2 = nearest_pulse_times[2:-2] - np.roll(nearest_pulse_times, 2)[2:-2]

    return delays / 1000, prime_1 / 1000, prime_2 / 1000


def calculate_3d_diffs(data_tags, nearest_pulse_times, delay):
    delays = data_tags[2:-2] + delay - nearest_pulse_times[2:-2]

    # this makes more sense to me than using np.diff
    prime_1 = nearest_pulse_times[2:-2] - np.roll(nearest_pulse_times, 1)[2:-2]

    # square
    # prime_2 = np.roll(nearest_pulse_times,1)[2:-2] - np.roll(nearest_pulse_times, 2)[2:-2]

    # triangle
    prime_2 = nearest_pulse_times[2:-2] - np.roll(nearest_pulse_times, 2)[2:-2]
    prime_3 = nearest_pulse_times[2:-2] - np.roll(nearest_pulse_times, 3)[2:-2]

    return delays / 1000, prime_1 / 1000, prime_2 / 1000, prime_3 / 1000


def do_correction(corr_params, calibration_obj, data):
    if corr_params["type"] == "1d":
        return do_1d_correction(corr_params, calibration_obj, data)
    if corr_params["type"] == "2d":
        return do_2d_correction(corr_params, calibration_obj, data)
    if corr_params["type"] == "3d":
        return do_3d_correction(corr_params, calibration_obj, data)


def custom_diff(tags):
    res = tags - np.roll(tags, 1)
    return res[1:]


def custom_diff2(tags):
    res = tags - np.roll(tags, 2)
    return res[1:]


def custom_diff3(tags):
    res = tags - np.roll(tags, 3)
    return res[1:]


def do_3d_correction(corr_params, calibration_obj, data):
    r = DataObj()  # results object
    uncorrected_diffs = data.data_tags - data.nearest_pulse_times
    uncorrected_diffs = uncorrected_diffs[3:-2]
    diff_1 = custom_diff(data.data_tags)
    diff_2 = custom_diff2(data.data_tags)
    diff_3 = custom_diff3(data.data_tags)
    diff_1 = diff_1[2:-2]
    diff_2 = diff_2[2:-2]
    diff_3 = diff_3[2:-2]
    if True:
        plt.figure()
        hist, bins = np.histogram(diff_1, bins=500)
        plt.plot(bins[1:], hist)
        plt.title("diff distribution")

    medians = np.array(calibration_obj.medians)
    medians = np.roll(np.roll(np.roll(medians, 1, axis=0), 1, axis=1), 1, axis=2)
    # make sure edges has the correct scaling from the calibration file
    # put edges in units of picoseconds
    edges = (np.arange(len(medians)) / calibration_obj.stats["pulse_rate"]) * 1000
    spline_2d = interpolate.RectBivariateSpline(edges, edges, medians[:, :, -1])
    spline_3d = interpolate.RegularGridInterpolator(
        (edges, edges, edges), medians, method="linear"
    )
    where_zero = diff_1 > 140000
    where_1d_interp = diff_2 > 130000
    where_2d_interp = diff_3 > 130000
    where_3d_interp = ~(where_zero | where_1d_interp | where_2d_interp)

    shifts = np.zeros(len(diff_1))
    medians_1d = medians[:, -1, -1]
    # last row of the 2d interpolation used for far field
    t_prime = (np.arange(len(medians)) / calibration_obj.stats["pulse_rate"]) * 1000

    shifts[where_zero] = np.zeros(
        np.count_nonzero(where_zero)
    )  # shouldn't even be needed
    shifts[where_1d_interp] = 1000 * np.interp(
        diff_1[where_1d_interp], t_prime, medians_1d
    )  # in picoseconds. Remove the 1st couple data
    shifts[where_2d_interp] = 1000 * spline_2d.ev(
        diff_1[where_2d_interp], diff_2[where_2d_interp]
    )
    shifts[where_3d_interp] = 1000 * spline_3d(
        (diff_1[where_3d_interp], diff_2[where_3d_interp], diff_3[where_3d_interp])
    )

    corrected = data.data_tags[3:-2] - shifts
    corrected_diffs = corrected - data.nearest_pulse_times[3:-2]
    # mostly included for compatibility with 1d correction
    data_tags = data.data_tags[3:-2]
    nearest_pulse_times = data.nearest_pulse_times[3:-2]
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

    r = plot_and_analyze_histogram(
        r,
        data,
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
            + corr_params["type"]
            + "_"
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


def do_2d_correction(corr_params, calibration_obj, data):
    r = DataObj()  # results object
    uncorrected_diffs = data.data_tags - data.nearest_pulse_times
    uncorrected_diffs = uncorrected_diffs[3:-2]
    diff_1 = custom_diff(data.data_tags)
    diff_2 = custom_diff2(data.data_tags)
    diff_1 = diff_1[2:-2]
    if True:
        plt.figure()
        hist, bins = np.histogram(diff_1, bins=500)
        plt.plot(bins[1:], hist)
        plt.title("diff distribution")
    diff_2 = diff_2[2:-2]
    medians = np.array(calibration_obj.medians)
    medians = np.roll(np.roll(medians, 1, axis=0), 1, axis=1)
    # make sure edges has the correct scaling from the calibration file
    # put edges in units of picoseconds
    edges = (np.arange(len(medians)) / calibration_obj.stats["pulse_rate"]) * 1000
    spline = interpolate.RectBivariateSpline(edges, edges, medians)
    where_zero = diff_1 > 140000
    where_1d_interp = diff_2 > 130000
    where_2d_interp = ~(where_zero | where_1d_interp)
    shifts = np.zeros(len(diff_1))
    offsets = medians[:, -1]  # last row of the 2d interpolation used for far field
    t_prime = (np.arange(len(medians)) / calibration_obj.stats["pulse_rate"]) * 1000
    shifts[where_1d_interp] = 1000 * np.interp(
        diff_1[where_1d_interp], t_prime, offsets
    )  # in picoseconds. Remove the 1st couple data
    shifts[where_zero] = np.zeros(
        np.count_nonzero(where_zero)
    )  # shouldn't even be needed
    shifts[where_2d_interp] = 1000 * spline.ev(
        diff_1[where_2d_interp], diff_2[where_2d_interp]
    )
    corrected = data.data_tags[3:-2] - shifts
    corrected_diffs = corrected - data.nearest_pulse_times[3:-2]
    # mostly included for compatibility with 1d correction
    data_tags = data.data_tags[3:-2]
    nearest_pulse_times = data.nearest_pulse_times[3:-2]
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

    r = plot_and_analyze_histogram(
        r,
        data,
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
        r,
        data,
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

    np.savez_compressed(
        "clocks_and_tags_jitterate_05",
        snspd_tags=snspd_tags,
        clock_tags=clock_tags,
        channels=channels,
        timetags=timetags,
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
def do_2d_scan(prime_1_masks, prime_2_masks, delays, prime_steps, min_sub_delay):

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

            if len(sub_delays) > min_sub_delay:
                # print(len(sub_delays))
                medians[i, j] = np.median(sub_delays)
                means[i, j] = np.mean(sub_delays)
                std[i, j] = np.std(sub_delays)

    valid = counts > 10  # boolean mask
    adjustment = np.mean(medians[130, -5:])
    # the - number should be less 150 - 130 by several units...

    medians[valid] = medians[valid] - adjustment
    means[valid] = means[valid] - adjustment

    if True:
        plt.figure()
        plt.plot(np.arange(len(medians[:, -5])), medians[:, -10])
        plt.plot(np.arange(len(medians[:, -5])), medians[:, -5])
        plt.plot(np.arange(len(medians[:, -5])), medians[:, -3])
        plt.plot(np.arange(len(medians[:, -5])), medians[:, -1])

    return medians, means, std, counts


def do_3d_scan(
    prime_1_masks, prime_2_masks, prime_3_masks, delays, prime_steps, min_sub_delay
):

    medians = np.zeros((prime_steps, prime_steps, prime_steps))
    means = np.zeros((prime_steps, prime_steps, prime_steps))
    std = np.zeros((prime_steps, prime_steps, prime_steps))
    counts = np.zeros((prime_steps, prime_steps, prime_steps))

    for i in tqdm(range(len(prime_1_masks))):
        print("iteration: ", i, "of ", len(prime_1_masks))
        mask_1 = prime_1_masks[i]
        for j in range(len(prime_2_masks)):
            # print("     iteration: ", j, "of ", len(prime_1_masks))
            mask_2 = mask_1 & prime_2_masks[j]
            for k in range(len(prime_3_masks)):
                mask_3 = mask_2 & prime_3_masks[k]
                sub_delays = delays[mask_3]
                counts[i, j, k] = len(sub_delays)

                if len(sub_delays) > min_sub_delay:
                    # print(len(sub_delays))
                    medians[i, j, k] = np.median(sub_delays)
                    means[i, j, k] = np.mean(sub_delays)
                    std[i, j, k] = np.std(sub_delays)

    valid = counts > 10  # boolean mask
    adjustment = np.mean(medians[130, -5:, -5:])
    # the - number should be less 150 - 130 by several units...

    medians[valid] = medians[valid] - adjustment
    means[valid] = means[valid] - adjustment

    if False:
        plt.figure()
        plt.plot(np.arange(len(medians[:, -5, 130])), medians[:, -10])
        plt.plot(np.arange(len(medians[:, -5, 130])), medians[:, -5])
        plt.plot(np.arange(len(medians[:, -5, 130])), medians[:, -3])
        plt.plot(np.arange(len(medians[:, -5, 130])), medians[:, -1])

    return medians, means, std, counts


def do_calibration(cal_params, data):
    if cal_params["type"] == "1d":
        return do_1d_calibration(cal_params, data)
    if cal_params["type"] == "2d":
        return do_2d_calibration(cal_params, data)
    if cal_params["type"] == "3d":
        return do_3d_calibration(cal_params, data)
    else:
        print("calibration type unknown: [use '1d' or '2d' or '3d']")
        quit()


@njit
def linear_3d_scan(prime_1, prime_2, prime_3, delays, prime_steps):
    print("length of delays: ", len(delays))
    stack_length = int(len(delays) / (prime_steps**3)) * 5
    print("stack length: ", stack_length)
    master_counts = np.empty((prime_steps, prime_steps, prime_steps, stack_length))
    placement = np.zeros((prime_steps, prime_steps, prime_steps)).astype("int")
    master_counts[:] = np.nan
    printed_100 = True
    printed_10 = True

    for i in range(len(delays)):
        if i == 0:
            print("starting")
        if (i > len(delays) / 100) and printed_100:
            print("more than 1%")
            printed_100 = False

        if (i > len(delays) / 10) and printed_10:
            print("more than 10%")
            printed_10 = False

        if i < 3:
            continue
        if prime_1[i] < prime_steps:
            loc1 = prime_1[i]
        else:
            continue

        if prime_2[i] < prime_steps:
            loc2 = prime_2[i]
        else:
            continue

        if prime_3[i] < prime_steps:
            loc3 = prime_3[i]
        else:
            continue

        place = placement[loc1, loc2, loc3]
        if place < stack_length:
            master_counts[loc1, loc2, loc3, placement[loc1, loc2, loc3]] = delays[i]
            placement[loc1, loc2, loc3] = placement[loc1, loc2, loc3] + 1

    return master_counts, placement, stack_length


def compute_3d_stats(master_counts, placement, stack_length):
    medians = np.nanmedian(master_counts, axis=3)
    means = np.nanmean(master_counts, axis=3)
    std = np.nanstd(master_counts, axis=3)
    # counts = stack_length - np.count_nonzero(np.isnan(master_counts), axis=3)
    counts = placement - 1

    # far field corner should average to zero
    adjustment = np.nanmean(medians[120:, 120:, 140])

    # print(medians[120:, 120:, 140])
    print("adjustment: ", adjustment)
    medians = medians - adjustment
    means = means - adjustment

    return medians, means, std, counts


def do_3d_calibration(cal_params, data):
    cal_results_obj = DataObj()  # for storing results of calibration
    cal_results_obj.calibration_type = "3d"
    cal_results_obj.stats = data.stats
    cal_results_obj.data_params = data.params
    cal_results_obj.cal_params = cal_params

    delays, prime_1, prime_2, prime_3 = calculate_3d_diffs(
        data.data_tags, data.nearest_pulse_times, data.params["delay"]
    )  # returns diffs in nanoseconds

    prime_steps = cal_params["prime_steps"]
    prime_1 = (
        prime_1 / data.stats["pulse_rate"]
    )  # prime_1 must be in units of ns before division
    prime_2 = prime_2 / data.stats["pulse_rate"]
    prime_3 = prime_3 / data.stats["pulse_rate"]

    prime_1 = prime_1.astype("int")  # these would never be zero. They are accurate.
    prime_2 = prime_2.astype("int")
    prime_3 = prime_3.astype("int")

    master_counts, placement, stack_length = linear_3d_scan(
        prime_1, prime_2, prime_3, delays, prime_steps
    )

    medians, means, std, counts = compute_3d_stats(
        master_counts, placement, stack_length
    )

    x = np.arange(0, prime_steps)
    y = np.arange(0, prime_steps)
    x, y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_ylabel("prime 2")
    ax.set_xlabel("prime_1")
    ax.plot_surface(x, y, medians[:, :, 149])
    ax.set_zlim(-0.1, 0.1)

    cal_results_obj.medians = np.nan_to_num(medians)
    cal_results_obj.means = np.nan_to_num(means)
    cal_results_obj.std = np.nan_to_num(std)
    cal_results_obj.counts = np.nan_to_num(counts)

    if cal_params["output"]["save_analysis_result"]:
        save_name = cal_results_obj.export(
            cal_params["output"]["save_name"] + "3d_",
            include_time=True,
            print_info=True,
        )

    fig, ax = plt.subplots(1, 5, figsize=(13, 7), dpi=180)
    ax[0].imshow(
        medians[: cal_params["prime_steps"], : cal_params["prime_steps"], 20],
        vmin=-0.05,
        vmax=0.30,
    )
    ax[0].set_title(number_manager(data.stats["count_rate"]) + "  t''' = 20")

    ax[1].imshow(
        medians[: cal_params["prime_steps"], : cal_params["prime_steps"], 50],
        vmin=-0.05,
        vmax=0.30,
    )
    ax[1].set_title(number_manager(data.stats["count_rate"]) + "  t''' = 50")

    ax[2].imshow(
        medians[: cal_params["prime_steps"], : cal_params["prime_steps"], 149],
        vmin=-0.05,
        vmax=0.30,
    )
    ax[2].set_title(number_manager(data.stats["count_rate"]) + "  t''' = 149")

    ax[3].imshow(counts[: cal_params["prime_steps"], : cal_params["prime_steps"], 140])
    ax[3].set_title(f"max counts at 140: {np.max(counts)}")

    im = ax[4].imshow(
        std[: cal_params["prime_steps"], : cal_params["prime_steps"], 100]
    )
    ax[4].set_title(f"standard deviation at 100")
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plt.savefig(f"{save_name}.png")

    return cal_results_obj


def do_2d_calibration(cal_params, data):
    cal_results_obj = DataObj()  # for storing results of calibration
    cal_results_obj.calibration_type = "2d"
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
    prime_1 = prime_1.astype("int")  # these would never be zero. They are accurate.
    prime_2 = prime_2.astype("int")

    prime_1_masks = []
    prime_2_masks = []

    for i in tqdm(range(prime_steps)):
        prime_1_masks.append(prime_1 == i)
        prime_2_masks.append(prime_2 == i)

    medians, means, std, counts = do_2d_scan(
        prime_1_masks, prime_2_masks, delays, prime_steps, cal_params["min_sub_delay"]
    )

    # remove some outliers?
    # mask = np.roll(np.roll(np.eye(len(medians)), 3, axis=1), -3, axis=0).astype("bool")
    # medians[mask] = 0
    # medians[0:7, 27:] = 0
    # medians[0:7, :14] = 0

    x = np.arange(0, prime_steps)
    y = np.arange(0, prime_steps)
    x, y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_ylabel("prime 2")
    ax.set_xlabel("prime_1")
    ax.plot_surface(x, y, medians)
    ax.set_zlim(-0.1, 0.1)

    cal_results_obj.medians = medians
    cal_results_obj.means = means
    cal_results_obj.std = std
    cal_results_obj.counts = counts

    if cal_params["output"]["save_analysis_result"]:
        save_name = cal_results_obj.export(
            cal_params["output"]["save_name"] + "2d_",
            include_time=True,
            print_info=True,
        )

    fig, ax = plt.subplots(1, 3, figsize=(8, 4), dpi=300)
    ax[0].imshow(
        medians[: cal_params["prime_steps"], : cal_params["prime_steps"]],
        vmin=-0.05,
        vmax=0.30,
    )
    ax[0].set_title(number_manager(data.stats["count_rate"]))
    ax[1].imshow(counts[: cal_params["prime_steps"], : cal_params["prime_steps"]])
    ax[1].set_title(f"max counts: {np.max(counts)}")

    im = ax[2].imshow(std[: cal_params["prime_steps"], : cal_params["prime_steps"]])
    ax[2].set_title(f"standard deviation")
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plt.savefig(f"{save_name}.png")

    return cal_results_obj


def do_1d_calibration(cal_params, data):
    cal_results_obj = DataObj()  # for storing results of calibration
    cal_results_obj.calibration_type = "1d"
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
            cal_params["mask_from_period"],
            cal_results_obj,
        )

    elif cal_params["mask_method"] == "from_peaks":
        cal_results_obj = mask_manager.apply_mask_from_peaks(
            cal_params["mask_from_peaks"], cal_results_obj
        )
    else:
        print("Unknown decoding method")
        return 1

    cal_results_obj.data_params = data.params

    fig = cal_results_obj.fig
    ax = cal_results_obj.ax

    del cal_results_obj.fig
    del cal_results_obj.ax

    if cal_params["output"]["save_analysis_result"]:
        save_name = cal_results_obj.export(
            cal_params["output"]["save_name"], include_time=True, print_info=True
        )
    fig.savefig(f"{save_name}.png")
    # hmm I could incorporate figure saving into dataobj

    return cal_results_obj


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


def main():
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
            print(
                "using calibration file generated from file: ",
                calibration_obj.data_params["data_file"],
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
    if (not params["do_calibration"]) and (not params["do_correction"]):
        print("Calibration and correction turned off. Nothing to do. ")


if __name__ == "__main__":
    main()
