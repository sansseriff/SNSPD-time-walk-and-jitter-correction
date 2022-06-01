from TimeTagger import createTimeTagger, FileWriter, FileReader
import numpy as np

from time import sleep

import struct
import matplotlib.pyplot as plt
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
    n = back + (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (((x - mu) / sigma) ** 2))
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


def save_incrementor(dic):

    pass  # see log on 3/30/2022 for what I want to make. A json saver using ()local


def runAnalysisJit(path_, file_, modu_params, DERIV, PROP, delayScan=False, delay=0, Figures=True):
    # pulses_per_clock = modu_params["cycles_per_sequence"]
    # pulse_rate = modu_params["system"]["laser_rate"]/modu_params["regular"]["data"]["pulse_divider"]
    pulses_per_clock = modu_params["pulses_per_clock"]
    # pulse_rate = modu_params["pulse_rate"]
    # pulse_rate = 1.000692
    # pulses_per_clock = 500
    # pulse_rate = 1 # GHz

    print("pulses per clock: ", pulses_per_clock)

    # get data set up
    snspd_channel = -5
    clock_channel = 9
    full_path = os.path.join(path_, file_)
    file_reader = FileReader(full_path)
    n_events = 1000000000  # Number of events to read at once

    n_events = 10000000  # Number of events to read at once
    data = file_reader.getData(n_events)
    print("Size of the returned data chunk: {:d} events\n".format(data.size))
    print("Showing a few selected timetags")
    channels = data.getChannels()  # these are numpy arrays
    timetags = data.getTimestamps()
    SNSPD_tags = timetags[channels == snspd_channel]
    CLOCK_tags = timetags[channels == clock_channel]
    print("SNSPD TAGS:   ", len(SNSPD_tags))
    count_rate = 1e12 * (len(SNSPD_tags) / (SNSPD_tags[-1] - SNSPD_tags[0]))
    print("Count rate is: ", count_rate)
    clock_rate = 1e12 * (len(CLOCK_tags) / (CLOCK_tags[-1] - CLOCK_tags[0]))
    pulse_rate = (clock_rate * pulses_per_clock) / 1e9
    print("Clock rate is: ", clock_rate)
    print("pulse rate: ", pulse_rate)
    inter_pulse_time = 1 / pulse_rate  # time between pulses in nanoseconds
    print("inter_pulse_time: ", inter_pulse_time)
    time_elapsed = 1e-12 * (SNSPD_tags[-1] - SNSPD_tags[0])
    print("time elapsed: ", time_elapsed)

    # optional delay analysis
    if delayScan:  # to be done with a low detection rate file (high attentuation)
        dataNumbers = []
        delayRange = np.array([i - 500 for i in range(1000)])
        Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(
            channels[:100000],
            timetags[:100000],
            clock_channel,
            snspd_channel,
            pulses_per_clock,
            delay,
            window=0.01,
            deriv=DERIV,
            prop=PROP,
        )
        checkLocking(Clocks, RecoveredClocks)
        for i, delay in enumerate(delayRange):
            Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(
                channels[:100000],
                timetags[:100000],
                clock_channel,
                snspd_channel,
                pulses_per_clock,
                delay,
                window=0.01,
                deriv=DERIV,
                prop=PROP,
            )
            deltaTimes = dataTags[1:-1] - np.roll(dataTags, 1)[1:-1]
            dataNumbers.append(len(deltaTimes))
        dataNumbers = np.array(dataNumbers)
        # delay = delayRange[np.argmax(dataNumbers)]
        plt.figure()
        plt.plot(delayRange, dataNumbers)
        print("Max counts found at delay: ", delayRange[np.argmax(dataNumbers)])
        plt.title("peak value is phase (ps) bewteen clock and SNSPD tags")
        print("Offset time: ", delayRange[np.argmax(dataNumbers)])
        return 0  # after

    # decode clocks and tags with correct delay
    Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(
        channels,
        timetags,
        clock_channel,
        snspd_channel,
        pulses_per_clock,
        delay,
        window=0.50,
        deriv=DERIV,
        prop=PROP,
        guardPeriod=600,
    )

    print("length of clocks: ", len(Clocks))
    print("length of nearestPulseTimes: ", len(nearestPulseTimes))
    if Figures:
        print("length of reovered clocks: ", len(RecoveredClocks))
        checkLocking(Clocks[2000:150000], RecoveredClocks[2000:150000])

    print("length of dataTags: ", len(dataTags))
    diffsorg = dataTags[1:-1] - nearestPulseTimes[1:-1]
    guassDiffs = diffsorg + delay

    guassEdges = np.linspace(int(-inter_pulse_time*1000*.5), int(inter_pulse_time*1000*.5), 4001) # 1 period width
    print("length of guassDiffs: ", len(guassDiffs))
    guassHist, guassBins = np.histogram(guassDiffs, guassEdges, density=True)
    gaussianBG = gaussian_bg(a=guassDiffs.min() / 1000, b=guassDiffs.max() / 1000, name="gaussianBG")
    start = time.time()
    print("starting fit")
    scalefactor = 1000
    guassStd2, guassAvg2, back, flock, fscale = gaussianBG.fit(guassDiffs[-30000:] / scalefactor, floc=0, fscale=1)
    guassStd = np.std(guassDiffs[-30000:])
    end = time.time()
    print("time of fit: ", end - start)
    guassStd2 = guassStd2 * scalefactor
    guassAvg2 = guassAvg2 * scalefactor
    # print("guassStd2: ", guassStd2)
    # print("guassAvg2: ", guassAvg2)
    if Figures:
        fig, ax = plt.subplots(2,1)
        ax[0].plot(guassBins[1:], guassHist)
        ax[1].plot(guassBins[1:], guassHist)
        ax[1].set_yscale('log')
        # plt.plot(
        #     guassBins[1:],
        #     gaussianBG.pdf(
        #         guassBins[1:] / scalefactor, back=back, sigma=guassStd2 / scalefactor, mu=guassAvg2 / scalefactor,
        #     )
        #     / scalefactor,
        # )
        ax[0].set_title("histogram of counts wrt clock")


    diffsR = dataTags[1:-1] - nearestPulseTimes[1:-1]
    nearestPulseTimes = np.roll(nearestPulseTimes, 1)
    dataTagsRolled = np.roll(dataTags, 1)
    diffs = dataTags[1:-1] - nearestPulseTimes[1:-1]
    diffs = diffs + delay
    diffs = diffs / 1000  # Now in nanoseconds

    ############################################

    # Set up red plot
    down_sample = 80
    # make two histograms. One high res for visualization. One low res for finding peaks
    # the peaks are used to specify the timing bounds used to form the groups for which the median is found
    max = 200000 # 200 ns
    bins = np.linspace(0, 200, max + 1)
    bins_peaks = np.linspace(0, 200, max//down_sample + 1)  # lower res for peak finding
    print("starting large histograms: ")
    hist, bins = np.histogram(diffs, bins, density=True)
    hist_peaks, bins_peaks = np.histogram(diffs, bins_peaks, density=True)
    print("ending large histograms")
    inter_pulse_time_ps = inter_pulse_time * 1000


    pulses = np.array([i * inter_pulse_time for i in range(1, 400)])

    # find peaks
    peaks, props = find_peaks(hist_peaks, height=0.01)

    peaks = np.sort(peaks)
    if Figures:
        plt.figure()
        print("length of bins", len(bins))
        print("length of hist: ", len(hist))
        # plt.plot(bins[:-1],hist, color = 'black')
        # plt.vlines(x, 0, 10000, color = 'red', alpha = 0.3)
        plt.plot(bins[:-1], hist, color="black")
        plt.plot(bins_peaks[:-1], hist_peaks, color="blue")
        # plt.vlines(bins[peaks*10], .01, 1, color='red', alpha=0.8)
        # plt.vlines(pulses - .0908, 0.01, 2, color="green", alpha=0.8)
        # plt.yscale('log')
        print("pulses: ", pulses[:40])
        print(bins[peaks * 10][:10])

    # adjustment = [(i ** 2.9) * 0.00023 for i in range(20)]
    adjustment = [(i ** 2.9) * 0 for i in range(20)]
    adjustment.reverse()
    st = 1
    t_start = np.zeros(len(pulses))
    t_end = np.zeros(len(pulses))
    for i in range(len(pulses)):
        if i < st:
            continue
        time_start = pulses[i] - inter_pulse_time / 2.1
        time_end = pulses[i] + inter_pulse_time / 2.1
        if (i >= st) and i < (st + len(adjustment)):
            time_start = time_start + adjustment[i - st]
            time_end = time_end + adjustment[i - st]
        if Figures:
            map = cm.get_cmap("viridis")
            plt.axvspan(time_start, time_end, alpha=0.3, color=map(i / 120))
        t_start[i] = time_start
        t_end[i] = time_end

    if Figures:
        plt.xlim(0, 120)
        # plt.yscale('log')
        plt.grid()

    # Set up for fitting histograms
    avgOffset = np.zeros(len(pulses))
    stdOffset = np.zeros(len(pulses))
    background = np.zeros(len(pulses))
    counts = np.zeros(len(pulses))
    tPrime = np.zeros(len(pulses))
    # if Figures:
    #     plt.figure()
    scalefactor = 300

    # print(t_start[20])
    map = cm.get_cmap("viridis")

    # pulses = pulses - .0908
    peaks_rh = bins[peaks * down_sample]

    pulses = pulses[pulses < 1000]
    peaks_rh = peaks_rh[peaks_rh < 1000]

    # for i in range(len(peaks_rh) - 1, -1, -1):
    plt.vlines(pulses, 0.01, 1, color="orange", alpha=0.8)
    plt.vlines(peaks_rh, 0.01, 1, color="purple", alpha=0.8)

    offsets = []
    pulses_x = []

    if Figures:
        plt.figure()

    j = 0
    pulses = np.sort(pulses)
    peaks_rh = np.sort(peaks_rh)
    peaks_rh = peaks_rh.tolist()
    pulses = pulses.tolist()

    # this is really dumb!!!
    while len(peaks_rh) != len(pulses):
        pulses.pop(0)
    # make the pulses and peaks_rh arrays the same length. They are aligned from the end.

    for i in tqdm(range(len(peaks_rh) - 2, 0, -1)):
        # print(i)
        j = j + 1
        bin_center = peaks_rh[i]
        bin_left = bin_center - (peaks_rh[i] - peaks_rh[i - 1]) / 2
        bin_right = bin_center + (peaks_rh[i + 1] - peaks_rh[i]) / 2

        bin_left_choked = bin_center - (peaks_rh[i] - peaks_rh[i - 1]) / 2.1
        bin_right_choked = bin_center + (peaks_rh[i + 1] - peaks_rh[i]) / 2.1
        mask = (diffs > bin_left) & (diffs < bin_right)
        mask_choked = (diffs > bin_left_choked) & (diffs < bin_right_choked)
        bin_tags = diffs[mask]
        bin_tags_choked = diffs[mask_choked]

        mini_bins = np.linspace(bin_left, bin_right, 100)
        # the individual histograms take some time...
        # mini_hist, mini_bins = np.histogram(bin_tags, mini_bins)
        # if Figures:
        #     plt.plot(mini_bins[:-1], mini_hist)
        #     plt.axvspan(bin_left_choked, bin_right_choked, alpha=0.3, color=map(i / len(peaks_rh)))
        #     plt.axvline(np.median(bin_tags), color = 'red')
        offset = np.median(bin_tags) - pulses[i]
        # print("pulses: ", pulses[i])
        # print("peak: ", peaks_rh[i])
        # print("np.median: ", np.median(bin_tags))
        # print("####################")
        offset_choked = np.median(bin_tags_choked) - pulses[i]
        offsets.append(offset_choked)
        pulses_x.append(pulses[i])

    pulses_x = np.array(pulses_x)
    offsets = np.array(offsets)

    sorter = np.argsort(pulses_x)
    pulses_x = pulses_x[sorter]
    offsets = offsets[sorter]

    zero_offset = np.mean(offsets[-40:])
    offsets = offsets - zero_offset  # offsets gets converted to a numpy array, from a list, because zero_offset is np.
    if Figures:
        plt.figure()
        plt.plot(pulses_x, offsets)
        plt.xlabel("time (ns)")
        plt.ylabel("offsets (ps)")
        plt.plot(pulses_x[-40:], offsets[-40:], lw=2.4, color="red")
        plt.grid()

    time_date = str(datetime.datetime.now()).replace(":", "_").replace("-", "_")[:-7]
    data_str = json.dumps({"pulses_x": pulses_x.tolist(), "offsets": offsets.tolist()})
    file_name_base = f"peacoq_1GHz_jitterate_curve_{time_date}"
    file_name = file_name_base + ".json"
    i = 0
    print("does it exist: ", exists(file_name))
    while exists(file_name):
        i += 1
        file_name = file_name_base + f"_{i}.json"
    with open(file_name, "w") as file:
        print(f"saving as {file_name}")
        file.write(data_str)


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

        return runAnalysisJit(
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
    dBScan = False
    LS = []
    if dBScan:
        # ** this is very specific to the dataset collected on April 5
        dBlist = [i * 2 + 26 for i in range(21)]
        path = "..//data//537.5MHz_0.1s//"
        file_list = ["jitterate_537.5MHz_-.025V_" + str(dB) + ".0.1.ttbin" for dB in dBlist]
        params_file = "..//modu//537.5MHz//0_21.08.27.13.24_params.yml"
        with open(params_file, "r") as f:
            modu_params = yaml.full_load(f)
        sleep_list = [(1.05 ** i) for i in range(10)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            dictList = executor.map(R(path, modu_params, 200000, 1e-14, False, -230, False), file_list,)
            # takes 10 - 20 minutes on 16 threads
        LS = []
        for dB, item in zip(dBlist, dictList):
            print(item)
            item["dB"] = dB
            LS.append(item)
        today_now = datetime.now().strftime("%y.%m.%d.%H.%M")
        with open("jitterate_swabianHighRes_537.5MHz_" + today_now + ".json", "w") as outfile:
            json.dump(LS, outfile, indent=4)

    else:
        path = "C://Users//Andrew//Documents//peacoq 1_GHz//Wire_1//41mV"
        file = "W1_41mV_3.0s_40.5.1.ttbin"
        # params_file = "..//modu//custom_4ghz.yml"
        # with open(params_file, 'r') as f:
        #     modu_params = yaml.full_load(f)

        modu_params = {"pulses_per_clock": 500, "pulse_rate": 1}
        jitters = []
        basic_jitters = []
        voltages = []

        fig, ax = plt.subplots()
        # delay should be determined at a low count rate
        dic = runAnalysisJit(path, file, modu_params, DERIV=500, PROP=1e-11, delayScan=False, delay=-204, Figures=True,)
        # -160 delay for sets higher power than 80
        # -140 delay for sets lower power than 80
