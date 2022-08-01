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

from tqdm import tqdm
from mask_generators import MaskGenerator

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


def load_snspd_and_clock_tags(file: str, path: str, snspd_ch: int, clock_ch: int, read_events = 1e9, debug= False):
    # print(file)
    # print(path)
    full_path = os.path.join(path, file)
    file_reader = FileReader(full_path)
    print(type(read_events))
    data = file_reader.getData(int(read_events))
    if debug:
        print("load_snspd_and_clock_tags: Size of the returned data chunk: {:d} events\n".format(data.size))
    channels = data.getChannels()
    timetags = data.getTimestamps()
    print("length of timetags: ", len(timetags))

    SNSPD_tags = timetags[channels == snspd_ch]
    CLOCK_tags = timetags[channels == clock_ch]
    return SNSPD_tags, CLOCK_tags, channels, timetags


def data_statistics(modu_params, snspd_tags, clock_tags, debug = True):
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

    return {"count_rate": count_rate,
            "clock_rate": clock_rate,
            "pulse_rate": pulse_rate,
            "inter_pulse_time": inter_pulse_time,
            "time_elapsed": time_elapsed}


def delay_analysis(channels, timetags, clock_channel, snspd_channel, stats, delay, deriv, prop):
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
    print("You can update the analysis_params.yaml file with this delay and turn off delay_scan")
    plt.title("peak value is phase (ps) bewteen clock and SNSPD tags")
    print("Offset time: ", delay)
    return delay  # after


def make_histogram(dataTags, nearestPulseTimes, delay, stats, Figures):
    diffsorg = dataTags[1:-1] - nearestPulseTimes[1:-1]
    guassDiffs = diffsorg + delay

    guassEdges = np.linspace(int(-stats["inter_pulse_time"] * 1000 * .5), int(stats["inter_pulse_time"] * 1000 * .5),
                             4001)  # 1 period width
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

    if Figures:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(guassBins[1:], guassHist)
        ax[1].plot(guassBins[1:], guassHist)
        ax[1].set_yscale('log')
        ax[0].set_title("histogram of counts wrt clock")


def calculate_diffs(data_tags, nearest_pulse_times, delay):
    # this function subtracts the previous laser-based timing from the timing of snspd tags
    # output is in nanoseconds

    nearest_pulse_times = np.roll(nearest_pulse_times, 1)
    diffs = data_tags[1:-1] - nearest_pulse_times[1:-1]
    diffs = diffs + delay
    diffs = diffs / 1000  # Now in nanoseconds
    return diffs


# def runAnalysis(path_, file_, modu_params, analysis_params, DERIV, PROP, delayScan=False, delay=0, Figures=True):

def runAnalysis(params):

    t = time.time()
    print("t1: ", t)

    Figures = params["show_figures"]
    print("starting analysis")
    pulses_per_clock = params["modulation_params"]["pulses_per_clock"]
    # get data set up
    snspd_channel = params["snspd_channel"]
    clock_channel = params["clock_channel"]
    snspd_tags, clock_tags, channels, timetags = \
        load_snspd_and_clock_tags(params["data_file"],
                                  params["data_path"],
                                  params["snspd_channel"],
                                  params["clock_channel"],
                                  params["data_limit"])

    print("t2: ", time.time() - t)
    t = time.time()

    stats = data_statistics(params["modulation_params"], snspd_tags, clock_tags, debug=False)

    delay = params["delay"]
    # optional delay analysis
    if params["delay_scan"]:  # to be done with a low detection rate file (high attentuation)
        delay = delay_analysis(channels,
                               timetags,
                               clock_channel,
                               snspd_channel,
                               stats,
                               delay,
                               params["phase_locked_loop"]["deriv"],
                               params["phase_locked_loop"]["prop"])

    print("type of params[phase_locked_loop][prop]: ", type(params["phase_locked_loop"]["prop"]))

    # print("prop: ", params["phase_locked_loop"]["prop"])
    # decode clocks and tags with correct delay
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
        guardPeriod=params["phase_locked_loop"]["guard_period"])

    print("t3: ", time.time() - t)
    t = time.time()

    print("clock lock finished")
    if Figures:
        checkLocking(clocks[2000:150000], recovered_clocks[2000:150000])

    print('length of data tags: ', len(data_tags))
    make_histogram(data_tags[:params["histogram_max_tags"]], nearest_pulse_times[:params["histogram_max_tags"]], delay, stats, Figures)


    print("t4: ", time.time() - t)
    t = time.time()


    diffs = calculate_diffs(data_tags, nearest_pulse_times, delay) # returns diffs in nanoseconds

    print("t5: ", time.time() - t)
    t = time.time()


    mask_manager = MaskGenerator(diffs,
                                 analysis_params["analysis_range"],
                                 stats["inter_pulse_time"],
                                 figures=params["show_figures"],
                                 main_hist_downsample=1) # analysis out to 200 ns


    print("t6: ", time.time() - t)
    t = time.time()

    if params["mask_method"] == 'from_period':
        mask_manager.apply_mask_from_period(params["mask_from_period"])

    elif params["mask_method"] == 'from_peaks':
        mask_manager.apply_mask_from_peaks(params["mask_from_peaks"])
    else:
        print("Unknown decoding method")
        return 1



    print("t7: ", time.time() - t)
    t = time.time()

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

        return runAnalysis(
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

    with open("analysis_params.yaml", 'r') as f:
        analysis_params = yaml.safe_load(f)["params"]

    if analysis_params["analyze_set"]:
        LS = []
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
        print("t0: ", time.time())
        runAnalysis(analysis_params)

