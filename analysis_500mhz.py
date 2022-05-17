
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
from datetime import datetime
import phd.viz
import concurrent.futures
import glob
from matplotlib import cm

Colors, palette = phd.viz.phd_style(text = 1)





def delayCorrect(_dataTags):
    delays = np.load("Delays_1.npy") # in picoseconds
    delayTimes = np.load("Delay_Times.npy") # in nanoseconds
    f2 = interp1d(delayTimes, delays, kind='cubic')
    print(max(delayTimes))
    print(min(delayTimes))
    xnew = np.linspace(min(delayTimes), max(delayTimes), num=200, endpoint=True)
    random = [5, 15,22,17,100]
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
        deltaTime = (tag - prevTag)/1000 # now in ns
        if deltaTime <= min(delayTimes) + .01:
            out = delays[0]
        elif deltaTime >= max(delayTimes) - .01:
            out = delays[-1]
        else:
            out = f2(deltaTime)
            #unprocessed[i] = tag

        newTags[i] = tag - out

        prevTag = tag
    print("end")
    return newTags#, unprocessed


def checkLocking(Clocks, RecoveredClocks):
    basis = np.linspace(Clocks[0], Clocks[-1], len(Clocks))
    diffs = Clocks - basis
    diffsRecovered = RecoveredClocks - basis
    x = np.arange(0, len(diffs))
    # fig1 = plt.figure()
    plt.plot(x, diffs)
    plt.plot(x, diffsRecovered)
    plt.title("Raw Clock and PLL Clock")
    # plt.plot(x,diffsRecovered)
    #plt.ylim(-1000, 1000)



def guassian_background(x,sigma,mu,back,l,r):
    "d was found by symbolically integrating in mathematica"
    n = back + (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*(((x-mu)/sigma)**2))
    d = 0.5*(2*back*(-l + r) + special.erf((-l + mu)/(np.sqrt(2)*sigma)) - special.erf((mu - r)/(np.sqrt(2)*sigma)))
    return n/d


class gaussian_bg(rv_continuous):
    "Gaussian distributionwithj Background parameter 'back'"
    def _pdf(self, x,sigma ,mu ,back):
        return guassian_background(x,sigma,mu,back,self.a,self.b)


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




def runAnalysisJit(path_, file_, modu_params, DERIV, PROP, delayScan = False, delay = 0, Figures = True):
    pulses_per_clock = modu_params["cycles_per_sequence"]
    pulse_rate = modu_params["system"]["laser_rate"]/modu_params["regular"]["data"]["pulse_divider"]

    print("pulses per clock: ", pulses_per_clock)
    print("pulse rate: ", pulse_rate)
    inter_pulse_time = 1/pulse_rate  # time between pulses in nanoseconds
    print("inter_pulse_time: ", inter_pulse_time)
    snspd_channel = -14
    clock_channel = 9
    # full_path = os.path.join(path_, file_)
    # file_reader = FileReader(full_path)
    # n_events = 1000000000  # Number of events to read at once
    # Read at most n_events.
    # data is an instance of TimeTagStreamBuffer
    # data = file_reader.getData(n_events)
    # print('Size of the returned data chunk: {:d} events\n'.format(data.size))
    # print('Showing a few selected timetags')
    # channels = data.getChannels() # these are numpy arrays
    # timetags = data.getTimestamps()

    data = np.load(os.path.join(path,file))
    channels = data["channels"]
    timetags = data["timetags"]

    SNSPD_tags = timetags[channels == snspd_channel]
    CLOCK_tags = timetags[channels == clock_channel]
    print("SNSPD TAGS:   ", len(SNSPD_tags))
    count_rate = 1e12*(len(SNSPD_tags)/(SNSPD_tags[-1] - SNSPD_tags[0]))
    print("Count rate is: ", count_rate)
    clock_rate = 1e12*(len(CLOCK_tags)/(CLOCK_tags[-1] - CLOCK_tags[0]))
    print("Clock rate is: ", clock_rate)
    # delay analysis
    delayRange = np.array([i*2.5 - 1000 for i in range(1200)])
    dataNumbers = []
    if delayScan: # to be done with a low detection rate file (high attentuation)

        Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(channels[:100000], timetags[:100000],
                                                                                 clock_channel, snspd_channel,
                                                                                 pulses_per_clock, delay, window=0.01,
                                                                                 deriv=DERIV, prop=PROP)
        checkLocking(Clocks, RecoveredClocks)
        for i, delay in enumerate(delayRange):
            Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(channels[:100000], timetags[:100000],
                                                                                     clock_channel, snspd_channel,
                                                                                     pulses_per_clock, delay,
                                                                                     window=0.01, deriv=DERIV,
                                                                                     prop=PROP)
            deltaTimes = dataTags[1:-1] - np.roll(dataTags,1)[1:-1]
            dataNumbers.append(len(deltaTimes))
        dataNumbers = np.array(dataNumbers)
        # print("index is:", np.argmax(dataNumbers))
        # print("delay is:",delayRange[np.argmax(dataNumbers)])
        delay = delayRange[np.argmax(dataNumbers)]
        plt.figure()
        plt.plot(delayRange,dataNumbers)
        plt.title("peak value is phase (ps) bewteen clock and SNSPD tags")
        print("Offset time: ", delayRange[np.argmax(dataNumbers)])
        return 0 # after


    Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(channels, timetags, clock_channel,
                                                                             snspd_channel, pulses_per_clock, delay,
                                                                             window=0.3, deriv=DERIV, prop=PROP,
                                                                             guardPeriod=600)

    print("diff dataTags and nearest pulse: ", nearestPulseTimes[:100] - dataTags[:100])
    # print(nearestPulseTimes[:100])

    print("length of clocks: ", len(Clocks))
    print("length of nearestPulseTimes: ", len(nearestPulseTimes))
    if Figures:
        plt.figure()
        plt.title("what is this")
        print("length of reovered clocks: ", len(RecoveredClocks))
        checkLocking(Clocks[2000:150000], RecoveredClocks[2000:150000])

    print(np.shape(nearestPulseTimes))
    print(np.shape(RecoveredClocks))
    print("DATATAGS: ", len(dataTags))
    diffsorg = dataTags[1:-1] - nearestPulseTimes[1:-1]
    guassDiffs = diffsorg + delay
    guassEdges = np.linspace(-2000, 2000, 800)
    guassHist, guassBins = np.histogram(guassDiffs, guassEdges,density = True)
    gaussianBG = gaussian_bg(a=guassDiffs.min()/1000, b=guassDiffs.max()/1000, name='gaussianBG')
    start = time.time()
    #print(guassDiffs[:50])
    print("starting fit")
    scalefactor = 1000
    guassStd2, guassAvg2, back, flock, fscale = gaussianBG.fit(guassDiffs[-30000:]/scalefactor , floc=0, fscale=1)
    guassStd = np.std(guassDiffs[-30000:])
    end = time.time()
    print("time of fit: ", end - start)
    guassStd2 = guassStd2*scalefactor
    guassAvg2 = guassAvg2*scalefactor
    print("guassStd2: ", guassStd2)
    print("guassAvg2: ", guassAvg2)
    if Figures:
        plt.figure()
        plt.plot(guassBins[1:],guassHist)
        plt.plot(guassBins[1:], gaussianBG.pdf(guassBins[1:] / scalefactor, back=back, sigma=guassStd2 / scalefactor,
                                               mu=guassAvg2 / scalefactor) / scalefactor)
        plt.title("compiled histogram with a fitting guassian function. Not expected to be a good fit")
    diffsR = dataTags[1:-1] - nearestPulseTimes[1:-1]
    nearestPulseTimes = np.roll(nearestPulseTimes,1)
    dataTagsRolled = np.roll(dataTags,1)
    diffs = dataTags[1:-1] - nearestPulseTimes[1:-1]
    diffs = diffs + delay
    diffs = diffs/1000  # Now in nanoseconds

    # Set up red plot
    bins = np.arange(0,200,.001)
    hist, bins = np.histogram(diffs, bins)
    inter_pulse_time_ps = inter_pulse_time * 1000
    x = np.linspace(inter_pulse_time_ps, inter_pulse_time_ps*200, 200) / 1000
    pulses = np.array([i * inter_pulse_time for i in range(1, 100)])


    if Figures:
        plt.figure()
        plt.plot(bins[:-1],hist, color = 'black')
        plt.vlines(x, 0, 10000, color = 'red', alpha = 0.3)
        plt.yscale('log')

    adjustment = [(i**2.9)*.00023 for i in range(20)]
    adjustment.reverse()
    st = 1
    t_start = np.zeros(len(pulses))
    t_end = np.zeros(len(pulses))
    for i in range(len(pulses)):
        if i < st:
            continue
        time_start = pulses[i] - inter_pulse_time / 2.4
        time_end = pulses[i] + inter_pulse_time / 2.4
        if (i >= st) and i < (st + len(adjustment)):
            time_start = time_start + adjustment[i-st]
            time_end = time_end + adjustment[i - st]
        if Figures:
            map = cm.get_cmap('viridis')
            plt.axvspan(time_start, time_end, alpha=0.3, color = map(i/120))
        t_start[i] = time_start
        t_end[i] = time_end

    print(t_start)
    print(t_end)
    if Figures:
        plt.xlim(0,120)
        plt.yscale('log')
        plt.grid()

    # Set up for fitting histograms
    pulses = np.array([i*inter_pulse_time for i in range(1,100)])
    avgOffset = np.zeros(len(pulses))
    stdOffset = np.zeros(len(pulses))
    background = np.zeros(len(pulses))
    counts = np.zeros(len(pulses))
    tPrime = np.zeros(len(pulses))
    if Figures:
        plt.figure()
    scalefactor = 300

    # print(t_start[20])
    map = cm.get_cmap('viridis')
    for jj, pulse in enumerate(pulses):
        section = diffs[(diffs > t_start[jj]) & (diffs < t_end[jj])]
        if len(section) > 5:  # if more than 5 counts, try to fit the counts to a guassian
            histp_density, bins = np.histogram(section, bins, density=True)
            gaussianBG = gaussian_bg(a=t_start[jj]/scalefactor, b=t_end[jj]/scalefactor, name='gaussianBG')
            # for some reason it works better with larger numbers
            # print("length of section: ", len(section))
            Std2, Mu2, back, flock, fscale = gaussianBG.fit(section[:5000]/scalefactor , np.std(section[:50])/scalefactor, np.mean(section[:50])/scalefactor, floc=0, fscale=1 )
            Std2 = Std2*scalefactor
            Mu2 = Mu2*scalefactor
            tPrime[jj] = pulse
            avgOffset[jj] = Mu2 - pulse
            stdOffset[jj] = Std2
            background[jj] = back
            counts[jj] = len(section)
            if Figures:
                plt.plot(bins[1:],len(section)*histp_density)
                plt.plot(bins, len(section)*gaussianBG.pdf(bins / scalefactor, back=back, sigma=Std2 / scalefactor,
                                        mu=Mu2 / scalefactor) / scalefactor, color = map(jj/120), alpha = 0.5)
                plt.grid()
    plt.vlines(x, 0, 2e6, color='red', alpha=0.3)

    zeroMask = (tPrime != 0)
    avgOffset = avgOffset[zeroMask].tolist()
    stdOffset = stdOffset[zeroMask].tolist()
    background = background[zeroMask].tolist()
    counts = counts[zeroMask].tolist()
    tPrime = tPrime[zeroMask].tolist()
    if Figures:
        plt.figure()
        plt.plot(tPrime,avgOffset)
        plt.title("delays")

        plt.figure()
        plt.plot(tPrime,stdOffset)
        plt.title("sigmas")
    print("count rate: ", count_rate)
    dict = {"tPrime": tPrime, "avgOffset": avgOffset, "stdOffset": stdOffset, "background": background,
            "counts": counts, "guassAvg2": guassAvg2, "guassStd2": guassStd2, "count_rate": count_rate, "guassBins": guassBins.tolist(),
            "guassHist": guassHist.tolist()}

    return dict
    # guassStd2 is the standard deviation of the jitter before correction
    # return avgOffset, stdOffset, background, counts, guassStd2


def sleeper(t, iter, tbla = 0):
    #time.sleep(t)
    for i in range(1000):
        q = np.sin(np.linspace(0,5,1000000))
    print("sleeping for: ", t)
    print("tbla is: ", tbla)
    return t

class runAnalysisCopier(object):
    def __init__(self, path, modu_params, DERIV, PROP, delayScan, delay, Figures):
        self.Path = path
        self.Deriv = DERIV
        self.Prop = PROP
        self.DelayScan = delayScan
        self.Delay = delay
        self.Figures = Figures
        self.modu_params = modu_params
    def __call__(self, file_iterator):

        return runAnalysisJit(self.Path, file_iterator, self.modu_params, DERIV=self.Deriv, PROP=self.Prop,
                              delayScan=self.DelayScan, delay=self.Delay, Figures=self.Figures)


if __name__ == '__main__':
    dBScan = False
    LS = []
    if dBScan:
        #** this is very specific to the dataset collected on April 5
        dBlist = [i * 2 + 26 for i in range(21)]
        path = "..//data//537.5MHz_0.1s//"
        file_list = ["jitterate_537.5MHz_-.025V_" + str(dB) + ".0.1.ttbin" for dB in dBlist]
        params_file = "..//modu//537.5MHz//0_21.08.27.13.24_params.yml"
        with open(params_file, 'r') as f:
            modu_params = yaml.full_load(f)
        sleep_list = [(1.05**i) for i in range(10)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            dictList = executor.map(runAnalysisCopier(path, modu_params, 200000, 1e-14, False, -230, False), file_list)
            # takes 10 - 20 minutes on 16 threads
        LS = []
        for dB,item in zip(dBlist,dictList):
            print(item)
            item["dB"] = dB
            LS.append(item)
        today_now = datetime.now().strftime("%y.%m.%d.%H.%M")
        with open('jitterate_swabianHighRes_537.5MHz_' + today_now + '.json', 'w') as outfile:
            json.dump(LS, outfile, indent = 4)

    else:
        # path = "..//data//537.5MHz_0.1s//"
        path = "..//data//500_MHz_peacoq_TT2_npz//"
        file = "TT2_dBScan37.npz"
        params_file = "..//modu//2503//0_22.03.10.14.01_params.yml"
        with open(params_file, 'r') as f:
            modu_params = yaml.full_load(f)
        jitters = []
        basic_jitters = []
        voltages = []

        fig, ax = plt.subplots()
        # delay should be determined at a low count rate
        dic = runAnalysisJit(path, file, modu_params, DERIV=500, PROP=8e-12, delayScan=False, delay=160, Figures=True)
        # print(dic)
        # 230
