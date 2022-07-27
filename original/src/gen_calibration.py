
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
from scipy import stats
from scipy.stats import rv_continuous
from scipy import special

from ClockTools import clockLock
from datetime import date
from datetime import datetime
import phd.viz
import concurrent.futures
import glob
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib as mpl

from scipy import signal
from sklearn.neighbors import KernelDensity

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
    plt.title("check locking")
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



def bootstrap_median(section, number = 100):
    meds = np.zeros(number)
    for i in range(number):
        meds[i] = np.median(np.random.choice(section, size=len(section)))

    return np.std(meds)


def bootstrap_kde(section, estimator, bins, bins_idx_left, bins_idx_right, number = 50):
    lims_l = np.zeros(number)
    lims_r = np.zeros(number)
    for i in range(number):
        sect = np.random.choice(section[:8000], size=len(section))
        estimator.fit(sect[:, None])
        logprob = estimator.score_samples(bins[bins_idx_left:bins_idx_right, None])
        peaks, _ = signal.find_peaks(np.exp(logprob))
        Max = peaks[np.argmax(logprob[peaks])]
        _, _, lw, rw = signal.peak_widths(np.exp(logprob), np.array([Max]), rel_height=0.5)

        lims_r[i] = np.interp(rw, np.arange(0, bins_idx_right - bins_idx_left), bins[bins_idx_left:bins_idx_right])[0]
        lims_l[i] = np.interp(lw, np.arange(0, bins_idx_right - bins_idx_left), bins[bins_idx_left:bins_idx_right])[0]

    return np.std(lims_l)*4, np.std(lims_r)*4


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


def calculate_hist2d(tags, nearest_pulses, delay):

    diffs = tags - np.roll(nearest_pulses,1)
    diffs_2nd = np.roll(tags, 1) - np.roll(nearest_pulses,2)

    # remove weird endings
    diffs = diffs[2:-2]
    diffs_2nd = diffs_2nd[2:-2]

    diffs = diffs + delay
    diffs_2nd = diffs_2nd + delay

    diffs = diffs / 1000  # Now in nanoseconds
    diffs_2nd = diffs_2nd / 1000

    Bins = np.linspace(15, 62, 2 * 370)
    Bins_2d = np.zeros((2, len(Bins)))
    Bins_2d[0] = Bins
    Bins_2d[1] = Bins


    plt.figure()
    plt.hist2d(diffs, diffs_2nd, bins=(Bins, Bins), cmap=plt.cm.jet)  # , norm=mpl.colors.LogNorm())


def make_absolute_hist(tags, nearest_pulses, delay, Figures = False):
    diffsorg = tags - nearest_pulses
    guassDiffs = diffsorg + delay
    guassEdges = np.linspace(-2000, 2000, 800)
    guassHist, guassBins = np.histogram(guassDiffs, guassEdges, density=True)
    gaussianBG = gaussian_bg(a=guassDiffs.min() / 1000, b=guassDiffs.max() / 1000, name='gaussianBG')

    start = time.time()
    scalefactor = 1000
    guassStd2, guassAvg2, back, flock, fscale = gaussianBG.fit(guassDiffs[-30000:] / scalefactor, floc=0, fscale=1)
    guassStd = np.std(guassDiffs[-30000:])
    end = time.time()
    print("time of absolute fit: ", end - start)
    guassStd2 = guassStd2 * scalefactor
    guassAvg2 = guassAvg2 * scalefactor
    print("guassStd2: ", guassStd2)
    print("guassAvg2: ", guassAvg2)
    if Figures:
        plt.figure()
        plt.plot(guassBins[1:], guassHist)
        plt.plot(guassBins[1:], gaussianBG.pdf(guassBins[1:] / scalefactor, back=back, sigma=guassStd2 / scalefactor,
                                               mu=guassAvg2 / scalefactor) / scalefactor)
        plt.title("compiled histogram with a fitting guassian function. Not expected to be a good fit")

    return guassAvg2, guassStd2, guassBins, guassHist


def run_analysis(path_, file_, modu_params, DERIV, PROP, delayScan = False, delay = 0, Figures = True):
    pulses_per_clock = modu_params["cycles_per_sequence"]
    pulse_rate = modu_params["system"]["laser_rate"]/modu_params["regular"]["data"]["pulse_divider"]
    inter_pulse_time = 1/pulse_rate  # time between pulses in nanoseconds
    snspd_channel = -5
    clock_channel = 9
    full_path = os.path.join(path_, file_)
    file_reader = FileReader(full_path)
    n_events = 1000000000  # Number of events to read at once
    # Read at most n_events.
    # data is an instance of TimeTagStreamBuffer
    data = file_reader.getData(n_events)
    print('Size of the returned data chunk: {:d} events\n'.format(data.size))
    print('Showing a few selected timetags')
    channels = data.getChannels() # these are numpy arrays
    timetags = data.getTimestamps()
    SNSPD_tags = timetags[channels == -5]
    print("SNSPD TAGS:   ", len(SNSPD_tags))
    count_rate = 1e12*(len(SNSPD_tags)/(SNSPD_tags[-1] - SNSPD_tags[0]))
    print("Count rate is: ", count_rate)
    # delay analysis
    delayRange = np.array([i*2.5 - 1000 for i in range(1200)])
    dataNumbers = []

    if delayScan: # to be done with a low detection rate file (high attentuation)
        Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(channels[:10000], timetags[:10000],
                                                                                 clock_channel, snspd_channel,
                                                                                 pulses_per_clock, delay, window=0.05,
                                                                                 deriv=DERIV, prop=PROP)
        checkLocking(Clocks, RecoveredClocks)
        for i, delay in enumerate(delayRange):
            Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(channels[:10000], timetags[:10000],
                                                                                     clock_channel, snspd_channel,
                                                                                     pulses_per_clock, delay,
                                                                                     window=0.05, deriv=DERIV,
                                                                                     prop=PROP)
            deltaTimes = dataTags[1:-1] - np.roll(dataTags,1)[1:-1]
            dataNumbers.append(len(deltaTimes))
        dataNumbers = np.array(dataNumbers)
        delay = delayRange[np.argmax(dataNumbers)]
        plt.figure()
        plt.plot(delayRange,dataNumbers)
        plt.title("for identifying delay")
        return 0

    Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(channels, timetags, clock_channel,
                                                                             snspd_channel, pulses_per_clock, delay,
                                                                             window=.5, deriv=DERIV, prop=PROP,
                                                                             guardPeriod=10)

    print("length of clocks: ", len(Clocks))
    print("length of nearestPulseTimes: ", len(nearestPulseTimes))
    if Figures:
        plt.figure()
        print("length of reovered clocks: ", len(RecoveredClocks))
        checkLocking(Clocks[2000:150000], RecoveredClocks[2000:150000])

    guassAvg2, guassStd2, guassBins, guassHist = make_absolute_hist(dataTags[1:-1], nearestPulseTimes[1:-1], delay,
                                                                    Figures=Figures)

    # main diff calculation
    diffs = dataTags[2:-2] - np.roll(nearestPulseTimes[2:-2],1)
    diffs = diffs + delay
    diffs = diffs/1000  # Now in nanoseconds

    calculate_hist2d(dataTags[2:-2], nearestPulseTimes[2:-2], delay)

    ##############################
    ##############################
    # Set up red plot
    bins = np.linspace(0, 299, 70000)
    x_bins = np.arange(len(bins))

    hist, bins = np.histogram(diffs, bins)
    inter_pulse_time_ps = inter_pulse_time * 1000
    x = np.linspace(inter_pulse_time_ps, inter_pulse_time_ps*200, 200) / 1000
    pulses = np.array([i * inter_pulse_time for i in range(1, 200)])
    if Figures:
        plt.figure()
        plt.plot(bins[:-1],hist, color = 'black')
        plt.vlines(x, 0, 10000, color = 'red', alpha = 0.3)
        plt.yscale('log')
    # set up bounds for fitting
    adjustment = [(i**2.86)*.00053 for i in range(20)]
    adjustment.reverse()
    st = 6
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
    if Figures:
        plt.xlim(0,120)
        plt.yscale('log')
        plt.grid()

    ##############################
    ##############################
    # Set up for fitting histograms
    avgOffset = np.zeros(len(pulses))
    stdOffset = np.zeros(len(pulses))
    background = np.zeros(len(pulses))
    counts = np.zeros(len(pulses))
    tPrime = np.zeros(len(pulses))
    Ranges = np.zeros(len(pulses))
    r_widths = np.zeros(len(pulses))
    l_widths = np.zeros(len(pulses))
    median = np.zeros(len(pulses))
    fwhm_ranges = np.zeros(len(pulses))
    if Figures:
        plt.figure()
    scalefactor = 300

    # fit each histogram in a loop
    map = cm.get_cmap('viridis')
    a = time.time()
    print("starting loop")
    print("This is bins")
    kde = KernelDensity(bandwidth=.025, kernel='gaussian') # bw has units of nanoseconds
    for jj, pulse in enumerate(pulses):

        if jj >= 160: # avoiding some error I don't want to deal with
            break


        section = diffs[(diffs > t_start[jj]) & (diffs < t_end[jj])]
        section_org = section
        section = section[:50000] # don't need more stats than this.
        if len(section) > 10:  # if more than 5 counts, try to fit the counts to a guassian
            # q = time.time()

            bins_idx_left = np.searchsorted(bins, t_start[jj], side="left")-1
            bins_idx_right = np.searchsorted(bins, t_end[jj], side="left")


            histp_density, bins = np.histogram(section, bins, density=True)
            # v = time.time()
            # print("tstart: ", v - q)
            print("size of section: ", len(section))
            print("length of bins: ", len(bins))
            gaussianBG = gaussian_bg(a=t_start[jj]/scalefactor, b=t_end[jj]/scalefactor, name='gaussianBG')
            # for some reason it works better with larger numbers
            # q = time.time()
            Std2, Mu2, back, flock, fscale = gaussianBG.fit(section[:2000]/scalefactor , np.std(section[:50])/scalefactor, np.mean(section[:50])/scalefactor, floc=0, fscale=1 )
            print("fitting: ", jj)

            t1 = time.time()
            fwhm_l_ranges, fwhm_r_ranges = bootstrap_kde(section, kde, bins, bins_idx_left, bins_idx_right, 18)
            print("bootstrap range time: ", time.time() - t1)

            kde.fit(section[:8000, None])
            logprob = kde.score_samples(bins[bins_idx_left:bins_idx_right, None])
            peaks, _ = signal.find_peaks(np.exp(logprob))
            Max = peaks[np.argmax(logprob[peaks])]
            width, w_height, lw, rw = signal.peak_widths(np.exp(logprob), np.array([Max]), rel_height=0.5)
            r_widths[jj] = np.interp(rw, np.arange(0, bins_idx_right-bins_idx_left),bins[bins_idx_left:bins_idx_right])[0]
            l_widths[jj] = np.interp(lw, np.arange(0, bins_idx_right-bins_idx_left), bins[bins_idx_left:bins_idx_right])[0]

            # l_widths[jj] = np.interp(lw, x_bins, bins)
            # # I should work on slices of 'bins
            # print(max(np.exp(logprob)))

            Range = bootstrap_median(section,500)*4 # 4x sigma for 95% confidence interval
            print("Range: ", Range)
            print("this is rwidth: ", r_widths[jj])
            print("this is lwidth: ", l_widths[jj])


            Std2 = Std2*scalefactor
            Mu2 = Mu2*scalefactor
            tPrime[jj] = pulse
            avgOffset[jj] = Mu2 - pulse
            stdOffset[jj] = Std2
            background[jj] = back
            Ranges[jj] = Range
            counts[jj] = len(section_org)
            median[jj] = np.median(section) - pulse
            fwhm_ranges[jj] = math.sqrt(fwhm_l_ranges**2 + fwhm_r_ranges**2)


            if Figures:
                plt.plot(bins[1:], histp_density)
                # plt.plot(bins[1:], histp_density)
                # plt.plot(bins[bins_idx_left:bins_idx_right], np.exp(logprob), alpha=1, color='red')
                #
                # plt.plot(bins, len(section)*gaussianBG.pdf(bins / scalefactor, back=back, sigma=Std2 / scalefactor,
                #                         mu=Mu2 / scalefactor) / scalefactor, color = map(jj/120), alpha = 0.5)
                plt.plot(bins[bins_idx_left:bins_idx_right], np.exp(logprob), alpha=1, color='red')
                plt.axvspan(r_widths[jj], l_widths[jj], color='green', alpha = 0.3)
                # plt.axvline(x=Mu2, color = 'red')
                # plt.axvline(x=np.median(section), color = 'green')
                plt.axvline(median[jj], color='blue', alpha=0.4)
                plt.axvline(x[jj], color='red', alpha=0.3)
                plt.grid()
            # if jj > 9:
            #     break

    print("ending loop")
    b = time.time()
    print("loop time is: ", b - a)

    for X in x:
        plt.axvline(x=X, color='red', alpha=0.3)

    zeroMask = (tPrime != 0)
    avgOffset = avgOffset[zeroMask].tolist()
    stdOffset = stdOffset[zeroMask].tolist()
    background = background[zeroMask].tolist()
    counts = counts[zeroMask].tolist()
    tPrime = tPrime[zeroMask].tolist()
    Ranges = Ranges[zeroMask].tolist()
    FWHM_widths = r_widths - l_widths
    FWHM_widths = FWHM_widths[zeroMask].tolist()
    median = median[zeroMask].tolist()
    fwhm_ranges = fwhm_ranges[zeroMask].tolist()


    if Figures:
        plt.figure()
        plt.plot(tPrime,median)
        plt.errorbar(tPrime,median, yerr=Ranges, elinewidth=5, capsize=0, alpha = 0.35, marker='o', ms = 3)
        plt.title("delays")
        #plt.ylim(-1,5)

        # plt.figure()
        # plt.plot(tPrime,stdOffset)
        # plt.title("sigmas")

        # plt.figure()
        # plt.plot(tPrime, Ranges)
        # plt.title("Ranges")

        plt.figure()
        plt.plot(tPrime, FWHM_widths)
        plt.errorbar(tPrime, FWHM_widths, yerr=fwhm_ranges, elinewidth=5, capsize=0, alpha=0.35, marker='o', ms=3)
        plt.title("density est FWHM widths")



    print("count rate: ", count_rate)
    Dict = {"tPrime": tPrime, "avgOffset": avgOffset, "stdOffset": stdOffset, "background": background,
            "counts": counts, "guassAvg2": guassAvg2, "guassStd2": guassStd2, "count_rate": count_rate,
            "guassBins": guassBins.tolist(), "guassHist": guassHist.tolist(), "FWHM_widths": FWHM_widths,
             "median": median, "fwhm_ranges": fwhm_ranges, "ranges": Ranges}

    return Dict
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

        return run_analysis(self.Path, file_iterator, self.modu_params, DERIV=self.Deriv, PROP=self.Prop,
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
        path = "..//data//537.5MHz//"
        file = "jitterate_4s_-0.049_537.5MHz_44.0.2.ttbin"
        params_file = "..//data//537.5MHz//0_21.08.27.13.24_params.yml"
        with open(params_file, 'r') as f:
            modu_params = yaml.full_load(f)
        jitters = []
        basic_jitters = []
        voltages = []

        dB = file.split('_')[-1].split('.')[0]



        fig, ax = plt.subplots()
        # delay should be determined at a low count rate
        print("This script takes a while to run because it uses a computationally expensive method for calculating \n"
              "the error bars on the t'-vs-delay and t'-vs-FWHM curves. \n"
              "\n"
              "It outputs a file 'jitterate_537.5MHz_XX_XXXXX.json that contains the calibration curve found. This \n"
              "file is loaded into the gen_correction scripts. "
              "\n"
              "\n")
        dic = run_analysis(path, file, modu_params, DERIV=500, PROP=5e-12, delayScan=False, delay=-191, Figures=True)
        # dic = run_analysis(path, file, modu_params, DERIV=500, PROP=5e-12, delayScan=False, delay=1103, Figures=True)
        dic["dB"] = dB
        today_now = datetime.now().strftime("%y.%m.%d.%H.%M")

        out_path = ''
        out_file = f'jitterate_537.5MHz_{dB}_{today_now}.json'
        with open(os.path.join(out_path,out_file), 'w') as outfile:
            print("saving to ", os.path.join(out_path,out_file))
            json.dump(dic, outfile, indent = 4)