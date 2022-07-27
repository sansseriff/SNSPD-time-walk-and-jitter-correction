
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

from ClockTools import clockLock
from datetime import date
from datetime import datetime

import matplotlib.cm as cm
import glob

import phd.viz
import concurrent.futures

from scipy import signal
from sklearn.neighbors import KernelDensity


Colors, palette = phd.viz.phd_style(text = 1)


def delayCorrectOld(_dataTags):
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



def count_rate(array):
    if len(array) == 0:
        return 0
    return len(array) / (1e-12 * (array[-1] - array[0]))

def delayCorrect(results_path, results_file, file, dataTags, nearestPulseTimes, singleFigure = False, Figures = True, Fig = 0, Ax = 0):
    results = os.path.join(results_path, results_file)
    with open(results, 'r') as f:
        data = json.load(f)
    dB = file.split('_')[-1][:-10]

    # match file dB with correction data for that dB
    # for i in range(len(data)):
    #     if data[i]["dB"] == int(dB):
    #         break

    tPrime = data['tPrime']
    offset = data['median']

    print("Tprime: ", tPrime)
    print("offset: ", offset)

    deltaTs = dataTags - np.roll(dataTags,1)
    deltaTs = deltaTs[1:-1]/1000 # now in nanoseconds
    dataTags = dataTags[1:-1]
    nearestPulseTimes = nearestPulseTimes[1:-1]


    shifts = 1000*np.interp(deltaTs,tPrime[2:],offset[2:]) # in picoseconds. Remove the 1st couple data
    # points because they are not generally valid.

    # print("shifts: ", shifts[2000:2100])

    correctedTags = dataTags - shifts
    uncorrected_diffs = dataTags - nearestPulseTimes
    corrected_diffs = correctedTags - nearestPulseTimes

    # print("len of corrected tags: ", len(correctedTags))

    cleaned_counts_150 = dataTags[deltaTs > 150]
    cleaned_counts_100 = dataTags[deltaTs > 100]
    cleaned_counts_50 = dataTags[deltaTs > 50]
    # print("len of suppressed tags: ", len(dataTags[deltaTs > 150]))

    cleaned_50_diffs = cleaned_counts_50 - nearestPulseTimes[deltaTs > 50]
    cleaned_100_diffs = cleaned_counts_100 - nearestPulseTimes[deltaTs > 100]
    cleaned_150_diffs = cleaned_counts_150 - nearestPulseTimes[deltaTs > 150]



    print("original count rate: ", count_rate(dataTags))
    print("cleaned count rate 50: ", count_rate(cleaned_counts_50))
    print("cleaned count rate 100: ", count_rate(cleaned_counts_100))
    print("cleaned count rate 150: ", count_rate(cleaned_counts_150))


    guassEdges = np.linspace(-2000, 2000, 800)
    uncorrected_Hist, guassBins = np.histogram(uncorrected_diffs, guassEdges, density=True)
    corrected_Hist, guassBins = np.histogram(corrected_diffs, guassEdges, density=True)

    cleaned_50_hist, guassBins = np.histogram(cleaned_50_diffs, guassEdges, density=True)
    cleaned_100_hist, guassBins = np.histogram(cleaned_100_diffs, guassEdges, density=True)
    cleaned_150_hist, guassBins = np.histogram(cleaned_150_diffs, guassEdges, density=True)

    cmap = cm.plasma
    colors = cmap(np.linspace(0, 1, 35))
    if not singleFigure:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=275)
    else:
        ax = Ax
        fig = Fig

    if Figures:
        if int(dB) > 30:
            ax.plot(guassBins[1:], uncorrected_Hist, '--', color = colors[int(int(dB)/2)],alpha = 0.8, label = "uncorrected data")
            ax.plot(guassBins[1:], corrected_Hist, color = colors[int(int(dB)/2)], label = "corrected data")
    scalefactor = 1000
    uncorr = gaussian_bg(a=uncorrected_diffs.min()/scalefactor, b=uncorrected_diffs.max()/scalefactor, name='gaussianBG')
    uncorr_std, uncorr_avg, uncorr_back, flock, fscale = uncorr.fit(uncorrected_diffs[:500000]/scalefactor,
                                                                    np.std(uncorrected_diffs[:500])/scalefactor,
                                                                    np.mean(uncorrected_diffs[:500])/scalefactor,
                                                                    floc=0, fscale=1)


    corr = gaussian_bg(a=corrected_diffs.min() / scalefactor, b=corrected_diffs.max() / scalefactor, name='gaussianBG')
    corr_std, corr_avg, corr_back, flock, fscale = corr.fit(corrected_diffs[:500000] / scalefactor,
                                                            np.std(corrected_diffs[:500]) / scalefactor,
                                                            np.mean(corrected_diffs[:500]) / scalefactor, floc=0,
                                                            fscale=1)
    corr_std = corr_std * scalefactor
    corr_avg = corr_avg * scalefactor
    uncorr_std = uncorr_std * scalefactor
    uncorr_avg = uncorr_avg * scalefactor

    # UNCORRECTED
    peaks_uncorrected, _ = signal.find_peaks(uncorrected_Hist)
    max_uncorrected = peaks_uncorrected[np.argmax(uncorrected_Hist[peaks_uncorrected])]
    uc_half_results = signal.peak_widths(uncorrected_Hist, np.array([max_uncorrected]), rel_height=0.5)
    uc_tenth_results = signal.peak_widths(uncorrected_Hist, np.array([max_uncorrected]), rel_height=0.9)
    uc_hund_results = signal.peak_widths(uncorrected_Hist, np.array([max_uncorrected]), rel_height=0.99)
    # go from index space to ps space
    uc_rw_half = np.interp(uc_half_results[3], np.arange(0, 800), guassBins)[0]
    uc_lw_half = np.interp(uc_half_results[2], np.arange(0, 800), guassBins)[0]
    uc_rw_tenth = np.interp(uc_tenth_results[3], np.arange(0, 800), guassBins)[0]
    uc_lw_tenth = np.interp(uc_tenth_results[2], np.arange(0, 800), guassBins)[0]
    uc_rw_hund = np.interp(uc_hund_results[3], np.arange(0, 800), guassBins)[0]
    uc_lw_hund = np.interp(uc_hund_results[2], np.arange(0, 800), guassBins)[0]

    # ordering: [0] is width, [1] is height, [2] is left location, [3] is right location


    # CORRECTED
    peaks_corrected, _ = signal.find_peaks(corrected_Hist)
    max_corrected = peaks_corrected[np.argmax(corrected_Hist[peaks_corrected])]

    c_half_results = signal.peak_widths(corrected_Hist, np.array([max_corrected]), rel_height=0.5)
    c_tenth_results = signal.peak_widths(corrected_Hist, np.array([max_corrected]), rel_height=0.9)
    c_hund_results = signal.peak_widths(corrected_Hist, np.array([max_corrected]), rel_height=0.99)
    # go from index space to ps space
    c_rw_half = np.interp(c_half_results[3], np.arange(0, 800), guassBins)[0]
    c_lw_half = np.interp(c_half_results[2], np.arange(0, 800), guassBins)[0]
    c_rw_tenth = np.interp(c_tenth_results[3], np.arange(0, 800), guassBins)[0]
    c_lw_tenth = np.interp(c_tenth_results[2], np.arange(0, 800), guassBins)[0]
    c_rw_hund = np.interp(c_hund_results[3], np.arange(0, 800), guassBins)[0]
    c_lw_hund = np.interp(c_hund_results[2], np.arange(0, 800), guassBins)[0]

    uncorrPdf = uncorr.pdf(guassBins[1:] / scalefactor, back=uncorr_back, sigma=uncorr_std / scalefactor,
                           mu=uncorr_avg / scalefactor) / scalefactor

    corrPdf = corr.pdf(guassBins[1:] / scalefactor, back=corr_back, sigma=corr_std / scalefactor,
                       mu=corr_avg / scalefactor) / scalefactor
    if Figures:
        ax.axvline(x=c_lw_half, color="blue")
        ax.axvline(x=c_rw_half, color="blue")
        ax.axvline(x=c_lw_tenth, color="green")
        ax.axvline(x=c_rw_tenth, color="green")

        ax.axvline(x=uc_lw_half, color="orange")
        ax.axvline(x=uc_rw_half, color="orange")
        ax.axvline(x=uc_lw_tenth, color="red")
        ax.axvline(x=uc_rw_tenth, color="red")
        ax.legend()
        ax.grid()
    # print(uncorr_std)
    # print(corr_std)

    print("c_half_results height: ", c_half_results[1][0])
    corrected = {"lw_half": c_lw_half, "rw_half": c_rw_half, "height_half": float(c_half_results[1][0]),
                 "lw_tenth": c_lw_tenth, "rw_tenth": c_rw_tenth, "height_tenth": float(c_tenth_results[1][0]),
                 "lw_hund": c_lw_hund, "rw_hund": c_rw_hund, "height_hund": float(c_hund_results[1][0])}

    uncorrected = {"lw_half": uc_lw_half, "rw_half": uc_rw_half, "height_half": float(uc_half_results[1][0]),
                   "lw_tenth": uc_lw_tenth, "rw_tenth": uc_rw_tenth, "height_tenth": float(uc_tenth_results[1][0]),
                   "lw_hund": uc_lw_hund, "rw_hund": uc_rw_hund, "height_hund": float(uc_hund_results[1][0])}

    dict = {"corr_std": corr_std, "uncorr_std": uncorr_std, "dB": dB, "guassBins": guassBins[1:].tolist(),
            "uncorrected_Hist": uncorrected_Hist.tolist(), "corrected_Hist": corrected_Hist.tolist(),
            "uncorrPdf": uncorrPdf.tolist(), "corrPdf": corrPdf.tolist(), "corrected": corrected, "uncorrected": uncorrected,

            "cleaned_cr_50": count_rate(cleaned_counts_50), "cleaned_cr_100": count_rate(cleaned_counts_100),
            "cleaned_cr_150":count_rate(cleaned_counts_150),

            "cleaned_50_hist": cleaned_50_hist.tolist(), "cleaned_100_hist": cleaned_100_hist.tolist(),
            "cleaned_150_hist": cleaned_150_hist.tolist()}

    return dict







def checkLocking(Clocks, RecoveredClocks, time = 0):
    basis = np.linspace(Clocks[0], Clocks[-1], len(Clocks))
    diffs = Clocks - basis
    diffsRecovered = RecoveredClocks - basis
    x = np.arange(0, len(diffs))
    # fig1 = plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=250)
    ax.plot(x, diffs)
    ax.plot(x, diffsRecovered)
    if time != 0:
        ax.set_title(f"Original and PLL Clock over {round(time,2)} s")
    # plt.plot(x,diffsRecovered)
    #ax.set_ylim(-1000, 1000)



def guassian_background(x,sigma,mu,back,l,r):
    "d was found by symbolically integrating in mathematica"
    n = back + (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*(((x-mu)/sigma)**2))
    d = 0.5*(2*back*(-l + r) + special.erf((-l + mu)/(np.sqrt(2)*sigma)) - special.erf((mu - r)/(np.sqrt(2)*sigma)))
    return n/d


class gaussian_bg(rv_continuous):
    "Gaussian distributionwithj Background parameter 'back'"
    def _pdf(self, x,sigma ,mu ,back):
        return guassian_background(x,sigma,mu,back,self.a,self.b)


def prepare_tags(path_, file_, modu_params, DERIV, PROP, delay = 0, Figures = True):
    pulses_per_clock = modu_params["cycles_per_sequence"]
    pulse_rate = modu_params["system"]["laser_rate"] / modu_params["regular"]["data"]["pulse_divider"]
    print("pulses per clock: ", pulses_per_clock)
    print("pulse_rate: ", pulse_rate)
    inter_pulse_time = 1 / pulse_rate  # time between pulses in nanoseconds
    snspd_channel = -5
    clock_channel = 9
    full_path = os.path.join(path_, file_)
    file_reader = FileReader(full_path)
    #while file_reader.hasData():
    n_events = 100000000  # Number of events to read at once
    # Read at most n_events.
    # data is an instance of TimeTagStreamBuffer
    data = file_reader.getData(n_events)
    channels = data.getChannels() # these are numpy arrays
    timetags = data.getTimestamps()
    count_rate = 1e12*(len(timetags)/(timetags[-1] - timetags[0]))

    Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(channels, timetags, clock_channel,
                                                                             snspd_channel, pulses_per_clock, delay,
                                                                             window=.5, deriv=DERIV, prop=PROP,
                                                                             guardPeriod=50000)
    print(Clocks[:20])
    print(RecoveredClocks[:20])
    print(Clocks[:20] - RecoveredClocks[:20])
    if Figures:
        time_elapsed = 1e-12*(Clocks[-1]-Clocks[0])
        checkLocking(Clocks, RecoveredClocks, time = time_elapsed)

    print("Clock rate: ", len(Clocks)/(1e-12*(Clocks[-1]-Clocks[0])))

    #print(len(dataTags))
    #print(len(nearestPulseTimes))
    return dataTags, nearestPulseTimes

class AnalysisCorrectCopier(object):
    def __init__(self, _path, _modu_params, _deriv, _prop, _delay,_Figures, _fig, _ax, _rpath, _rfile):
        self.path = _path
        self.deriv = _deriv
        self.prop = _prop
        self.delay = _delay
        self.Figures = _Figures
        self.fig = _fig
        self.ax = _ax
        self.rpath = _rpath
        self.rfile = _rfile
        self.modu_params = _modu_params
    def __call__(self, file_iterator):
        return AnalysisCorrect(self.path, file_iterator, self.modu_params, self.seriv, self.prop, self.delay, self.Figures,
                               self.fig, self.ax, self.rpath, self.rfile)



def AnalysisCorrect(_path, _file, modu_params, _DERIV, _PROP, _delay,_Figures, _fig, _ax, _rpath, _rfile):

    dataTags, nearestPulseTimes = prepare_tags(_path, _file, modu_params, DERIV=_DERIV, PROP=_PROP, delay=_delay,
                                                 Figures=_Figures)

    dict = delayCorrect(_rpath, _rfile, _file, dataTags, nearestPulseTimes, singleFigure=True, Fig=_fig, Ax=_ax)
    return dict


if __name__ == '__main__':
    dBScan = True
    LS = []
    if dBScan:

        path = "..//data//537.5MHz//"
        params_file = "..//data//537.5MHz//0_21.08.27.13.24_params.yml"
        with open(params_file, 'r') as f:
            modu_params = yaml.full_load(f)
        results_path = ""
        results_file = "jitterate_537.5MHz_44_22.07.26.19.09.json"

        dBlist = [i * 2 + 34 for i in range(17)]
        print(dBlist)

        dictList = []
        file_list = []
        for dB in dBlist:
            print(f"jitterate_4s_-0.049_537.5MHz_{dB}.0.2.ttbin")
            file = f"jitterate_4s_-0.049_537.5MHz_{dB}.0.2.ttbin"
            file_list.append(file)
            dataTags, nearestPulseTimes = prepare_tags(path, file, modu_params, DERIV=500, PROP=1e-12, delay=-700,
                                                       Figures=False)
            print("count rate: ", len(dataTags) / (1e-12 * (dataTags[-1] - dataTags[0])))
            dict = delayCorrect(results_path, results_file, file, dataTags, nearestPulseTimes, singleFigure=True,
                                Figures=False)
            dict["countRate"] = len(dataTags) / (1e-12 * (dataTags[-1] - dataTags[0]))
            dictList.append(dict)

        LS = []
        for file, item in zip(file_list, dictList):
            item["file"] = file
            LS.append(item)

        today_now = datetime.now().strftime("%y.%m.%d.%H.%M")
        with open('Jitterate_singlePixel_correction_withHists_final_' + today_now + '.json', 'w') as outfile:
            json.dump(LS, outfile, indent=4)


    else:
        path = "..//data//537.5MHz//"
        file = "jitterate_4s_-0.049_537.5MHz_44.0.2.ttbin"
        params_file = "..//data//537.5MHz//0_21.08.27.13.24_params.yml"
        with open(params_file, 'r') as f:
            modu_params = yaml.full_load(f)
        results_path = ""
        results_file = "jitterate_537.5MHz_44_22.07.26.19.09.json"
        dataTags, nearestPulseTimes = prepare_tags(path,file, modu_params, DERIV = 500, PROP = 1e-13, delay = -700, Figures = True)
        print("count rate: ", len(dataTags) / (1e-12 * (dataTags[-1] - dataTags[0])))
        dict = delayCorrect(results_path, results_file, file, dataTags, nearestPulseTimes, singleFigure = False, Figures = True)
        dict["countRate"] =  len(dataTags) / (1e-12 * (dataTags[-1] - dataTags[0]))
