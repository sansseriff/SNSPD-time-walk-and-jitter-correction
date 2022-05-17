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

from clock_tools import agnostic_clock_lock
from clock_tools import clockLock
from datetime import date
from datetime import datetime

import matplotlib.cm as cm
import glob

import phd.viz
import concurrent.futures

from scipy import signal
from sklearn.neighbors import KernelDensity


Colors, palette = phd.viz.phd_style(text = 1)


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


def prepare_tags(path_, file_, modu_params, DERIV, PROP, delay = 0, Figures = True):
    pulses_per_clock = modu_params["cycles_per_sequence"]
    pulse_rate = modu_params["system"]["laser_rate"] / modu_params["regular"]["data"]["pulse_divider"]

    print("pulses per clock: ", pulses_per_clock)
    print("pulse rate: ", pulse_rate)
    inter_pulse_time = 1 / pulse_rate  # time between pulses in nanoseconds
    print("inter_pulse_time: ", inter_pulse_time)
    snspd_channel = -14
    clock_channel = 9

    data = np.load(os.path.join(path_, file_))
    channels = data["channels"]
    timetags = data["timetags"]

    count_rate = 1e12*(len(timetags)/(timetags[-1] - timetags[0]))

    Clocks, RecoveredClocks, dataTags, nearestPulseTimes, Cycles = clockLock(channels, timetags, clock_channel,
                                                                             snspd_channel, pulses_per_clock, delay,
                                                                             window=0.50, deriv=DERIV, prop=PROP,
                                                                             guardPeriod=600)
    if Figures:
        time_elapsed = 1e-12*(Clocks[-1]-Clocks[0])
        checkLocking(Clocks, RecoveredClocks, time = time_elapsed)
    print("Clock rate: ", len(Clocks)/(1e-12*(Clocks[-1]-Clocks[0])))

    return dataTags, nearestPulseTimes


def delay_correct(results_path, results_file, file, dataTags, rel_clocks, singleFigure = False, Figures = True, Fig = 0, Ax = 0):
    results = os.path.join(results_path, results_file)
    with open(results, 'r') as f:
        data = json.load(f)
    dB = file.split('_')[-1][:-10]

    # match file dB with correction data for that dB
    # for i in range(len(data)):
    #     if data[i]["dB"] == int(dB):
    #         break

    tPrime = data['pulses_x']
    offset = data['offsets']

    print("tPrime: :", tPrime[:40])
    print("offset: :", offset[:40])
    plt.figure()
    plt.plot(tPrime, offset)
    plt.title("tPrime and offset")



    deltaTs = dataTags - np.roll(dataTags,1)
    deltaTs = deltaTs[1:-1]/1000 # now in nanoseconds
    dataTags = dataTags[1:-1]
    # nearestPulseTimes = nearestPulseTimes[1:-1]
    rel_clocks = rel_clocks[1:-1]

    plt.figure()
    hist, bins = np.histogram(deltaTs, bins=500)
    plt.plot(bins[:-1], hist)
    plt.title("dist of deltaTs")

    print("delta Ts: ", deltaTs[:100])
    # GENERATE SHIFTS
    shifts = 1000*np.interp(deltaTs,tPrime,offset) # in picoseconds. Remove the 1st couple data
    print("shifts ", shifts[:100])

    plt.figure()
    hist, bins = np.histogram(shifts, bins = 500)
    plt.plot(bins[:-1], hist)
    plt.title("dist of shifts")
    # points because they are not generally valid.

    correctedTags = dataTags - shifts
    uncorrected_diffs = dataTags - rel_clocks
    corrected_diffs = correctedTags - rel_clocks

    guassEdges = np.linspace(-2000, 8000, 2000)
    uncorrected_Hist, guassBins = np.histogram(uncorrected_diffs, guassEdges, density=False)
    corrected_Hist, guassBins = np.histogram(corrected_diffs, guassEdges, density=False)

    cmap = cm.plasma
    colors = cmap(np.linspace(0, 1, 35))
    if not singleFigure:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=275)
    else:
        ax = Ax
        fig = Fig

    if Figures:
        # if int(dB) > 30:
        ax.plot(guassBins[1:], uncorrected_Hist, '--', color = 'black',alpha = 0.8, label = "uncorrected data")
        ax.plot(guassBins[1:], corrected_Hist, color = 'black', label = "corrected data")
        # ax.set_yscale('log')
        ax.grid(which='both')
        plt.legend()


    return 0

    guassBins = guassBins.tolist()
    uncorrected_Hist = uncorrected_Hist.tolist()
    corrected_Hist = corrected_Hist.tolist()
    print(type(corrected_Hist[0]))
    print(type(guassBins[0]))
    dic = {"guassBins": guassBins, "uncorrected_Hist": uncorrected_Hist, "corrected_Hist": corrected_Hist}

    return dic

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
    print(uncorr_std)
    print(corr_std)

    print("c_half_results height: ", c_half_results[1][0])
    corrected = {"lw_half": c_lw_half, "rw_half": c_rw_half, "height_half": float(c_half_results[1][0]),
                 "lw_tenth": c_lw_tenth, "rw_tenth": c_rw_tenth, "height_tenth": float(c_tenth_results[1][0]),
                 "lw_hund": c_lw_hund, "rw_hund": c_rw_hund, "height_hund": float(c_hund_results[1][0])}

    uncorrected = {"lw_half": uc_lw_half, "rw_half": uc_rw_half, "height_half": float(uc_half_results[1][0]),
                   "lw_tenth": uc_lw_tenth, "rw_tenth": uc_rw_tenth, "height_tenth": float(uc_tenth_results[1][0]),
                   "lw_hund": uc_lw_hund, "rw_hund": uc_rw_hund, "height_hund": float(uc_hund_results[1][0])}

    dict = {"corr_std": corr_std, "uncorr_std": uncorr_std, "dB": dB, "guassBins": guassBins[1:].tolist(),
            "uncorrected_Hist": uncorrected_Hist.tolist(), "corrected_Hist": corrected_Hist.tolist(),
            "uncorrPdf": uncorrPdf.tolist(), "corrPdf": corrPdf.tolist(), "corrected": corrected, "uncorrected": uncorrected}
    return dict

if __name__ == '__main__':
    dBScan = False
    LS = []
    if dBScan:
        pass

    else:
        # path = "..//data//3.5833GHz"
        # file = "3.5833GHz_-.025V_dBscan_60.0.1.ttbin"

        path = "..//data//4_GHz_peacoq_TT2_npz//"
        file = "TT2_4GHz_dBScan68.npz"
        params_file = "..//modu//custom_4ghz.yml"
        with open(params_file, 'r') as f:
            modu_params = yaml.full_load(f)

        results_path = "./"
        results_file = "peacoq_4GHz_jitterate_curve_3.json"
        dataTags, rel_clocks = prepare_tags(path, file, modu_params, DERIV=500, PROP=1e-11, delay = -190, Figures = True)


        count_rate = len(dataTags) / (1e-12 * (dataTags[-1] - dataTags[0]))
        print("count rate: ", count_rate)
        dic = delay_correct(results_path, results_file, file, dataTags, rel_clocks, singleFigure = False, Figures = True)

        # today_now = datetime.now().strftime("%y.%m.%d.%H.%M")
        # dic["count_rate"] = count_rate
        # with open('Jitterate_high_rate_correction_60dB_' + today_now + '.json', 'w') as outfile:
        #     json.dump(dic, outfile, indent=4)