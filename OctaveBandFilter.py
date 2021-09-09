from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Based on ISO61260. Heavily based on code from Encida

def getFrequencies(lowestFreq, highestFreq, bandsPerOct, base=10, refFreq=1000): #Get frequencies for filter band design
# refFreq - The reference frequency which will be the center of a band and around which all other bands will be created
    # frequency matrix
    freqs = np.empty((0, 3))

    if base == 10:
        base = 10 ** (3 / 10)  # Base ten
    elif base == 2:
        base = 2
    else:
        print('The base system is not permitted. G must be 10 or 2')

    x = -1000
    f2 = 0
    while f2 <= highestFreq:
        # Exact midband frequencies
        if bandsPerOct % 2 == 0:  # even
            fm = (base ** ((2 * x - 59) / (2 * bandsPerOct))) * (refFreq)
        else:  # odd
            fm = (base ** ((x - 30) / bandsPerOct)) * (refFreq)
        # Bandedge frequencies
        f1 = (base ** (-1 / (2 * bandsPerOct))) * (fm)
        f2 = (base ** (1 / (2 * bandsPerOct))) * (fm)
        if f2 >= lowestFreq:
            freqs = np.append(freqs, np.array([[f1, fm, f2]]), axis=0)
        x += 1
    return {'frequencies': freqs, 'Bands': bandsPerOct, 'Base': base}


def designFilters(filterSpec, samplingRate=48000, plot=False): # Produces filter coefficients based on the specifications provided

    frequencies = filterSpec['frequencies']
    bands = filterSpec['Bands']
    base = filterSpec['Base']

    order = 4
    filters = np.empty((0, 6))
    for index in range(np.size(frequencies, axis=0)):
        lowCutoff = 2 * frequencies[index, 0] / samplingRate
        highCutoff = 2 * frequencies[index, 2] / samplingRate
        sos = signal.butter(order, [lowCutoff, highCutoff], btype='bandpass', output='sos')
        filters = np.append(filters, sos, axis=0)

    # Plots the curves of the filters to show how the bands intersect
    if plot is True:
        # Requirements
        maskLevelmax = -np.array([0.15, 0.2, 0.4, 1.1, 4.5, 4.5, 200, 200, 200, 200])
        maskLevelmin = -np.array([-0.15, -0.15, -0.15, -0.15, -0.15, 2.3, 18, 42.5, 62, 75])
        breakpoints = np.array([0, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 1 / 2, 1, 2, 3, 4])

        freqH = 1 + ((base ** (1 / (2 * bands)) - 1) / (base ** (1 / 2) - 1)) * (base ** (breakpoints) - 1)
        freqL = 1 / freqH

        plt.figure('Filter responses with {} bands from {} Hz to {} Hz'.format(np.size(frequencies, 0), np.int(frequencies[0, 0]), np.int(frequencies[-1, 2])))
        for index in range(np.size(frequencies, axis=0)):
            # Plot
            w, h = signal.sosfreqz(filters[(order * index):(order * index + order), :], worN=fs)
            plt.semilogx(freqH * frequencies[index, 1], maskLevelmax, 'k--')
            plt.semilogx(freqH * frequencies[index, 1], maskLevelmin, 'k--')
            plt.semilogx(freqL * frequencies[index, 1], maskLevelmax, 'k--')
            plt.semilogx(freqL * frequencies[index, 1], maskLevelmin, 'k--')
            plt.semilogx((samplingRate * 0.5 / np.pi) * w, 20 * np.log10(abs(h)))
            plt.ylim([-100, 1])
        plt.show()
    return filters


def filterData(filters, data): # Filters the data with filter coefficients provided

    order = 4
    # Construct signal
    if data.ndim == 1:
        filteredSignal = np.zeros([np.size(data, axis=0), int(np.size(filters, 0) / order)])
        for index in range(int(np.size(filters, axis=0) / order)):
            filteredSignal[:, index] = signal.sosfilt(filters[(order * index):(order * index + order), :], data)
    elif data.ndim == 2:
        filteredSignal = np.zeros([np.size(data, axis=0), int(np.size(filters, 0) / order), np.size(data, axis=1)])
        for dataIndex in range(np.size(data, axis=1)):
            for bandIndex in range(int(np.size(filters, axis=0) / order)):
                filteredSignal[:, bandIndex, dataIndex] = signal.sosfilt(filters[(order * bandIndex):(order * bandIndex + order), :], data[:, dataIndex])
    return filteredSignal