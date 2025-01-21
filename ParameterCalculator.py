"""
This file implements all function and calculations that are needed in the process of analysing impulses

"""
import numpy as np
import math
import scipy
import librosa
import iracema
import OctaveBandFilter as obf
import mosqito
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d


class Harmonics:
    def __init__(self, frequencies, amplitudes):
        self.frequencies = frequencies
        self.amplitudes = amplitudes


def InsertIntoVstack(vector, stack):
    if len(stack) == 0:
        stack = [vector]
    else:
        stack = np.vstack([stack, vector])
    return stack


def CalculateAverageVector(Vectors): # Takes a vector of vectors and calculates a average vector
    averageVector = np.mean(Vectors, axis=0)
    return averageVector


def CalculateFFTs(takes, samplingRate, attackTime, sustainTime): # Takes array of impulses and creates array of spectrums
    attackSpectrums, sustainSpectrums, decaySpectrums, fullSpectrums = [], [], [], []
    attackFrequencies, sustainFrequencies, decayFrequencies, fullFrequencies = 0, 0, 0, 0

    #These are experimental values used to calibrate to full scale
    full, attack, sustain, decay = 48.87, 35.19, 40.42, 47.98

    for take in takes:
        fullSpectrum = scipy.fft.fft(take, norm='ortho')
        fullFrequencies = scipy.fft.fftfreq(len(take), 1/samplingRate)
        attackSpectrum = scipy.fft.fft(take[:int(attackTime*samplingRate)], norm='ortho')
        attackFrequencies = scipy.fft.fftfreq(len(take[:int(attackTime*samplingRate)]), 1/samplingRate)
        sustainSpectrum = scipy.fft.fft(take[int(attackTime*samplingRate):int(sustainTime*samplingRate)], norm='ortho')
        sustainFrequencies = scipy.fft.fftfreq(len(take[int(attackTime*samplingRate):int(sustainTime*samplingRate)]), 1/samplingRate)
        decaySpectrum = scipy.fft.fft(take[int(sustainTime*samplingRate):], norm='ortho')
        decayFrequencies = scipy.fft.fftfreq(len(take[int(sustainTime*samplingRate):]), 1/samplingRate)

        # Scaling to be decibels Full Scale
        fullSpectrum = 10 * np.log10(abs(fullSpectrum)) - full
        attackSpectrum = 10 * np.log10(abs(attackSpectrum)) - attack
        sustainSpectrum = 10 * np.log10(abs(sustainSpectrum)) - sustain
        decaySpectrum = 10 * np.log10(abs(decaySpectrum)) - decay

        fullSpectrums = InsertIntoVstack(fullSpectrum, fullSpectrums)
        attackSpectrums = InsertIntoVstack(attackSpectrum, attackSpectrums)
        sustainSpectrums = InsertIntoVstack(sustainSpectrum, sustainSpectrums)
        decaySpectrums = InsertIntoVstack(decaySpectrum, decaySpectrums)

    return fullFrequencies, fullSpectrums, attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums


# Creates vector of mathematical harmonic frequencies based on the fundamental pitch
def CreateMathematicalHarmonicFrequencyVector(pitch, n):
    freq = []
    for i in range (1, n):
        freq.append(pitch * i)
    return freq


# Based on harmonic data from iracema estimates the fundamental pitch
def EstimateFundamentalPitch(harmonicFrequencies):
    fundamentalPitches = []
    for i in range (0, len(harmonicFrequencies)):
        fundamentalPitches.append(harmonicFrequencies[i]/(i+1))
    return np.mean(fundamentalPitches)


# Takes spectrum of impulse and the mathematical harmonic frequencies and returns the amplitudes and frequencies of harmonics in the signal
def ExtractHarmonicDataFromSpectrums(spectrums, spectrumFrequencies, mathHarmonicFrequencies, bufforInHZ = 5):
    buffor = int(bufforInHZ/(spectrumFrequencies[1]-spectrumFrequencies[0]))
    harmonicData = []

    for spectrum in spectrums:
        amplitudes, harmonicFrequencies, harmonicNumber = [], [], 0

        for fftSample in range (0, len(spectrum)):
            if spectrumFrequencies[fftSample] > mathHarmonicFrequencies[harmonicNumber]:
                peakValue, peakFrequency = -200, 0

                for i in range (fftSample-buffor, fftSample+buffor):
                    if pow(10, spectrum[i]/10) >= peakValue:
                        peakValue = pow(10, spectrum[i]/10)
                        peakFrequency = spectrumFrequencies[i]

                amplitudes.append(peakValue)
                harmonicFrequencies.append(peakFrequency)

                if harmonicNumber < len(mathHarmonicFrequencies) - 1:
                    harmonicNumber = harmonicNumber + 1
                else:
                    break

        # showing found harmonics and spectrum for debugging
        """
        print(amplitudes)
        print(harmonicFrequencies)

        x = []
        for y in range(0, len(harmonicFrequencies)):
            x.append(y + 1)
        plt.subplot(211)
        plt.plot(spectrumFrequencies, spectrum)
        plt.xlim([0,4000])
        plt.subplot(212)
        plt.bar(x, amplitudes)
        plt.xlim([0, 30])
        plt.show()
        """

        harmonicData.append(Harmonics(harmonicFrequencies, amplitudes))
    return harmonicData


def SmoothEnvelope(envelope, window_size, sigma=20, padding_mode='reflect'):

    # Pad the envelope to minimize boundary effects
    pad_size = int(3 * sigma)
    padded_envelope = np.pad(envelope, pad_size, mode=padding_mode)

    smoothed_padded_envelope = np.convolve(padded_envelope, np.ones(window_size)/window_size, mode='same')

    # Remove the padding
    smoothed_envelope = smoothed_padded_envelope[pad_size:-pad_size]

    return smoothed_envelope


def CalculateNoisiness(spectrums, frequencies, harmonicsData, harmonicWidth = 0.5):
    noisinesses = []
    for takeNumber in range (0, len(spectrums)):
        fullEnergy, harmonicEnergy = 0, 0

        for sample in spectrums[takeNumber]:
            fullEnergy += pow(sample, 2)
        for harmonicFrequency in harmonicsData[takeNumber].frequencies:
            for frequency in range (0, len(frequencies)):
                if frequencies[frequency] >= harmonicFrequency-harmonicWidth and frequencies[frequency] <= harmonicFrequency+harmonicWidth:
                    harmonicEnergy += pow(spectrums[takeNumber][frequency], 2)

        noisinesses.append((fullEnergy-harmonicEnergy)/fullEnergy)
    return noisinesses


def CalculateIrregularity(harmonicData): # Calculates spectral harmonic irregularity according to Krimphoff, McAdams 1994
    irregularities = []
    for take in harmonicData:
        irregularity = 0
        for i in range (1, len(take.amplitudes)-1):
            irregularity += abs(take.amplitudes[i] - np.mean(take.amplitudes[i - 1] + take.amplitudes[i] + take.amplitudes[i + 1]))
        irregularities.append(np.log10(irregularity))
    return irregularities


def CalculateHighEnergyLowEnergyRatio(spectrums, frequencies, boundaryFrequency = 1500):
    highlowenergies = []
    for take in spectrums:
        highEnergy, lowEnergy = 0, 0
        for sample in range(0, len(take)):
            if frequencies[sample] < boundaryFrequency:
                lowEnergy += pow(take[sample], 2)
            else:
                highEnergy += pow(take[sample], 2)
        highlowenergies.append(highEnergy/lowEnergy)
    return highlowenergies


def CalculateTristimulus(harmonicsData):
    tristimulus1s, tristimulus2s, tristimulus3s = [], [], []
    for take in harmonicsData:
        allAmplitudes = 0
        fiveUpAmplitudes = 0
        for i in range (0, len(take.amplitudes)):
            allAmplitudes += take.amplitudes[i]
            if i >= 5:
                fiveUpAmplitudes += take.amplitudes[i]
        tristimulus1s.append(take.amplitudes[0]/allAmplitudes)
        tristimulus2s.append((take.amplitudes[1]+take.amplitudes[2]+take.amplitudes[3])/allAmplitudes)
        tristimulus3s.append(fiveUpAmplitudes/allAmplitudes)
    return tristimulus1s, tristimulus2s, tristimulus3s


def CalculateInharmonicity(harmonicsData):
    inharmonicities = []
    for take in harmonicsData:
        fundumentalPitch = EstimateFundamentalPitch(take.frequencies)
        inharmonicity, allAmplitudes = 0, 0
        for i in range(0,len(take.frequencies)):
            inharmonicity += (abs(take.frequencies[i]-fundumentalPitch*(i+1))*pow(take.amplitudes[i], 2))
            allAmplitudes += pow(take.amplitudes[i], 2)
        inharmonicities.append((2*inharmonicity)/(fundumentalPitch*allAmplitudes))
    return inharmonicities


def CalculateOERs(harmonicsData):
    oers = []
    for take in harmonicsData:
        odd, even = 0, 0
        for i in range (0, len(take.amplitudes)):
            if (i+1)%2 == 1:
                odd += pow(take.amplitudes[i], 2)
            else:
                even += pow(take.amplitudes[i], 2)
        oers.append(odd/even)
    return oers


def CalculateLoudness(args):
    N, N_spec, bark_axis, time_axis = mosqito.loudness_zwtv(args.impulseLIB, args.samplingRate, field_type="free")
    return max(N), np.mean(N)


def CalculateRoughness(args):
    r, r_spec, bark, time = mosqito.roughness_dw(args.impulseLIB, args.samplingRate)
    return np.mean(r)


def CalculateRMS(args):
    return np.sum(librosa.feature.rms(args.impulseLIB))


#Calculates the signals temporal centroid. Only takes into account signal over threshold to disguard silence.
# Watch out when using signals of different lengths.
def CalculateTemporalCentroid(args, windowLength = 128, hopsize = 64):
    envelope = iracema.features.peak_envelope(args.impulseIRA, windowLength, hopsize)
    envelope.data = 10 * np.log10(abs(envelope.data))

    # Smooth the envelope
    windowSize = int(args.samplingRate * 0.0007)
    smoothedEnvelope = SmoothEnvelope(envelope.data, windowSize)

    # Normalize the log envelope
    smoothedEnvelope -= np.min(smoothedEnvelope)
    smoothedEnvelope /= np.max(smoothedEnvelope)

    amplitudeSum = np.sum(smoothedEnvelope)
    ampXTimeSum = np.sum(smoothedEnvelope * envelope.time)

    tempCentroid = ampXTimeSum/amplitudeSum

    # Drawing the original envelope, the smoothed one and temporal centroid bands laid on top for both
    # smoothed_indicator = np.zeros(len(envelope.data))
    # smoothed_index = np.argmin(np.abs(envelope.time - tempCentroid))
    # smoothed_indicator[smoothed_index] = 1
    #
    # plt.figure()
    # plt.plot(smoothedEnvelope, color='silver', label='Smoothed Envelope')
    # plt.plot(envelope.data, color='grey', label='Envelope')
    # plt.plot(smoothed_indicator, color='chartreuse', label='Smoothed Temporal Centroid')
    # plt.legend(loc='lower right')
    # #plt.xlim(0, 1500)
    # plt.show()

    return tempCentroid

# Calculate log of attack time of signal. The algorythm was simplified when it comes to finding the start time of attack due to the it giving better results for guitar
def CalculateLogAttackTime(args, windowLength = 254, hopsize = 32, threshold = 0.1):
    envelope = iracema.features.peak_envelope(args.impulseIRA, windowLength, hopsize)
    envelope.data = 10 * np.log10(abs(envelope.data))

    startTime, stopTime = 0, 0

    windowSize = int(args.samplingRate * 0.0007)
    smoothedEnvelope = SmoothEnvelope(envelope.data, windowSize)
    maxEnv = max(smoothedEnvelope)

    startIndicator = np.zeros(len(envelope.data))
    stopIndicator = np.zeros(len(envelope.data))

    max_y_diff = 0
    start_index = 0
    end_index = 0

    current_start = 0
    for i in range(1, len(smoothedEnvelope)):
        if smoothedEnvelope[i] > smoothedEnvelope[i - 1]:  # Still rising
            continue
        else:
            # Calculate the y-axis difference for the current segment
            y_diff = smoothedEnvelope[i - 1] - smoothedEnvelope[current_start]
            if y_diff > max_y_diff:
                max_y_diff = y_diff
                start_index = current_start
                end_index = i - 1

            # Reset for the next segment
            current_start = i

    # Check the last segment
    y_diff = smoothedEnvelope[-1] - smoothedEnvelope[current_start]
    if y_diff > max_y_diff:
        max_y_diff = y_diff
        start_index = current_start
        end_index = len(smoothedEnvelope) - 1

    startTime = envelope.time[start_index]
    stopTime = envelope.time[end_index]

    startIndicator[start_index] = -50
    stopIndicator[end_index] = -50

    # Drawing the original envelope, the smoothed one and start end bands laid on top
    # plt.figure()
    # plt.plot(smoothedEnvelope+5, color='silver', label='Smoothed Envelope')
    # plt.plot(envelope.data, color='grey', label='Envelope')
    # plt.plot(startIndicator, 'orangered', label='Start Decay Time')
    # plt.plot(stopIndicator, color='chartreuse', label='Stop Decay Time')
    # plt.legend(loc='lower right')
    # plt.xlim(0, 1500)
    # plt.show()

    if stopTime != startTime:
        return math.log10(stopTime - startTime)
    else:
        return 0


# Calculates time between the peak of impulse and it decaying below the value of max-threshold in dB
def CalculateDecayTime(args, windowLength = 2048, hopsize = 128, threshold = 10):
    envelope = iracema.features.peak_envelope(args.impulseIRA, windowLength, hopsize)
    envelope.data = 10 * np.log10(abs(envelope.data))

    windowSize = int(args.samplingRate * 0.002)
    smoothedEnvelope = SmoothEnvelope(envelope.data, windowSize)
    maxEnv = max(smoothedEnvelope)

    startIndicator = np.zeros(len(envelope.data))
    stopIndicator = np.zeros(len(envelope.data))

    peakTime = 0
    decayTime = 0
    flag = False

    for i in range(0, len(smoothedEnvelope)):
        if smoothedEnvelope[i] == maxEnv:
            peakTime = envelope.time[i]
            startIndicator[i] = -50
            flag = True

        if (maxEnv - threshold > smoothedEnvelope[i]) & flag:
            decayTime = envelope.time[i] - peakTime
            stopIndicator[i] = -50
            break

    # Drawing the original envelope, the smoothed one and start end bands laid on top
    # plt.figure()
    # plt.plot(smoothedEnvelope+5, color='silver', label='Smoothed Envelope')
    # plt.plot(envelope.data, color='grey', label='Envelope')
    # plt.plot(startIndicator, 'orangered', label='Start Decay Time')
    # plt.plot(stopIndicator, color='chartreuse', label='Stop Decay Time')
    # plt.legend(loc='lower right')
    # plt.show()

    return decayTime


# Calculating Euclidean Distance between adjacent points of signal. Used in sub-band flux calculation
def CalculateEuclideanDistance(signal):
    distance = 0
    for i in range (1, len(signal)):
        distance += pow(signal[i] - signal[i-1], 2)

    return math.sqrt(distance)


# Calculates sub-band spectral flux as defined in "Exploring perceptual and acoustical correlates of polyphonic timbre"
def CalculateSubBandSpectralFlux(args, samplingRate):
    # Filter into 10 sub-band octave filters
    if samplingRate < 22500: #You can't filter over the nyquist frequency
        freqLimit = 4000
        print("Sampling Rate too low for 10 octave bands, fs < 22450")
    elif samplingRate < 45000:
        freqLimit = 8000
        print("Sampling Rate too low for 10 octave bands, fs < 45000")
    else:
        freqLimit = 16000

    filterFreqs = obf.getFrequencies(30, freqLimit, 1)
    filters = obf.designFilters(filterFreqs, samplingRate)
    filteredSignal = obf.filterData(filters, args.impulseIRA.data)

    # Calculate spectral flux for each of the bands
    subBandFlux = []
    filteredAudio = args.impulseIRA # iracema feature extraction works on it's own class so we have to put the filtered audio into it
    for i in range (0, len(filteredSignal[0,:])):
        filteredAudio.data = filteredSignal[:,i]
        subBandFlux.append(CalculateEuclideanDistance(iracema.features.spectral_flux(iracema.spectral.fft(filteredAudio, window_size=2048, hop_size=1024)).data))
    return subBandFlux