import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
import librosa
import iracema

# This file implements all function and calculations that are needed in the process of analysing impulses

class Harmonics:
    def __init__(self, frequencies, amplitudes):
        self.frequencies = frequencies
        self.amplitudes = amplitudes

def InsertIntoVstack(vector, stack):
    if stack == []:  # Not very elegant way to make sure the first impulse is loaded in correctly
        stack = vector
    else:
        stack = np.vstack([stack, vector])

    return stack

def CalculateAverageVector(Vectors): # Takes a vector of vectors and calculates a average vector
    averageVector = np.mean(Vectors, axis=0)
    return averageVector

def CalculateFFTs(takes, samplingRate, attackTime, sustainTime): # Takes array of impulses and creates array of spectrums
    #No idea why thid is needed but it seems to be needed. If harmonics don't work correctly start here!
    samplingRate = samplingRate/2
    attackSpectrums, sustainSpectrums, decaySpectrums, fullSpectrums = [], [], [], []
    attackFrequencies, sustainFrequencies, decayFrequencies, fullFrequencies = 0, 0, 0, 0
    for take in takes:
        fullFrequencies, fullSpectrum = signal.periodogram(take, samplingRate, scaling="spectrum")
        attackFrequencies, attackSpectrum = signal.periodogram(take[:int(attackTime*samplingRate)], samplingRate, scaling="spectrum")
        sustainFrequencies, sustainSpectrum = signal.periodogram(take[int(attackTime*samplingRate):int(sustainTime*samplingRate)], samplingRate, scaling="spectrum")
        decayFrequencies, decaySpectrum = signal.periodogram(take[int(sustainTime*samplingRate):], samplingRate, scaling="spectrum")

        fullSpectrums = InsertIntoVstack(fullSpectrum.real, fullSpectrums)
        attackSpectrums = InsertIntoVstack(attackSpectrum.real, attackSpectrums)
        sustainSpectrums = InsertIntoVstack(sustainSpectrum.real, sustainSpectrums)
        decaySpectrums = InsertIntoVstack(decaySpectrum.real, decaySpectrums)

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
                peakValue, peakFrequency = 0, 0

                for i in range (fftSample-buffor, fftSample+buffor):
                    if spectrum[i] >= peakValue:
                        peakValue = spectrum[i]
                        peakFrequency = spectrumFrequencies[i]

                amplitudes.append(peakValue)
                harmonicFrequencies.append(peakFrequency)

                if harmonicNumber < len(mathHarmonicFrequencies) - 1:
                    harmonicNumber = harmonicNumber + 1
                else:
                    break

        # showing found harmonics and spectrum for debugging
        '''
        print(amplitudes)
        print(harmonicFrequencies)

        x = []
        for y in range(0, len(harmonicFrequencies)):
            x.append(y)
        plt.subplot(121)
        plt.plot(spectrumFrequencies, spectrum)
        plt.xlim([0,10000])
        plt.subplot(122)
        plt.bar(x, amplitudes)
        plt.xlim([0, 30])
        plt.show()
        '''

        harmonicData.append(Harmonics(harmonicFrequencies, amplitudes))
    return harmonicData

def CalculateNoisiness(spectrums, frequencies, harmonicsData, harmonicWidth = 0.5):
    noisinesses = []
    for takeNumber in range (0, len(spectrums)): #Dla każdego dźwięku
        fullEnergy, harmonicEnergy = 0, 0

        for sample in spectrums[takeNumber]:
            fullEnergy += pow(sample, 2)
        for harmonicFrequency in harmonicsData[takeNumber].frequencies:
            for frequency in range (0, len(frequencies)):
                if frequencies[frequency] >= harmonicFrequency-harmonicWidth and frequencies[frequency] <= harmonicFrequency+harmonicWidth:
                    harmonicEnergy += pow(spectrums[takeNumber][frequency], 2)

        noisinesses.append((fullEnergy-harmonicEnergy)/fullEnergy)
    return noisinesses

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
        tristimulus2s.append((take.amplitudes[1] + take.amplitudes[2] + take.amplitudes[3])/allAmplitudes)
        tristimulus3s.append(fiveUpAmplitudes/allAmplitudes)
    return tristimulus1s, tristimulus2s, tristimulus3s

def CalculateInharmonicity(harmonicsData):
    inharmonicities = []
    for take in harmonicsData:
        fundumentalPitch = EstimateFundamentalPitch(take.frequencies)
        Inharmonicity, allAmplitudes = 0, 0
        for i in range(0,len(take.frequencies)):
            Inharmonicity += (abs(take.frequencies[i]-fundumentalPitch*(i+1))*pow(take.amplitudes[i], 2))
            allAmplitudes += pow(take.amplitudes[i], 2)
        inharmonicities.append((2*Inharmonicity)/(fundumentalPitch*allAmplitudes))
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

def CalculateRMS(args):
    return np.sum(librosa.feature.rmse(args.impulseLIB))

#Calculates the signals temporal centroid. Only takes into account signal over threshold to disguard silence.
# Watch out when using signals of different lengths.

def CalculateTemporalCentroid(args, windowLength = 2048, hopsize = 1024, threshold = 0.1):
    envelope = iracema.features.peak_envelope(args.impulseIRA, windowLength, hopsize)
    maxEnv = max(envelope.data)
    amplitudeSum = sum(envelope.data)
    ampXTimeSum = 0

    for i in range(0, len(envelope)):
        if envelope.data[i] > (maxEnv * threshold):
            ampXTimeSum += envelope.time[i] * envelope.data[i]
    return (ampXTimeSum/amplitudeSum)

# Calculate log of attack time of signal. The algorythm was simplified when it comes to finding the start time of attack due to the it giving better results for guitar
def CalculateLogAttackTime(args, windowLength = 256, hopsize = 128, threshold = 0.15):
    envelope = iracema.features.peak_envelope(args.impulseIRA, windowLength, hopsize)
    maxEnv = max(envelope.data)
    startTime, stopTime = 0, 0

    for i in range(0, len(envelope)):
        if envelope.data[i] > (maxEnv * threshold):
            startTime = envelope.time[i]
            break
    for i in range(0, len(envelope)):
        if envelope.data[i] == maxEnv:
            stopTime = envelope.time[i]
            break
    if stopTime != startTime:
        return math.log10(stopTime - startTime)
    else:
        return 0

# Calculates time between the peak of impulse and it decaying below the value of max*ratio
def CalculateDecayTime(args, windowLength = 2048, hopsize = 1024, ratio = 0.1):
    envelope = iracema.features.peak_envelope(args.impulseIRA, windowLength, hopsize)
    maxEnv = max(envelope.data)

    plt.plot(envelope.data)
    #plt.show()

    peakTime = 0
    decayTime = 0
    flag = False

    for i in range(0,len(envelope.data)):
        if envelope.data[i] == maxEnv:
            peakTime = envelope.time[i]
            flag = True

        if (maxEnv * ratio > envelope.data[i]) & flag:
            decayTime = envelope.time[i] - peakTime
            break
    return decayTime