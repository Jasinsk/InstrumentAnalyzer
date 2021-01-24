import numpy as np
from scipy import signal
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

    return fullSpectrums, fullFrequencies, attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums

def CreateMathematicalHarmonicFrequencyVector(pitch, n):
    freq = []
    for i in range (1, n):
        freq.append(pitch * i)
    return freq

def EstimateFundamentalPitch(harmonicFrequencies):
    fundamentalPitches = []
    for i in range (0, len(harmonicFrequencies)):
        fundamentalPitches.append(harmonicFrequencies[i]/(i+1))
    return np.mean(fundamentalPitches)

def ExtractHarmonicDataFromSpectrums(spectrums, spectrumFrequencies, mathHarmonicFrequencies, bufforInHZ = 5):
    buffor = int(bufforInHZ/(spectrumFrequencies[1]-spectrumFrequencies[0]))
    harmonicData = []

    for spectrum in spectrums:
        amplitudes = []
        harmonicFrequencies = []
        harmonicNumber = 0

        for fftSample in range (0, len(spectrum)):
            if spectrumFrequencies[fftSample] > mathHarmonicFrequencies[harmonicNumber]:
                peakValue = 0
                peakFrequency = 0

                for i in range (fftSample-buffor, fftSample+buffor):
                    if spectrum[i] > peakValue:
                        peakValue = spectrum[i]
                        peakFrequency = spectrumFrequencies[i]

                amplitudes.append(peakValue)
                harmonicFrequencies.append(peakFrequency)

                if harmonicNumber < len(mathHarmonicFrequencies) - 1:
                    harmonicNumber = harmonicNumber + 1
                else:
                    break

        # showing found harmonics and spectrum for debugging
        if False:
            x = []
            for y in range(0, len(harmonicFrequencies)):
                x.append(y)
            plt.subplot(121)
            plt.plot(spectrumFrequencies, spectrum)
            plt.subplot(122)
            plt.bar(x, amplitudes)
            plt.show()

        harmonicData.append(Harmonics(harmonicFrequencies, amplitudes))
    return harmonicData

def CalculateRMS(signal):
    return np.sum(librosa.feature.rmse(signal))

def CalculateDecayTime(impulse, windowLength = 2048, hopsize = 1024, ratio = 0.12): # Calculates time between the peak of impulse and it decaying below the value of max*ratio
    envelope = iracema.features.peak_envelope(impulse, windowLength, hopsize)
    maxEnv = max(envelope.data)
    peakTime = 0
    decayTime = 0

    for i in range(0,len(envelope.data)):
        if envelope.data[i] == maxEnv:
            peakTime = envelope.time[i]
        if maxEnv * ratio > envelope.data[i]:
            decayTime = envelope.time[i] - peakTime
            break
    return decayTime

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
        topInharmonicity = 0
        allAmplitudes = 0
        for i in range(0,len(take.frequencies)):
            topInharmonicity += (abs(take.frequencies[i]-fundumentalPitch*(i+1))*pow(take.amplitudes[i], 2))
            allAmplitudes += pow(take.amplitudes[i], 2)
        inharmonicities.append((2*topInharmonicity)/(fundumentalPitch*allAmplitudes))
    return inharmonicities
