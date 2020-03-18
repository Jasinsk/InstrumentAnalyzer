import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math

def CalculateAverageVector(Vectors): #Takes a vector of vectors and calculates a average vector
    averageVector = np.mean(Vectors, axis=0)
    return averageVector

def CalculateFFTs(takes, samplingRate, attackTime, sustainTime): #Takes array of impulses and creates array of spectrums

    attackSpectrums, sustainSpectrums, decaySpectrums = [], [], []
    attackFrequencies, sustainFrequencies, decayFrequencies = 0, 0, 0
    for take in takes:
        attackFrequencies, attackSpectrum = signal.periodogram(take[:int(attackTime*samplingRate)], samplingRate, scaling="spectrum")
        sustainFrequencies, sustainSpectrum = signal.periodogram(take[int(attackTime*samplingRate):int(sustainTime*samplingRate)], samplingRate, scaling="spectrum")
        decayFrequencies, decaySpectrum = signal.periodogram(take[int(sustainTime*samplingRate):], samplingRate, scaling="spectrum")

        if attackSpectrums == []: #Not very elegant way to make sure the first impulse is loaded in correctly
            attackSpectrums = attackSpectrum.real
        else:
            attackSpectrums = np.vstack([attackSpectrums, attackSpectrum.real])
        if sustainSpectrums == []: #Not very elegant way to make sure the first impulse is loaded in correctly
            sustainSpectrums = sustainSpectrum.real
        else:
            sustainSpectrums = np.vstack([sustainSpectrums, sustainSpectrum.real])
        if decaySpectrums == []: #Not very elegant way to make sure the first impulse is loaded in correctly
            decaySpectrums = decaySpectrum.real
        else:
            decaySpectrums = np.vstack([decaySpectrums, decaySpectrum.real])

    return attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums


inputDirectory = "ParserOutputFolder"
samplingRate = 1

attackTime = 0.3
sustainTime = 1.3

for seriesDirectory in os.listdir(os.fsencode(inputDirectory)):
    seriesDirectory = inputDirectory + "/" + os.fsdecode(seriesDirectory)
    print("Entering folder: " + seriesDirectory)
    impulses = []
    centroids = []
    rolloffs = []
    rmss = []

    for impulseFile in os.listdir(os.fsencode(seriesDirectory)):
        impulseFileName = seriesDirectory + "/" + os.fsdecode(impulseFile)
        print("Loading file: " + impulseFileName)
        impulse, samplingRate = librosa.load(impulseFileName)

        centroids.append(np.mean(librosa.feature.spectral_centroid(impulse)))
        rolloffs.append(np.mean(librosa.feature.spectral_rolloff(impulse)))
        rmss.append(np.mean(librosa.feature.rmse(impulse)))

        if impulses == []: #Not very elegant way to make sure the first impulse is loaded in correctly
            impulses = impulse
        else:
            impulses = np.vstack([impulses, impulse])

    #We have a working system that throws into a single data object all the tries for one situation
    #Now we should try to calculate the required parameters from each try.

    attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums = CalculateFFTs(impulses, samplingRate, attackTime, sustainTime)
    avrAttackSpectrum = CalculateAverageVector(attackSpectrums)
    avrSustainSpectrum = CalculateAverageVector(sustainSpectrums)
    avrDecaySpectrum = CalculateAverageVector(decaySpectrums)

    print("Spectral centroids")
    print("mean: ")
    print(np.mean(centroids))
    print("stdev: ")
    print(np.std(centroids))
    print("Spectral rolloff")
    print("mean: ")
    print(np.mean(rolloffs))
    print("stdev: ")
    print(np.std(rolloffs))
    print("RMS")
    print("mean: ")
    print(np.mean(rmss))
    print("stdev: ")
    print(np.std(rmss))

    plt.subplot(311)
    plt.plot(attackFrequencies, avrAttackSpectrum)
    plt.title(seriesDirectory)
    plt.xlim([0, 3000])
    plt.xlabel('frequency [Hz]')

    plt.subplot(312)
    plt.plot(sustainFrequencies, avrSustainSpectrum)
    plt.xlim([0, 3000])
    plt.xlabel('frequency [Hz]')

    plt.subplot(313)
    plt.plot(decayFrequencies, avrDecaySpectrum)
    plt.xlim([0, 3000])
    plt.xlabel('frequency [Hz]')

    plt.savefig(seriesDirectory)
    plt.show()






