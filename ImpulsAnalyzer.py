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

def CalculateFFT(takes, samplingRate): #Takes array of impulses and creates array of spectrums

    spectrums = []
    frequencies = 0
    for take in takes:
        frequencies, spectrum = signal.periodogram(take, samplingRate, scaling="spectrum")
        if spectrums == []: #Not very elegant way to make sure the first impulse is loaded in correctly
            spectrums = spectrum.real
        else:
            spectrums = np.vstack([spectrums, spectrum.real])

    return frequencies, spectrums


inputDirectory = "ParserOutputFolder"
samplingRate = 1

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

    frequencies, spectrums = CalculateFFT(impulses, samplingRate)
    avrSpectrum = CalculateAverageVector(spectrums)

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

    plt.plot(frequencies, avrSpectrum)
    plt.title(seriesDirectory)
    #plt.ylim([0.5e-3, 1])
    plt.xlim([0, 3000])
    plt.xlabel('frequency [Hz]')
    plt.savefig(seriesDirectory)
    plt.show()






