import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math
import shutil
import csv

def InsertIntoVstack(vector, stack):
    if stack == []:  # Not very elegant way to make sure the first impulse is loaded in correctly
        stack = vector
    else:
        stack = np.vstack([stack, vector])

    return stack

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

        attackSpectrums = InsertIntoVstack(attackSpectrum.real, attackSpectrums)
        sustainSpectrums = InsertIntoVstack(sustainSpectrum.real, sustainSpectrums)
        decaySpectrums = InsertIntoVstack(decaySpectrum.real, decaySpectrums)

    return attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums


inputDirectory = "ParserOutputFolder"
outputDirectory = "AnalyzerOutputFolder"

if os.path.isdir(outputDirectory):
        shutil.rmtree(outputDirectory)
os.mkdir(outputDirectory)

samplingRate = 0
# The impulse is cut into three parts, beginning-attackTime, attackTime-sustainTime, sustainTime-end
attackTime = 0.3
sustainTime = 1.3

#These variables are used to save the parameter data to a csv file
data_array = []
series_names, centroid_values, centroid_deviations, rolloff_values, rolloff_deviations, rms_values, rms_deviations = [" "], ["Spectrum Centroid"], ["Centroid Deviation"], ["Rolloff"], ["Rolloff Deviation"], ["RMS"], ["RMS Deviation"]

for seriesDirectory in os.listdir(os.fsencode(inputDirectory)):
    seriesDirectory = inputDirectory + "/" + os.fsdecode(seriesDirectory)
    print("Entering folder: " + seriesDirectory)
    impulses, centroids, rolloffs, rmss = [], [], [], []

    for impulseFile in os.listdir(os.fsencode(seriesDirectory)):
        impulseFileName = seriesDirectory + "/" + os.fsdecode(impulseFile)
        print("Loading file: " + impulseFileName)
        impulse, samplingRate = librosa.load(impulseFileName)

        centroids.append(np.mean(librosa.feature.spectral_centroid(impulse)))
        rolloffs.append(np.mean(librosa.feature.spectral_rolloff(impulse)))
        rmss.append(np.mean(librosa.feature.rmse(impulse)))

        impulses = InsertIntoVstack(impulse, impulses)

    attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums = CalculateFFTs(impulses, samplingRate, attackTime, sustainTime)
    avrAttackSpectrum = CalculateAverageVector(attackSpectrums)
    avrSustainSpectrum = CalculateAverageVector(sustainSpectrums)
    avrDecaySpectrum = CalculateAverageVector(decaySpectrums)

    seriesName = seriesDirectory.replace(inputDirectory + "/", "")
    series_names.append(seriesName)
    centroid_values.append(np.mean(centroids))
    centroid_deviations.append(np.std(centroids))
    rolloff_values.append(np.mean(rolloffs))
    rolloff_deviations.append(np.std(rolloffs))
    rms_values.append(np.mean(rmss))
    rms_deviations.append(np.std(rmss))

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

    outputFile = seriesDirectory.replace(inputDirectory, outputDirectory)
    print(outputFile)
    #if os.path.isfile(seriesDirectory + ".png"):
    #    os.remove(seriesDirectory + ".png")
    plt.savefig(outputFile)
    #plt.show()

data_array = np.vstack([series_names, centroid_values, centroid_deviations, rolloff_values, rolloff_deviations, rms_values, rms_deviations])
np.save('data.npy', data_array)

with open('data.csv', 'w', newline='') as csvfile:
    dataWriter = csv.writer(csvfile, delimiter=',', quotechar=';', quoting=csv.QUOTE_MINIMAL)
    dataWriter.writerow(series_names)
    dataWriter.writerow(centroid_values)
    dataWriter.writerow(centroid_deviations)
    dataWriter.writerow(rms_values)
    dataWriter.writerow(rms_deviations)
    dataWriter.writerow(rolloff_values)
    dataWriter.writerow(rolloff_deviations)







