import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import math
import shutil
import csv

# This script accepts folders with individually parsed impulses and calculates an averaged spectrum for each directory.
# The plots of these results are saved into the output directory.
# Spectral and energy parameters are also calculated and saved into the output directory.

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

def CalculateRMS(signal):
    return np.sum(librosa.feature.rmse(signal))


inputDirectory = "ParserOutputFolder"
outputDirectory = "AnalyzerOutputFolder"
dataFileName = "ParameterData"

if os.path.isdir(outputDirectory):
        shutil.rmtree(outputDirectory)
os.mkdir(outputDirectory)

samplingRate = 0
# The impulse is cut into three parts, beginning-attackTime, attackTime-sustainTime, sustainTime-end
attackTime = 0.3
sustainTime = 1.3

#These variables are used to save the parameter data to a csv file
data_array = []
series_names, centroid_values, centroid_deviations, rolloff_values, rolloff_deviations, rms_values, rms_deviations, \
    bandwidth_values, bandwidth_deviation, attackRMS_values, attackRMS_deviations, sustainRMS_values, \
    sustainRMS_deviations, decayRMS_values, decayRMS_deviations = [" "], ["Spectrum Centroid"], ["Centroid Deviation"], \
                                                    ["Rolloff"], ["Rolloff Deviation"], ["RMS"], ["RMS Deviation"], \
                                                    ["Bandwidth"], ["Bandwidth Deviation"], ["Attack RMS"], \
                                                    ["Attack RMS Deviation"], ["Sustain RMS"], ["Sustain RMS Deviation"], \
                                                    ["Decay RMS"], ["Decay RMS Deviation"]



#These are used as a y axis limits in the final spectrums to have them all have the same axis limits for easier comparison
useLimits = True
attackMaxValue = 0.007
sustainMaxValue = 0.00085
decayMaxValue = 1.5e-05

for seriesDirectory in os.listdir(os.fsencode(inputDirectory)):
    seriesDirectory = inputDirectory + "/" + os.fsdecode(seriesDirectory)
    print("Entering folder: " + seriesDirectory)
    impulses, centroids, rolloffs, rmss, bandwidths = [], [], [], [], []

    for impulseFile in os.listdir(os.fsencode(seriesDirectory)):
        impulseFileName = seriesDirectory + "/" + os.fsdecode(impulseFile)
        print("Loading file: " + impulseFileName)
        impulse, samplingRate = librosa.load(impulseFileName)

        centroids.append(np.mean(librosa.feature.spectral_centroid(impulse)))
        rolloffs.append(np.mean(librosa.feature.spectral_rolloff(impulse)))
        bandwidths.append(np.mean(librosa.feature.spectral_bandwidth((impulse))))
        rmss.append(CalculateRMS(impulse))

        impulses = InsertIntoVstack(impulse, impulses)

    attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums = \
        CalculateFFTs(impulses, samplingRate, attackTime, sustainTime)
    avrAttackSpectrum = CalculateAverageVector(attackSpectrums)
    avrSustainSpectrum = CalculateAverageVector(sustainSpectrums)
    avrDecaySpectrum = CalculateAverageVector(decaySpectrums)

    seriesName = seriesDirectory.replace(inputDirectory + "/", "")
    series_names.append(seriesName)
    centroid_values.append(np.mean(centroids))
    centroid_deviations.append(np.std(centroids))
    rolloff_values.append(np.mean(rolloffs))
    rolloff_deviations.append(np.std(rolloffs))
    bandwidth_values.append(np.mean(bandwidths))
    bandwidth_deviation.append((np.std(bandwidths)))
    rms_values.append(np.mean(rmss))
    rms_deviations.append(np.std(rmss))

    ylimit = max(avrAttackSpectrum)

    plt.suptitle(seriesDirectory.replace(inputDirectory + '/', ''), fontsize='xx-large')

    if useLimits == True:
        attackYLimit = attackMaxValue * 1.05
        sustainYLimit = sustainMaxValue * 1.05
        decayYLimit = decayMaxValue * 1.05

    else:
        attackYLimit = max(avrAttackSpectrum) * 1.05
        sustainYLimit = max(avrSustainSpectrum) * 1.05
        decayYLimit = max(avrDecaySpectrum) * 1.05

    attackMaxValue = max(attackMaxValue, max(avrAttackSpectrum))
    sustainMaxValue = max(sustainMaxValue, max(avrSustainSpectrum))
    decayMaxValue = max(decayMaxValue, max(avrDecaySpectrum))

    attackRMS_values.append(CalculateRMS(avrAttackSpectrum))
    attackRMS_deviations.append(0)
    sustainRMS_values.append(CalculateRMS(avrSustainSpectrum))
    sustainRMS_deviations.append(0)
    decayRMS_values.append(CalculateRMS(avrDecaySpectrum))
    decayRMS_deviations.append(0)

    plt.subplot(131)
    plt.plot(attackFrequencies, avrAttackSpectrum)
    plt.xlim([0, 3000])
    plt.ylim([0,attackYLimit])
    plt.xlabel('frequency [Hz]')
    plt.title('0 - ' + str(attackTime) + ' [s]')
    plt.text(1500, attackYLimit*0.95, str('Energy = ' + str(CalculateRMS(avrAttackSpectrum))))

    plt.subplot(132)
    plt.plot(sustainFrequencies, avrSustainSpectrum)
    plt.title(seriesDirectory)
    plt.xlim([0, 3000])
    plt.ylim([0, sustainYLimit])
    plt.xlabel('frequency [Hz]')
    plt.title(str(attackTime) + ' - ' + str(sustainTime) + ' [s]')
    plt.text(1500, sustainYLimit * 0.95, str('Energy = ' + str(CalculateRMS(avrSustainSpectrum))))

    plt.subplot(133)
    plt.plot(decayFrequencies, avrDecaySpectrum)
    plt.xlim([0, 3000])
    plt.ylim([0, decayYLimit])
    plt.xlabel('frequency [Hz]')
    plt.title(str(sustainTime) + ' - ' + str(round(len(impulses[0,:])/samplingRate, 2)) + ' [s]')
    plt.text(1500, decayYLimit * 0.95, str('Energy = ' + str(CalculateRMS(avrDecaySpectrum))))


    outputFile = seriesDirectory.replace(inputDirectory, outputDirectory)
    print("Outputing to: " + outputFile)

    figure = plt.gcf()
    figure.set_size_inches(19.2, 10.8)
    plt.savefig(outputFile, dpi = 100)
    #plt.show()
    plt.clf()

data_array = np.vstack([series_names, centroid_values, centroid_deviations, rolloff_values, rolloff_deviations, rms_values,
                        rms_deviations, bandwidth_values, bandwidth_deviation, attackRMS_values, attackRMS_deviations,
                        sustainRMS_values, sustainRMS_deviations, decayRMS_values, decayRMS_deviations])

np.save(outputDirectory + '/' + dataFileName + '.npy', data_array)
with open(outputDirectory + '/' + dataFileName + '.csv', 'w', newline='') as csvfile:
    dataWriter = csv.writer(csvfile, delimiter=',', quotechar=';', quoting=csv.QUOTE_MINIMAL)
    dataWriter.writerow(series_names)
    dataWriter.writerow(centroid_values)
    dataWriter.writerow(centroid_deviations)
    dataWriter.writerow(rms_values)
    dataWriter.writerow(rms_deviations)
    dataWriter.writerow(rolloff_values)
    dataWriter.writerow(rolloff_deviations)
    dataWriter.writerow(bandwidth_values)
    dataWriter.writerow(bandwidth_deviation)
    dataWriter.writerow(attackRMS_values)
    dataWriter.writerow(attackRMS_deviations)
    dataWriter.writerow(sustainRMS_values)
    dataWriter.writerow(sustainRMS_deviations)
    dataWriter.writerow(decayRMS_values)
    dataWriter.writerow(decayRMS_deviations)

print("Data saved to: " + dataFileName)

print("Max Attack Value: " + str(attackMaxValue))
print("Max Sustain Value: " + str(sustainMaxValue))
print("Max Decay Value: " + str(decayMaxValue))








