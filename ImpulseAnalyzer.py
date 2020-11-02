import numpy as np
from scipy import signal
import os
import obspy
import obspy.signal
import obspy.signal.filter
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

def CalculateDecayTime(impulse, samplingRate, windowLength = 1000, ratio = 10):
    envelope = []
    decayTime = 0
    maxEnv = 0
    peakTime = 0
    for i in range(0,len(impulse)):

        energy = CalculateRMS(impulse[i:i+windowLength])
        if energy > maxEnv:
            maxEnv = energy
            peakTime = i

        if energy * ratio < maxEnv:
            decayTime = (i - peakTime)/samplingRate
            #print('Decay Time: ' + str(decayTime))
            break

    return decayTime

def run(inputDirectory, outputDirectory, dataFileName, fileNameAppendix, attackTime, sustainTime, decayTime_flag):

    if os.path.isdir(outputDirectory):
            shutil.rmtree(outputDirectory)
    os.mkdir(outputDirectory)

    samplingRate = 0
    #These variables are used to save the parameter data to a csv file
    data_array = []
    series_names, centroid_values, centroid_deviations, rolloff_values, rolloff_deviations, rms_values, rms_deviations, \
        bandwidth_values, bandwidth_deviation, decayTime_values, decayTime_deviations, attackRMS_values, attackRMS_deviations, sustainRMS_values, \
        sustainRMS_deviations, decayRMS_values, decayRMS_deviations, tuning_values, tuning_deviations = [" "], ["Spectrum Centroid"], ["Centroid Deviation"], \
                                                        ["Rolloff"], ["Rolloff Deviation"], ["RMS"], ["RMS Deviation"], \
                                                        ["Bandwidth"], ["Bandwidth Deviation"], ["Decay Time"], ["Decay Time Deviation"], ["Attack RMS"], \
                                                        ["Attack RMS Deviation"], ["Sustain RMS"], ["Sustain RMS Deviation"], \
                                                        ["Decay RMS"], ["Decay RMS Deviation"], ["Tuning"], ["Tuning Deviation"]
    allAttackSpectrums, allSustainSpectrums, allDecaySpectrums, allAttackFrequencies, allSustainFrequencies, \
            allDecayFrequencies, seriesNames = [], [], [], [], [], [], []
    impulseTime, maxAttack, maxSustain, maxDecay = 0, 0, 0, 0

    # Calculating spectrums and parameters
    for seriesDirectory in os.listdir(os.fsencode(inputDirectory)):
        seriesDirectory = inputDirectory + "/" + os.fsdecode(seriesDirectory)
        print("Entering folder: " + seriesDirectory)
        impulses, attackSpectrums, sustainSpectrums, decaySpectrums, centroids, rolloffs, rmss, bandwidths, decayTimes, tunings = [], [], [], [], [], [], [], [], [], []

        for impulseFile in os.listdir(os.fsencode(seriesDirectory)):
            impulseFileName = seriesDirectory + "/" + os.fsdecode(impulseFile)
            print("Loading file: " + impulseFileName)
            impulse, samplingRate = librosa.load(impulseFileName)

            centroids.append(np.mean(librosa.feature.spectral_centroid(impulse)))
            rolloffs.append(np.mean(librosa.feature.spectral_rolloff(impulse)))
            bandwidths.append(np.mean(librosa.feature.spectral_bandwidth((impulse))))
            rmss.append(CalculateRMS(impulse))
            tunings.append(np.mean(librosa.estimate_tuning(impulse)))
            if decayTime_flag:
                decayTimes.append(CalculateDecayTime(impulse, samplingRate))
            else:
                decayTimes.append(0)

            impulses = InsertIntoVstack(impulse, impulses)

        attackFrequencies, attackSpectrums, sustainFrequencies, sustainSpectrums, decayFrequencies, decaySpectrums = \
            CalculateFFTs(impulses, samplingRate, attackTime, sustainTime)

        avrAttackSpectrum = CalculateAverageVector(attackSpectrums)
        avrSustainSpectrum = CalculateAverageVector(sustainSpectrums)
        avrDecaySpectrum = CalculateAverageVector(decaySpectrums)

        allAttackSpectrums.append(avrAttackSpectrum)
        allSustainSpectrums.append(avrSustainSpectrum)
        allDecaySpectrums.append(avrDecaySpectrum)
        allAttackFrequencies.append(attackFrequencies)
        allSustainFrequencies.append(sustainFrequencies)
        allDecayFrequencies.append(decayFrequencies)
        seriesNames.append(seriesDirectory)
        impulseTime = len(impulses[0,:])/samplingRate
        maxAttack = max([maxAttack, max(avrAttackSpectrum)])
        maxSustain = max([maxSustain, max(avrSustainSpectrum)])
        maxDecay = max([maxDecay, max(avrDecaySpectrum)])

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
        decayTime_values.append(np.mean(decayTimes))
        decayTime_deviations.append(np.std(decayTimes))
        tuning_values.append(np.mean(tunings))
        tuning_deviations.append(np.std(tunings))

        attackRMS_values.append(CalculateRMS(avrAttackSpectrum))
        attackRMS_deviations.append(0)
        sustainRMS_values.append(CalculateRMS(avrSustainSpectrum))
        sustainRMS_deviations.append(0)
        decayRMS_values.append(CalculateRMS(avrDecaySpectrum))
        decayRMS_deviations.append(0)

    # Saving the parameter data
    data_array = np.vstack(
        [series_names, centroid_values, centroid_deviations, rolloff_values, rolloff_deviations, rms_values,
         rms_deviations, bandwidth_values, bandwidth_deviation, decayTime_values, decayTime_deviations, attackRMS_values, attackRMS_deviations,
         sustainRMS_values, sustainRMS_deviations, decayRMS_values, decayRMS_deviations, tuning_values, tuning_deviations])

    np.save(outputDirectory + '/' + dataFileName + '_' + fileNameAppendix + '.npy', data_array)
    with open(outputDirectory + '/' + dataFileName + '_' + fileNameAppendix + '.csv', 'w', newline='') as csvfile:
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
        dataWriter.writerow(decayTime_values)
        dataWriter.writerow(decayTime_deviations)
        dataWriter.writerow(attackRMS_values)
        dataWriter.writerow(attackRMS_deviations)
        dataWriter.writerow(sustainRMS_values)
        dataWriter.writerow(sustainRMS_deviations)
        dataWriter.writerow(decayRMS_values)
        dataWriter.writerow(decayRMS_deviations)
        dataWriter.writerow(tuning_values)
        dataWriter.writerow(tuning_deviations)

    print("Data saved to: " + dataFileName + '_' + fileNameAppendix)

    # Drawing spectrum plots
    for iterator in range(0, len(seriesNames)):

        #plt.suptitle(seriesNames[iterator].replace(inputDirectory, ''), fontsize='xx-large')

        plt.subplot(131)
        plt.plot(allAttackFrequencies[iterator], allAttackSpectrums[iterator])
        plt.xlim([0, 3000])
        plt.ylim([0, maxAttack * 1.02])
        plt.xlabel('Częstotliwość [Hz]', fontsize='xx-large')
        plt.xticks(fontsize='x-large')
        plt.ylabel('Amplituda [jednostka arbitralna]', fontsize='xx-large')
        plt.title('0 - ' + str(attackTime) + ' [s]', fontsize='xx-large')
        #plt.text(1500, maxAttack, str('Energy = ' + str(CalculateRMS(allAttackSpectrums[iterator]))))

        plt.subplot(132)
        plt.plot(allSustainFrequencies[iterator], allSustainSpectrums[iterator])
        plt.xlim([0, 3000])
        plt.ylim([0, maxSustain * 1.02])
        plt.xlabel('Częstotliwość [Hz]', fontsize='xx-large')
        plt.xticks(fontsize='x-large')
        plt.title(str(attackTime) + ' - ' + str(sustainTime) + ' [s]', fontsize='xx-large')
        #plt.text(1500, maxSustain, str('Energy = ' + str(CalculateRMS(allSustainSpectrums[iterator]))))

        plt.subplot(133)
        plt.plot(allDecayFrequencies[iterator], allDecaySpectrums[iterator])
        plt.xlim([0, 3000])
        plt.ylim([0, maxDecay * 1.02])
        plt.xlabel('Częstotliwość [Hz]', fontsize='xx-large')
        plt.xticks(fontsize='x-large')
        plt.title(str(sustainTime) + ' - ' + str(round(impulseTime, 2)) + ' [s]', fontsize='xx-large')
        #plt.text(1500, maxDecay, str('Energy = ' + str(CalculateRMS(allDecaySpectrums[iterator]))))


        outputFile = seriesNames[iterator].replace(inputDirectory, outputDirectory)
        print("Outputing to: " + outputFile)

        figure = plt.gcf()
        figure.set_size_inches(17.28, 9.72)
        plt.savefig(outputFile, dpi = 100)
        #plt.show()
        plt.clf()








